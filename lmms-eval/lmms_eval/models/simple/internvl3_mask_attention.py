"""
File deprecated since it does not work with flash attention and removing flash attention seems to not work. New implementation only done with hf. 
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import torch
from tqdm import tqdm

from lmms_eval.api.instance import Instance
from lmms_eval.api.registry import register_model
from lmms_eval.models.simple.internvl3 import (
    DEFAULT_GEN_KWARGS,
    InternVL3,
    load_image,
    load_video,
)


eval_logger = logging.getLogger("lmms-eval")


@register_model("internvl3_mask_attention")
class InternVL3_MaskAttention(InternVL3):
    """InternVL3 wrapper with optional cross-image attention masking.

    The implementation mirrors InternVL3 behavior when ``mask_layers`` is unset.
    When enabled, selected language-model attention layers receive a pre-forward
    hook that blocks token-to-token attention across different image token blocks.
    """

    def __init__(self, mask_layers: Optional[str] = None, **kwargs):
        mask_layers_str = None if (mask_layers is None or mask_layers == "baseline") else str(mask_layers)
        if mask_layers_str:
            kwargs["use_flash_attn"] = False 

        super().__init__(**kwargs)
        self.mask_layers = mask_layers_str

        self.hooks: List[torch.utils.hooks.RemovableHandle] = []
        self._phase_hook: Optional[torch.utils.hooks.RemovableHandle] = None
        self.mask_layers_indices: List[int] = []
        self._original_attn_impl: Optional[str] = None
        self._last_attn_impl: Optional[str] = None

        # Set before generation, consumed by hooks.
        self.image_token_indices: Optional[List[List[Dict[str, int]]]] = []
        self._pending_num_patches_list: Optional[List[int]] = None
        self._cached_mask: Optional[torch.Tensor] = None

        # save the delimiter tokens for the images to be used to find the image tokens
        self.vision_start_id = self.tokenizer.convert_tokens_to_ids("<img>")
        self.vision_end_id = self.tokenizer.convert_tokens_to_ids("</img>")
        self.vision_context_id = self.tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
        
        if self.mask_layers:
            self._setup_hybrid_attn_switching()
            self._parse_mask_layers()
            self._register_attention_hooks()


    def _setup_hybrid_attn_switching(self) -> None:
        """Switch LM attention backend by phase: prefill->eager, decode->flash_attention_2."""
        language_model = getattr(self.model, "language_model", None)
        if language_model is None or not hasattr(language_model, "config"):
            eval_logger.warning("[InternVL3_MaskAttention] Could not find language_model config for hybrid attention switching.")
            return

        current_impl = getattr(language_model.config, "_attn_implementation", None)
        self._original_attn_impl = current_impl
        self._last_attn_impl = current_impl

        if current_impl != "flash_attention_2":
            eval_logger.warning(
                f"[InternVL3_MaskAttention] Base attention implementation is '{current_impl}', "
                "hybrid flash/eager switching will be limited."
            )

        def _phase_hook(module, args, kwargs):
            # If there are no image patches in this sample, keep original backend.
            if not self._pending_num_patches_list:
                target_impl = self._original_attn_impl
            else:
                # Resolve q_len from kwargs/args to detect prefill vs decode.
                input_ids = kwargs.get("input_ids", args[0] if len(args) > 0 else None)
                inputs_embeds = kwargs.get("inputs_embeds", args[2] if len(args) > 2 else None)

                q_len = None
                if inputs_embeds is not None and hasattr(inputs_embeds, "shape"):
                    q_len = int(inputs_embeds.shape[1])
                elif input_ids is not None and hasattr(input_ids, "shape"):
                    q_len = int(input_ids.shape[1])

                if q_len is None:
                    target_impl = self._original_attn_impl
                elif q_len > 1:
                    # Prefill: force eager so additive 4D masks are supported.
                    target_impl = "eager"
                else:
                    # Decode: restore original (typically flash_attention_2).
                    target_impl = self._original_attn_impl

            if target_impl and getattr(module.config, "_attn_implementation", None) != target_impl:
                module.config._attn_implementation = target_impl
                if self._last_attn_impl != target_impl:
                    eval_logger.info(f"[InternVL3_MaskAttention] Switched attn implementation to '{target_impl}'.")
                    self._last_attn_impl = target_impl

            return args, kwargs

        self._phase_hook = language_model.register_forward_pre_hook(_phase_hook, with_kwargs=True)

    def _parse_mask_layers(self) -> None:
        layers = self._resolve_decoder_layers()
        if layers is None:
            raise RuntimeError("Could not find decoder layers in InternVL3 model for mask hooks.")

        if self.mask_layers == "all":
            self.mask_layers_indices = list(range(len(layers)))
            return

        indices = set()
        # Split by '+' to separate groups (e.g., "1-3+5" -> ["1-3", "5"])
        segments = self.mask_layers.split("+")
        for segment in segments:
            segment = segment.strip()
            if not segment:
                continue
            if "-" in segment:
                try:
                    start, end = map(int, segment.split("-", 1))
                    if start > end:
                        start, end = end, start
                    indices.update(range(start, end + 1))
                except ValueError:
                    eval_logger.warning(f"[InternVL3_MaskAttention] Invalid mask layer range: {segment}")
            else:
                try:
                    indices.add(int(segment))
                except ValueError:
                    eval_logger.warning(f"[InternVL3_MaskAttention] Invalid mask layer index: {segment}")

        self.mask_layers_indices = sorted(idx for idx in indices if idx >= 0)

    def _resolve_decoder_layers(self):
        """Best-effort lookup for the decoder layer stack."""
        candidates = [
            getattr(self.model, "language_model", None),
            getattr(self.model, "model", None),
            self.model,
        ]

        for root in candidates:
            if root is None:
                continue

            paths = [
                getattr(root, "layers", None),
                getattr(getattr(root, "model", None), "layers", None),
                getattr(getattr(root, "decoder", None), "layers", None),
                getattr(getattr(getattr(root, "model", None), "decoder", None), "layers", None),
            ]
            for layers in paths:
                if layers is not None and len(layers) > 0 and hasattr(layers[0], "self_attn"):
                    return layers

        return None


    def _extract_image_token_positions(self, input_ids):
        batch_size = input_ids.shape[0]
        image_ranges = []
        image_context = []

        for batch_idx in range(batch_size):
            batch_ranges = []
            # Pure tensor operations on the GPU
            starts = torch.nonzero(input_ids[batch_idx] == self.vision_start_id, as_tuple=True)[0]
            ends = torch.nonzero(input_ids[batch_idx] == self.vision_end_id, as_tuple=True)[0]
            context_tokens = torch.nonzero(input_ids[batch_idx] == self.vision_context_id, as_tuple=True)[0]

            if len(context_tokens)<len(starts) or len(context_tokens)<1:
                image_ranges.append([])
                image_context.append([])
                continue
            
            if len(starts) != len(ends):
                eval_logger.warning(
                    f"Batch {batch_idx}: Mismatched vision tokens - "
                    f"{len(starts)} starts vs {len(ends)} ends"
                )
                # sometimes happens that the image delimiters apear as part of the question. (e.g. "(...) The total number of <img> tags found on the (...)")
                # In this case we sill only take those delimiters that have context_tokens after start and end
                context_tokens_set = set(context_tokens.tolist())
                new_starts, new_ends = [], []
                for start, end in zip(starts.tolist(), ends.tolist()):
                    if start+1 in context_tokens_set and end-1 in context_tokens_set:
                        new_starts.append(start)
                        new_ends.append(end)
                starts = new_starts
                ends = new_ends

                if len(starts) != len(ends):
                    eval_logger.warning(
                        f"Second time Batch {batch_idx}: Mismatched vision tokens - "
                        f"{len(starts)} starts vs {len(ends)} ends"
                    )
            
            for start, end in zip(starts, ends):
                # Store range (inclusive of start, inclusive of end)
                # The range includes both delimiter tokens and the image tokens between them
                if isinstance(start, torch.Tensor):
                    start_val = start.item()
                else:
                    start_val = start

                if isinstance(end, torch.Tensor):
                    end_val = end.item()
                else:
                    end_val = end

                batch_ranges.append((start_val, end_val + 1))
            
            image_ranges.append(batch_ranges)
            image_context.append(context_tokens)
            
        return image_ranges, image_context
    

    def _register_attention_hooks(self) -> None:

        # Hook generate to get input_ids
        original_generate = self.model.generate

        def hooked_generate(*args, **kwargs):
            input_ids = kwargs.get("input_ids", args[0] if len(args) > 0 else None)
            if isinstance(input_ids, torch.Tensor) and input_ids.ndim >= 2:
                q_len = input_ids.shape[1]
                if q_len > 1: # Prefill phase
                    self.image_token_indices, self.image_token_context_indices = self._extract_image_token_positions(input_ids) 
                else:
                    self.image_token_indices, self.image_token_context_indices = None, None

            return original_generate(*args, **kwargs)

        self.model.generate = hooked_generate

        # Register attention layers hooks
        layers = self._resolve_decoder_layers()
        if layers is None:
            raise RuntimeError("Could not find attention layers to register mask hooks.")

        for layer_idx in self.mask_layers_indices:
            if layer_idx >= len(layers):
                eval_logger.warning(f"[InternVL3_MaskAttention] Layer index {layer_idx} is out of range.")
                continue

            layer = layers[layer_idx]
            if not hasattr(layer, "self_attn"):
                eval_logger.warning(f"[InternVL3_MaskAttention] Layer {layer_idx} has no self_attn module.")
                continue

            handle = layer.self_attn.register_forward_pre_hook(
                self._create_attention_mask_hook(layer_idx),
                with_kwargs=True,
            )
            self.hooks.append(handle)
            eval_logger.info(f"[InternVL3_MaskAttention] Registered pre-forward hook on layer {layer_idx}")

    def _remove_hooks(self) -> None:
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        if self._phase_hook is not None:
            self._phase_hook.remove()
            self._phase_hook = None

    def __del__(self):
        self._remove_hooks()

    def _create_attention_mask_hook(self, layer_idx: int):
        def hook(module, args, kwargs):
            hidden_states = kwargs.get("hidden_states", args[0] if len(args) > 0 else None)
            if hidden_states is None:
                return args, kwargs

            batch_size, q_len, _ = hidden_states.shape

            # Skip decode phase.
            if q_len == 1:
                return args, kwargs

            # Skip if image index not saved
            if not self.image_token_indices or not self.image_token_context_indices:
                return args, kwargs

            attention_mask = kwargs.get("attention_mask", args[1] if len(args) > 1 else None)
            device = hidden_states.device
            kv_len = q_len

            if self._cached_mask is None:
                self._cached_mask = self._create_cross_image_attention_mask(
                    batch_size=batch_size,
                    q_len=q_len,
                    kv_len=kv_len,
                    batch_image_indices=self.image_token_indices,
                    batch_image_context_indices=self.image_token_context_indices,
                    device=device,
                    dtype=hidden_states.dtype,
                )
            cross_mask = self._cached_mask

            kwargs["attention_mask"] = cross_mask if attention_mask is None else attention_mask + cross_mask
            return args, kwargs

        return hook

    def _create_cross_image_attention_mask(
        self,
        batch_size: int,
        q_len: int,
        kv_len: int,
        batch_image_indices: List[List[Dict[str, int]]],
        batch_image_context_indices: List[List[Dict[str, int]]],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Create additive mask with -inf over cross-image attention blocks."""
        # mask = torch.zeros(batch_size, 1, q_len, kv_len, device=device, dtype=dtype)
        # Start with causal mask only — much cheaper than full zeros + fill
        causal = torch.triu(
            torch.full((q_len, kv_len), torch.finfo(dtype).min, device=device, dtype=dtype),
            diagonal=1
        )
        mask = causal.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1).clone()
        
        min_val = torch.finfo(dtype).min
        # For each batch item
        for b in range(batch_size):
            if b >= len(batch_image_indices):
                continue

            spans = batch_image_indices[b]
            in_img = batch_image_context_indices[b]
            if len(spans) <= 1:
                continue
            
            # Block attention between different images
            for i, (start_i, end_i) in enumerate(spans):
                for j, (start_j, end_j) in enumerate(spans):
                    if i == j: # Same image
                        continue
                    
                    si, ei = max(start_i, 0), min(end_i, q_len)
                    sj, ej = max(start_j, 0), min(end_j, kv_len)
                    if si < ei and sj < ej:
                        mask[b, 0, si:ei, sj:ej] = min_val
                    else:
                        eval_logger.warning(f"[InternVL3_MaskAttention] Mask intervals are incorrect \n \
                                           \t start_i: {start_i} -> {si}, end_i: {end_i} -> {ei} \n \
                                           \t start_j: {start_j} -> {sj}, end_j: {end_j} -> {ej} .")


        # Keep causal behavior explicit for models/layers expecting additive masks.
        # causal = torch.triu(torch.ones((q_len, kv_len), device=device, dtype=torch.bool), diagonal=1)
        # mask = mask.masked_fill(causal, min_val)
        return mask
