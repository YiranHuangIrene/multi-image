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
        mask_layers_str = None if mask_layers is None else str(mask_layers)

        super().__init__(**kwargs)
        self.mask_layers = mask_layers_str

        self.hooks: List[torch.utils.hooks.RemovableHandle] = []
        self._phase_hook: Optional[torch.utils.hooks.RemovableHandle] = None
        self.mask_layers_indices: List[int] = []
        self._original_attn_impl: Optional[str] = None
        self._last_attn_impl: Optional[str] = None

        # Set before generation, consumed by hooks.
        self.image_token_indices: Optional[List[List[Dict[str, int]]]] = None
        self._pending_num_patches_list: Optional[List[int]] = None
        self._cached_mask: Optional[torch.Tensor] = None

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

    def _register_attention_hooks(self) -> None:
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

    def _get_tokens_per_patch(self) -> int:
        """Infer how many LM tokens represent one visual patch in InternVL."""
        for obj in (self.model, self._config):
            for attr in ("num_image_token", "vision_token_num", "image_token_num"):
                if hasattr(obj, attr):
                    value = int(getattr(obj, attr))
                    if value > 0:
                        return value

        # Common InternVL default; used only as fallback when attrs are absent.
        return 256

    def _estimate_image_token_indices(self, q_len: int) -> Optional[List[List[Dict[str, int]]]]:
        """Estimate image token spans for the current sample.

        InternVL's `chat(...)` does not expose exact per-image LM token boundaries,
        so we estimate contiguous visual blocks based on patch counts.
        If the estimate is not plausible, masking is skipped.
        """
        if not self._pending_num_patches_list:
            return None

        tokens_per_patch = self._get_tokens_per_patch()
        per_image_tokens = [int(n) * tokens_per_patch for n in self._pending_num_patches_list]
        total_image_tokens = sum(per_image_tokens)

        if total_image_tokens <= 0:
            return None

        # Conservative guard: if estimate exceeds sequence length, skip masking.
        if total_image_tokens >= q_len:
            eval_logger.warning(
                "[InternVL3_MaskAttention] Estimated image tokens exceed sequence length; skipping mask. "
                f"estimated={total_image_tokens}, q_len={q_len}."
            )
            return None

        # InternVL prompt construction typically places visual blocks before text.
        spans: List[Dict[str, int]] = []
        cursor = 0
        for token_count in per_image_tokens:
            start = cursor
            end = min(cursor + token_count, q_len)
            if end > start:
                spans.append({"start": start, "end": end})
            cursor = end

        return [spans] if len(spans) > 1 else None

    def _create_attention_mask_hook(self, layer_idx: int):
        def hook(module, args, kwargs):
            hidden_states = kwargs.get("hidden_states", args[0] if len(args) > 0 else None)
            if hidden_states is None:
                return args, kwargs

            batch_size, q_len, _ = hidden_states.shape

            # Skip decode phase.
            if q_len == 1:
                return args, kwargs

            if self.image_token_indices is None:
                self.image_token_indices = self._estimate_image_token_indices(q_len)

            if not self.image_token_indices:
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
        for b in range(batch_size):
            if b >= len(batch_image_indices):
                continue

            spans = batch_image_indices[b]
            if len(spans) <= 1:
                continue

            for i, span_i in enumerate(spans):
                for j, span_j in enumerate(spans):
                    if i == j:
                        continue

                    si, ei = min(span_i["start"], q_len), min(span_i["end"], q_len)
                    sj, ej = min(span_j["start"], kv_len), min(span_j["end"], kv_len)
                    if si < ei and sj < ej:
                        mask[b, 0, si:ei, sj:ej] = min_val

        # Keep causal behavior explicit for models/layers expecting additive masks.
        # causal = torch.triu(torch.ones((q_len, kv_len), device=device, dtype=torch.bool), diagonal=1)
        # mask = mask.masked_fill(causal, min_val)
        return mask

    def generate_until(self, requests: List[Instance]) -> List[str]:
        """InternVL3 generation with optional cross-image attention masking."""
        res: List[str] = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        for contexts, gen_kwargs, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            if "until" in gen_kwargs:
                gen_kwargs.pop("until")
            for k, v in DEFAULT_GEN_KWARGS.items():
                if k not in gen_kwargs:
                    gen_kwargs[k] = v

            pop_keys = [k for k in gen_kwargs.keys() if k not in DEFAULT_GEN_KWARGS]
            for k in pop_keys:
                gen_kwargs.pop(k)

            visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
            visuals = self.flatten(visuals)

            # Reset per-sample tracking used by hooks.
            self.image_token_indices = None
            self._pending_num_patches_list = None
            self._cached_mask = None

            if self.modality == "image":
                if visuals:
                    image_num = len(visuals)
                    dynamic_max_num = max(1, min(self.max_num, self.total_max_num // image_num))

                    processed_visuals = [load_image(visual, max_num=dynamic_max_num).to(torch.bfloat16).to(self._device) for visual in visuals]
                    pixel_values = torch.cat(processed_visuals, dim=0)
                    num_patches_list = [v.size(0) for v in processed_visuals]

                    existing_tags = contexts.count("<image>")
                    if existing_tags == 0:
                        image_tokens = " ".join(["<image>"] * len(processed_visuals))
                        contexts = image_tokens + "\n" + contexts
                    elif existing_tags != len(processed_visuals):
                        eval_logger.warning(
                            f"[InternVL3_MaskAttention] Token mismatch: {existing_tags} tags in text, "
                            f"{len(processed_visuals)} images provided. Prepending image tags."
                        )
                        image_tokens = " ".join(["<image>"] * len(processed_visuals))
                        contexts = image_tokens + "\n" + contexts

                    if self.mask_layers:
                        self._pending_num_patches_list = num_patches_list
                else:
                    pixel_values = None
                    num_patches_list = None

                response, _ = self.model.chat(
                    self.tokenizer,
                    pixel_values,
                    contexts,
                    gen_kwargs,
                    num_patches_list=num_patches_list,
                    history=None,
                    return_history=True,
                )

            elif self.modality == "video":
                assert len(visuals) == 1, f"Only one video is supported, but got {len(visuals)} videos."
                video_path = visuals[0]

                dynamic_max_num = max(1, min(self.max_num, self.total_max_num // self.num_frame))
                pixel_values, num_patches_list = load_video(video_path, num_segments=self.num_frame, max_num=dynamic_max_num)
                pixel_values = pixel_values.to(torch.bfloat16).to(self._device)
                video_prefix = "".join([f"Frame{i + 1}: <image>\n" for i in range(len(num_patches_list))])
                question = video_prefix + contexts

                # Masking is intentionally disabled for video mode until exact span
                # tracking is added for frame-level visual token blocks.
                response, _ = self.model.chat(
                    self.tokenizer,
                    pixel_values,
                    question,
                    gen_kwargs,
                    num_patches_list=num_patches_list,
                    history=None,
                    return_history=True,
                )
            else:
                raise ValueError(f"Unsupported modality: {self.modality}")

            res.append(response)
            torch.cuda.empty_cache()
            pbar.update(1)

        pbar.close()
        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood is not implemented for InternVL3_MaskAttention.")

    def generate_until_multi_round(self, requests: List[Instance]) -> List[str]:
        raise NotImplementedError("Multi-round generation is not implemented for InternVL3_MaskAttention.")
