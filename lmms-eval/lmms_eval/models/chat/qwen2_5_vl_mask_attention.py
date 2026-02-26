import copy
from collections.abc import Callable

import torch
from loguru import logger as eval_logger
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.utils.import_utils import is_torch_flex_attn_available

from lmms_eval.api.registry import register_model
from lmms_eval.models.chat.qwen2_5_vl import Qwen2_5_VL


@register_model("qwen2_5_vl_mask_attention")
class Qwen2_5_VL_Mask_Attention(Qwen2_5_VL):
    def __init__(self, mask_layers=None, **kwargs):
        super().__init__(**kwargs)
        self.mask_layers = mask_layers
        self.hooks = []
        self.image_token_indices = []
        self._image_group_ids = None
        self._has_cross_image_pairs = False
        self._masked_attention_types = {}
        self._base_text_attn_impl = None
        self._base_mask_config = None
        self._flex_mask_config = None

        if self.mask_layers not in (None, "", "none", "None"):
            self._validate_attention_backend()
            self.parse_mask_layers()
            self._configure_layer_attention_types()
            self._register_attention_hooks()

    def _validate_attention_backend(self):
        if not is_torch_flex_attn_available():
            raise RuntimeError(
                "Cross-image masking with mask_layers requires PyTorch FlexAttention support "
                "(torch.nn.attention.flex_attention)."
            )
        language_model = self.model.model.language_model
        self._base_text_attn_impl = language_model.config._attn_implementation
        self._base_mask_config = language_model.config
        self._flex_mask_config = copy.copy(language_model.config)
        self._flex_mask_config._attn_implementation = "flex_attention"
        if self._base_text_attn_impl != "flash_attention_2":
            eval_logger.warning(
                "Hybrid masking works best when base attn_implementation is 'flash_attention_2'. "
                f"Current base implementation is '{self._base_text_attn_impl}'."
            )

    def parse_mask_layers(self):
        if self.mask_layers == "all":
            self.mask_layers_indices = list(range(len(self.model.model.language_model.layers)))
            return

        indices = set()
        segments = str(self.mask_layers).split("+")

        for segment in segments:
            segment = segment.strip()
            if "-" in segment:
                try:
                    start, end = map(int, segment.split("-"))
                    if start > end:
                        start, end = end, start
                    indices.update(range(start, end + 1))
                except ValueError:
                    eval_logger.warning(f"Could not parse range '{segment}'")
            else:
                try:
                    indices.add(int(segment))
                except ValueError:
                    eval_logger.warning(f"Could not parse layer index '{segment}'")

        self.mask_layers_indices = sorted(list(indices))

    def _configure_layer_attention_types(self):
        layers = self.model.model.language_model.layers
        num_layers = len(layers)

        valid_indices = []
        for layer_idx in self.mask_layers_indices:
            if 0 <= layer_idx < num_layers:
                valid_indices.append(layer_idx)
            else:
                eval_logger.warning(f"Layer index {layer_idx} is out of range [0, {num_layers - 1}]")

        self.mask_layers_indices = valid_indices
        eval_logger.info(
            f"Cross-image masking active on layers: {self.mask_layers_indices}"
        )
        if not self.mask_layers_indices:
            eval_logger.warning("No valid layers selected for cross-image masking after validation.")
        for layer_idx in self.mask_layers_indices:
            layer = layers[layer_idx]
            base_attention_type = layer.attention_type
            masked_attention_type = f"masked_{base_attention_type}_{layer_idx}"
            layer.attention_type = masked_attention_type
            self._masked_attention_types[masked_attention_type] = base_attention_type
            # Keep unmasked layers on base backend while forcing masked layers to FlexAttention.
            layer.self_attn.config = copy.copy(layer.self_attn.config)
            layer.self_attn.config._attn_implementation = "flex_attention"

    def _extract_image_token_positions(self, input_ids):
        vision_start_id = self.model.config.vision_start_token_id
        vision_end_id = self.model.config.vision_end_token_id

        batch_size = input_ids.shape[0]
        image_ranges = []

        for batch_idx in range(batch_size):
            batch_ranges = []
            starts = torch.nonzero(input_ids[batch_idx] == vision_start_id, as_tuple=True)[0]
            ends = torch.nonzero(input_ids[batch_idx] == vision_end_id, as_tuple=True)[0]

            if len(starts) != len(ends):
                eval_logger.warning(
                    f"Batch {batch_idx}: Mismatched vision tokens - "
                    f"{len(starts)} starts vs {len(ends)} ends"
                )

            for start, end in zip(starts, ends):
                batch_ranges.append((start, end + 1))

            image_ranges.append(batch_ranges)

        return image_ranges

    def _build_image_group_ids(self, image_token_indices, batch_size, q_len, device):
        group_ids = torch.full((batch_size, q_len), -1, dtype=torch.int32, device=device)
        for batch_idx, image_ranges in enumerate(image_token_indices):
            for image_id, (start, end) in enumerate(image_ranges):
                if start >= q_len:
                    continue
                group_ids[batch_idx, start : min(end, q_len)] = image_id
        return group_ids

    def _create_cross_image_and_mask_function(self) -> Callable | None:
        if self._image_group_ids is None or not self._has_cross_image_pairs:
            return None

        image_group_ids = self._image_group_ids

        def cross_image_and_mask(batch_idx, head_idx, q_idx, kv_idx):
            q_img = image_group_ids[batch_idx, q_idx]
            kv_img = image_group_ids[batch_idx, kv_idx]
            q_is_img = q_img >= 0
            kv_is_img = kv_img >= 0
            mask = ~(q_is_img & kv_is_img & (q_img != kv_img))
            return mask

        return cross_image_and_mask

    def _create_cache_position(self, inputs_embeds, past_key_values, cache_position):
        if cache_position is not None:
            return cache_position
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        return torch.arange(
            past_seen_tokens,
            past_seen_tokens + inputs_embeds.shape[1],
            device=inputs_embeds.device,
        )

    def _register_attention_hooks(self):
        extract_hook = self.model.register_forward_pre_hook(
            self._create_extract_indices_hook(),
            with_kwargs=True,
        )
        self.hooks.append(extract_hook)
        language_hook = self.model.model.language_model.register_forward_pre_hook(
            self._create_language_model_mask_hook(),
            with_kwargs=True,
        )
        self.hooks.append(language_hook)
        eval_logger.info("Registered hooks for layer-selective cross-image masking")

    def _create_extract_indices_hook(self):
        def extract_indices_hook(module, args, kwargs):
            input_ids = kwargs.get("input_ids", args[0] if len(args) > 0 else None)
            if isinstance(input_ids, torch.Tensor) and input_ids.ndim >= 2:
                q_len = input_ids.shape[1]
                if q_len > 1:
                    self.image_token_indices = self._extract_image_token_positions(input_ids)
                    batch_size = input_ids.shape[0]
                    self._image_group_ids = self._build_image_group_ids(
                        self.image_token_indices, batch_size, q_len, input_ids.device
                    )
                    self._has_cross_image_pairs = any(len(image_ranges) > 1 for image_ranges in self.image_token_indices)
                else:
                    self._image_group_ids = None
                    self._has_cross_image_pairs = False

            return args, kwargs

        return extract_indices_hook

    def _create_language_model_mask_hook(self):
        def language_model_mask_hook(module, args, kwargs):
            if not self._masked_attention_types:
                return args, kwargs

            inputs_embeds = kwargs.get("inputs_embeds", args[4] if len(args) > 4 else None)
            if inputs_embeds is None:
                return args, kwargs

            attention_mask = kwargs.get("attention_mask", args[1] if len(args) > 1 else None)
            position_ids = kwargs.get("position_ids", args[2] if len(args) > 2 else None)
            past_key_values = kwargs.get("past_key_values", args[3] if len(args) > 3 else None)
            # Keep positional fallback aligned with Qwen2.5-VLTextModel.forward signature.
            cache_position = kwargs.get("cache_position", args[9] if len(args) > 9 else None)

            cache_position = self._create_cache_position(inputs_embeds, past_key_values, cache_position)

            if position_ids is not None and position_ids.ndim == 3 and position_ids.shape[0] == 4:
                text_position_ids = position_ids[0]
            else:
                text_position_ids = None

            common_mask_kwargs = {
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": text_position_ids,
            }

            if isinstance(attention_mask, dict):
                base_masks = dict(attention_mask)
            else:
                base_masks = {
                    "full_attention": create_causal_mask(config=self._base_mask_config, **common_mask_kwargs),
                }
                if module.has_sliding_layers:
                    base_masks["sliding_attention"] = create_sliding_window_causal_mask(
                        config=self._base_mask_config, **common_mask_kwargs
                    )

            mask_mapping = dict(base_masks)
            apply_cross_image_mask = inputs_embeds.shape[1] > 1 and self._has_cross_image_pairs
            and_mask_function = self._create_cross_image_and_mask_function() if apply_cross_image_mask else None

            for masked_attention_type, base_attention_type in self._masked_attention_types.items():
                if base_attention_type not in base_masks:
                    raise ValueError(
                        f"Base attention_type '{base_attention_type}' missing in mask mapping keys: "
                        f"{list(base_masks.keys())}"
                    )

                if and_mask_function is None:
                    masked_mask = base_masks[base_attention_type]
                elif base_attention_type == "full_attention":
                    masked_mask = create_causal_mask(
                        config=self._flex_mask_config,
                        **common_mask_kwargs,
                        and_mask_function=and_mask_function,
                    )
                elif base_attention_type == "sliding_attention":
                    masked_mask = create_sliding_window_causal_mask(
                        config=self._flex_mask_config,
                        **common_mask_kwargs,
                        and_mask_function=and_mask_function,
                    )
                else:
                    raise ValueError(f"Unsupported attention_type '{base_attention_type}' for masked layer.")
                mask_mapping[masked_attention_type] = masked_mask

            kwargs["attention_mask"] = mask_mapping
            kwargs["cache_position"] = cache_position
            return args, kwargs

        return language_model_mask_hook