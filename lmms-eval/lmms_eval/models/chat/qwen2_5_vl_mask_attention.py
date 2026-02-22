import time
import torch
from typing import List
import sys
from loguru import logger as eval_logger
from tqdm import tqdm

try:
    import decord
except ImportError:
    decord = None

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.registry import register_model
from lmms_eval.imports import optional_import
from lmms_eval.models.model_utils.gen_metrics import log_metrics
from lmms_eval.models.model_utils.reasoning_model_utils import (
    parse_reasoning_model_answer,
)
from lmms_eval.models.chat.qwen2_5_vl import Qwen2_5_VL

process_vision_info, _has_qwen_vl = optional_import("qwen_vl_utils", "process_vision_info")
if not _has_qwen_vl:
    eval_logger.warning("Failed to import qwen_vl_utils; Please install it via `pip install qwen-vl-utils`")


@register_model("qwen2_5_vl_mask_attention")
class Qwen2_5_VL_Mask_Attention(Qwen2_5_VL):
    def __init__(self, mask_layers=None, **kwargs):
        super().__init__(**kwargs)
        self.mask_layers = str(mask_layers)
        self.hooks = []
        self.image_token_indices = []
        self._cached_cross_mask = None
        if self.mask_layers:
            self.parse_mask_layers()
            self._register_attention_hooks()
    
    def parse_mask_layers(self):
        if self.mask_layers == "all":
            self.mask_layers_indices = list(range(len(self.model.model.language_model.layers)))
            return

        indices = set()
        # Split by '+' to separate groups (e.g., "1-3+5" -> ["1-3", "5"])
        segments = self.mask_layers.split('+')
        
        for segment in segments:
            segment = segment.strip() # Remove potential whitespace
            if '-' in segment:
                # Handle Range: "start-end" (inclusive)
                try:
                    start, end = map(int, segment.split('-'))
                    if start > end:
                        start, end = end, start # Auto-correct if user types "10-5"
                    indices.update(range(start, end + 1))
                except ValueError:
                    print(f"Warning: Could not parse range '{segment}'")
            else:
                # Handle Individual Layer
                try:
                    indices.add(int(segment))
                except ValueError:
                    print(f"Warning: Could not parse layer index '{segment}'")
        
        self.mask_layers_indices = sorted(list(indices))
        
    def _extract_image_token_positions(self, input_ids):
        vision_start_id = self.model.config.vision_start_token_id
        vision_end_id = self.model.config.vision_end_token_id
        
        batch_size = input_ids.shape[0]
        image_ranges = []
        
        for batch_idx in range(batch_size):
            batch_ranges = []
            # Pure tensor operations on the GPU
            starts = torch.nonzero(input_ids[batch_idx] == vision_start_id, as_tuple=True)[0]
            ends = torch.nonzero(input_ids[batch_idx] == vision_end_id, as_tuple=True)[0]
            
            if len(starts) != len(ends):
                eval_logger.warning(
                    f"Batch {batch_idx}: Mismatched vision tokens - "
                    f"{len(starts)} starts vs {len(ends)} ends"
                )
            
            for start, end in zip(starts, ends):
                # Store range (inclusive of start, inclusive of end)
                # The range includes both delimiter tokens and the image tokens between them
                batch_ranges.append((start, end + 1))
            
            image_ranges.append(batch_ranges)
            
        return image_ranges
        
    def _register_attention_hooks(self):
        # --- Register top-level hook for extracting indices ---
        extract_hook = self.model.register_forward_pre_hook(
            self._create_extract_indices_hook(),
            with_kwargs=True
        )
        self.hooks.append(extract_hook)
        print("Registered top-level pre-forward hook to extract image token indices")
        eval_logger.info("Registered top-level pre-forward hook to extract image token indices")
        # Fallback for models that might nest the language model differently
        layers = self.model.model.language_model.layers
        for layer_idx in self.mask_layers_indices:
            if layer_idx < len(layers):
                layer = layers[layer_idx]
                if hasattr(layer, 'self_attn'):
                    # Use with_kwargs=True (Requires PyTorch 2.0+) to capture kwargs like attention_mask
                    hook = layer.self_attn.register_forward_pre_hook(
                        self._create_attention_mask_hook(layer_idx),
                        with_kwargs=True
                    )
                    self.hooks.append(hook)
                    print(f"Registered pre-forward hook on layer {layer_idx}")
                    eval_logger.info(f"Registered pre-forward hook on layer {layer_idx}")
                else:
                    print(f"Layer {layer_idx} does not have self_attn module")
                    eval_logger.warning(f"Layer {layer_idx} does not have self_attn module")
            else:
                print(f"Layer index {layer_idx} is out of range")
                eval_logger.warning(f"Layer index {layer_idx} is out of range")
    
    def _create_extract_indices_hook(self):
        def extract_indices_hook(module, args, kwargs):
            input_ids = kwargs.get('input_ids', args[0] if len(args) > 0 else None)
            if isinstance(input_ids, torch.Tensor) and input_ids.ndim >= 2:
                q_len = input_ids.shape[1]
                if q_len > 1: # Prefill phase
                    self.image_token_indices = self._extract_image_token_positions(input_ids)
                    
                    # PRE-COMPUTE AND CACHE THE MASK HERE
                    batch_size = input_ids.shape[0]
                    self._cached_cross_mask = self._create_cross_image_attention_mask(
                        batch_size, q_len, q_len, 
                        self.image_token_indices, 
                        input_ids.device, 
                        self.model.dtype
                    )
                else:
                    # Clear cache during decoding to save memory
                    self._cached_cross_mask = None 
                    
            return args, kwargs
            
        return extract_indices_hook
    
    def _create_attention_mask_hook(self, layer_idx):
        def attention_mask_hook(module, args, kwargs):
            if not self.image_token_indices or self._cached_cross_mask is None:
                return args, kwargs

            hidden_states = kwargs.get('hidden_states', args[0] if len(args) > 0 else None)
            if hidden_states is None:
                return args, kwargs
                
            q_len = hidden_states.shape[1]
            if q_len == 1: # Decoding phase
                return args, kwargs

            attention_mask = kwargs.get('attention_mask', args[1] if len(args) > 1 else None)
            if attention_mask is not None:
                combined_mask = attention_mask + self._cached_cross_mask
            else:
                combined_mask = self._cached_cross_mask

            kwargs['attention_mask'] = combined_mask
            return args, kwargs
            
        return attention_mask_hook
    
    def _create_cross_image_attention_mask(self, batch_size, q_len, kv_len, image_token_indices, device, dtype):
        """
        Create an attention mask that blocks cross-image attention.
        This prevents tokens from one image attending to tokens from other images.
        
        Args:
            batch_size: Batch size
            q_len: Query sequence length
            kv_len: Key/Value sequence length
            image_token_indices: List of image token ranges per batch item
                                 Format: [[batch_0_img_ranges], [batch_1_img_ranges], ...]
                                 where each range is (start_idx, end_idx)
            device: Device to create the mask on
            dtype: Data type for the mask (typically float)
            
        Returns:
            Attention mask of shape [batch_size, 1, q_len, kv_len]
            Values are 0 (no masking) or -inf (block attention)
        """
        # Initialize mask with zeros (no masking)
        mask = torch.zeros((batch_size, 1, q_len, kv_len), dtype=dtype, device=device)
        
        # For each batch item
        for batch_idx in range(batch_size):
            if batch_idx >= len(image_token_indices):
                continue
                
            image_ranges = image_token_indices[batch_idx]
            num_images = len(image_ranges)
            
            if num_images <= 1:
                # No cross-image attention to block if there's 0 or 1 image
                continue
            
            eval_logger.debug(
                f"Layer masking: Batch {batch_idx} has {num_images} images, "
                f"blocking cross-image attention"
            )
            
            # Block attention between different images
            for i, (start_i, end_i) in enumerate(image_ranges):
                for j, (start_j, end_j) in enumerate(image_ranges):
                    if i != j:  # Different images

                        mask[batch_idx, 0, start_i:end_i, start_j:end_j] = float('-inf')
        # Apply causal mask
        causal_mask = torch.triu(
            torch.ones((q_len, kv_len), device=device, dtype=torch.bool), 
            diagonal=1
        )
        mask = mask.masked_fill(causal_mask, float('-inf'))
        return mask
    