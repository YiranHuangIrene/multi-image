import math
import re
import copy
import json
import logging
import warnings
import numpy as np
import PIL
import torch
import torch.nn as nn

import sys
import os
# Append the path to the util
llava_utils_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "utils")
sys.path.append(llava_utils_path)

from tqdm import tqdm
from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.load_video import read_video_pyav
from lmms_eval.models.simple.llava_onevision import Llava_OneVision


# Assuming standard Llava imports are available as in your original script
from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import KeywordsStoppingCriteria, process_images, tokenizer_image_token

eval_logger = logging.getLogger("lmms-eval")

@register_model("llava_onevision_mask_attention")
class Llava_OneVision_Mask_Attention(Llava_OneVision):
    """
    Llava OneVision model with cross-image attention masking capability.
    """
    def __init__(self, mask_layers=None, **kwargs):
        super().__init__(**kwargs)
        self.mask_layers = str(mask_layers)
        self.hooks = []
        self.image_token_indices = None
        self.mask_layers_indices = None
        
        if self.mask_layers:
            self.parse_mask_layers()
            self._register_attention_hooks()
            

    
    def parse_mask_layers(self):
        if self.mask_layers == "all":
            self.mask_layers_indices = list(range(len(self.model.model.layers)))
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

    def _register_attention_hooks(self):
        # Fallback for models that might nest the language model differently
        language_model = self.model.model
        layers = language_model.layers
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

    def _create_attention_mask_hook(self, layer_idx):
        def attention_mask_hook(module, args, kwargs):
            if not self.image_token_indices:
                print("No image token indices found! Skipping attention masking.")
                return args, kwargs

            # Retrieve hidden_states to determine query sequence length
            hidden_states = kwargs.get('hidden_states', args[0] if len(args) > 0 else None)
            if hidden_states is None:
                return args, kwargs
            batch_size, q_len, _ = hidden_states.shape
            
            # Skip custom masking during the decoding phase (q_len == 1)
            # We only mask images from each other during the prefill phase.
            if q_len == 1:
                return args, kwargs

            attention_mask = kwargs.get('attention_mask', args[1] if len(args) > 1 else None)
            device = hidden_states.device
            kv_len = q_len # Assumes prefill where q_len == kv_len

            # Create cross-image mask [batch_size, 1, q_len, kv_len]
            cross_mask = self._create_cross_image_attention_mask(
                batch_size, q_len, kv_len, self.image_token_indices, device, hidden_states.dtype
            )
            
            # Additive masking
            if attention_mask is not None:
                combined_mask = attention_mask + cross_mask
            else:
                combined_mask = cross_mask

            # Inject the combined mask back into kwargs/args
            kwargs['attention_mask'] = combined_mask
                
            return args, kwargs
        return attention_mask_hook

    def _create_cross_image_attention_mask(self, batch_size, q_len, kv_len, batch_image_indices, device, dtype):
        """
        Creates a 4D mask tensor where cross-image attention scores are set to -inf.
        """
        # Ensure mask broadcasts over heads: [batch, 1, q_len, kv_len]
        mask = torch.zeros(batch_size, 1, q_len, kv_len, device=device, dtype=dtype)
        min_val = torch.finfo(dtype).min

        for b in range(batch_size):
            if b >= len(batch_image_indices):
                continue
                
            image_indices = batch_image_indices[b]
            num_images = len(image_indices)
            
            if num_images <= 1:
                continue
                
            for i in range(num_images):
                for j in range(num_images):
                    if i != j:
                        start_i, end_i = image_indices[i]['start'], image_indices[i]['end']
                        start_j, end_j = image_indices[j]['start'], image_indices[j]['end']
                        
                        # Guard against out-of-bounds due to truncation
                        start_i, end_i = min(start_i, q_len), min(end_i, q_len)
                        start_j, end_j = min(start_j, kv_len), min(end_j, kv_len)
                        
                        if start_i < end_i and start_j < end_j:
                            mask[b, 0, start_i:end_i, start_j:end_j] = min_val
        # Apply causal mask
        causal_mask = torch.triu(
            torch.ones((q_len, kv_len), device=device, dtype=torch.bool), 
            diagonal=1
        )
        mask = mask.masked_fill(causal_mask, min_val)
        return mask


    def generate_until(self, requests: list[Instance]) -> list[str]:
        res = []

        def _collate(x):
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end
            toks = self.tok_encode(x[0])
            return -len(toks), x[0]

        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        metadata = requests[0].metadata
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        num_iters = len(requests) // self.batch_size if len(requests) % self.batch_size == 0 else len(requests) // self.batch_size + 1
        pbar = tqdm(total=num_iters, disable=(self.rank != 0), desc="Model Responding")

        origin_image_aspect_ratio = getattr(self._config, "image_aspect_ratio", None)

        for chunk in chunks:
            (
                batched_contexts,
                all_gen_kwargs,
                batched_doc_to_visual,
                batched_doc_id,
                batched_task,
                batched_split,
            ) = zip(*chunk)
            task = batched_task[0]
            split = batched_split[0]
            batched_visuals = [batched_doc_to_visual[0](self.task_dict[task][split][ids]) for ids in batched_doc_id]  # [B, N]
            assert len(batched_visuals) == 1

            # we assume all gen kwargs in the batch are the same
            # this is safe to assume because the `grouper` object ensures it.
            gen_kwargs = all_gen_kwargs[0]
            if "until" in gen_kwargs:
                gen_kwargs.pop("until")

            question_input = []
            # import ipdb; ipdb.set_trace()
            for visual, context in zip(batched_visuals, batched_contexts):
                if origin_image_aspect_ratio is not None and self._config.image_aspect_ratio != origin_image_aspect_ratio:
                    self._config.image_aspect_ratio = origin_image_aspect_ratio
                    eval_logger.info(f"Resetting image aspect ratio to {origin_image_aspect_ratio}")

                if visual is None or visual == []:  # for text-only tasks.
                    visual = None
                    task_type = "text"
                    placeholder_count = 0
                    image_tensor = None
                else:
                    if len(visual) > 1 or "image_aspect_ratio" not in self._config.__dict__:  # for multi image case, we treat per image aspect ratio as "pad" by default.
                        self._config.image_aspect_ratio = getattr(gen_kwargs, "image_aspect_ratio", "pad")
                        eval_logger.info(f"In Multi-Image setting, image aspect ratio: {self._config.image_aspect_ratio}")

                    if "task_type" in metadata and metadata["task_type"] == "video" and "sample_frames" in metadata:  # overwrite logic for video task with multiple static image frames
                        assert type(visual) == list, "sample_frames must be specified for video task"
                        sample_indices = np.linspace(0, len(visual) - 1, metadata["sample_frames"], dtype=int)
                        visual = [visual[i] for i in sample_indices]
                        assert len(visual) == metadata["sample_frames"]

                        image_tensor = process_images(visual, self._image_processor, self._config)
                        if type(image_tensor) is list:
                            image_tensor = [_image.to(dtype=torch.float16, device=self.device) for _image in image_tensor]
                        else:
                            image_tensor = image_tensor.to(dtype=torch.float16, device=self.device)

                        task_type = "video"
                        placeholder_count = 1

                    elif type(visual[0]) == PIL.Image.Image:  # For image, multi-image tasks
                        image_tensor = process_images(visual, self._image_processor, self._config)
                        if type(image_tensor) is list:
                            image_tensor = [_image.to(dtype=torch.float16, device=self.device) for _image in image_tensor]
                        else:
                            image_tensor = image_tensor.to(dtype=torch.float16, device=self.device)

                        task_type = "image"
                        placeholder_count = len(visual) if isinstance(visual, list) else 1

                    elif type(visual[0]) == str:  # For video task
                        image_tensor = []
                        try:
                            if self.video_decode_backend == "decord":
                                frames = self.load_video(visual, self.max_frames_num)
                            elif self.video_decode_backend == "pyav":
                                frames = read_video_pyav(visual[0], num_frm=self.max_frames_num)
                            frames = self._image_processor.preprocess(frames, return_tensors="pt")["pixel_values"].half().to(self._device)
                            image_tensor.append(frames)
                        except Exception as e:
                            eval_logger.error(f"Error {e} in loading video")
                            image_tensor = None

                        task_type = "video"
                        placeholder_count = len(frames) if self.token_strategy == "multiple" else 1

                if image_tensor is not None and len(image_tensor) != 0 and DEFAULT_IMAGE_TOKEN not in context:
                    """
                    Three senarios:
                    1. No image, and there for, no image token should be added.
                    2. image token is already specified in the context, so we don't need to add it.
                    3. image token is not specified in the context and there is image inputs, so we need to add it. In this case, we add the image token at the beginning of the context and add a new line.
                    4. For video tasks, we could add a <image> token or multiple <image> tokens for each frame in the context. This depends on the training strategy and should balance in test to decide which is better
                    """
                    # if task_type == "image": # indeed in multi-image case, not the video in frames.
                    #     image_tokens = [DEFAULT_IMAGE_TOKEN] * placeholder_count if isinstance(visual, list) else [DEFAULT_IMAGE_TOKEN]
                    # elif task_type == "video":
                    # image_tokens = [DEFAULT_IMAGE_TOKEN] * placeholder_count if self.token_strategy == "multiple" else [DEFAULT_IMAGE_TOKEN]
                    image_tokens = [DEFAULT_IMAGE_TOKEN] * placeholder_count
                    image_tokens = " ".join(image_tokens)
                    question = image_tokens + "\n" + context
                else:
                    question = context

                # This is much safer for llama3, as we now have some object type in it
                if "llama_3" in self.conv_template:
                    conv = copy.deepcopy(conv_templates[self.conv_template])
                else:
                    conv = conv_templates[self.conv_template].copy()

                if utils.is_json(question):  # conversational question input
                    question = json.loads(question)
                    for idx, item in enumerate(question):
                        role = conv.roles[idx % 2]
                        message = item["value"]
                        conv.append_message(role, message)

                    assert len(conv.messages) % 2 == 1
                    conv.append_message(conv.roles[1], None)
                    prompt_question = conv.get_prompt()
                    question_input.append(prompt_question)
                else:  # only simple string for question
                    conv.append_message(conv.roles[0], question)
                    conv.append_message(conv.roles[1], None)
                    prompt_question = conv.get_prompt()
                    question_input.append(prompt_question)

            input_ids_list = [tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt") for prompt in question_input]
            pad_token_ids = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
            input_ids = self.pad_sequence(input_ids_list, batch_first=True, padding_value=pad_token_ids).to(self.device)
            attention_masks = input_ids.ne(pad_token_ids).to(self.device)
            
            if task_type == "image":
                gen_kwargs["image_sizes"] = [batched_visuals[0][idx].size for idx in range(len(batched_visuals[0]))]
            elif task_type == "video":
                stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
                keywords = [stop_str]
                stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
                gen_kwargs["modalities"] = ["video"]
                gen_kwargs["stopping_criteria"] = [stopping_criteria]
                self._config.mm_spatial_pool_stride = self.mm_spatial_pool_stride
                self._config.mm_spatial_pool_mode = self.mm_spatial_pool_mode

            # These steps are not in LLaVA's original code, but are necessary for generation to work
            # TODO: attention to this major generation step...
            # preconfigure gen_kwargs with defaults
            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 1024

            if "image_aspect_ratio" in gen_kwargs.keys():
                gen_kwargs.pop("image_aspect_ratio")
            # When do_sample=False, remove sampling-related parameters to avoid warnings
            # These might be in gen_kwargs or in the model's generation_config
            if self.mask_layers:
                (inputs, position_ids, attention_mask, _, inputs_embeds, _, all_image_token_indices) = self.model.prepare_inputs_labels_for_multimodal_w_tracking(
                    input_ids=input_ids, 
                    position_ids=gen_kwargs.pop("position_ids", None), 
                    attention_mask=attention_masks, 
                    past_key_values=gen_kwargs.get("past_key_values", None),
                    labels=gen_kwargs.get("labels", None),
                    images=image_tensor,
                    image_sizes=gen_kwargs.get("image_sizes", None)
                )
                self.image_token_indices = all_image_token_indices
            
            if not gen_kwargs.get("do_sample", False):
                gen_kwargs.pop("temperature", None)
                gen_kwargs.pop("top_p", None)
                gen_kwargs.pop("top_k", None)
            try:
                if self.mask_layers:
                    with torch.inference_mode():
                        cont = self.model.generate_with_tracking(
                            position_ids=position_ids,
                            attention_mask=attention_mask,
                            inputs_embeds=inputs_embeds,
                            **gen_kwargs,
                        )
                else:
                    with torch.inference_mode():
                        cont = self.model.generate(
                            input_ids,
                            attention_mask=attention_masks,
                            pad_token_id=pad_token_ids,
                            images=image_tensor,
                            use_cache=self.use_cache,
                            **gen_kwargs,
                        )
                text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)
                
            except Exception as e:
                raise e

            text_outputs = [response.strip() for response in text_outputs]
            res.extend(text_outputs)
            self.cache_hook.add_partial("generate_until", (context, gen_kwargs), text_outputs)
            pbar.update(1)
            # reorder this group of results back to original unsorted form
        res = re_ords.get_original(res)

        pbar.close()
        return res
        
    # def _remove_hooks(self):
    #     for hook in self.hooks:
    #         hook.remove()
    #     self.hooks = []

    # def __del__(self):
    #     self._remove_hooks()