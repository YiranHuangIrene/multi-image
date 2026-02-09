import time
from typing import List
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../")))
from delimiter_scaling import DelimiterTokenScaler

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


@register_model("qwen2_vl_scaled")
class Qwen2_5_VL_Delimiter_Scaling(Qwen2_5_VL):
    def __init__(self, scaling_factor=5.0, scaling_layers="0,1,2,3", **kwargs):
        super().__init__(**kwargs)
        # Parse arguments
        self.scaling_factor = float(scaling_factor)
        self.target_layers = [int(x) for x in scaling_layers.split(",")]
        
        # Identify delimiter tokens 
        assert self.model.config.vision_start_token_id is not None, "Vision start token ID is not found in the model config"
        assert self.model.config.vision_end_token_id is not None, "Vision end token ID is not found in the model config"
        vision_start_id = self.model.config.vision_start_token_id
        vision_end_id = self.model.config.vision_end_token_id
        eval_logger.info(f"Applying Delimiter Scaling: Factor {self.scaling_factor} on Layers {self.target_layers}")
        eval_logger.info(f"Delimiter Tokens: {vision_start_id}, {vision_end_id}")
        self.scaler = DelimiterTokenScaler(
            model=self.model,
            scaling_factor=self.scaling_factor,
            target_layers=self.target_layers,
            delimiter_token_ids=[vision_start_id, vision_end_id]
        )
        self.scaler.register()

    