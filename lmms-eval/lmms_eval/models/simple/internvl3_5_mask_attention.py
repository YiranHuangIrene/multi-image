"""
File deprecated Check internvl3_mask_attention
"""

from lmms_eval.api.registry import register_model
from lmms_eval.models.simple.internvl3_mask_attention import InternVL3_MaskAttention


@register_model("internvl3_5_mask_attention")
class InternVL3_5_MaskAttention(InternVL3_MaskAttention):
    """InternVL3.5 model wrapper with optional cross-image attention masking.

    Uses the same implementation as InternVL3 since both share identical interfaces.
    Default pretrained model is set to InternVL3_5-8B.
    """

    def __init__(
        self,
        pretrained: str = "OpenGVLab/InternVL3_5-8B",
        **kwargs,
    ):
        super().__init__(pretrained=pretrained, **kwargs)
