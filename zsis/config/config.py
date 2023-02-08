#!/usr/bin/env python

from detectron2.config import CfgNode as CN
from detectron2_timm.config.defaults import _C as _c


_C = _c.clone()
# resnet layers for modified resnet architecture
_C.MODEL.RESNETS.LAYERS = [3, 4, 6, 3]

# roi heads
_C.MODEL.ROI_HEADS.USE_DROPLOSS = False
_C.MODEL.ROI_HEADS.DROPLOSS_IOU_THRESH = 0.0

# clip params
_C.MODEL.CLIP = CN()
_C.MODEL.CLIP.ARCHITECTURE = ''
_C.MODEL.CLIP.TOPK = 5
_C.MODEL.CLIP.PROB_SCALE = 100.0
_C.MODEL.CLIP.CROP_SCALE = 1.2
_C.MODEL.CLIP.EVAL_ONLY = True

_C.MODEL.CLIP.TEXT_ENCODER = CN()
_C.MODEL.CLIP.TEXT_ENCODER.FROZEN = True
_C.MODEL.CLIP.TEXT_ENCODER.CONTEXT_LENGTH = 77  # positional embedding length
_C.MODEL.CLIP.TEXT_ENCODER.TRANSFORMER_WIDTH = 512
_C.MODEL.CLIP.TEXT_ENCODER.TRANSFORMER_LAYERS = 12
_C.MODEL.CLIP.TEXT_ENCODER.TRANSFORMER_HEADS = 8
