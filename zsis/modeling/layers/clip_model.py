#!/usr/bin/env python

from typing import Any, Dict
from detectron2.config import CfgNode

import clip
from clip.model import AttentionPool2d


def build_clip_model(cfg: CfgNode) -> Dict[str, Any]:
    clip_cfg = cfg.MODEL.CLIP
    model, preprocessing = clip.load(clip_cfg.ARCHITECTURE)

    if clip_cfg.EVAL_ONLY:
        model.to(cfg.MODEL.DEVICE).eval()

    logger.info(f'Model parameters: {np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}')
    logger.info(f'Input resolution: {model.visual.input_resolution}')
    logger.info(f'Context length: {model.context_length}')
    logger.info(f'Vocab size: {model.vocab_size}')

    logger.warn('Changing of floating precision is currently not supported!')
    return {'clip_model': model, 'preprocessing': preprocessing}


def build_attention_pool(cfg: CfgNode) -> AttentionPool2d:
    return AttentionPool2d(
        cfg.MODEL.CLIP.IMAGE_ENCODER.ROI_HEAD.POOLER_RESOLUTION,
        cfg.MODEL.CLIP.IMAGE_ENCODER.ATTN_EMBED_DIM,
        cfg.MODEL.CLIP.IMAGE_ENCODER.ATTN_NUM_HEADS,
        cfg.MODEL.CLIP.IMAGE_ENCODER.ATTN_OUTPUT_DIM,
    )


def build_attention_mask(cfg: CfgNode) -> torch.Tensor:
    context_length = cfg.MODEL.CLIP.TEXT_ENCODER.CONTEXT_LENGTH
    mask = torch.empty(context_length, context_length)
    mask.fill_(float("-inf"))
    mask.triu_(1)  # zero out the lower diagonal
    return mask
