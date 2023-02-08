#!/usr/bin/env python

from typing import Dict

from detectron2.modeling import BACKBONE_REGISTRY, ShapeSpec, FPN
from detectron2.modeling.backbone.fpn import LastLevelMaxPool
from detectron2_timm.models.backbone import Backbone

from clip.model import ModifiedResNet


__all__ = ['build_modified_resnet_backbone', 'build_modified_resnet_fpn_backbone']


def _build_backbone(cfg):
    layers = cfg.MODEL.RESNETS.LAYERS
    output_dim, heads = 4, 1  # dummpy not used
    model = ModifiedResNet(layers=layers, output_dim=output_dim, heads=heads)
    return Backbone(cfg, model=model)


@BACKBONE_REGISTRY.register()
def build_modified_resnet_backbone(cfg, input_shape: ShapeSpec = None) -> Backbone:
    return _build_backbone(cfg)


@BACKBONE_REGISTRY.register()
def build_modified_resnet_fpn_backbone(cfg, input_shape: ShapeSpec = None) -> Backbone:
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    norm = cfg.MODEL.FPN.NORM
    fuse_type = cfg.MODEL.FPN.FUSE_TYPE
    return FPN(
        bottom_up=_build_backbone(cfg),
        in_features=in_features,
        out_channels=out_channels,
        norm=norm,
        top_block=LastLevelMaxPool(),
        fuse_type=fuse_type,
    )
