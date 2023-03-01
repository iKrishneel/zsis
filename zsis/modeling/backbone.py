#!/usr/bin/env python

from typing import Dict

from detectron2.modeling import BACKBONE_REGISTRY, ShapeSpec, FPN
from detectron2.modeling.backbone.fpn import LastLevelMaxPool
from detectron2_timm.models.backbone import Backbone

from clip.model import ModifiedResNet


__all__ = ['build_modified_resnet_backbone', 'build_modified_resnet_fpn_backbone']


def replace_non_trainable_batchnorm(model):
    import torch.nn as nn
    from torchvision.ops import FrozenBatchNorm2d

    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            requires_grad = False
            for param in module.parameters():
                requires_grad |= param.requires_grad

            if not requires_grad:
                setattr(model, name, FrozenBatchNorm2d(num_features=module.num_features))
    return model


def _build_backbone(cfg):
    layers = cfg.MODEL.RESNETS.LAYERS
    output_dim, heads = 4, 1  # dummpy not used
    model = ModifiedResNet(layers=layers, output_dim=output_dim, heads=heads)
    # model = replace_non_trainable_batchnorm(model)
    backbone = Backbone(cfg, model=model)
    return backbone


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
