#!/usr/bin/env python

from typing import Any, Dict, List, Optional, Tuple

import torch
from detectron2.config import configurable
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.roi_heads import ROI_HEADS_REGISTRY, ROIHeads, select_foreground_proposals
from detectron2.structures import ImageList, Instances


__all__ = [
    'build_roi_pooler',
    'SimpleROIHeads',
]


def build_roi_pooler(cfg, input_shape):
    name = cfg.MODEL.CLIP.IMAGE_ENCODER.ROI_HEAD.NAME
    return ROI_HEADS_REGISTRY.get(name)(cfg, input_shape)


@ROI_HEADS_REGISTRY.register()
class SimpleROIHeads(ROIHeads):
    @configurable
    def __init__(self, *, in_features: List[str], roi_pooler: ROIPooler, **kwargs):
        super(SimpleROIHeads, self).__init__(**kwargs)
        self.in_features = in_features
        self.roi_pooler = roi_pooler

    @classmethod
    def from_config(cls, cfg, input_shape) -> Dict[str, Any]:
        ret = super().from_config(cfg)
        in_features = cfg.MODEL.CLIP.IMAGE_ENCODER.ROI_HEAD.IN_FEATURES
        pooler_resolution = cfg.MODEL.CLIP.IMAGE_ENCODER.ROI_HEAD.POOLER_RESOLUTION
        pooler_type = cfg.MODEL.CLIP.IMAGE_ENCODER.ROI_HEAD.POOLER_TYPE
        sampling_ratio = cfg.MODEL.CLIP.IMAGE_ENCODER.ROI_HEAD.POOLER_SAMPLING_RATIO
        pooler_scales = tuple(1.0 / input_shape[k].stride for k in in_features)

        roi_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )

        ret.update(
            {
                'in_features': in_features,
                'roi_pooler': roi_pooler,
            }
        )
        return ret

    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:
        del images

        if self.training:
            assert targets, "'targets' argument is required during training"
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets

        roi_features, proposals = self._roi_features(features, proposals)
        return roi_features, proposals

    def _roi_features(self, features: Dict[str, torch.Tensor], instances: List[Instances]):
        if self.training:
            # trained on positive proposals only
            instances, _ = select_foreground_proposals(instances, self.num_classes)

        if self.roi_pooler is not None:
            features = [features[f] for f in self.in_features]
            boxes = [x.proposal_boxes if self.training else x.pred_boxes for x in instances]
            features = self.roi_pooler(features, boxes)
        else:
            features = {f: features[f] for f in self.in_features}
        return features, instances
