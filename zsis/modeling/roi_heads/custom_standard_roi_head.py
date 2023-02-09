#!/usr/bin/env python

import inspect
from typing import Dict, List, Optional, Tuple
import torch
from torch import nn

from detectron2.config import configurable
from detectron2.layers import ShapeSpec
from detectron2.structures import Boxes
from detectron2.structures import ImageList, Instances

from detectron2.config import configurable
from detectron2.modeling.roi_heads import ROI_HEADS_REGISTRY, ROIHeads, StandardROIHeads
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.roi_heads.box_head import build_box_head
from detectron2.modeling.roi_heads.keypoint_head import build_keypoint_head
from detectron2.modeling.roi_heads.mask_head import build_mask_head
from detectron2.modeling.roi_heads import select_foreground_proposals

from zsis.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers
from zsis.modeling.structures import pairwise_iou_max_scores


@ROI_HEADS_REGISTRY.register()
class CustomStandardROIHeads(StandardROIHeads):
    @configurable
    def __init__(self, **kwargs):
        self.use_droploss = kwargs.pop('use_droploss')
        self.box2box_transform = kwargs.pop('box2box_transform')
        self.droploss_iou_thresh = kwargs.pop('droploss_iou_thresh')
        super(CustomStandardROIHeads, self).__init__(**kwargs)

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        ret['use_droploss'] = True
        ret['droploss_iou_thresh'] = cfg.MODEL.ROI_HEADS.DROPLOSS_IOU_THRESH
        ret['box2box_transform'] = Box2BoxTransform(weights=cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS)
        return ret

    def _forward_box(self, features: Dict[str, torch.Tensor], proposals: List[Instances]):
        """
        Forward logic of the box prediction branch. If `self.train_on_pred_boxes is True`,
            the function puts predicted boxes in the `proposal_boxes` field of `proposals` argument.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".

        Returns:
            In training, a dict of losses.
            In inference, a list of `Instances`, the predicted instances.
        """
        features = [features[f] for f in self.box_in_features]
        box_features = self.box_pooler(
            features, [x.proposal_boxes for x in proposals]
        )  # torch.Size([512 * batch_size, 256, 7, 7])
        box_features = self.box_head(box_features)  # torch.Size([512 * batch_size, 1024])
        predictions = self.box_predictor(
            box_features
        )  # [torch.Size([512 * batch_size, 2]), torch.Size([512 * batch_size, 4])]

        no_gt_found = False
        if self.use_droploss and self.training:
            # the first K proposals are GT proposals
            try:
                box_num_list = [len(x.gt_boxes) for x in proposals]
                gt_num_list = [torch.unique(x.gt_boxes.tensor[:100], dim=0).size()[0] for x in proposals]
            except:
                box_num_list = [0 for _ in proposals]
                gt_num_list = [0 for _ in proposals]
                no_gt_found = True

        if self.use_droploss and self.training and not no_gt_found:
            # NOTE: maximum overlapping with GT (IoU)
            predictions_delta = predictions[1]
            proposal_boxes = Boxes.cat([x.proposal_boxes for x in proposals])
            predictions_bbox = self.box2box_transform.apply_deltas(predictions_delta, proposal_boxes.tensor)
            idx_start = 0
            iou_max_list = []
            for idx, x in enumerate(proposals):
                idx_end = idx_start + box_num_list[idx]
                iou_max_list.append(
                    pairwise_iou_max_scores(predictions_bbox[idx_start:idx_end], x.gt_boxes[: gt_num_list[idx]].tensor)
                )
                idx_start = idx_end
            iou_max = torch.cat(iou_max_list, dim=0)

        del box_features

        if self.training:
            if self.use_droploss and not no_gt_found:
                weights = iou_max.le(self.droploss_iou_thresh).float()
                weights = 1 - weights.ge(1.0).float()
                losses = self.box_predictor.losses(predictions, proposals, weights=weights.detach())
            else:
                losses = self.box_predictor.losses(predictions, proposals)
            if self.train_on_pred_boxes:  # default is false
                with torch.no_grad():
                    pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(predictions, proposals)
                    for proposals_per_image, pred_boxes_per_image in zip(proposals, pred_boxes):
                        proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)
            return losses
        else:
            pred_instances, _ = self.box_predictor.inference(predictions, proposals)
            return pred_instances
