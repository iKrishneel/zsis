#!/usr/bin/env python

from typing import Callable, Dict, List, Optional, Tuple, Union
import torch

from detectron2.layers import cat, cross_entropy
from detectron2.structures import Instances

from detectron2.modeling.roi_heads.fast_rcnn import _log_classification_stats
from detectron2.modeling.roi_heads import FastRCNNOutputLayers as _FastRCNNOutputLayers


__all__ = ['FastRCNNOutputLayers']


class FastRCNNOutputLayers(_FastRCNNOutputLayers):
    def losses(self, predictions, proposals, weights=None):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were used
                to compute predictions. The fields ``proposal_boxes``, ``gt_boxes``,
                ``gt_classes`` are expected.
            weights: weights for reweighting the loss of each instance based on IoU
        Returns:
            Dict[str, Tensor]: dict of losses
        """
        scores, proposal_deltas = predictions

        # parse classification outputs
        gt_classes = cat([p.gt_classes for p in proposals], dim=0) if len(proposals) else torch.empty(0)
        _log_classification_stats(scores, gt_classes)

        # parse box regression outputs
        if len(proposals):
            proposal_boxes = cat([p.proposal_boxes.tensor for p in proposals], dim=0)  # Nx4
            assert not proposal_boxes.requires_grad, "Proposals should not require gradients!"
            # If "gt_boxes" does not exist, the proposals must be all negative and
            # should not be included in regression loss computation.
            # Here we just use proposal_boxes as an arbitrary placeholder because its
            # value won't be used in self.box_reg_loss().
            gt_boxes = cat(
                [(p.gt_boxes if p.has("gt_boxes") else p.proposal_boxes).tensor for p in proposals],
                dim=0,
            )
        else:
            proposal_boxes = gt_boxes = torch.empty((0, 4), device=proposal_deltas.device)

        if self.use_sigmoid_ce:
            loss_cls = self.sigmoid_cross_entropy_loss(scores, gt_classes)
        else:
            if weights != None:
                loss_cls = (weights * cross_entropy(scores, gt_classes, reduction='none')).mean()
            else:
                loss_cls = cross_entropy(scores, gt_classes, reduction="mean")

        losses = {
            "loss_cls": loss_cls,
            "loss_box_reg": self.box_reg_loss(proposal_boxes, gt_boxes, proposal_deltas, gt_classes),
        }
        return {k: v * self.loss_weight.get(k, 1.0) for k, v in losses.items()}
