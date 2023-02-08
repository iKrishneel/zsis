#!/usr/bin/env python

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from detectron2.config import configurable
from detectron2.modeling import Backbone
from detectron2.structures import Instances, Boxes
from detectron2.modeling.meta_arch import META_ARCH_REGISTRY, GeneralizedRCNN
from detectron2.utils.logger import setup_logger

import clip
from clip.model import AttentionPool2d, Transformer


__all__ = ['GeneralizedRCNNWithText', 'GeneralizedRCNNClip']


logger = setup_logger(name='clip')


def build_attention_mask(cfg) -> torch.Tensor:
    context_length = cfg.MODEL.CLIP.TEXT_ENCODER.CONTEXT_LENGTH
    mask = torch.empty(context_length, context_length)
    mask.fill_(float("-inf"))
    mask.triu_(1)  # zero out the lower diagonal
    return mask


def build_text_encoder(cfg) -> Transformer:
    transformer_width = cfg.MODEL.CLIP.TEXT_ENCODER.TRANSFORMER_WIDTH
    transformer_layers = cfg.MODEL.CLIP.TEXT_ENCODER.TRANSFORMER_LAYERS
    transformer_heads = cfg.MODEL.CLIP.TEXT_ENCODER.TRANSFORMER_HEADS
    transformer = Transformer(
        width=transformer_width, layers=transformer_layers, heads=transformer_heads, attn_mask=build_attention_mask(cfg)
    )

    if cfg.MODEL.CLIP.TEXT_ENCODER.FROZEN:
        logger.info('Full text encoder freeze')
        for param in transformer.parameters():
            param.requires_grad_(False)

    return transformer


def build_clip_model(cfg):
    clip_cfg = cfg.MODEL.CLIP
    model, preprocessing = clip.load(clip_cfg.ARCHITECTURE)

    if clip_cfg.EVAL_ONLY:
        model.to(cfg.MODEL.DEVICE).eval()

    logger.info(f"Model parameters: {np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
    logger.info(f"Input resolution: {model.visual.input_resolution}")
    logger.info(f"Context length: {model.context_length}")
    logger.info(f"Vocab size: {model.vocab_size}")

    return {'clip_model': model, 'preprocessing': preprocessing}


def get_roi_size(bbox: np.ndarray, im_size: List[int], scale: float = 1.0, use_max_len: bool = False) -> np.ndarray:
    h, w = im_size[:2]
    bbox = bbox.reshape(2, -1)
    center = np.average(bbox, axis=0)
    lenght = (np.max(np.diff(bbox, axis=0) / 2) if use_max_len else (np.diff(bbox, axis=0))[0] / 2) * scale
    lenght = np.array([lenght, lenght]) if isinstance(lenght, float) else lenght
    x1, y1, x2, y2 = [*(center - lenght), *(center + lenght)]
    x1, x2 = (0, x2 - x1) if x1 < 0 else (x1, x2)
    x1, x2 = (x1 - (x2 - w), w) if x2 > w else (x1, x2)
    y1, y2 = (0, y2 - y1) if y1 < 0 else (y1, y2)
    y1, y2 = (y1 - (y2 - h), h) if y2 > h else (y1, y2)
    return np.array([x1, y1, x2, y2], dtype=np.intp)


@META_ARCH_REGISTRY.register()
class GeneralizedRCNNClip(GeneralizedRCNN):
    @configurable
    def __init__(self, **kwargs):
        clip_model = kwargs.pop('clip_model', None)
        self.preprocessing = kwargs.pop('preprocessing', None)
        self.topk = kwargs.pop('topk', 1)
        self.crop_scale = kwargs.pop('crop_scale', 1.0)
        self.prob_scale = torch.FloatTensor([kwargs.pop('prob_scale', 100.0)]).to(torch.float16)

        assert clip_model is not None, 'Clip model is required'
        assert isinstance(self.topk, int) and self.topk > 0
        assert self.crop_scale > 0, 'Crop scale must be greater then 0'
        super(GeneralizedRCNNClip, self).__init__(**kwargs)

        self.clip_model = clip_model

    @classmethod
    def from_config(cls, cfg) -> Dict[str, Any]:
        attrs = super().from_config(cfg)
        attrs.update(build_clip_model(cfg))
        attrs['topk'] = cfg.MODEL.CLIP.TOPK
        attrs['prob_scale'] = cfg.MODEL.CLIP.PROB_SCALE
        attrs['crop_scale'] = cfg.MODEL.CLIP.CROP_SCALE
        return attrs

    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        if not self.training:
            return self.inference(batched_inputs)

        raise NotImplementedError('Training method for clip + rcnn is not yet implemented')

    def inference(self, batched_inputs: List[Dict[str, torch.Tensor]]) -> List[Dict[str, torch.Tensor]]:
        index = 0
        instances = super().inference(batched_inputs)[index]

        # extract features if not provided from cache
        if 'text_descriptions' in batched_inputs[index]:
            for batch in batched_inputs:
                batch.update(self.get_text_features(batch.pop('text_descriptions')))

        text_features = [inp['text_features'] for inp in batched_inputs][index]

        """
        assert (
            len(instances) == len(text_features)
        ), f'Lenght of instances {len(instances)} and text features {len(text_features)} are not same'
        """

        top_probs, top_labels = self.patchwise_similarity(
            batched_inputs[index]['original_image'], instances['instances'].get('pred_boxes'), text_features
        )

        instances['top_probs'] = top_probs
        instances['top_labels'] = top_labels
        return [instances]

    def patchwise_similarity(self, image: np.ndarray, bboxes: Boxes, text_features: torch.Tensor):
        image = image[:, :, ::-1]
        im_crops = []
        for bbox in bboxes:
            bbox = bbox.int().cpu().numpy()
            x1, y1, x2, y2 = get_roi_size(bbox, image.shape[:2], scale=self.crop_scale, use_max_len=False)
            im_crop = self.preprocessing(Image.fromarray(image[y1:y2, x1:x2].copy()))
            im_crops.append(im_crop)

        image_features = self.clip_model.encode_image(torch.stack(im_crops).to(self.device))
        image_features /= image_features.norm(dim=-1, keepdim=True)

        text_probs = (self.prob_scale.to(self.device) * image_features @ text_features.T).softmax(dim=-1)
        top_probs, top_labels = text_probs.topk(self.topk, dim=1)
        return top_probs, top_labels

    def get_text_features(self, text_descriptions: List[str]) -> Dict[str, torch.Tensor]:
        assert isinstance(
            text_descriptions, (list, tuple)
        ), f'Expects text descriptions as list but got {text_descriptions}'
        for text in text_descriptions:
            assert isinstance(text, str), f'Text description must be a str but got a str: {text}'

        text_tokens = clip.tokenize(text_descriptions).to(self.device)
        text_features = self.clip_model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        return {'text_features': text_features, 'text_tokens': text_tokens}


@META_ARCH_REGISTRY.register()
class GeneralizedRCNNWithText(GeneralizedRCNN):
    @configurable
    def __init__(self, **kwargs):
        text_encoder = kwargs.pop('text_encoder', None)
        assert text_encoder is not None, 'Text encoding model is required'
        super(GeneralizedRCNNWithText, self).__init__(**kwargs)
        self.text_encoder = text_encoder

        # self.attenpool = AttentionPool2d()

    @classmethod
    def from_config(cls, cfg) -> Dict[str, Any]:
        attrs = super().from_config(cfg)
        attrs['text_encoder'] = build_text_encoder(cfg)
        return attrs

    def forward_images(self, batched_inputs: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Forward the images through the backbone and the network heads
        """

        images = self.preprocess_image(batched_inputs)
        gt_instances = (
            [x['instances'].to(self.device) for x in batched_inputs] if 'instances' in batched_inputs[0] else None
        )
        features = self.backbone(images.tensor)

        if self.proposal_generator is not None:
            proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        else:
            assert 'proposals' in batched_inputs[0]
            proposals = [x['proposals'].to(self.device) for x in batched_inputs]
            proposal_losses = {}

        _, detector_losses = self.roi_heads(images, features, proposals, gt_instances)

        import IPython, sys; IPython.embed(header='Embedding in Forward'); sys.exit()

        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses

    def forward_text(self, batched_inputs: List[Dict[str, torch.Tensor]]) -> Dict:
        """
        Forward the text descriptions through the text encoding network
        """
        pass

    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        if not self.training:
            return self.inference(batched_inputs)
        
        image_losses = self.forward_images(batched_inputs)
        # text_losses = self.text_encoder(batched_inputs)

    def inference(
        self,
        batched_inputs: List[Dict[str, torch.Tensor]],
        detected_instances: Optional[List[Instances]] = None,
        do_postprocess: bool = True,
    ):
        pass
