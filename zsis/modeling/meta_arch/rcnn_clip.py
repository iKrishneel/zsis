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
from clip.model import AttentionPool2d, Transformer, LayerNorm

from ..roi_heads import build_roi_pooler


__all__ = ['GeneralizedRCNNWithText', 'GeneralizedRCNNClip']


logger = setup_logger(name='clip')


def build_attention_mask(cfg) -> torch.Tensor:
    context_length = cfg.MODEL.CLIP.TEXT_ENCODER.CONTEXT_LENGTH
    mask = torch.empty(context_length, context_length)
    mask.fill_(float("-inf"))
    mask.triu_(1)  # zero out the lower diagonal
    return mask


def build_text_encoder(cfg) -> Transformer:
    vocab_size = cfg.MODEL.CLIP.TEXT_ENCODER.VOCAB_SIZE
    context_length = cfg.MODEL.CLIP.TEXT_ENCODER.CONTEXT_LENGTH
    embed_dim = cfg.MODEL.CLIP.TEXT_ENCODER.EMBED_DIM
    transformer_width = cfg.MODEL.CLIP.TEXT_ENCODER.TRANSFORMER_WIDTH
    transformer_layers = cfg.MODEL.CLIP.TEXT_ENCODER.TRANSFORMER_LAYERS
    transformer_heads = cfg.MODEL.CLIP.TEXT_ENCODER.TRANSFORMER_HEADS
    transformer = TextTransformer(
        vocab_size=vocab_size,
        context_length=context_length,
        embed_dim=embed_dim,
        width=transformer_width,
        layers=transformer_layers,
        heads=transformer_heads,
        attn_mask=build_attention_mask(cfg),
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

    logger.info(f'Model parameters: {np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}')
    logger.info(f'Input resolution: {model.visual.input_resolution}')
    logger.info(f'Context length: {model.context_length}')
    logger.info(f'Vocab size: {model.vocab_size}')

    logger.warn('Changing of floating precision is currently not supported!')
    return {'clip_model': model, 'preprocessing': preprocessing}


def build_attention_pool(cfg) -> AttentionPool2d:
    return AttentionPool2d(
        cfg.MODEL.CLIP.IMAGE_ENCODER.ROI_HEAD.POOLER_RESOLUTION,
        cfg.MODEL.CLIP.IMAGE_ENCODER.ATTN_EMBED_DIM,
        cfg.MODEL.CLIP.IMAGE_ENCODER.ATTN_NUM_HEADS,
        cfg.MODEL.CLIP.IMAGE_ENCODER.ATTN_OUTPUT_DIM,
    )


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


def l2_norm(x: torch.Tensor, dim: int = 1, keepdim: bool = True) -> torch.Tensor:
    return x / x.norm(dim=dim, keepdim=keepdim)


class TextTransformer(Transformer):
    def __init__(self, vocab_size: int, context_length: int, embed_dim: int, **kwargs):
        super(TextTransformer, self).__init__(**kwargs)
        self.context_length = context_length
        transformer_width = kwargs.get('width')

        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.ln_final = LayerNorm(transformer_width)
        self.initialize_parameters()

        self._embed_dim = embed_dim

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.width**-0.5) * ((2 * self.layers) ** -0.5)
        attn_std = self.width**-0.5
        fc_std = (2 * self.width) ** -0.5
        for block in self.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.width**-0.5)

    def forward(self, text: List[torch.Tensor]):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.resblocks(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    @property
    def dtype(self) -> str:
        return self.resblocks[0].attn.out_proj.weight.dtype


@META_ARCH_REGISTRY.register()
class GeneralizedRCNNClip(GeneralizedRCNN):
    @configurable
    def __init__(self, **kwargs):
        clip_model = kwargs.pop('clip_model', None)
        self.preprocessing = kwargs.pop('preprocessing', None)
        self.topk = kwargs.pop('topk', 1)
        self.crop_scale = kwargs.pop('crop_scale', 1.0)
        self.prob_scale = torch.FloatTensor([kwargs.pop('prob_scale', 100.0)]).to(torch.float16)
        self.image_format = kwargs.pop('image_format')

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
        attrs['image_format'] = cfg.INPUT.FORMAT
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
        # clip takes takes RGB as input
        if self.image_format == 'BGR':
            image = image[:, :, ::-1]
        im_crops = []
        for bbox in bboxes:
            bbox = bbox.int().cpu().numpy()
            x1, y1, x2, y2 = get_roi_size(bbox, image.shape[:2], scale=self.crop_scale, use_max_len=False)
            im_crop = self.preprocessing(Image.fromarray(image[y1:y2, x1:x2].copy()))
            im_crops.append(im_crop)

        image_features = l2_norm(self.clip_model.encode_image(torch.stack(im_crops).to(self.device)), dim=-1)
        text_probs = (self.prob_scale.to(self.device) * image_features @ text_features.t()).softmax(dim=-1)
        top_probs, top_labels = text_probs.topk(self.topk, dim=1)
        return top_probs, top_labels

    def get_text_features(self, text_descriptions: List[str]) -> Dict[str, torch.Tensor]:
        assert isinstance(
            text_descriptions, (list, tuple)
        ), f'Expects text descriptions as list but got {text_descriptions}'
        for text in text_descriptions:
            assert isinstance(text, str), f'Text description must be a str but got a str: {text}'

        text_tokens = clip.tokenize(text_descriptions).to(self.device)
        text_features = l2_norm(self.clip_model.encode_text(text_tokens), dim=-1)
        return {'text_features': text_features, 'text_tokens': text_tokens}


@META_ARCH_REGISTRY.register()
class GeneralizedRCNNWithText(GeneralizedRCNN):
    @configurable
    def __init__(self, **kwargs):
        text_encoder = kwargs.pop('text_encoder', None)
        roi_pooler = kwargs.pop('roi_pooler', None)
        attnpool = kwargs.pop('attnpool', None)

        assert text_encoder is not None, 'Text encoding model is required'
        super(GeneralizedRCNNWithText, self).__init__(**kwargs)

        self.transformer = text_encoder
        self.roi_pooler = roi_pooler
        self.attnpool = attnpool

        embed_dim = self.transformer.embed_dim
        self.text_proj = nn.Linear(embed_dim, embed_dim)

    @classmethod
    def from_config(cls, cfg) -> Dict[str, Any]:
        attrs = super().from_config(cfg)

        output_shape = attrs['backbone'].output_shape()
        attrs['roi_pooler'] = build_roi_pooler(cfg, output_shape)
        attrs['text_encoder'] = build_text_encoder(cfg)
        attrs['attnpool'] = build_attention_pool(cfg)
        return attrs

    def forward_images(self, batched_inputs: List[Dict[str, torch.Tensor]]) -> List[Any]:
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

        # try:
        _, detector_losses = self.roi_heads(images, features, proposals, gt_instances)
        # except IndexError:
        # import IPython, sys; IPython.embed(header='Embedded'); sys.exit()

        # roi pool the positive region features
        roi_features, proposals = self.roi_pooler(images, features, proposals, gt_instances)
        roi_features = self.attnpool(roi_features)

        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)

        losses = {**detector_losses, **proposal_losses}
        return losses, roi_features, proposals

    def forward_text(self, proposals: List[Dict[str, torch.Tensor]]) -> Dict[str, Any]:
        """
        Forward the text descriptions through the text encoding network
        """
        text_tokens = torch.cat([p.get('gt_text_tokens') for p in proposals])
        text_features = self.transformer(text_tokens)
        text_features = self.text_proj(text_features)

        # TODO: Implement loss function
        losses = {}
        return losses, text_features

    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        if not self.training:
            return self.inference(batched_inputs)

        image_losses, roi_features, proposals = self.forward_images(batched_inputs)
        text_losses, text_features = self.forward_text(proposals)

        # l2 normalization
        roi_features, text_features = [l2_norm(f) for f in [roi_features, text_features]]

        # cosine similarity as logits
        logit_scale = self.transformer.logit_scale.exp()
        logits_per_image = logit_scale * roi_features @ text_features.t()
        logits_per_text = logits_per_image.t()
        
        losses = {**self.losses(logits_per_image), **image_losses, **text_losses}
        return losses

    def losses(self, logits: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {'loss_clip': ((self.cross_entropy(logits, 0) + self.cross_entropy(logits, 1)) / 2.0) * 1.0}

    def cross_entropy(self, logits: torch.Tensor, dim: int) -> torch.Tensor:
        log_probs = nn.functional.log_softmax(logits, dim=dim)
        nll = torch.diag(log_probs)
        return -torch.mean(nll)

    def inference(
        self,
        batched_inputs: List[Dict[str, torch.Tensor]],
        detected_instances: Optional[List[Instances]] = None,
        do_postprocess: bool = True,
    ):
        pass
