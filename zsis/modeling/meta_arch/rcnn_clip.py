#!/usr/bin/env python

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as T

from detectron2.config import configurable
from detectron2.structures import Instances, Boxes, ROIMasks
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

    def encode_text(self, text: List[str]) -> torch.Tensor:
        # TODO: Fix the device
        text_tokens = clip.tokenize(text).cuda()
        return self.forward(text_tokens)

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    @property
    def dtype(self) -> str:
        return self.resblocks[0].attn.out_proj.weight.dtype


@META_ARCH_REGISTRY.register()
class GeneralizedRCNN2(GeneralizedRCNN):
    def __init__(self, *args, **kwargs):
        super(GeneralizedRCNN2, self).__init__(*args, **kwargs)

        from detectron2.layers import FrozenBatchNorm2d

        for name, module in self.backbone.named_modules():
            if not isinstance(module, nn.BatchNorm2d):
                continue

            requires_grad = False
            for param in module.parameters():
                requires_grad |= param.requires_grad

            if requires_grad:
                continue

            names = name.split('.')
            module = getattr(self.backbone, names[0])
            for name in names[1:-1]:
                module = getattr(module, name)

            num_features = getattr(module, names[-1]).num_features
            setattr(module, names[-1], FrozenBatchNorm2d(num_features))

    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        if not self.training:
            return super().forward(batched_inputs)

        images = self.preprocess_image(batched_inputs)
        if 'instances' in batched_inputs[0]:
            gt_instances = [x['instances'].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        features = self.backbone(images.tensor)

        if self.proposal_generator is not None:
            proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        else:
            assert 'proposals' in batched_inputs[0]
            proposals = [x['proposals'].to(self.device) for x in batched_inputs]
            proposal_losses = {}

        _, detector_losses = self.roi_heads(images, features, proposals, gt_instances)

        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses


@META_ARCH_REGISTRY.register()
class GeneralizedRCNNClip(GeneralizedRCNN):
    @configurable
    def __init__(self, **kwargs):
        clip_model = kwargs.pop('clip_model', None)
        roi_pooler = kwargs.pop('roi_pooler', None)

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
        self.roi_pooler = roi_pooler

    @classmethod
    def from_config(cls, cfg) -> Dict[str, Any]:
        attrs = super().from_config(cfg)
        attrs.update(build_clip_model(cfg))
        attrs['topk'] = cfg.MODEL.CLIP.TOPK
        attrs['topk'] = cfg.MODEL.CLIP.TOPK
        attrs['prob_scale'] = cfg.MODEL.CLIP.PROB_SCALE
        attrs['crop_scale'] = cfg.MODEL.CLIP.CROP_SCALE
        attrs['image_format'] = cfg.INPUT.FORMAT

        # TODO: organize
        if isinstance(attrs['clip_model'].visual, clip.model.VisionTransformer):
            from detectron2.modeling.poolers import ROIPooler

            roi_pooler = ROIPooler(
                output_size=cfg.MODEL.CLIP.IMAGE_ENCODER.ROI_HEAD.POOLER_RESOLUTION,
                scales=(1.0 / 32.0,),
                sampling_ratio=cfg.MODEL.CLIP.IMAGE_ENCODER.ROI_HEAD.POOLER_SAMPLING_RATIO,
                pooler_type=cfg.MODEL.CLIP.IMAGE_ENCODER.ROI_HEAD.POOLER_TYPE,
            )
        else:
            roi_pooler = None

        attrs['roi_pooler'] = roi_pooler
        return attrs

    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        if not self.training:
            return self.inference(batched_inputs)

        raise NotImplementedError('Training method for clip + rcnn is not yet implemented')

    def inference(self, batched_inputs: List[Dict[str, torch.Tensor]]) -> List[Dict[str, torch.Tensor]]:
        index = 0
        instances = super().inference(batched_inputs)[index]

        if len(instances['instances']) == 0:
            return

        text_features = self.get_text_features(batched_inputs, index)

        if isinstance(self.clip_model.visual, clip.model.VisionTransformer) or not self.roi_pooler:
            top_probs, top_labels = self.patchwise_similarity(
                batched_inputs[index]['original_image'], instances['instances'].get('pred_boxes'), text_features
            )
        else:
            image = batched_inputs[index]['original_image']
            roi_features = self._image_features(image, instances['instances'])
            top_probs, top_labels = self.similarity(roi_features, text_features)

        instances['top_probs'] = top_probs
        instances['top_labels'] = top_labels
        return [instances]

    @torch.no_grad()
    def _image_features(self, image: np.ndarray, instances: Instances):
        image = Image.fromarray(image) if not isinstance(image, Image.Image) else image
        image = T.Compose([T.Resize((800, 1280)), *self.preprocessing.transforms[2:]])(image)
        image = (image.unsqueeze(0) if image.dim() == 3 else image).to(self.device)
        im_feats = self._visual_forward(image)
        roi_feats = self.roi_pooler([im_feats], [instances.pred_boxes])
        roi_feats = self.clip_model.visual.attnpool(roi_feats)
        return l2_norm(roi_feats, dim=-1, keepdim=True).float()

    def _visual_forward(self, x: torch.Tensor):
        model = self.clip_model.visual

        def stem(x):
            x = model.relu1(model.bn1(model.conv1(x)))
            x = model.relu2(model.bn2(model.conv2(x)))
            x = model.relu3(model.bn3(model.conv3(x)))
            x = model.avgpool(x)
            return x

        x = x.type(model.conv1.weight.dtype)
        x = stem(x)
        x = model.layer1(x)
        x = model.layer2(x)
        x = model.layer3(x)
        x = model.layer4(x)
        return x

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
        return self.similarity(image_features, text_features)

    def similarity(self, image_features, text_features):
        text_probs = (self.prob_scale.to(self.device) * image_features.float() @ text_features.float().t()).softmax(
            dim=-1
        )
        top_probs, top_labels = text_probs.topk(self.topk, dim=1)
        return top_probs, top_labels

    def get_text_features(self, batched_inputs: List[Dict[str, torch.Tensor]], index: int = 0) -> torch.Tensor:
        if 'text_descriptions' in batched_inputs[index]:
            for batch in batched_inputs:
                batch.update(self._text_features(batch.pop('text_descriptions')))
        return [inp['text_features'] for inp in batched_inputs][index]

    def _text_features(self, text_descriptions: List[str]) -> Dict[str, torch.Tensor]:
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
        frozen_image_encoder = kwargs.pop('frozen_image_encoder', False)
        frozen_text_encoder = kwargs.pop('frozen_text_encoder', False)

        text_encoder = kwargs.pop('text_encoder', None)
        roi_pooler = kwargs.pop('roi_pooler', None)
        attnpool = kwargs.pop('attnpool', None)
        self.topk = kwargs.pop('topk', 1)
        self.prob_scale = torch.FloatTensor([kwargs.pop('prob_scale', 100.0)])

        assert isinstance(self.topk, int) and self.topk > 0
        assert text_encoder is not None, 'Text encoding model is required'
        super(GeneralizedRCNNWithText, self).__init__(**kwargs)

        self.transformer = text_encoder
        self.roi_pooler = roi_pooler
        self.attnpool = attnpool

        embed_dim = self.transformer.embed_dim
        self.text_proj = nn.Linear(embed_dim, embed_dim)

        if frozen_image_encoder:
            logger.info('The Image backbone is completely frozen')
            for param in self.backbone.parameters():
                param.requires_grad_(False)
            for param in self.attnpool.parameters():
                param.requires_grad_(False)

        if frozen_text_encoder:
            logger.info('The text transformer is completely frozen')
            for param in self.transformer.parameters():
                param.requires_grad_(False)
            # transformer.logit_scale.requires_grad_(True)

    @classmethod
    def from_config(cls, cfg) -> Dict[str, Any]:
        """
        if cfg.MODEL.CLIP.IMAGE_ENCODER.FROZEN:
            from detectron2.modeling import build_backbone

            backbone = build_backbone(cfg)
            attrs = {
                'backbone': backbone,
                'proposal_generator': None,
                'roi_heads': None,
                'input_format': cfg.INPUT.FORMAT,
                'vis_period': cfg.VIS_PERIOD,
                'pixel_mean': cfg.MODEL.PIXEL_MEAN,
                'pixel_std': cfg.MODEL.PIXEL_STD,
            }
        else:
        """
        attrs = super().from_config(cfg)
        backbone = attrs['backbone']

        attrs['roi_pooler'] = build_roi_pooler(cfg, backbone.output_shape())
        attrs['text_encoder'] = build_text_encoder(cfg)
        attrs['attnpool'] = build_attention_pool(cfg)
        attrs['topk'] = cfg.MODEL.CLIP.TOPK
        attrs['prob_scale'] = cfg.MODEL.CLIP.PROB_SCALE
        attrs['frozen_image_encoder'] = cfg.MODEL.CLIP.IMAGE_ENCODER.FROZEN
        attrs['frozen_text_encoder'] = cfg.MODEL.CLIP.TEXT_ENCODER.FROZEN
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

        if self.proposal_generator is None and self.roi_heads is None:
            proposals, proposal_losses = [], {}
            for i, gt_instance in enumerate(gt_instances):
                gt_instance.set('proposal_boxes', gt_instance.get('gt_boxes'))
                gt_instance.set('objectness_logits', torch.ones(len(gt_instance)) * 10)
                proposals.append(gt_instance)
            detector_losses = proposal_losses = {}
        else:
            if self.proposal_generator is not None:
                proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
            else:
                assert 'proposals' in batched_inputs[0]
                proposals = [x['proposals'].to(self.device) for x in batched_inputs]
                proposal_losses = {}

            _, detector_losses = self.roi_heads(images, features, proposals, gt_instances)

        # roi pool the positive region features
        roi_features, proposals = self.roi_pooler(images, features, proposals, gt_instances)

        # clean up of memory that no longer required for this iteration
        del features, gt_instances, images

        roi_features = self.attnpool(roi_features)

        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)

        losses = {**detector_losses, **proposal_losses}
        return losses, roi_features, proposals

    def forward_text(self, batched_inputs: List[Dict[str, torch.Tensor]]) -> Dict[str, Any]:
        """
        Forward the text descriptions through the text encoding network
        """
        key = 'gt_descriptions' if self.training else 'descriptions'
        text_descriptions = []
        for batched_input in batched_inputs:
            text_descriptions.extend(batched_input[key])
        text_features = self.transformer.encode_text(text_descriptions)
        text_features = self.text_proj(text_features)

        if not self.training:
            return text_features

        # TODO: Implement loss function
        losses = {}
        return losses, text_features

    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        import IPython, sys

        IPython.embed()
        sys.exit()

        if not self.training:
            return self.inference(batched_inputs)

        if self.transformer.logit_scale.requires_grad:
            # clamp to ln(100), as in the paper
            with torch.no_grad():
                self.transformer.logit_scale.clamp_(0, np.log(100))

        image_losses, roi_features, proposals = self.forward_images(batched_inputs)
        text_losses, text_features = self.forward_text(batched_inputs)

        # accumate the text features for the proposals
        assert len(batched_inputs) == len(proposals), 'Lenght of batched input and proposals doesnt match'

        _roi_text_features, k = [], 0
        for batched_input, proposal in zip(batched_inputs, proposals):
            indices = proposal.get('gt_instance_labels').long() + k
            _roi_text_features.append(text_features[indices])
            k += len(batched_input['instances'])
        text_features = torch.vstack(_roi_text_features)

        # l2 normalization
        roi_features, text_features = [l2_norm(f) for f in [roi_features, text_features]]

        # cosine similarity as logits
        logit_scale = self.transformer.logit_scale.exp()
        logits = logit_scale * roi_features @ text_features.t()
        # logits_per_text = logits.t()

        losses = {**self.losses(logits), **image_losses, **text_losses}

        del roi_features, text_features, logits, image_losses, text_losses

        return losses

    def inference(
        self,
        batched_inputs: List[Dict[str, torch.Tensor]],
        detected_instances: Optional[List[Instances]] = None,
        do_postprocess: bool = True,
    ):
        assert not self.training, 'Running inference in training mode'

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        if detected_instances is None:
            if self.proposal_generator is not None:
                proposals, _ = self.proposal_generator(images, features, None)
            else:
                assert 'proposals' in batched_inputs[0]
                proposals = [x['proposals'].to(self.device) for x in batched_inputs]

            results, _ = self.roi_heads(images, features, proposals, None)
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results = self.roi_heads.forward_with_given_boxes(features, detected_instances)

        # get roi wise features
        roi_features, _ = self.roi_pooler(images, features, results, None)
        roi_features = self.attnpool(roi_features)

        # run the text encoding
        text_features = self.forward_text(batched_inputs)

        # l2 normalization
        roi_features, text_features = [l2_norm(f) for f in [roi_features, text_features]]
        text_probs = (self.prob_scale.to(self.device) * roi_features @ text_features.t()).softmax(dim=-1)

        if do_postprocess:
            assert not torch.jit.is_scripting(), 'Scripting is not supported for postprocess.'
            results, valid_indices = self.postprocess(results, batched_inputs, images.image_sizes)
            text_probs = text_probs[valid_indices]

        top_probs, top_labels = text_probs.topk(self.topk, dim=1)

        # TODO: support for multiple batch size
        for i in range(len(results)):
            results[i]['top_probs'] = top_probs
            results[i]['top_labels'] = top_labels
        return results

    def losses(self, logits: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {'loss_clip': ((self.cross_entropy(logits, 0) + self.cross_entropy(logits, 1)) / 2.0) * 1.0}

    def cross_entropy(self, logits: torch.Tensor, dim: int) -> torch.Tensor:
        log_probs = nn.functional.log_softmax(logits, dim=dim)
        return -torch.mean(torch.diag(log_probs))

    def postprocess(
        self, instances, batched_inputs: List[Dict[str, torch.Tensor]], image_sizes: Tuple[int, int]
    ) -> Tuple[List[Dict[str, Instances]], List[torch.Tensor]]:
        """
        Rescale the output instances to the target size.
        """
        processed_results = []
        valid_indices = []
        for results_per_image, input_per_image, image_size in zip(instances, batched_inputs, image_sizes):
            height = input_per_image.get('height', image_size[0])
            width = input_per_image.get('width', image_size[1])
            r, indices = detector_postprocess(results_per_image, height, width)
            valid_indices.append(indices)
            processed_results.append({'instances': r})
        return processed_results, valid_indices


def detector_postprocess(results: Instances, output_height: int, output_width: int, mask_threshold: float = 0.5):
    """
    Resize the output instances.
    The input images are often resized when entering an object detector.
    As a result, we often need the outputs of the detector in a different
    resolution from its inputs.
    This function will resize the raw outputs of an R-CNN detector
    to produce outputs according to the desired output resolution.
    Args:
        results (Instances): the raw outputs from the detector.
            `results.image_size` contains the input image resolution the detector sees.
            This object might be modified in-place.
        output_height, output_width: the desired output resolution.
    Returns:
        Instances: the resized output from the model, based on the output resolution
    """
    if isinstance(output_width, torch.Tensor):
        # This shape might (but not necessarily) be tensors during tracing.
        # Converts integer tensors to float temporaries to ensure true
        # division is performed when computing scale_x and scale_y.
        output_width_tmp = output_width.float()
        output_height_tmp = output_height.float()
        new_size = torch.stack([output_height, output_width])
    else:
        new_size = (output_height, output_width)
        output_width_tmp = output_width
        output_height_tmp = output_height

    scale_x, scale_y = (
        output_width_tmp / results.image_size[1],
        output_height_tmp / results.image_size[0],
    )
    results = Instances(new_size, **results.get_fields())

    if results.has("pred_boxes"):
        output_boxes = results.pred_boxes
    elif results.has("proposal_boxes"):
        output_boxes = results.proposal_boxes
    else:
        output_boxes = None
    assert output_boxes is not None, "Predictions must contain boxes!"

    output_boxes.scale(scale_x, scale_y)
    output_boxes.clip(results.image_size)

    results = results[output_boxes.nonempty()]

    if results.has("pred_masks"):
        if isinstance(results.pred_masks, ROIMasks):
            roi_masks = results.pred_masks
        else:
            # pred_masks is a tensor of shape (N, 1, M, M)
            roi_masks = ROIMasks(results.pred_masks[:, 0, :, :])
        results.pred_masks = roi_masks.to_bitmasks(
            results.pred_boxes, output_height, output_width, mask_threshold
        ).tensor  # TODO return ROIMasks/BitMask object in the future

    if results.has("pred_keypoints"):
        results.pred_keypoints[:, :, 0] *= scale_x
        results.pred_keypoints[:, :, 1] *= scale_y

    return results, output_boxes.nonempty().cpu().numpy()
