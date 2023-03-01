#!/usr/bin/env python

from typing import List

import numpy as np
import torch
from detectron2.engine.defaults import DefaultPredictor as _DefaultPredictor

__all__ = [
    'DefaultPredictor',
]


class DefaultPredictor(_DefaultPredictor):
    def __init__(self, cfg, **kwargs):
        super(DefaultPredictor, self).__init__(cfg)

        self._text_descriptions = None
        self._text_features = None
        self._text_tokens = None

    @torch.no_grad()
    def __call__(self, original_image: np.ndarray, text_descriptions: List[str] = None):
        assert text_descriptions is not None or self._text_descriptions is not None, 'Set or pass the text description'

        height, width = original_image.shape[:2]
        image = self.process_rgb(original_image)

        inputs = {'image': image, 'height': height, 'width': width, 'original_image': original_image}
        if text_descriptions is not None:
            inputs['descriptions'] = text_descriptions
        else:
            assert self._text_features is not None, 'Text features is None'
            inputs['text_features'] = self._text_features

        predictions = self.model([inputs])[0]
        return predictions

    def process_rgb(self, original_image: np.ndarray) -> torch.Tensor:
        if self.input_format == 'RGB':
            original_image = original_image[:, :, ::-1]
        image = self.aug.get_transform(original_image).apply_image(original_image)
        return torch.as_tensor(image.astype('float32').transpose(2, 0, 1))

    @torch.no_grad()
    def set_text_descriptions(self, text_descriptions: List[str]):
        text_feats = self.model.get_text_features(text_descriptions)
        self._text_features = text_feats['text_features']
        self._text_tokens = text_feats['text_tokens']
        self._text_descriptions = text_descriptions

    @property
    def text_descriptions(self):
        return self._text_descriptions
