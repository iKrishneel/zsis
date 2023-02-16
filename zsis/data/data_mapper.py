#!/usr/bin/env python

from copy import deepcopy
import os
from typing import Any, Dict, List
from dataclasses import dataclass, field

import json
import numpy as np
import torch
from detectron2.config import configurable
from detectron2.data import detection_utils
from detectron2.data import transforms as T
from detectron2.data.dataset_mapper import DatasetMapper as _DM

import clip


class DatasetMapper(_DM):
    @configurable
    def __init__(self, *args, **kwargs):
        self.is_class_agnostic = kwargs.pop('is_class_agnostic')
        self.with_text = kwargs.pop('with_text', False)
        super(DatasetMapper, self).__init__(*args, **kwargs)

        self.label_name_map = {0: 'bean', 1: 'leaf', 2: 'bean', 7: 'leaf'}

    @classmethod
    def from_config(cls, cfg, is_train: bool = True) -> Dict[str, Any]:
        ret = _DM.from_config(cfg, is_train)
        ret['is_class_agnostic'] = cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG
        ret['with_text'] = cfg.MODEL.META_ARCHITECTURE in ['GeneralizedRCNNWithText', 'GeneralizedRCNNClip']
        return ret

    def __call__(self, dataset_dict):
        ddict = self._process(deepcopy(dataset_dict))
        return ddict

    def _process(self, dataset_dict: Dict[str, Any]):
        image = detection_utils.read_image(dataset_dict['file_name'], format=self.image_format)
        detection_utils.check_image_size(dataset_dict, image)

        # transformation
        aug_input = T.AugInput(image)
        transforms = self.augmentations(aug_input)
        image = aug_input.image

        image_shape = image.shape[:2]
        dataset_dict['image'] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        if not self.is_train:
            for key in ['annotations', 'sem_seg_file_name']:
                dataset_dict.pop('key', None)
            return dataset_dict

        text_descriptions, class_labels = self.get_instance_meta_data(dataset_dict)

        if 'annotations' in dataset_dict:
            self._transform_annotations(dataset_dict, transforms, image_shape)

        if self.with_text:
            dataset_dict['instances'].set('gt_class_labels', torch.Tensor(class_labels))
            dataset_dict['instances'].set('gt_text_tokens', clip.tokenize(text_descriptions))
            dataset_dict['gt_descriptions'] = text_descriptions

        return dataset_dict

    def get_instance_meta_data(self, dataset_dict) -> List[List[Any]]:
        return self.create_synthetic_caption(dataset_dict)

    def create_synthetic_caption(self, dataset_dict) -> List[List[Any]]:
        text_descriptions = []
        class_labels = []

        # LOAD CAPTION FROM FILE
        caption_data = self.load_caption_from_file(dataset_dict)

        key = 'category_id'
        for i, annotation in enumerate(dataset_dict['annotations']):
            if self.with_text:
                text = self.label_name_map[annotation[key]]
                description = f'This is a image of a {text}'
                for cap in caption_data['captions']:
                    if annotation['bbox'] == cap['bbox']:
                        description = cap['caption']
                        break

                # textual description of the image roi
                text_descriptions.append(description)
                class_labels.append(annotation[key])

            # for class agnostic segmentation, it will be binary classification bg / fg
            if self.is_class_agnostic:
                annotation[key] = 0

        return text_descriptions, class_labels

    def load_caption_from_file(self, dataset_dict: Dict[str, Any]) -> Dict[str, Any]:
        # TEMP
        return {'captions': []}
        
        file_name = dataset_dict['file_name'].split(os.sep)[-1]
        ext = file_name.split('.')[1]
        file_name = file_name.replace(ext, 'json')
        root = os.path.join(
            f'{os.sep}'.join(dataset_dict['file_name'].split(os.sep)[:-3]), 'caption/train2017/{file_name}'
        )
        with open(root, 'r') as fp:
            data = json.load(fp)

        assert data['image_id'] == dataset_dict['image_id'], f'The image_id is not same {root}'
        return data


if __name__ == '__main__':
    import sys
    from detectron2.data.datasets import load_coco_json
    from zsis.config import get_cfg

    c = sys.argv[1]
    c1 = sys.argv[2]

    cfg = get_cfg()
    cfg.merge_from_file(c)

    print('loading dataset annotations')
    dataset_dict = load_coco_json(c1 + '/trainval.json', c1)

    mapper = DatasetMapper(cfg)
    x = mapper(dataset_dict[2])

    import IPython

    IPython.embed()
