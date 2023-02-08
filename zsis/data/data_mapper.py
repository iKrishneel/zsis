#!/usr/bin/env python

from copy import deepcopy
import os
from typing import Any, Dict, List

import numpy as np
import torch
from detectron2.config import configurable
from detectron2.data import detection_utils
from detectron2.data import transforms as T
from detectron2.data.dataset_mapper import DatasetMapper as _DM


class DatasetMapper(_DM):
    @configurable
    def __init__(self, *args, **kwargs):
        super(DatasetMapper, self).__init__(*args, **kwargs)

        self.label_name_map = {0: 'bean', 1: 'leaf'}

    @classmethod
    def from_config(cls, cfg, is_train: bool = True) -> Dict[str, Any]:
        ret = _DM.from_config(cfg, is_train)
        return ret

    def __call__(self, dataset_dict):
        ddict = self._process(deepcopy(dataset_dict))
        if len(ddict['instances'].gt_boxes) > 0:
            return ddict
        return self._process(deepcopy(dataset_dict), False)

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

        text_descriptions = self.get_text_descriptions(dataset_dict)

        if 'annotations' in dataset_dict:
            self._transform_annotations(dataset_dict, transforms, image_shape)

        dataset_dict['text_descriptions'] = text_descriptions
        return dataset_dict

    def get_text_descriptions(self, dataset_dict) -> List[str]:
        text_descriptions = []
        for annotation in dataset_dict['annotations']:
            text = self.label_name_map[annotation['category_id']]
            description = f'This is a image of a {text}'
            text_descriptions.append(description)
        return text_descriptions


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
