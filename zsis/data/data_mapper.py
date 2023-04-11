#!/usr/bin/env python

from copy import deepcopy
import os
from typing import Any, Dict, List, Union, Callable

import importlib
import numpy as np
import torch
from detectron2.config import configurable
from detectron2.data import detection_utils, MetadataCatalog
from detectron2.data import transforms as T
from detectron2.data.dataset_mapper import DatasetMapper as _DM


def get_loader(filename: str) -> Callable:
    if '.json' in filename:
        module, func = 'json', 'load'
    elif '.yaml' in filename or '.yml' in filename:
        module, func = 'yaml', 'safe_load'
    else:
        raise TypeError('fUnknown file type {filename}')

    return getattr(importlib.import_module(module), func)


def load_file(filename: str) -> Dict[str, Any]:
    assert os.path.isfile(filename), f'{filename} not found!'
    loader = get_loader(filename)
    with open(filename, 'r') as fp:
        data = loader(fp)
    return data


class DatasetMapper(_DM):
    @configurable
    def __init__(self, *args, **kwargs) -> None:
        self.is_class_agnostic = kwargs.pop('is_class_agnostic')
        self.with_text = kwargs.pop('with_text', False)
        self.label_name_map = kwargs.pop('label_name_map', None)
        self.min_bbox_wh = kwargs.pop('min_bbox_wh', [30, 30])
        super(DatasetMapper, self).__init__(*args, **kwargs)

        self.sample_descriptions = (
            # 'This is an image of a %s',
            # 'This image contains %s',
            # '%s is in this image',
            # 'This is a photo of a %s',
            '%s',
        )

    @classmethod
    def from_config(cls, cfg, is_train: bool = True) -> Dict[str, Any]:
        ret = _DM.from_config(cfg, is_train)
        ret['is_class_agnostic'] = cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG
        ret['with_text'] = cfg.MODEL.META_ARCHITECTURE in [
            'GeneralizedRCNNWithText',
            'GeneralizedRCNNClip',
            'GeneralizedRCNNClipPrompter',
        ]

        # TODO: Use argument for size
        ret['min_bbox_wh'] = [30, 30]

        if is_train:
            meta = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
            # label_name_map = {i: k for i, k in enumerate(meta.thing_classes)}
            label_name_map = {0: 'leaf', 1: 'bean'}
            ret['label_name_map'] = label_name_map
        return ret

    def __call__(self, dataset_dict: Dict[str, Any]):
        if len(dataset_dict['annotations']) == 0:
            return

        dataset_dict = deepcopy(dataset_dict)
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

        self.update_annotation_with_instance_wise_captions(dataset_dict)

        if 'annotations' in dataset_dict:
            self._transform_annotations(dataset_dict, transforms, image_shape)

        return dataset_dict

    def update_annotation_with_instance_wise_captions(self, dataset_dict) -> List[Dict[str, Union[str, int]]]:
        caption_data = self.load_caption_from_file(dataset_dict)

        key = 'category_id'
        for i, annotation in enumerate(dataset_dict['annotations']):
            try:
                caption = caption_data['captions'][i]
            except IndexError:
                caption = None

            if self.with_text:
                text = self.label_name_map[annotation[key]]
                description = np.random.choice(self.sample_descriptions) % text

                if np.random.choice([True, False]) and (caption is not None and annotation['bbox'] == caption['bbox']):
                    if np.all(annotation['bbox'][2:] > self.min_bbox_wh):
                        description = caption['caption']

                annotation['caption'] = {
                    'text_descriptions': description,
                    'class_labels': annotation[key],
                    'instance_labels': i,
                }

            # For class agnostic segmentation, it will be binary classification bg / fg
            if self.is_class_agnostic:
                annotation[key] = 0

        return dataset_dict

    def load_caption_from_file(self, dataset_dict: Dict[str, Any]) -> Dict[str, Any]:
        file_name = dataset_dict['file_name'].split(os.sep)[-1]
        ext = file_name.split('.')[1]
        file_name = file_name.replace(ext, 'json')
        path_to_json = os.path.join(
            f'{os.sep}'.join(dataset_dict['file_name'].split(os.sep)[:-2]), f'captions/train2017/{file_name}'
        )

        if not os.path.isfile(path_to_json):
            return {'captions': []}

        data = load_file(path_to_json)
        assert data['image_id'] == dataset_dict['image_id'], f'The image_id is not same {root}'
        return data

    def _transform_annotations(self, dataset_dict, transforms, image_shape):
        for anno in dataset_dict['annotations']:
            if not self.use_instance_mask:
                anno.pop('segmentation', None)
            if not self.use_keypoint:
                anno.pop('keypoints', None)

        annos = [
            detection_utils.transform_instance_annotations(
                obj, transforms, image_shape, keypoint_hflip_indices=self.keypoint_hflip_indices
            )
            for obj in dataset_dict.pop('annotations')
            if obj.get('iscrowd', 0) == 0
        ]

        # gather all the captions
        instance_captions = [anno.pop('caption', None) for anno in annos]

        instances = detection_utils.annotations_to_instances(annos, image_shape, mask_format=self.instance_mask_format)

        if self.recompute_boxes:
            instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
        dataset_dict['instances'], mask = detection_utils.filter_empty_instances(instances, return_mask=True)

        if self.with_text:
            labels = torch.Tensor(
                [(x['instance_labels'], x['class_labels']) for i, x in enumerate(instance_captions) if mask[i]]
            )
            dataset_dict['instances'].set('gt_instance_labels', labels[:, 0])
            dataset_dict['instances'].set('gt_class_labels', labels[:, 1])
            dataset_dict['gt_descriptions'] = [x['text_descriptions'] for x in instance_captions]


if __name__ == '__main__':
    import sys
    from detectron2.data.datasets import load_coco_json
    from zsis.config import get_cfg

    c = sys.argv[1]
    c1 = sys.argv[2]

    cfg = get_cfg()
    cfg.merge_from_file(c)

    print('loading dataset annotations')

    if os.path.isdir(c1):
        dataset_dict = load_coco_json(c1 + '/trainval.json', c1)
    else:
        root = sys.argv[3]
        dataset_dict = load_coco_json(c1, root)

    mapper = DatasetMapper(cfg)
    for i in range(120000):
        print(i)
        x = mapper(dataset_dict[i])

    import IPython

    IPython.embed()
