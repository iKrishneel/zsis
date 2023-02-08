#!/usr/bin/env python

from detectron2.modeling import build_model
from detectron2.data.detection_utils import read_image

from zsis.config import get_cfg
from zsis.modeling import GeneralizedRCNNClip
from zsis.engine import DefaultPredictor


c = get_cfg()
c.merge_from_file('../../config/culter/cascade_mask_rcnn_R_50_FPN_clip.yaml')
c.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

d = DefaultPredictor(c)

labels = [
    'fish', 'leaf', 'white bean', 'red bean', 'cashew nut', 'nut', 'broccoli', 'plate', 'yellowish bean', 'broccoli', 'greenish bean'
]
text_descriptions = [f'This is a photo of a {label}' for label in labels]

d.set_text_descriptions(text_descriptions)

p = '/root/krishneel/Downloads/real_veggie_test_images/image_10.png'

img = read_image(p, 'BGR')

d(img)
