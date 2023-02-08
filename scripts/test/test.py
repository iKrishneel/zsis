#!/usr/bin/env python

from detectron2.data import MetadataCatalog
from detectron2.modeling import build_model
from detectron2.data.detection_utils import read_image
from detectron2.utils.visualizer import ColorMode, Visualizer

from zsis.config import get_cfg
from zsis.modeling import GeneralizedRCNNClip
from zsis.engine import DefaultPredictor

import matplotlib.pyplot as plt

cfg = get_cfg()
cfg.merge_from_file('../../config/culter/cascade_mask_rcnn_R_50_FPN_clip.yaml')

cfg.MODEL.CLIP.TOPK = 1
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.DATASETS.TEST = ['leaf_bean']
cfg.MODEL.WEIGHTS = './cascade_mask_rcnn_r50_clip_vit_b_32.pth'

predictor = DefaultPredictor(cfg)

labels = [
    'fish',
    'leaf',
    'white bean',
    'red bean',
    'cashew nut',
    'nut',
    'broccoli',
    'plate',
    'yellowish bean',
    'broccoli',
    'greenish bean',
]
text_descriptions = [f'This is a photo of a {label}' for label in labels]

predictor.set_text_descriptions(text_descriptions)

im_path = '/root/krishneel/Downloads/real_veggie_test_images/image_10.png'
image = read_image(im_path, 'BGR')

predictions = predictor(image)

metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused")
metadata.thing_classes = labels
visualizer = Visualizer(image, metadata, instance_mode=ColorMode.IMAGE)

instances = predictions["instances"].to('cpu')
top_probs, top_labels = predictions['top_probs'][:, 0].cpu(), predictions['top_labels'][:, 0].cpu()
instances.set('pred_classes', top_labels)
instances.set('scores', top_probs)

print(predictions['top_probs'])
vis_output = visualizer.draw_instance_predictions(predictions=instances)

plt.imshow(vis_output.get_image())
plt.show()
