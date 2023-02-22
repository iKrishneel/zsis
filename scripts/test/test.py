#!/usr/bin/env python

import click

from detectron2.data import MetadataCatalog
from detectron2.modeling import build_model
from detectron2.data.detection_utils import read_image
from detectron2.utils.visualizer import ColorMode, Visualizer

from zsis.config import get_cfg
from zsis.modeling import GeneralizedRCNNClip, GeneralizedRCNNWithText  # NOQA: F401
from zsis.engine import DefaultPredictor

import matplotlib.pyplot as plt


@click.command()
@click.option('--config-file', required=False, default='../../config/culter/cascade_mask_rcnn_R_50_FPN_clip.yaml')
@click.option('--image', required=True)
@click.option('--weights', required=False)
@click.option('--threshold', default=0.5)
@click.option('--rgb/--bgr', default=True)
def main(config_file: str, image: str, weights: str, threshold: float, rgb: bool) -> None:
    cfg = get_cfg()
    cfg.merge_from_file(config_file)

    cfg.MODEL.CLIP.TOPK = 1
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
    cfg.DATASETS.TEST = ['leaf_bean']

    if weights is not None:
        cfg.MODEL.WEIGHTS = weights

    image = read_image(image, 'RGB' if rgb else 'BGR')

    predictor = DefaultPredictor(cfg)
    labels = [
        # 'fish',
        # 'white leaf',
        'greenish leaf',
        'white bean',
        'red bean',
        'yellow cashew nut',
        'nut',
        'broccoli',
        'plate',
        'black olive',
        'yellowish bean',
        'greenish bean',
        'black keyboard',
    ]
    text_descriptions = [f'This is a photo of a {label}' for label in labels]

    if cfg.MODEL.META_ARCHITECTURE == 'GeneralizedRCNNClip':
        predictor.set_text_descriptions(text_descriptions)
        predictions = predictor(image)
    else:
        predictions = predictor(image, text_descriptions)

    metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused")
    metadata.thing_classes = labels
    visualizer = Visualizer(image, metadata, instance_mode=ColorMode.IMAGE)

    instances = predictions["instances"].to('cpu')
    top_probs, top_labels = predictions['top_probs'][:, 0].cpu(), predictions['top_labels'][:, 0].cpu()
    instances.set('pred_classes', top_labels)
    instances.set('scores', top_probs)

    vis_output = visualizer.draw_instance_predictions(predictions=instances)
    plt.imshow(vis_output.get_image())
    plt.show()


if __name__ == '__main__':
    main()
