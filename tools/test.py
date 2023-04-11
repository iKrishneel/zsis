#!/usr/bin/env python

import click
import time

from detectron2.data import MetadataCatalog
from detectron2.modeling import build_model
from detectron2.data.detection_utils import read_image
from detectron2.utils.visualizer import ColorMode, Visualizer

from zsis.config import get_cfg
from zsis.modeling import GeneralizedRCNNClip, GeneralizedRCNNWithText  # NOQA: F401
from zsis.engine import DefaultPredictor

import matplotlib.pyplot as plt


@click.command()
@click.option('--config-file', required=True)
@click.option('--image', required=True)
@click.option('--labels', type=str, required=True)
@click.option('--weights', required=False)
@click.option('--threshold', default=0.5)
@click.option('--rgb/--bgr', default=True)
def main(config_file: str, image: str, labels: str, weights: str, threshold: float, rgb: bool) -> None:
    labels = labels.split(',')
    assert len(labels) > 0, f'labels is required.'

    cfg = get_cfg()
    cfg.merge_from_file(config_file)

    cfg.MODEL.CLIP.TOPK = 1
    cfg.MODEL.CLIP.ARCHITECTURE = "RN50"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
    cfg.MODEL.CLIP.IMAGE_ENCODER.FROZEN = False

    if weights is not None:
        cfg.MODEL.WEIGHTS = weights

    image = read_image(image, 'RGB' if rgb else 'BGR')

    predictor = DefaultPredictor(cfg)
    text_descriptions = [f'This is a photo of a {label}' for label in labels]

    for _ in range(1):
        time_start = time.time()
        if cfg.MODEL.META_ARCHITECTURE == 'GeneralizedRCNNClip':
            predictor.set_text_descriptions(text_descriptions)
            predictions = predictor(image)
        elif cfg.MODEL.META_ARCHITECTURE == 'GeneralizedRCNNWithText':
            predictions = predictor(image, text_descriptions)
        else:
            predictions = predictor(image)

        print(f'\033[32mProcessing time {time.time() - time_start}\033[0m')

    metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused")
    try:
        metadata.thing_classes = labels
    except AssertionError:
        pass
    visualizer = Visualizer(image, metadata, instance_mode=ColorMode.IMAGE)

    instances = predictions['instances'].to('cpu')
    if 'top_probs' in predictions:
        top_probs, top_labels = predictions['top_probs'][:, 0].cpu(), predictions['top_labels'][:, 0].cpu()
        instances.set('pred_classes', top_labels)
        instances.set('scores', top_probs)

    vis_output = visualizer.draw_instance_predictions(predictions=instances)

    image = image[:, :, ::-1] if not rgb else image

    plt.close()
    _, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(image)
    ax1.set_title('Input Image')
    ax1.axis('off')

    ax2.imshow(vis_output.get_image())
    ax2.set_title('Detection results')
    ax2.axis('off')

    plt.show()

    # import IPython;IPython.embed()


if __name__ == '__main__':
    main()
