#!/usr/bin/env python

import open_clip
import torch
from PIL import Image
from copy import deepcopy

import os
import os.path as osp
from pycocotools.coco import COCO
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json

from detectron2.data.datasets import load_coco_json
from detectron2.data.detection_utils import read_image
from detectron2.data import MetadataCatalog

import click


def write_json(filename, data):
    with open(filename, 'w') as fp:
        json.dump(data, fp)


def get_caption(images, model):
    images = [images] if not isinstance(images, list) else images
    for i, image in enumerate(images):
        image = Image.fromarray(image) if isinstance(image, np.ndarray) else image
        images[i] = model.transform(image)

    images = torch.stack(images).to(model.device)
    with torch.no_grad(), torch.cuda.amp.autocast():
        generated = model.generate(images)

    caps = [open_clip.decode(g).split("<end_of_text>")[0].replace("<start_of_text>", "") for g in generated]
    return caps


def inference(model, images, batch_size=24):
    captions = []
    for i in range(0, len(images), batch_size):
        chunck = images[i : i + batch_size]
        captions.extend(get_caption(chunck, model))
    return captions


def get_model(device):
    model, _, transform = open_clip.create_model_and_transforms(
        model_name="coca_ViT-L-14", pretrained="mscoco_finetuned_laion2B-s13B-b90k"
    )
    model = model.to(device).eval()
    model.transform = transform
    model.device = device
    return model


def process(model, data_dicts, write_dir):
    for i, data_dict in enumerate(tqdm(data_dicts)):
        filename = data_dict['file_name']
        annotations = data_dict['annotations']

        image = read_image(file_name=filename)
        im_crops = []
        bboxes = []
        for _, annotation in enumerate(annotations):
            x, y, w, h = annotation['bbox']
            bboxes.append(annotation['bbox'])

            x, y, w, h = np.intp([x, y, w, h])
            im_crop = image[y : y + h, x : x + w].copy()
            im_crops.append(im_crop)

        captions = inference(model, im_crops)

        caption_dicts = {
            'file_name': filename,
            'image_id': data_dict['image_id'],
            'captions': [{'bbox': bbox, 'caption': cap} for cap, bbox in zip(captions, bboxes)],
        }

        image_id = data_dict['image_id']
        filename = f'{osp.join(write_dir, str(image_id).zfill(12))}.json'
        write_json(filename, caption_dicts)


@click.command()
@click.option('--start', type=int, required=True)
@click.option('--end', type=int, required=True)
@click.option('--device', type=str, default='cuda:0')
@click.option('--root', type=str, required=False, default='/workspace/Downloads/datasets/coco/')
@click.option('--year', type=str, default='2017')
@click.option('--type', type=str, default='train')
def main(start, end, device, root, year, type):
    data_type = f'{type}{year}'
    ann_file = '{}/annotations/instances_{}.json'.format(root, data_type)

    write_dir = osp.join(root, f'captions/{data_type}')
    if not osp.isdir(write_dir):
        os.makedirs(write_dir, exist_ok=True)

    # load coco data

    print("loading coco data")
    filename_resolver = lambda x: osp.join(root, data_type)
    data_dicts = load_coco_json(ann_file, filename_resolver(''))
    meta = MetadataCatalog.get(f'coco_{year}_{type}')
    classes = meta.thing_classes
    cat_id_mapping = meta.get('thing_dataset_id_to_contiguous_id')

    print("loading model")
    model = get_model(device=device)

    process(model, data_dicts[start:end], write_dir)


if __name__ == '__main__':
    main()
