# Zero Shot Instance Segmentation (ZSIS)

This repo is for vision-language based zero-shot instance segmentation built on top of [detectron2](https://github.com/facebookresearch/detectron2) framework.. 

## Installation
To use this package, you will need to install it

&rarr; via source
```bash
$ git clone https://github.com/iKrishneel/zsis.git
$ pip install -e zsis
```

&rarr; via pip
```bash
$ pip install git+https://github.com/iKrishneel/zsis@master
```

The vision-language model is based of the OpenAI [CLIP](https://github.com/openai/CLIP.git) and is fully configurable through the detectron2 config.

## Models
#### Class Agnostic Instance Segmentation
This model uses the two stage MaskRCNN pipeline to generate class agnostic object bounding boxes and uses CLIP for classifying the proposal bounding boxes.
The pretrained [cutlter](https://github.com/facebookresearch/CutLER.git) is used as a class agnostic proposal generator. The configs for Culter + CLIP can be found [here](https://github.com/iKrishneel/zsis/blob/master/config/culter/cascade_mask_rcnn_R_50_FPN_clip.yaml).

##### Running Demo
To run the Culter + CLIP demo
```bash
$ python tools/test.py --config-file config/culter/cascade_mask_rcnn_R_50_FPN_clip.yaml --image PATH_TO_IMAGE --labels LIST_OF_VOCABS
```
