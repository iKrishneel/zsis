#!/usr/bin/env python

def add_clip_config(cfg):
    cfg.MODEL.RESNETS.LAYERS = [3, 4, 6, 3]

    cfg.MODEL.ROI_HEADS.USE_DROPLOSS = False
    cfg.MODEL.ROI_HEADS.DROPLOSS_IOU_THRESH = 0.0
