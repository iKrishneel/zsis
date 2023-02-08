#!/usr/bin/env python

from detectron2.config import CfgNode as CN
from .config import _C


def get_cfg() -> CN:
    return _C.clone()
