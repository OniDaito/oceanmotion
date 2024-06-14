#!/usr/bin/env python
r"""
     ___                   _  _      _  o          
     )) ) __  __ ___  _ _  )\/,) __  )L _  __  _ _ 
    ((_( ((_ (('((_( ((\( ((`(( ((_)(( (( ((_)((\( 

Some small utility functions that return a rough number
of detections over the course of training.
"""

from __future__ import annotations

__all__ = []
__version__ = "0.7.0"
__author__ = "Benjamin Blundell <bjb8@st-andrews.ac.uk>"


import skimage as sk
import numpy as np
from typing import List
from sealhits.bbox import XYBox


def blobs_in_stack(pred_single: np.ndarray):
    """ Take a single pred image and 
    see how many blobs we have.
    
    Args:
        pred_single (np.ndarray): the data image.
    """
    regions = sk.morphology.label(pred_single)
    props = sk.measure.regionprops(regions)
    return props

def bbs_in_image(pred_single: np.ndarray) -> List[XYBox]:
    """ Given an image, return a list of bounding boxes. Requires
    linking up the blobs on a half-size image.
    
     Args:
        pred_single (np.ndarray): The image frame we are looking at. 
    """
    # REMOVED half size for now as the detections are being removed, which is USEFUL! :D
    #half_size = (int(pred_single.shape[0] / 2), int(pred_single.shape[1] / 2))
    #pred_single_small = resize(pred_single, half_size)
    #regions = sk.morphology.label(pred_single_small)
    regions = sk.morphology.label(pred_single)
    props = sk.measure.regionprops(regions)
    boxes = []

    for prop in props:
        minr, minc, maxr, maxc = prop.bbox
        #box = XYBox(int(minc * 2), int(minr * 2), int(maxc * 2), int(maxr * 2))
        box = XYBox(minc, minr, maxc, maxr)
        boxes.append(box)

    return boxes