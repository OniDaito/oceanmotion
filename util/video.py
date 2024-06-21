#!/usr/bin/env python
r"""
     ___                   _  _      _  o          
     )) ) __  __ ___  _ _  )\/,) __  )L _  __  _ _ 
    ((_( ((_ (('((_( ((\( ((`(( ((_)(( (( ((_)((\( 

video.py - generate videos of predictions
"""

from __future__ import annotations

__all__ = ["pred_to_video"]
__version__ = "0.9.0"
__author__ = "Benjamin Blundell <bjb8@st-andrews.ac.uk>"

import ffmpegio
import numpy as np
from typing import Tuple, List
from sealhits.bbox import XYBox
from palettable.scientific.sequential import Batlow_20 # https://jiffyclub.github.io/palettable/scientific/sequential/
from palettable.colorbrewer.qualitative import Set1_8

def binary_to_colour(frames: np.array, colour: Tuple[float, float, float]):
    """ Binary images must be either 0 or 1 in uint8. Colour is RGB in the 
    range 0 to 1.0.
    
    Args:
        frames (np.array): the frames to convert.
        colour (Tuple[float, float, float]): The RGB Colour in 0 to 1 ranges.
    """
    frames = np.clip(frames, 0, 1)
    # Using memmap for these large videos that need conversion.
    coloured = np.zeros(shape=(*frames.shape, 3), dtype=np.uint8)

    # Take entries from RGB LUT according to greyscale values in image
    lut = [[0,0,0], [int(colour[0] * 255), int(colour[1] * 255), int(colour[2] * 255)]]
    np.take(lut, frames, axis=0, out=coloured)

    return coloured

def class_to_colour(frames: np.array):
    """ Class images have one int per pixel corresponding to a class.
    
    Args:
        frames (np.array): the frames to convert.
    """
    # Using memmap for these large videos that need conversion.
    coloured = np.zeros(shape=(*frames.shape, 3), dtype=np.uint8)

    # Take entries from RGB LUT according to the class category values in image
    # TODO - Maximum of 8 classes for now
    lut = [Set1_8.mpl_colormap(x / 8.0) for x in range(8)]
    lut = [[int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)] for x in lut]
    np.take(lut, frames, axis=0, out=coloured)

    return coloured

def intensity_to_colour(frames: np.array, colourmap=Batlow_20):
    """Frames must be uint8 0 to 255.

    Args:
        frames (np.array): the frames to convert.
        colourmap (Palette): a Palette from the Palettable library.
    """
    assert(np.max(frames) - np.min(frames) > 1)
    assert(frames.dtype == np.uint8)
    # Using memmap for these large videos that need conversion.
    coloured = np.zeros(shape=(*frames.shape, 3), dtype=np.uint8)

    # Take entries from RGB LUT according to greyscale values in image
    lut = [colourmap.mpl_colormap(x / 255.0) for x in range(256)]
    lut = [[int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)] for x in lut]
    np.take(lut, frames, axis=0, out=coloured)

    return coloured

def alpha_blend(fg: np.array, bg: np.array, fg_alpha: float, bg_alpha: float):
    """Alpha blend two coloured 0 to 255 int videos.
    
    Args:
        fg (np.array): the foreground frames.
        bg (np.array): the background frames.
        fg_alpha (float): the alpha of the foreground.
        bg_alpha (float): the alpha of the background.
    """
    fg = fg.astype(float) / 255.0 
    bg = bg.astype(float) / 255.0

    alpha = 1.0 - (1.0 - fg_alpha) * (1.0 - bg_alpha)
    final = fg * fg_alpha / alpha + bg * bg_alpha * (1.0 - fg_alpha) / alpha
    final = np.clip((final * 255), 0, 255).astype(np.uint8)

    return final
  
def add_blend(fg: np.array, bg: np.array, alpha=1.0):
    """Additive blend but sticking in the 255 space."
    
    Args:
        fg (np.array): the foreground frames.
        bg (np.array): the background frames.
        alpha (float): the alpha of the foreground.    
    """
    assert(alpha > 0 and alpha <= 1.0)
    assert(fg.dtype == np.uint8)
    assert(bg.dtype == np.uint8)
    final = bg
    mixed = (fg * alpha).astype(np.uint8)
    final += mixed
    final[final < mixed]=255

    return final


def pred_to_video(
    source_np: np.array, mask_np: np.array, pred_np: np.array, outpath: str, multiclass=False
):
    """ Convert our numpy arrays from the process to video. source_np will be 
    float from 0 to 1. mask_np and pred_np should be uint8, either 0 or 1.
    mask_np can be None.
    
    Args:
        source_np (np.array): the source frames.
        mask_np (np.array): the mask frames.
        pred_np (np.array): the predicted frames.
        outpath (str): the output path with filename.
        multiclass (bool): is this a multiclass prediction?
    
    """
    mask_colour = None

    if mask_np is not None:
        if mask_np.dtype == float:
            mask_np = np.where(mask_np > 0.5, 1, 0)
            mask_np = mask_np.astype(np.uint8)
        mask_colour = binary_to_colour(mask_np, [0.0, 1.0, 0.0])

    # base should be 0-255. Mask and pred should be 0 or 1. All should be uint8
    source_np = np.clip(source_np * 255, 0, 255).astype(np.uint8)
    base_colour = intensity_to_colour(source_np)
    pred_colour = None

    if multiclass:
        pred_colour = class_to_colour(pred_np)
    else:
        if pred_np.dtype == float:
            pred_np = np.where(pred_np > 0.5, 1, 0)
            pred_np = pred_np.astype(np.uint8)
        pred_colour = binary_to_colour(pred_np,[1.0, 0.0, 0.0])

    # Combine
    combined = base_colour

    if mask_colour is not None:
        combined = add_blend(mask_colour, base_colour, 0.4)

    combined = add_blend(pred_colour, base_colour, 0.8)
    combined = np.clip(combined, 0, 255).astype(np.uint8)

    ffmpegio.video.write(outpath, 4, combined, overwrite=True, show_log=False)


def bbraw_to_np(bbs: List[XYBox], img_size: Tuple[int, int]) -> np.array:
    """Convert a list of raw BBS to an np array we can compare.
    
    Args:
        bbs (List[XYBox]): the list of bounding box.
        img_size (Tuple[int, int]): The image size (width then height in pixels).
    """
    og = np.zeros((img_size[1], img_size[0]), dtype=np.uint8)

    for b in bbs:
        og[b.y_min : b.y_max, b.x_min : b.x_max] = 1

    return og
