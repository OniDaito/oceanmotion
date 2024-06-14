#!/usr/bin/env python
r"""
     ___                   _  _      _  o          
     )) ) __  __ ___  _ _  )\/,) __  )L _  __  _ _ 
    ((_( ((_ (('((_( ((\( ((`(( ((_)(( (( ((_)((\( 

image.py - Utilities for images.
"""

from __future__ import annotations

__all__ = ["pred_to_gif"]
__version__ = "0.9.0"
__author__ = "Benjamin Blundell <bjb8@st-andrews.ac.uk>"

import numpy as np
from PIL import Image


def pred_to_gif(np_stack: np.array, out_path: str):
    """Create a gif from a prediction stack or an original mask.
    
    Args:
        np_stack: the frames to convert.
        out_path: the full path to the output gif.
    """
    # Spit out an npz and a gif of the prediction
    # First, the gif
    gif = np.where(np_stack[0] >= 1, 255, 0)
    gif = gif.astype(np.uint8)
    gif = Image.fromarray(gif)

    gif_stack = []

    for img in np_stack[1:]:
        gif_frame = np.where(img >= 1, 255, 0)
        gif_frame = gif_frame.astype(np.uint8)
        gif_frame = Image.fromarray(gif_frame)
        gif_stack.append(gif_frame)

    gif.save(
        out_path,
        save_all=True,
        append_images=gif_stack,
        duration=len(gif_stack) + 1,
        loop=0,
    )


def nought_to_one_np(func):
    """ A check to make sure we are in the right range."""

    def wrapper(*args, **kwargs):
        for x in args:
            if type(x) == np.ndarray:
                assert(x.dtype) == float
                assert(np.max(x) <= 1.0)
                assert(np.min(x) >= 0.0)
        
        func(*args, **kwargs)
     
    return wrapper