#!/usr/bin/env python
r"""
     ___                   _  _      _  o          
     )) ) __  __ ___  _ _  )\/,) __  )L _  __  _ _ 
    ((_( ((_ (('((_( ((\( ((`(( ((_)(( (( ((_)((\( 

tensors.py - a few useful tensor operations
"""

from __future__ import annotations

__all__ = ["count_unique_values", "convert_softmax_to_onehot"]
__version__ = "0.9.0"
__author__ = "Benjamin Blundell <bjb8@st-andrews.ac.uk>"

import torch
import torch.nn.functional as F


def count_unique_values(tt: torch.Tensor):
    """ Returns the counts of the unique values in this tensor, in descending order.

    Args:
        tt (torch.Tensor): the tensor to count.
    """
    unique_values, counts = torch.unique(tt, return_counts=True)
    sorted_counts, sorted_indices = torch.sort(counts, descending=True)
    sorted_values = unique_values[sorted_indices]
    return sorted_values, sorted_counts

def convert_softmax_to_onehot(softmax_tensor: torch.Tensor, num_classes:int, dim=1):
    """ Take a softmaxed tensor and return a onehot where the largest value 
    is 1.0 and all others are 0.
    
    Args:
        softmax_tensor (torch.Tensor): the tensor to convert.
        num_classes (int): the number of classes.
        dim (int): which dimension to max out?
    """
    _, max_indices = torch.max(softmax_tensor, dim=dim)
    onehot_tensor = F.one_hot(max_indices, num_classes=num_classes)
    onehot_tensor = onehot_tensor.permute(3, 0, 1, 2)
    return onehot_tensor