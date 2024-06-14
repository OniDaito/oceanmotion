#!/usr/bin/env python
r"""
     ___                   _  _      _  o          
     )) ) __  __ ___  _ _  )\/,) __  )L _  __  _ _ 
    ((_( ((_ (('((_( ((\( ((`(( ((_)(( (( ((_)((\( 

loss.py - A collection of loss functions for use
with OceanMotion in binary and multiclass modes."""

from __future__ import annotations

__all__ = ["OceanLoss"]
__version__ = "0.9.0"
__author__ = "Benjamin Blundell <bjb8@st-andrews.ac.uk>"

import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
from typing import Optional
from functools import partial

""" The (symmetric/asymetric) unified focal loss is the tversky and focal loss combined.
    https://www.sciencedirect.com/science/article/pii/S0895611121001750
    https://arxiv.org/html/2312.05391v1
    https://github.com/mlyg/unified-focal-loss
    https://socket.dev/pypi/package/unified-focal-loss-pytorch
"""

class OceanLoss(_Loss):
    """A small class that combines the two loss functions in a combo loss style.
    This version peforms the binary moves loss.
    """
    def __init__(self, tmix=0.65):
        """ 
            Initialise the loss.

        Args:
            tmix (float): the balance of the tversky loss (between 0.0 and 1.0).
        """
        super(OceanLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.tversky = TverskyLoss(tmix, 1.0 - tmix)
        self.alpha = 0.6

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """The input tensor must be logits and in n,c,d,h,w format.
        Target is n,d,h,w.
        
        Args:
            inputs (torch.Tensor): the predicted tensors.
            targets (torch.Tensor): the target tensors.
        """
        targets = targets.unsqueeze(
            1
        ).float()  # Get the C channel into the targets. Equivalent of one hot
        loss = self.alpha * self.bce(inputs, targets) + (
            1.0 - self.alpha
        ) * self.tversky.forward(inputs, targets)

        return loss


class TverskyLoss(nn.Module):
    """ The Tversky loss. Thought to be better than Dice in certain situations.
    https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch
    """
    def __init__(self, alpha, beta, weight=None, size_average=True):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()
        FP = ((1 - targets) * inputs).sum()
        FN = (targets * (1 - inputs)).sum()

        Tversky = (TP + smooth) / (TP + self.alpha * FP + self.beta * FN + smooth)

        return 1 - Tversky