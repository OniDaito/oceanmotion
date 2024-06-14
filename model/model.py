#!/usr/bin/env python
r"""
     ___                   _  _      _  o          
     )) ) __  __ ___  _ _  )\/,) __  )L _  __  _ _ 
    ((_( ((_ (('((_( ((\( ((`(( ((_)(( (( ((_)((\( 

Parts of the U-Net model 

Taken from https://github.com/milesial/Pytorch-UNet

Modified to work in 3D.
"""

from __future__ import annotations

__all__ = ["UNet3D", "count_parameters"]
__version__ = "0.9.0"
__author__ = "Benjamin Blundell <bjb8@st-andrews.ac.uk>"

import torch
import torch.nn as nn
from .model_parts import DoubleConv, Down, Up, OutConv

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class UNet3D(nn.Module):
    """ The classic U-Net but with a shorter decoder side, 4 steps down
    but only two steps up."""
    def __init__(self, n_channels, n_classes):
        super(UNet3D, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = DoubleConv(n_channels, 16, kernel=(5,5,5), dilation=(3,3,3))
        self.down1 = Down(16, 32, kernel=(3,3,3), dilation=(2,2,2))
        self.down2 = Down(32, 64, kernel=(3,3,3))
        self.down3 = Down(64, 128, kernel=(1,3,3))
        self.down4 = Down(128, 256, kernel=(1,3,3))
        self.up1 = Up(256, 64)
        self.up2 = Up(64, 16)
        self.outc = OutConv(16, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        # Note the one sided-ness
        x = self.up1(x5, x3)
        x = self.up2(x, x1)
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.outc = torch.utils.checkpoint(self.outc)

class Sector3D(nn.Module):
    """ Sector3D is similar to UNet3D but doensn't have the decoder
    branch, and no concatenations. It's job is to produce a sector
    map instead of a full segmentation."""
    def __init__(self, n_channels, n_classes):
        super(Sector3D, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = DoubleConv(n_channels, 16, kernel=(5,5,5), dilation=(3,3,3))
        self.down1 = Down(16, 32, kernel=(3,3,3), dilation=(2,2,2))
        self.down2 = Down(32, 64, kernel=(3,3,3))
        self.down3 = Down(64, 128, kernel=(1,3,3))
        self.down4 = Down(128, 256, kernel=(1,3,3))
        self.outc = OutConv(256, n_classes) # This is a big crunch! Might need some interleaving layers

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        logits = self.outc(x5)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.outc = torch.utils.checkpoint(self.outc)


class UNetTRed(nn.Module):
    """ The classic U-Net but with a shorter decoder side, 4 steps down
    but only two steps up. This version also reduces on the time dimension."""
    def __init__(self, n_channels, n_classes):
        super(UNetTRed, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = DoubleConv(n_channels, 16, kernel=(5,5,5), dilation=(3,3,3))
        self.down1 = Down(16, 32, kernel=(3,3,3), dilation=(2,2,2), down_stride=(2,2,2), down_kernel=(2,2,2))
        self.down2 = Down(32, 64, kernel=(3,3,3), down_stride=(2,2,2), down_kernel=(2,2,2))
        self.down3 = Down(64, 128, kernel=(1,3,3), down_stride=(2,2,2), down_kernel=(2,2,2))
        self.down4 = Down(128, 256, kernel=(1,3,3), down_stride=(2,2,2), down_kernel=(2,2,2))
        self.up1 = Up(256, 64)
        self.up2 = Up(64, 16)
        self.outc = OutConv(16, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        # Note the one sided-ness
        x = self.up1(x5, x3)
        x = self.up2(x, x1)
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.outc = torch.utils.checkpoint(self.outc)

class UNetApricot(nn.Module):
    def __init__(self, n_channels, n_classes,):
        super(UNetApricot, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = DoubleConv(n_channels, 16, kernel=(5,5,5), dilation=(3,3,3))
        self.down1 = Down(16, 32, kernel=(3,3,3), dilation=(2,2,2), down_kernel=(2,2,2), down_stride=(2,2,2))
        self.down2 = Down(32, 64, kernel=(3,3,3), down_kernel=(2,2,2), down_stride=(2,2,2))
        self.down3 = Down(64, 128, kernel=(1,3,3), down_kernel=(2,2,2), down_stride=(2,2,2))
        self.down4 = Down(128, 256, down_kernel=(2,2,2), down_stride=(2,2,2))
        self.up1 = Up(256, 64)
        self.up2 = Up(64, 16)
        self.outc = OutConv(16, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        # Note the one sided-ness
        x = self.up1(x5, x3)
        x = self.up2(x, x1)
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.outc = torch.utils.checkpoint(self.outc)
