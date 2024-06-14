#!/usr/bin/env python
r"""
     ___                   _  _      _  o          
     )) ) __  __ ___  _ _  )\/,) __  )L _  __  _ _ 
    ((_( ((_ (('((_( ((\( ((`(( ((_)(( (( ((_)((\( 

The U-Net model itself.

Taken from https://github.com/milesial/Pytorch-UNet

Modified to work in 3D.
"""

from __future__ import annotations

__all__ = ["DoubleConv", "Down", "Up", "OutConv"]
__version__ = "0.9.0"
__author__ = "Benjamin Blundell <bjb8@st-andrews.ac.uk>"

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel=(3, 5, 5),
        dilation=(1, 1, 1),
        mid_channels=None,
    ):
        """
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel (Tuple[int,int,int]): the convolution kernel.
            dilation (Tuple[int,int,int]): The dilation values.
            mid_channels (Union[None, int]): Optional mid channels.
        """
        super(DoubleConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels

        ppad0 = (kernel[0] - 1) // 2 * dilation[0]
        ppad1 = (kernel[1] - 1) // 2 * dilation[1]
        ppad2 = (kernel[2] - 1) // 2 * dilation[2]

        self.double_conv = nn.Sequential(
            nn.Conv3d(
                in_channels,
                mid_channels,
                kernel_size=kernel,
                padding=(ppad0, ppad1, ppad2),
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                mid_channels,
                out_channels,
                kernel_size=kernel,
                padding=(ppad0, ppad1, ppad2),
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv. Reduce the X/Y but not the time dimension."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel=(3, 5, 5),
        dilation=(1, 1, 1),
        down_kernel=(1, 2, 2),
        down_stride=(1, 2, 2),
    ):
        """
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel (Tuple[int,int,int]): the convolution kernel.
            dilation (Tuple[int,int,int]): The dilation values.
            down_kernel (Tuple[int,int,int]): The kernel on the down step.
            down_stride (Tuple[int,int,int]): The stride on the down kernel.
        """
        super(Down, self).__init__()

        self.maxpool_conv = nn.Sequential(
            nn.Conv3d(
                in_channels,
                in_channels,
                kernel_size=down_kernel,
                padding=0,
                dilation=1,
                stride=down_stride,
                bias=False,
            ),
            DoubleConv(in_channels, out_channels, kernel=kernel, dilation=dilation),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv. This one does quad instead of double for the wonky decoder"""

    def __init__(self, in_channels, out_channels, dilation=(1, 1, 1)):
        """
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            dilation (Tuple[int,int,int]): The dilation values.
        """
        super(Up, self).__init__()
        self.up = nn.ConvTranspose3d(
            in_channels, in_channels // 4, kernel_size=(4, 4, 4), stride=(4, 4, 4)
        )
        self.conv = DoubleConv(in_channels // 2, out_channels, dilation=dilation)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input is CDHW
        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]

        x1 = F.pad(
            x1,
            [
                diffX // 2,
                diffX - diffX // 2,
                diffY // 2,
                diffY - diffY // 2,
                diffZ // 2,
                diffZ - diffZ // 2,
            ],
        )

        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
