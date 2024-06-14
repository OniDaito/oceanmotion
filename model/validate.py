#!/usr/bin/env python
r"""
     ___                   _  _      _  o          
     )) ) __  __ ___  _ _  )\/,) __  )L _  __  _ _ 
    ((_( ((_ (('((_( ((\( ((`(( ((_)(( (( ((_)((\( 

validate.py - Run the validation pass during training.

Used to set new learning rates.
"""

from __future__ import annotations

__all__ = ["validate"]
__version__ = "0.9.0"
__author__ = "Benjamin Blundell <bjb8@st-andrews.ac.uk>"

import torch

def validate(model, val_loader, loss_func, device):
    """Calculate the validation loss so we can adjust hyper-parameters.
    
    Args:
        model (): Current model instance.
        val_loader (torch.utils.data.DataLoader): a DataLoader instance.
        loss_func (): our loss function.
        device (str): The device we are running on.
    """
    model.eval()
    val_loss = 0  # We'll take the mean of the losses

    with torch.no_grad():
        test_num = 0

        for data, target in val_loader:
            # Move the data to the current device
            data, target = data.to(device), target.to(device)
            pred = model(data)

            test_num += 1
            val_loss += loss_func(pred, target)

    val_loss /= test_num

    return val_loss