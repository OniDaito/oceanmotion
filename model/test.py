#!/usr/bin/env python
r"""
     ___                   _  _      _  o          
     )) ) __  __ ___  _ _  )\/,) __  )L _  __  _ _ 
    ((_( ((_ (('((_( ((\( ((`(( ((_)(( (( ((_)((\( 

The test pass, run throughout training.
"""

__all__ = []
__version__ = "0.9.0"
__author__ = "Benjamin Blundell <bjb8@st-andrews.ac.uk>"

import torch
from tqdm import tqdm
import logging
import wandb
import random
import torch.nn.functional as F


def test_model(
    model,
    test_loader: torch.utils.data.DataLoader,
    loss_func,
    device: str,
    send_to_wandb: bool,
    confidence=0.7,
):
    """Test the model against the test set.
    
    Args:
        model (): Current model instance.
        test_loader (torch.utils.data.DataLoader): a DataLoader instance.
        loss_func (): our loss function.
        device (str): The device we are running on.
        send_to_wandb (bool): send results to Weights and Biases.
        confidence (float): The meet or beat score for a positive classification.

    """
    # TODO - eventually pass in the write for cluster tensorboard stuff.
    model.eval()
    test_loss = 0  # We'll take the mean of the losses
    targets = []
    predictions = []

    with torch.no_grad():
        test_num = 0

        for data, target in tqdm(test_loader, desc="Running test loss"):
            target = torch.where(target > 0, 1, 0).to(dtype=torch.uint8)

            # Move the data to the current device
            data_dev, target_dev = (
                data.to(device),
                target.to(device),
            )

            pred_dev = model(data_dev)
            test_num += 1
            test_loss += loss_func(pred_dev, target_dev)

            # Go through each item in the batch, one at a time
            for i in range(target_dev.shape[0]):
                tt = target_dev[i].cpu()
                tp = pred_dev[i].cpu()

                a = torch.where(tt > 0, 1, 0).to(dtype=torch.uint8)
                a = (
                    (torch.clip(torch.sum(a, dim=0), 0, 1) * 255)
                    .cpu()
                    .to(dtype=torch.uint8)
                )
                targets.append(a)

                # With BCE use a sigmoid and confidence score for ther prediction
                b = (
                    (
                        torch.clip(
                            torch.sum((F.sigmoid(tp) > confidence).long(), dim=1), 0, 1
                        )
                        * 255
                    )
                    .cpu()
                    .to(dtype=torch.uint8)
                )
                predictions.append(b)

    test_loss /= test_num
    logging.info("Test set: mean loss %s", str(test_loss.item()))

    if send_to_wandb:
        table = wandb.Table(columns=["ID", "Target", "Pred"])
        choices = list(range(len(targets)))
        random.shuffle(choices)

        for i in range(4):  # Save four images
            c = choices.pop(0)
            a = targets[c]
            b = predictions[c]

            # Find the target and the majority predicted class.
            table.add_data(c, wandb.Image(a), wandb.Image(b))

        wandb.log({"predictions": table})
        wandb.log(
            {
                "Test Loss": test_loss,
            }
        )
