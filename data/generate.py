#!/usr/bin/env python
r"""
     ___                   _  _      _  o          
     )) ) __  __ ___  _ _  )\/,) __  )L _  __  _ _ 
    ((_( ((_ (('((_( ((\( ((`(( ((_)(( (( ((_)((\( 

Generate the datasets for use with the network.
"""

from __future__ import annotations

__all__ = ["generate_sets_presplit", "generate_sets"]
__version__ = "0.9.0"
__author__ = "Benjamin Blundell <bjb8@st-andrews.ac.uk>"

import torch
import logging
import numpy as np
from data.dataset import MoveDataset
from data.dataset_presplit import MoveDatasetPreSplit, SetType
from torch.utils.data.sampler import SubsetRandomSampler

def generate_sets_presplit(args, num_classes):
    """Generate the sets from a list of images. This function returns
    three MoveDatasetPreSplits for training, testing and validations sets.
    
    Args:
        args (args): the args object from the main function.
        num_classes (int): how many classes are there in this dataset?
    """
    # Train first
    dataset_train = MoveDatasetPreSplit(args.data_path, settype=SetType.TRAIN, num_classes=num_classes)
    dataset_train_size = len(dataset_train)
    indices_train = list(range(dataset_train_size))
    np.random.seed(args.seed)
    np.random.shuffle(indices_train)

    # Now the test
    dataset_test = MoveDatasetPreSplit(args.data_path, settype=SetType.TEST, num_classes=num_classes)
    dataset_test_size = len(dataset_test)
    indices_test = list(range(dataset_test_size))
    np.random.seed(args.seed)
    np.random.shuffle(indices_test)

    # Finally the val sampler
    dataset_val = MoveDatasetPreSplit(args.data_path, settype=SetType.VAL, num_classes=num_classes)
    dataset_val_size = len(dataset_val)
    indices_val = list(range(dataset_val_size))
    np.random.seed(args.seed)
    np.random.shuffle(indices_val)

    logging.info(
        "Set Sizes: %d, %d, %d", len(indices_train), len(indices_test), len(indices_val)
    )

    # Now make the final loaders
    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = torch.utils.data.DataLoader(
        dataset_val, batch_size=args.batch_size,
        shuffle=True,
    )

    return (
        train_loader,
        test_loader,
        val_loader,
        indices_train,
        indices_test,
        indices_val,
    )


def generate_sets(args):
    """Generate the sets from a list of images.
    
    Args:
        args (args): the args object from the main function.    
    """
    dataset = MoveDataset(args.data_path)

    # 90 / 8 / 2 split on train / test / val
    test_split = 0.1  # test and val
    val_split = 0.25  # 2 of 8 of 100
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(test_split * dataset_size))

    # Shuffle and split the dataset into train and test
    np.random.seed(args.seed)
    np.random.shuffle(indices)
    train_indices, temp_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)

    # Split temp into test and val
    temp_size = len(temp_indices)
    split = int(np.floor(val_split * temp_size))

    test_indices, val_indices = temp_indices[split:], temp_indices[:split]

    test_sampler = SubsetRandomSampler(test_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    logging.info(
        "Set Sizes: %d, %d, %d", len(train_indices), len(test_indices), len(val_indices)
    )

    # Now make the final loaders
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, sampler=test_sampler
    )
    val_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, sampler=val_sampler
    )

    return (
        dataset,
        train_loader,
        test_loader,
        val_loader,
        train_indices,
        test_indices,
        val_indices,
    )