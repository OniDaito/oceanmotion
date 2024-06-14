#!/usr/bin/env python
r"""
     ___                   _  _      _  o          
     )) ) __  __ ___  _ _  )\/,) __  )L _  __  _ _ 
    ((_( ((_ (('((_( ((\( ((`(( ((_)(( (( ((_)((\( 

Load our npz files from our collection and 
create a pytorch dataset.
"""

from __future__ import annotations

__all__ = ["MoveDataset"]
__version__ = "0.9.0"
__author__ = "Benjamin Blundell <bjb8@st-andrews.ac.uk>"

import os
import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms.functional as F

class MoveDataset(Dataset):
    """ Create a dataset from a set of files that haven't already been partitioned."""
    
    def __init__(self, data_dir):
        """ Initialise with a path to the directory holding the data.
            
            Args:
                data_dir (string): full path to the data directory.
        """
        self.data_dir = data_dir
        
        if os.path.isdir(os.path.join(data_dir, "images")):
            self.data_dir = os.path.join(data_dir, "images")

        files_base = [f for f in os.listdir(self.data_dir) if os.path.isfile(os.path.join(self.data_dir, f)) and "base.npz" in f]
        files_mask = [f for f in os.listdir(self.data_dir) if os.path.isfile(os.path.join(self.data_dir, f)) and "mask.npz" in f]

        files_base.sort()
        files_mask.sort()

        self.files = list(zip(files_base, files_mask))

        # record, to make sure we have consistent sizes. For some reaseon, crabseal still spits out wrong-uns
        base_file, _ = self.files[0]
        base_npz = os.path.join(self.data_dir, base_file)
        source = np.load(base_npz)
        self.width = source.shape[2]
        self.height = source.shape[1]
   
    def __len__(self):
        """ Return the length of the dataset."""
        return len(self.files)

    def __getitem__(self, idx):
        """ Return a datum with the given index.
        Args:
            idx (int): the index of the item to return.
        """
        base_file, mask_file = self.files[idx]
        base_npz = os.path.join(self.data_dir, base_file)
        mask_npz = os.path.join(self.data_dir, mask_file)
        
        target = np.load(mask_npz)
        source = np.load(base_npz)
        assert(np.max(source) > 0 and np.max(source) < 256) # Should be in the uint8 range

        target = torch.from_numpy(target).unsqueeze(0).to(dtype=torch.uint8)

        # Sometimes, the target is 0 or 255 cos visualisation?
        if torch.max(target) > 1:
            target = torch.where(target > 0, 1, 0).to(dtype=torch.uint8)

        # We need to alter the target to be a single classification - either 0 or 1
        #target = torch.where(target > 0, 1, 0).to(dtype=torch.uint8)
        source = torch.from_numpy(source).unsqueeze(0).to(dtype=torch.float32)
        source = source / 255.0 # Move to 0 to 1 range.

        # Double check that the sizes are correct. Some often aren't, off by one 
        # because of rounding it seems.
        if source.shape[3] != self.width or source.shape[2] != self.height:
            source = F.resize(source, (self.height, self.width))

        if target.shape[3] != self.width or target.shape[2] != self.height:
            target = F.resize(target, (self.height, self.width))
       
        return source, target