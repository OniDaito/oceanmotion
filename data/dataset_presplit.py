#!/usr/bin/env python
r"""
     ___                   _  _      _  o          
     )) ) __  __ ___  _ _  )\/,) __  )L _  __  _ _ 
    ((_( ((_ (('((_( ((\( ((`(( ((_)(( (( ((_)((\( 

Load our npz files from our collection and 
create a pytorch dataset. This uses pre-split
data from the later versions of crabseal.
"""

from __future__ import annotations

__all__ = ["MoveDatasetPreSplit"]
__version__ = "0.9.0"
__author__ = "Benjamin Blundell <bjb8@st-andrews.ac.uk>"

import os
import torch
import numpy as np
import logging
from torch.utils.data import Dataset
from enum import Enum


SetType = Enum('settype', ['TRAIN', 'TEST', 'VAL'])

class MoveDatasetPreSplit(Dataset):
    """Create a Moves Dataset using a pre-splut dataset from
    crabseal.
    """
    """ This class craeates a dataset (either train, test or val) from
    an already split set of npz files, made by crabseal.
    """
    def __init__(self, data_dir, settype=SetType.TRAIN, num_classes=1, sector=False):
        """ Initialise the class. Data_dir points to the crabseal directory of images.
        settype sets which set this is (TRAIN, TEST or VAL), num_classes should match
        the number of classes in the set. Sector should be set to true if the dataset
        is sectored.
    
        Args:
            data_dir (str): the full path to a directory.
            settype (SetType): which set at we reading into? Train, Test or Val?
            num_classes (int): how many classes are there?
            sector (bool): Is this a sectored dataset?

        """
        assert(os.path.exists(data_dir))
     
        if settype == SetType.TRAIN:
            self.data_dir  = os.path.join(data_dir, "images/train")
        elif settype == SetType.TEST:
            self.data_dir  = os.path.join(data_dir, "images/test")
        elif settype == SetType.VAL:
            self.data_dir  = os.path.join(data_dir, "images/val")
   
        assert(os.path.exists(self.data_dir))

        class_to_csv_path = None
        
        if num_classes != 1:
            class_to_csv_path = os.path.join(data_dir, "code_to_class.csv")
            assert(os.path.exists(class_to_csv_path))

        self.num_classes = num_classes
 
        # Load the images - specifically the half size ones for now
        files_base = [f for f in os.listdir(self.data_dir) if os.path.isfile(os.path.join(self.data_dir, f)) and "base.npz" in f]
        files_mask = [f for f in os.listdir(self.data_dir) if os.path.isfile(os.path.join(self.data_dir, f)) and "mask.npz" in f]

        files_base.sort()
        files_mask.sort()

        self.files = list(zip(files_base, files_mask))
        self.num_classes = num_classes

        # We need to find the smallest size from all of our images. Crabseal spits out images that are always the right width but 
        # different lengths due to sonar issues or a different range setting on the sonar. We read the first 100 and take the
        # lowest as we will crop down anything bigger.
        base_file, _ = self.files[0]
        base_npz = os.path.join(self.data_dir, base_file)
        source = np.load(base_npz)
        self.width = source.shape[2]
        self.height = source.shape[1]
        cache_path = os.path.join(data_dir, "cache_img_size_" + str(settype) + ".txt")
    
        if not os.path.exists(cache_path):
            class_examples = {}

            for i in range(1, len(self.files)):
                base_file, mask_file = self.files[i]
                base_npz = os.path.join(self.data_dir, base_file)
                mask_npz = os.path.join(self.data_dir, mask_file)
                source = np.load(base_npz)
                mask = np.load(mask_npz)
                
                if num_classes != 1: # Ignore if we are just doing a binary classification task
                    mclass = np.max(mask)

                    if (mclass >= num_classes):
                        logging.error("Rejected %s as it contains an incorrect class: %d", mask_file, mclass)
                        assert(False)

                    if mclass not in class_examples.keys():
                        class_examples[mclass] = 1
                    else:
                        class_examples[mclass] += 1

                    if source.shape[2] < self.width:
                        self.width = source.shape[2]

                    if source.shape[1] < self.height:
                        self.height = source.shape[1]

                    if sector:
                        # At present sectors are 32x32 at full size, 16 x 16 at half size
                        # As we are taking in half-size at the moment, lets check this is
                        # the right size
                        assert(mask.shape[1] / 16 == 51)
                        assert(mask.shape[2] / 16 == 16)


            # Make sure we have examples of all classes in the dataset
            # Might fail if the set is super small?
            # TODO - this appears to keep failing on the recent datasets. It shouldn't! Why?
            #assert(len(class_examples.keys()) == num_classes)
            if num_classes != 1:
                for k in class_examples.keys():
                    logging.info("Class %s has %d examples.", k, class_examples[k])

            with open(cache_path, 'w') as f:
                f.write(str(self.width) + "," + str(self.height) + "\n")
        else:
            with open(cache_path, 'r') as f:
                line = f.readline()
                tokens = line.replace("\n","").split(",")
                self.width = int(tokens[0])
                self.height = int(tokens[1])

        # Assert that all mask files have some tracks
        for i in range(1, len(self.files)):
            _, mask_file = self.files[i]
            mask_npz = os.path.join(self.data_dir, mask_file)
            mask = np.load(mask_npz)
            assert(np.max(mask) > 0)
        
        logging.info("Dataset image dimensions: %d %d", self.width, self.height)
   
    def __len__(self):
        """Return the length of the set."""
        return len(self.files)

    def __getitem__(self, idx):
        """Return a datum given the position in the file list.
        
        Args:
            idx (int): the index of the item we want.
        """
        base_file, mask_file = self.files[idx]
        base_npz = os.path.join(self.data_dir, base_file)
        mask_npz = os.path.join(self.data_dir, mask_file)
        
        target = np.load(mask_npz)
        source = np.load(base_npz)
        
        # The target doesn't change as we pass in D,W,H - we loose the target channel
        # as the crossentropyloss doesn't need it.
        target = torch.from_numpy(target).to(dtype=torch.long)

        if self.num_classes == 1:
            # Binary classification task so limit the maximum value to 1
            target = torch.where(target > 0, 1, 0).to(dtype=torch.uint8)
    
        source = torch.from_numpy(source).unsqueeze(0).to(dtype=torch.float32)
        source = source / 255.0 # Move to 0 to 1 range.
       
        return source, target