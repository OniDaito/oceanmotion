#!/usr/bin/env python
r"""
     ___                   _  _      _  o
     )) ) __  __ ___  _ _  )\/,) __  )L _  __  _ _
    ((_( ((_ (('((_( ((\( ((`(( ((_)(( (( ((_)((\(

eval.py - Evaluation functions for oceanmotion.
"""

from __future__ import annotations

__all__ = ["predict", "get_group_np", "get_group_og"]
__version__ = "0.9.0"
__author__ = "Benjamin Blundell <bjb8@st-andrews.ac.uk>"

import torch
import numpy as np
from typing import List, Tuple, Union
import torch.nn.functional as F
from eval.detection import blobs_in_stack
from sealhits import utils
from sealhits.db.dbschema import Groups, Images
from sealhits.db.db import DB
from sealhits.bbox import points_to_bb, XYBox, bb_overlap
from sealhits.image import fits_to_np
from sealhits.utils import fast_find
from skimage.transform import resize


def group_bbs_raw(
    seal_db: DB, group: Groups, img_rec: Images, img_size: Tuple[int, int]
) -> List[XYBox]:
    """Return the bounding boxes for a group.

    Args:
       seal_db (DB): connection to the database object.
       group (Groups): the Groups object we are interested in.
       img_rec (Images): the Images object we are interestd in.
       img_size (Tuple[int, int]): The image size (width then height in pixels).
    """
    fname = img_rec.filename
    points = seal_db.get_image_points_by_filename_group(fname, group.uid)
    bbs = []

    if (
        len(points) > 0
    ):  # Since we have buffer start / end images and intermediates, 0 points on an image is possible
        bb = points_to_bb(points, img_rec.range)
        bbox = bb.to_xy_raw(img_size)
        bbs.append(bbox)

    return bbs


def bbraw_to_np(bbs: List[XYBox], img_size: Tuple[int, int]) -> np.array:
    """Convert a list of raw BBS to an np array we can compare.

    Args:
        bbs (List[XYBox]): List of bounding boxes.
        img_size (Tuple[int, int]): the image size (width then height in pixels).
    """
    og = np.zeros((img_size[1], img_size[0]), dtype=np.uint8)

    for b in bbs:
        og[b.y_min : b.y_max, b.x_min : b.x_max] = 1

    return og


def group_has_fits(seal_db: DB, fits_path: str, group_uid: str, sonar_id: int):
    """Make sure this group has images. Sometimes they don't for reasons I'm not sure of.

    Args:
        seal_db (DB): The database object.
        fits_path (str): The path to the FITS file.
        group_uid (str): The UID of the group.
        sonar_id (int): The Sonar ID.
    """
    imgs = seal_db.get_images_group_sonarid(group_uid, sonar_id)

    for img in imgs:
        fname = img.filename
        fresult = utils.fast_find(fname, fits_path)
        if fresult is None:
            return False

    return True


def _predict(model, frames: np.array, device: str, confidence: float) -> np.array:
    """Perform the prediction, now we have frames,
    a prediction length and a device.

    Args:
        model (): The current model.
        frames (np.array): The frames to predict over.
        device (str): The curent device we are running on (cuda or cpu).
        confidence (float): The confidence value we must meet or beat.
    """
    assert frames.shape[0] > 1
    stack = torch.from_numpy(frames).to(device=device)
    unsqueezed = stack.unsqueeze(0).unsqueeze(0).to(dtype=torch.float32)
    del stack

    # TODO - could return probabilities here?
    pred = model(unsqueezed)
    pred = torch.where(F.sigmoid(pred) > confidence, 1, 0)
    pred = pred.squeeze().cpu().detach().long().numpy().astype(np.uint8)

    return pred


def counts(bbs: List[XYBox], pred_single: np.array):
    """Return the original counts, predicted counts and overlapping counts.

    Args:
        bbs (List[XYBox]): The list of boudning boxes.
        pred_single (np.array): A predicted frame.
    """
    og_count = len(bbs)
    overlap_count = 0
    pred_count = 0

    # Find the blobs if there are any
    if np.max(pred_single) > 0:
        blobs = blobs_in_stack(pred_single)
        count = len(blobs)

        if count != 0:
            # We now check the original bounding boxes
            # against the new bounding box blobs. It's not
            # pefect but it's quick and reasonable.
            for blob in blobs:
                minr, minc, maxr, maxc = blob.bbox
                detbox = XYBox(minc, minr, maxc, maxr)

                for bb in bbs:
                    if bb_overlap(bb, detbox):
                        overlap_count += 1
                        break

    return (og_count, pred_count, overlap_count)


def jaccard(og: np.array, pred: np.array):
    """Generate a jaccard score for this original mask and predicted mask.
    As this could be multiclass, we check the prediction and just go with
    a binary mask.

    Args:
        og (np.array): The first np.array to compare.
        pred (np.array): The second np.array to compare against the first.
    """

    if pred.shape[0] > 1:
        pred = np.argmax(pred, axis=0, keepdims=True)

    pred = np.where(pred > 0, 1, 0)
    og = np.where(og > 0, 1, 0)

    intersection = np.sum(og * pred) + 0.00001
    union = np.sum(np.clip(og + pred, 0, 1)) + 0.00001
    jaccard = intersection / union
    return jaccard


def get_group_og(
    seal_db: DB,
    group_huid: str,
    og_img_size: Tuple[int, int],
    small_img_size: Tuple[int, int],
    sonar_id: int,
    crop_height: int,
) -> np.array:
    """Get the original track, converted to bounding box format,
      and return it as a numpy mask. Rather than the original points, we draw a box around
      all of them and return that instead.
    Args:
        seal_db (DB): The Database Object.
        group_huid (str): the group huid.
        og_img_size (Tuple[int, int]): The original image size (width then height in pixels).
        small_img_size (Tuple[int, int]): The image size post crop and scale.
        sonar_id (int): The sonar id.
        crop_height (int): the height the original image was cropped to.
    """
    group = seal_db.get_group_huid(group_huid)
    group_images = seal_db.get_images_group_sonarid(group.uid, sonar_id)
    masks = []

    for idx in range(len(group_images)):
        # Get the BBS but at the original FITS image size
        # Remember, we perform a height crop then a resize.
        bbs = group_bbs_raw(seal_db, group, group_images[idx], og_img_size)
        mask = bbraw_to_np(bbs, og_img_size)
    
        # Crop to the cropheight and resize - nearest-neighbour accurate
        mask = mask[0:crop_height, :]
        mask = resize(mask, (small_img_size[1], small_img_size[0]), preserve_range=True)
        # Convert back to a uint8 and make sure its 0 or 1 on this resize
        mask = np.where(mask > 0.0, 1.0, 0.0)
        mask = mask.astype(np.uint8)
        masks.append(mask)

    return np.array(masks).astype(np.uint8)


def get_group_np(
    seal_db: DB,
    group_huid: str,
    small_img_size: Tuple[int, int],
    fits_path: str,
    crop_height: int,
    sonar_id: int,
    pred_length=16,
    cthulhu=False,
) -> Union[Tuple[np.array, Tuple[int, int]], None]:
    """Given an existing group, make a prediction as to what it is and where.

    Args:
        seal_db (DB): The Database Object.
        group_huid (str): the group huid.
        small_img_size (Tuple[int, int]): The image size post crop and scale.
        fits_path (str): the path to the FITS files.
        crop_height (int): the height the original image was cropped to.
        sonar_id (int): the sonar ID.
        pred_length (int): the prediction window size in frames.
        cthulhu (bool): invoke the un-nameable!
    """
    group = seal_db.get_group_huid(group_huid)
    group_images = seal_db.get_images_group_sonarid(group.uid, sonar_id)
    raw_img_size = None

    if len(group_images) <= pred_length:
        return None

    frames = []
    c = None

    if cthulhu:
        # Add lil Cthulthu to all the frames
        c = np.load("cthulhu_fan.npz")["x"]

    # Load the images from the FITS but then crop, resize, then
    # convert to float 0 to 1.0 range.
    for idx, img in enumerate(group_images):
        fname = img.filename
        fresult = fast_find(fname, fits_path)
        raw_img, _ = fits_to_np(fresult)

        if raw_img_size is None:
            raw_img_size = (raw_img.shape[1], raw_img.shape[0])

        final_img = raw_img[0:crop_height, :]  # Crop down

        final_img = resize(
            final_img, (small_img_size[1], small_img_size[0]), preserve_range=True
        )

        if cthulhu:
            # Add lil Cthulthu to all the frames
            c = np.load("cthulhu_fan.npz")["x"]
            c = resize(c, final_img.shape, preserve_range=True)
            final_img = np.clip(final_img + c, 0, 255)

        final_img = final_img.astype(float) / 255.0
        assert np.max(final_img) <= 1.0
        frames.append(final_img)

    frames = np.array(frames)

    return (frames, raw_img_size)


def predict(
    model, frames: np.array, device: str, pred_length=16, confidence=0.8
) -> Tuple[np.array, np.array, List[int]]:
    """Perform a prediction but we have an NPZ Stack. We look at the
    entire window in one go. We update with the latest as we move forward
    so later time points have precedence.

    Args:
        model (): The current model.
        frames (np.array): the frames we are predicting over.
        device (str): The device we are running on.
        pred_length (int): the prediction window size.
        confidence (float): the meet or beat score for a positive classification.
    """
    preds = np.zeros(frames.shape, dtype=np.uint8)

    for idx in range(0, len(frames) - pred_length + 1):
        start = idx
        end = idx + pred_length
        pred_stack = _predict(model, frames[start:end], device, confidence)

        # TODO - this line essentially means that data arriving later takes precidence
        # past predictions for a frame are overwritten by future results. Is that ideal?
        preds[start:end, :, :] = pred_stack
        assert(preds.base is None)
        del pred_stack

    return preds
