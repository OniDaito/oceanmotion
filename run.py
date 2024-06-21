#!/usr/bin/env python
r"""
     ___                   _  _      _  o
     )) ) __  __ ___  _ _  )\/,) __  )L _  __  _ _
    ((_( ((_ (('((_( ((\( ((`(( ((_)(( (( ((_)((\(

run.py -Run the model against a set of images, either
from a group or a timerange + glfs.
"""

from __future__ import annotations

from eval.detection import bbs_in_image

__all__ = ["main"]
__version__ = "0.9.0"
__author__ = "Benjamin Blundell <bjb8@st-andrews.ac.uk>"

import torch
import os
import pytz
from tqdm import tqdm
import numpy as np
from util.model import load_model_pt
from datetime import datetime
from eval.eval import bbraw_to_np, jaccard, predict, get_group_np, get_group_og
from eval.glf import get_glf_time_np
from util.image import pred_to_gif
from util.video import pred_to_video
from sealhits.db.db import DB
from sealhits.image import fan_distort, bearing_table
from sealhits.utils import get_fan_size


def predict_group(args, model, device: str):
    """We are predicting a group from the database.

    Args:
        args (): the program arguments.
        model (): the model we are using.
        device (str): the device we are on.

    """
    small_img_size = (args.img_width, args.img_height)

    # Find out which sonar is the one to use (or both)
    seal_db = DB(
        db_name=args.dbname,
        username=args.dbuser,
        password=args.dbpass,
        host=args.dbhost,
    )

    group_res = get_group_np(
        seal_db,
        args.group_huid,
        small_img_size,
        args.fits_path,
        args.crop_height,
        args.sonar_id,
        args.pred_length,
        args.cthulhu,
    )

    if group_res is None:
        if args.score_only:
            print(0)
        else:
            print(
                "Group",
                args.group_huid,
                "is shorter than the window with sonar",
                args.sonar_id,
            )
        return

    frames, og_img_size = group_res
    preds = predict(model, frames, device, args.pred_length)
    preds_bbs = []

    for pred in preds:
        bbs = bbs_in_image(pred)
        npred = bbraw_to_np(bbs,(frames.shape[2], frames.shape[1]))
        preds_bbs.append(npred)

    preds_bbs = np.array(preds_bbs)
        
    # Now get the original group as well. This original mask
    # is actually converted to a bounding box, so we should 
    # do the same with our prediction.
    mask = get_group_og(
        seal_db,
        args.group_huid,
        og_img_size,
        small_img_size,
        args.sonar_id,
        args.crop_height,
    )

    if args.score_only:
        print(jaccard(mask, preds_bbs))
    else:
        save_results(
            frames,
            preds_bbs,
            mask,
            args.out_path,
            args.group_huid + "_" + str(args.sonar_id),
            args.img_height,
            args.polar,
        )


def predict_times(args, model, start_date: datetime, end_date: datetime, device: str):
    """
    Predict but between two times.

    Args:
        args (): program args.
        model (): the model we are using.
        start_date (datetime): the start date.
        end_date (datetime): the end date.
        device (str): the device we are using.

    """
    # Find the GLFs and read through till we get to the times we want.
    small_img_size = (args.img_width, args.img_height)
    queue = []
    frames = []
    preds = []

    for frame in get_glf_time_np(
        args.glf_path,
        start_date,
        end_date,
        small_img_size,
        args.sonar_id,
        args.crop_height,
        args.halfrate,
        args.cthulhu,
    ):
        queue.append(frame)

        if len(queue) == args.pred_length:
            np_queue = np.array(queue)
            pred = predict(model, np_queue, device, args.pred_length, args.confidence)
            datename = (
                str(start_date).replace(" ", "_").replace(":", "").replace(".", "")
            )
            datename += "_" + str(end_date).replace(" ", "_").replace(":", "").replace(
                ".", ""
            )
            preds.append(pred[-1])
            frames.append(queue[-1])
            queue.pop(0)

    save_results(frames, preds, None, args.out_path, datename, args.polar)


def save_results(
    sources: np.array,
    preds: np.array,
    ogmask: np.array,
    out_path: str,
    files_prefix: str,
    img_height: int,
    polar=False,
):
    """Save the predictions and sources to gif, npz and webm.

    Args:
        sources (np.array): the sources.
        preds (np.array): the predictions.
        ogmask (np.array): the original masks.
        out_path (str): directory to save results in.
        files_prefix (str): the prefix to add to saved files.
        img_height (int): the height of the images being saved.
        polar (bool): convert to polar format.
    """
    fan_height = get_fan_size(img_height)

    # Start with sources
    if polar:
        sources_fan = []

        for source in tqdm(sources, desc="Fan Conversion of sources"):
            fan = np.fliplr(
                np.flipud(fan_distort(source, fan_height[1], bearing_table))
            )
            sources_fan.append(fan)

        sources = sources_fan

    source_np = np.array(sources)
    del sources

    # now the predictions
    if polar:
        preds_fan = []

        for pred in tqdm(preds, desc="Fan Conversion of Predictions"):
            fan = np.fliplr(np.flipud(fan_distort(pred, fan_height[1], bearing_table)))
            # As this messes the masks up a bit we need to do a little chopping
            fan = np.where(fan > 0.5, 1, 0)
            preds_fan.append(fan)

        preds = preds_fan

    pred_np = np.array(preds)
    del preds

    # Finally the original mask if we have it
    if ogmask is not None:
        if polar:
            ogmask_fan = []

            for mask in tqdm(ogmask, desc="Fan Conversion of OG Mask"):
                fan = np.fliplr(
                    np.flipud(fan_distort(mask, fan_height[1], bearing_table))
                )
                # As this messes the masks up a bit we need to do a little chopping
                fan = np.where(fan > 0.5, 1, 0)
                ogmask_fan.append(fan)

            ogmask = ogmask_fan
            ogmask = np.array(ogmask)

    print("Sizes of sources/preds", source_np.shape, pred_np.shape)

    full_outpath = os.path.join(out_path, files_prefix + ".webm")
    pred_to_video(source_np, ogmask, pred_np, full_outpath, False)
    full_outpath = os.path.join(out_path, files_prefix + ".gif")
    pred_to_gif(pred_np, full_outpath)
    full_outpath = os.path.join(out_path, files_prefix + "_base.npz")
    np.savez_compressed(full_outpath, x=source_np)
    full_outpath = os.path.join(out_path, files_prefix + "_pred.npz")
    np.savez_compressed(full_outpath, x=pred_np)

    if ogmask is not None:
        full_outpath = os.path.join(out_path, files_prefix + "_ogmask.npz")
        np.savez_compressed(full_outpath, x=ogmask)


def main(args):
    """Load the model, then predict based on the command line options."""
    # Load the model
    device = torch.device("cpu" if not torch.cuda.is_available() else args.device)
    model = load_model_pt(args.model_path, device, args.model_class)

    # Decide if we are predicting a group from the DB or a time range from
    # a bunch of GLF files.
    if len(args.group_huid) > 0:
        predict_group(args, model, device)

    elif len(args.start_time) > 0 and len(args.end_time) > 0:
        # Organise the start and end times for a GLF range prediction.
        try:
            start_time = datetime.strptime(
                args.start_time, "%Y-%m-%d %H:%M:%S.%f"
            ).replace(tzinfo=pytz.timezone("UTC"))
            end_time = datetime.strptime(args.end_time, "%Y-%m-%d %H:%M:%S.%f").replace(
                tzinfo=pytz.timezone("UTC")
            )

            predict_times(args, model, start_time, end_time, device)

        except ValueError as e:
            print(e)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        prog="oceanmotion - run",
        description="Run a model over an NPZ file and get a prediction back.",
        epilog="SMRU St Andrews",
    )
    parser.add_argument(
        "-m", "--model_path", default=".", help="The path to the saved model."
    )
    parser.add_argument(
        "-f", "--fits_path", default=".", help="The path to the fits image files."
    )
    parser.add_argument(
        "-o", "--out_path", default=".", help="The path for the output."
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--pred_length",
        type=int,
        default=16,
        help="The length of the prediction window (default: 16)",
    )
    parser.add_argument(
        "--crop_height",
        type=int,
        default=1632,
        help="Before any resize, what height do we crop raw images to?(default: 1632))",
    )
    parser.add_argument(
        "-p",
        "--polar",
        action="store_true",
        default=False,
        help="Convert to polar plot (default: False)",
    )
    parser.add_argument(
        "--halfrate",
        action="store_true",
        default=False,
        help="Drop half the frames (default: False)",
    )
    parser.add_argument(
        "--cthulhu",
        action="store_true",
        default=False,
        help="Summon unspeakable horrors from the deepest depths (default: False)",
    )
    parser.add_argument(
        "--img_width",
        type=int,
        default=256,
        help="The width of the input images (default: 256)",
    )
    parser.add_argument(
        "--img_height",
        type=int,
        default=816,
        help="The height of the polar image if we are using that (default: 816)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.8,
        help="For the (default: 0.8)?",
    )
    parser.add_argument(
        "-r",
        "--sonar_id",
        type=int,
        default=854,
        help="Which Sonar are we looking at (default: 854)?",
    )
    parser.add_argument(
        "-d",
        "--dbname",
        default="sealhits",
        help="The name of the postgresql database (default: sealhits)",
    )
    parser.add_argument(
        "-u",
        "--dbuser",
        default="sealhits",
        help="The username for the postgresql database (default: sealhits)",
    )
    parser.add_argument(
        "-w",
        "--dbpass",
        default="kissfromarose",
        help="The password for the postgresql database (default: kissfromarose)",
    )
    parser.add_argument(
        "-n",
        "--dbhost",
        default="localhost",
        help="The hostname for the postgresql database (default: localhost)",
    )
    parser.add_argument(
        "-g",
        "--group_huid",
        default="",
        help="(Optional) huid string for checking a group (default: " ")",
    )
    parser.add_argument(
        "-l",
        "--glf_path",
        default="",
        help="(Optional) path to the GLFs if using time range (default: " ")",
    )
    parser.add_argument(
        "-a",
        "--start_time",
        default="",
        help="The start time if we are using GLFs (Format - 2024-01-01 01:01:01.001) (default: none)",
    )
    parser.add_argument(
        "-b",
        "--end_time",
        default="",
        help="The end time if we are using GLFs (Format - 2024-01-01 01:01:01.001) (default: none)",
    )
    parser.add_argument(
        "--sector",
        action="store_true",
        default=False,
        help="Rather than predict full paths, predict sectors(default: false)",
    )
    parser.add_argument(
        "--score-only",
        action="store_true",
        default=False,
        help="Just output the Jaccard Score (default: false)",
    )
    parser.add_argument(
        "-t",
        "--model_class",
        default="UNetTRed",
        help="The model class to load (default: UNetTRed)",
    )

    args = parser.parse_args()
    main(args)
