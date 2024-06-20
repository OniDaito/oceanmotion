#!/usr/bin/env python
r"""
     ___                   _  _      _  o          
     )) ) __  __ ___  _ _  )\/,) __  )L _  __  _ _ 
    ((_( ((_ (('((_( ((\( ((`(( ((_)(( (( ((_)((\( 

run_onnx.py - run an onnx model.

"""

from __future__ import annotations

__all__ = ["main"]
__version__ = "0.9.0"
__author__ = "Benjamin Blundell <bjb8@st-andrews.ac.uk>"

import numpy as np
import onnxruntime as rt
from sealhits.db.db import DB
from eval.eval import get_group_np, get_group_og
from run import save_results
import time


def main(args):
    """ Run ONNX against a known group."""
    sess_options = rt.SessionOptions()
    sess_options.enable_profiling = True
    small_img_size = (args.img_width, args.img_height)

    # Find out which sonar is the one to use (or both)
    seal_db = DB(
        db_name=args.dbname,
        username=args.dbuser,
        password=args.dbpass,
        host=args.dbhost,
    )

    frames, og_img_size = get_group_np(
        seal_db,
        args.group_huid,
        small_img_size,
        args.fits_path,
        args.crop_height,
        args.sonar_id,
        args.pred_length,
        False,
    )
    
    def sigmoid(z):
        return 1/(1 + np.exp(-z))

    sess = rt.InferenceSession(
        args.model_file, sess_options, providers=rt.get_available_providers())
    input_name = sess.get_inputs()[0].name
    inf_times = []

    # Go through the group frames as before

    preds = np.zeros(frames.shape, dtype=np.uint8)

    for idx in range(0, len(frames) - args.pred_length + 1):
        start = idx
        end = idx + args.pred_length

        stack = frames[start:end]
        stack = np.expand_dims(stack, axis=0)
        stack = np.expand_dims(stack, axis=0)
        stack = stack.astype(np.float32)
    
        t0 = time.time()
        pred_stack = sess.run(None, {input_name: stack})[0]
        t1 = time.time()
        inf_times.appen(t1-t0)

        pred = np.where(sigmoid(pred_stack) > args.confidence, 1, 0)
        pred = pred.squeeze().astype(np.uint8)


        # TODO - this line essentially means that data arriving later takes precidence
        # past predictions for a frame are overwritten by future results. Is that ideal?
        preds[start:end, :, :] = pred

    prof_file = sess.end_profiling()
    print(prof_file)
    print("Average Inference Time:", sum(inf_times)/len(inf_times))
    # Now get the original group as well
    mask = get_group_og(seal_db, args.group_huid, og_img_size, small_img_size, args.sonar_id, args.crop_height)

    save_results(
        frames, preds, mask, args.out_path, args.group_huid + "_" + str(args.sonar_id), args.img_height, args.polar
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert a model to an onnx format")
    parser.add_argument(
        "-g",
        "--group_huid",
        default="",
        help="(Optional) huid string for checking a group (default: " ")",
    )
    parser.add_argument(
        "-o",
        "--out_path",
        default="./out",
        help="Path to save the output (default: ./out)",
    )
    parser.add_argument(
        "-f", "--fits_path", default=".", help="The path to the fits image files."
    )
    parser.add_argument(
        "-m",
        "--model_file",
        default="./model.onnx",
        help="Path to the model (default: ./model.onnx)",
    )
    parser.add_argument(
        "-t",
        "--model_class",
        default="UNet3D",
        help="The model class to load (default: UNet3D)",
    )
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

    args = parser.parse_args()
    main(args)
