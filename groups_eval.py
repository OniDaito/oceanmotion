#!/usr/bin/env python
"""
     ___                   _  _      _  o          
     )) ) __  __ ___  _ _  )\/,) __  )L _  __  _ _ 
    ((_( ((_ (('((_( ((\( ((`(( ((_)(( (( ((_)((\( 

Author - Benjamin Blundell - bjb8@st-andrews.ac.uk

Run the model against a set of groups, returning a csv of scores,
webm of the result and gifs of the original and predicted masks. 
"""

from __future__ import annotations

__all__ = ["main"]
__version__ = "0.9.0"
__author__ = "Benjamin Blundell <bjb8@st-andrews.ac.uk>"

import torch
import os
from tqdm import tqdm
import numpy as np
from eval.eval import (
    bbraw_to_np,
    group_bbs_raw,
    group_prediction,
    jaccard,
    pred_to_gif,
    pred_to_video,
)
from model.model import UNet3D
from sealhits.db.db import DB
from sealhits.db.dbschema import Groups


def main(args):
    device = torch.device("cpu" if not torch.cuda.is_available() else args.device)
    model = UNet3D(1, 1)
    model.load_state_dict(torch.load(args.modelpath))
    model.to(device=device)
    model.eval()

    seal_db = DB(
        db_name=args.dbname,
        username=args.dbuser,
        password=args.dbpass,
        host=args.dbhost,
    )

    groups = seal_db.get_groups_filters([Groups.code == "seal"])

    with open(os.path.join(args.outpath, "group_test.csv"), "w") as f:
        f.write("huid,time,areaog,areapred,jaccard\n")

        for group in tqdm(groups, "Evaluating groups"):
            group_images = seal_db.get_images_group_sonarid(group.uid, args.sonarid)

            if len(group_images) > 120: # avoid very long groups
                continue

            if len(group_images) <= args.pred_length: # Shouldn't happen but apparently does :S
                continue

            img_size = (args.img_width, args.img_height)

            sources, preds = group_prediction(
                model,
                group_images,
                args.fitspath,
                img_size,
                args.pred_length,
                args.confidence,
                device,
            )
            offset = args.pred_length - 1
            masks = []
            areas_og = []

            for idx in range(len(sources)):
                bbs = group_bbs_raw(
                    seal_db, group, group_images[offset + idx], img_size
                )
                area = 0

                for bb in bbs:
                    area += bb.area()

                areas_og.append(area)
                mask = bbraw_to_np(bbs, img_size)
                masks.append(mask)

            pred_np = np.array(preds)
            source_np = np.array(sources)
            mask_np = np.array(masks)
            outpath = os.path.join(args.outpath, group.huid + ".webm")
            pred_to_video(source_np, mask_np, pred_np, outpath)
            outpath = os.path.join(args.outpath, group.huid + "_pred.gif")
            pred_to_gif(pred_np, outpath)
            outpath = os.path.join(args.outpath, group.huid + "_mask.gif")
            pred_to_gif(mask_np, outpath)

            for idx, pred_single in enumerate(preds):
                op_single = masks[idx]
                area_pred = np.sum(pred_single)
                jscore = jaccard(op_single, pred_single)

                f.write(
                    group.huid
                    + ","
                    + str(group_images[offset + idx].time)
                    + ","
                    + str(areas_og[idx])
                    + ","
                    + str(area_pred)
                    + ","
                    + str(jscore)
                    + "\n"
                )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        prog="oceanmotion - run",
        description="Run a model against a set of groups and get a prediction back.",
        epilog="SMRU St Andrews",
    )
    parser.add_argument(
        "-m", "--modelpath", default=".", help="The path to the saved model."
    )
    parser.add_argument(
        "-f", "--fitspath", default=".", help="The path to the fits image files."
    )
    parser.add_argument("-o", "--outpath", default=".", help="The path for the output.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--pred_length",
        type=int,
        default=16,
        help="The length of the prediction window (default: 16)",
    )
    parser.add_argument(
        "-c",
        "--confidence",
        type=float,
        default=0.7,
        help="What confidence level do we want (default: 0.7)",
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
        default=829,
        help="The width of the input images (default: 829)",
    )
    parser.add_argument(
        "-r",
        "--sonarid",
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
