#!/usr/bin/env python
r"""
     ___                   _  _      _  o          
     )) ) __  __ ___  _ _  )\/,) __  )L _  __  _ _ 
    ((_( ((_ (('((_( ((\( ((`(( ((_)(( (( ((_)((\( 

fast_eval_onnx.py runs an OceanMotion model against a list of
GLFs but using the ONNX version of the model. This version
*should* be a little more stable than the pytorch one.

It previously requested images from a websocket server
until there are no more. But this was removed for brevity.

Fast eval saves detections as bounding boxes in an SQLLite
database, rather than the entire predicted mask, in order
to save space and time.

"""

from __future__ import annotations

__all__ = ["main"]
__version__ = "0.9.0"
__author__ = "Benjamin Blundell <bjb8@st-andrews.ac.uk>"

import pytz
import sys
import os
import numpy as np
import sqlite3
from sqlite3 import Error
from concurrent.futures import ThreadPoolExecutor, as_completed
from eval.detection import bbs_in_image
from datetime import datetime, timedelta
from skimage.transform import resize
from eval.glf import GLFBuffer
from tqdm import tqdm
from typing import Tuple
import onnxruntime as rt


def loop(
    model,
    gbuff: GLFBuffer,
    confidence: float,
    img_size: Tuple[int, int],
    crop_height: int,
    start_time: datetime,
    end_time: datetime,
    out_path: str,
    pbar: tqdm,
    qsize=16,
    halfrate=False,
):
    """ Loop through the range of GLFs, predicting and saving these predictions 
    as fast as possible.
    
    Args:
        model (): Our current model in onnx format.
        gbuff (GLFBuffer): a GLFBuffer instance.
        confidence (float): the meet or beat score for a positive classification.
        img_size (Tuple[int, int]): the image size (width then height in pixels).
        crop_height (int): the crop height of the large image.
        start_time (datetime): the start time.
        end_time (datetime): the end time.
        out_path (str): path we are saving things to.
        pbar (tqdm): the tqdm progress bar.
        qsize (int): the window size.
        halfrate (bool): drop every other frame.
    """
    queue = []
    conn = setup_sqlite(out_path, start_time, end_time)
    cur = conn.cursor()
    step = 0
    np_filename = ""
    detecting = False
    current_detection = []
    current_bbs = []
    current_base = []
    current_frame_times = []
    progress = [(start_time, 0)]
    pwindow = timedelta(seconds=30)  # number of seconds for our plotting window
    ptime = start_time
    pcount = 0
    rows = []

    sess_options = rt.SessionOptions()
    sess_options.enable_profiling = True

    def sigmoid(z):
        return 1/(1 + np.exp(-z))

    try:
        sess = rt.InferenceSession(
            model, sess_options, providers=rt.get_available_providers())
        input_name = sess.get_inputs()[0].name


        while True:
            # tick = time.perf_counter()
            np_img, img_time = gbuff.__next__()

            # Potentially skip every other frame
            if halfrate and step % 2 == 0:
                step += 1
                continue

            # Perform the crop and resize on our images.
            final_img = np_img[0:crop_height, :]
            final_img = resize(final_img, img_size, preserve_range=True)
            final_img = final_img.astype(float) / 255.0
            assert(np.max(final_img) <= 1.0) 
            queue.append((final_img, img_time))

            # Do we have a big enough queue yet?
            if len(queue) > qsize:
                queue.pop(0)
                np_queue = np.array([q[0] for q in queue])

                stack = np.expand_dims(np_queue, axis=0)
                stack = np.expand_dims(np_queue, axis=0)
                stack = stack.astype(np.float32)
    
                pred_stack = sess.run(None, {input_name: stack})[0]

                pred = pred_stack[-1]
                pred = np.where(sigmoid(pred) > confidence, 1, 0)
                pred = pred.squeeze().astype(np.uint8)

                # Need to see if there is any detection here in the preds
                # TODO - this is super simple and also, we are looking at the
                # most recent frame only.
                # As I recall, fancier methods didn't really seem to work.
                cbase = final_img

                if np.max(pred) > 0:
                    if not detecting:
                        detecting = True
                    
                    bbs = bbs_in_image(pred)

                    cpred = (pred * 255).astype(np.uint8)
                    current_detection.append(cpred)
                    current_bbs.append((img_time, bbs))
                    current_base.append(cbase)
                    current_frame_times.append(img_time)
                    pcount += len(bbs)

                elif detecting:
                    # We can stop detecting now.
                    detecting = False
                    current_detection = []
                    current_frame_times = []
                    # Now write out to the sqlite file
                    rows = []

                    for dd_img, bboxes in sorted(current_bbs):
                        for bbox in bboxes:
                            row = (
                                str(dd_img),
                                bbox.x_min,
                                bbox.y_min,
                                bbox.x_max,
                                bbox.y_max,
                                np_filename,
                            )
                            rows.append(row)

                    cur.executemany(
                        "INSERT INTO detections VALUES(?, ?, ?, ?, ?, ?)", rows
                    )
                    conn.commit()
                    rows = []
                    current_bbs = []

                sys.stdout.flush()

            # Update our stats for the count
            # Only store the last ten counts
            if img_time >= ptime + pwindow:
                progress.append((ptime, pcount))
                ptime = img_time
                pcount = 0

                if len(progress) > 10:
                    progress.pop(0)

            pbar.update(1)

    except StopIteration:
        if len(rows) > 0:
            cur.executemany("INSERT INTO detections VALUES(?, ?, ?, ?, ?, ?)", rows)
            conn.commit()
            rows = []

        print("GBuffer completed.")

    return "Completed"

def setup_sqlite(out_path: str, start_date: datetime, end_date: datetime):
    """Create the SQLITE file to hold the required outputs.
    The schema is:

        datetime - text
        x_min - integer (pixels on the image)
        y_min - integer (pixels on the image)
        x_max - integer (pixels on the image)
        y_max - integer (pixels on the image)
        filename - text

    There is no primary key sadly, but we do create an index on
    datetime.

    Args:
        out_path (str): The output path
        start_date (datetime): the start date.
        end_date (datetime): the end date.
    """
    conn = None
    sqlname = (
        start_date.strftime("%Y%m%d-%H%M%S")
        + "_"
        + end_date.strftime("%Y%m%d-%H%M%S")
        + ".sqlite3"
    )
    db_file = os.path.join(out_path, sqlname)

    if os.path.exists(db_file):
        # Delete the sqlite3 file if it already exists. Dangerous!
        os.remove(db_file)

    try:
        conn = sqlite3.connect(db_file)
        cur = conn.cursor()
        cur.execute(
            "CREATE TABLE IF NOT EXISTS detections(datetime text, x_min integer, y_min integer, x_max integer, y_max integer, filename text)"
        )
        cur.execute("CREATE INDEX datetime_index on detections(datetime)")

        return conn

    except Error as e:
        print(e)

    return None


def main(args):
    start_date = None
    end_date = None

    try:
        start_date = datetime.strptime(args.startdate, "%Y-%m-%d %H:%M:%S").replace(
            tzinfo=pytz.utc
        )
    except ValueError:
        start_date = None
        print("Not using start date.")

    try:
        end_date = datetime.strptime(args.enddate, "%Y-%m-%d %H:%M:%S").replace(
            tzinfo=pytz.utc
        )
    except ValueError:
        start_date = None
        print("Not using end date.")

    assert start_date < end_date
    
    estimated_total = (end_date - start_date)
    estimated_total = int(estimated_total.total_seconds() * 4)
    print("Estimated number of steps:", estimated_total)

    mid_date = (end_date - start_date) / 2 + start_date
    img_size = (args.img_width, args.img_height)
    jobs = []

    gbuff0 = GLFBuffer(
        args.glf_path, args.sonarid, args.img_width, args.img_height, start_date, mid_date
    )

    gbuff1 = GLFBuffer(
        args.glf_path, args.sonarid, args.img_width, args.img_height, mid_date, end_date
    )

    with tqdm(total=estimated_total) as pbar:
        jobs.append((args.model_path, gbuff0, start_date, mid_date))
        jobs.append((args.model_path, gbuff1, mid_date, end_date))
    
        with ThreadPoolExecutor(max_workers=len(jobs)) as ex:
            futures = [ex.submit(loop, m, g, args.confidence, img_size, args.crop_height, s, e, args.out_path, pbar, args.window_size, args.halfrate) for m,g,s,e in jobs]

            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as exc:
                    print('%r generated an exception: %s' % (future, exc))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        prog="oceanmotion - fast_eval",
        description="Run the model against a number of GLFs as fast as possible.",
        epilog="SMRU St Andrews",
    )
    parser.add_argument(
        "-m", "--model_path", default=".", help="The path to the saved model in ONNX format."
    )
    parser.add_argument("-o", "--out_path", default=".", help="The path for the output.")
    parser.add_argument(
        "-l", "--glf_path", default=".", help="The path to the GLF Files."
    )
    parser.add_argument(
        "-a",
        "--startdate",
        default="",
        help="An optional start date in Y-m-d H:M:S format (default: '').",
    )
    parser.add_argument(
        "-b",
        "--enddate",
        default="",
        help="An optional end date in Y-m-d H:M:S format (default: '').",
    )
    parser.add_argument(
        "-w",
        "--window_size",
        type=int,
        default=16,
        help="The window size (default: 16)",
    )
    parser.add_argument(
        "-x",
        "--img_width",
        type=int,
        default=256,
        help="The image width (default: 256)",
    )
    parser.add_argument(
        "-y",
        "--img_height",
        type=int,
        default=816,
        help="The image height (default: 816)",
    )
    parser.add_argument(
        "--crop_height",
        type=int,
        default=1632,
        help="Before any resize, what height do we crop raw images to?(default: 1632))",
    )
    parser.add_argument(
        "-r",
        "--sonarid",
        type=int,
        default=854,
        help="The sonar id (default: 854)",
    )
    parser.add_argument(
        "-c",
        "--confidence",
        type=float,
        default=0.8,
        help="What confidence level do we want (default: 0.8)",
    )
    parser.add_argument(
        "--halfrate",
        action="store_true",
        default=False,
        help="Drop every other frame (default: False)",
    )
    parser.add_argument(
        "-t",
        "--model_class",
        default="UNet3D",
        help="The model class to load (default: UNet3D)",
    )


    args = parser.parse_args()

    main(args)
