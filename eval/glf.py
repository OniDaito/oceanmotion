#!/usr/bin/env python
r"""
     ___                   _  _      _  o          
     )) ) __  __ ___  _ _  )\/,) __  )L _  __  _ _ 
    ((_( ((_ (('((_( ((\( ((`(( ((_)(( (( ((_)((\( 

Find the GLFS we want and create a buffer of images
ready to feed our network.
"""

from __future__ import annotations

__all__ = ["GLFBuffer", "images_from_glf", "get_glf_time_np"]
__version__ = "0.9.0"
__author__ = "Benjamin Blundell <bjb8@st-andrews.ac.uk>"

import os
import pytz
import numpy as np
from tqdm import tqdm
from PIL import Image
from datetime import datetime
from skimage.transform import resize
from sealhits.sources.files import glf_files_avail
from pytritech.glftimes import glf_times
from pytritech.glf import GLF
from typing import List, Generator, Tuple


class GLFBuffer:
    def __init__(
        self,
        path: str,
        sonar_id: int,
        crop_height: int,
        img_width: int,
        img_height: int,
        start_date: datetime,
        end_date: datetime,
    ):
        """ Initialise the GLFBuffer.
        
        Args:
            path (str): the path to the GLF files.
            sonar_id (int): the sonar id.
            crop_height (int): the initial crop height in pixels.
            img_width (int): the output image width in pixels.
            img_height (int): the output image height in pixels.
            start_date (datetime): the starting datetime.
            end_date (datetime): the ending datetime.
        """
        gfiles = glf_files_avail(path)
        self.dates_glfs = []  # (start, end, glffile)
        self.img_width = img_width
        self.img_height = img_height
        self.crop_height = crop_height

        ts = start_date
        te = end_date

        if ts is None:
            ts = datetime.strptime("2001-01-01", "%Y-%m-%d").replace(tzinfo=pytz.utc)

        if te is None:
            te = datetime.now().replace(tzinfo=pytz.utc)

        for g in tqdm(gfiles, desc="Reading glfs"):
            s, e = glf_times(g)

            if s <= te and e >= ts:
                self.dates_glfs.append((s, e, g))

        self.dates_glfs = sorted(self.dates_glfs, key=lambda x: x[0])

        if start_date is None:
            self.start_date = self.dates_glfs[0][0]
        else:
            self.start_date = start_date

        if end_date is None:
            self.end_date = self.dates_glfs[-1][1]
        else:
            self.end_date = end_date

        print("Buffering between", self.start_date, "and", self.end_date)

        self.current_date = self.start_date
        self.file_pos = 0
        self.sonar_id = sonar_id

        if len(self.dates_glfs) == 0:
            raise ValueError

        self.glf = GLF(self.dates_glfs[0][2]).__enter__()
        self._pop_records()

    def _pop_records(self):
        self.records = []

        for image_rec in self.glf.images:
            if image_rec.header.device_id == self.sonar_id:
                self.records.append(image_rec)

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            # Now go through the GLF and find our image
            # TODO - Does miss the last image at the moment
            for idx, image_rec in enumerate(self.records[:-1]):
                next_rec = self.records[idx + 1]
                image_time = image_rec.db_tx_time
                next_time = next_rec.db_tx_time

                if self.current_date >= image_time and self.current_date < next_time:
                    self.current_date = next_time

                    # Check to make sure current time is still within limit
                    if self.current_date >= self.end_date:
                        raise StopIteration

                    image_data, image_size = self.glf.extract_image(image_rec)
                    image_np = None

                    # Intial crop before resize.
                    if image_size[0] > self.crop_height:
                        image_data = image_data[0:self.crop_height, :]

                    if self.img_height > 0 and self.img_width > 0:
                        if (
                            image_size[0] != self.img_width
                            or image_size[1] != self.img_height
                        ):
                            pil_img = Image.frombuffer(
                                "L", image_size, image_data, "raw", "L", 0, 1
                            )
                            pil_img = pil_img.resize(
                                (self.img_width, self.img_height),
                                resample=Image.NEAREST,
                            )
                            image_np = np.array(pil_img)

                    if image_np is None:
                        image_np = np.frombuffer(image_data, dtype=np.uint8).reshape(
                            (image_size[1], image_size[0])
                        )
                    return (image_np, image_time)

            self.glf.__exit__()
            self.file_pos += 1

            if self.file_pos >= len(self.dates_glfs):
                raise StopIteration

            self.glf = GLF(self.dates_glfs[self.file_pos][2]).__enter__()
            self._pop_records()
            self.current_date = self.records[0].db_tx_time  # First spot in the new file


def images_from_glf(
    glf_files: List,
    start_date: datetime,
    end_date: datetime,
    sonarid: int,
    halfrate: bool,
) -> Generator[np.array]:
    """ Return images from a list of GLFs. This
    function is an iterator in order to save
    memory when running on long ranges.

    Args:
        glf_files (List): the list of GLF files.
        start_date (datetime): the starting datetime.
        end_date (datetime): the ending datetime.
        sonar_id (int): the sonar id.
        halfrate (bool): drop every other frame?
    """
    glf_files.sort()
    #frames = []  # Hold the images we've found
    skip_remaining = False
    start = False
    img_count = 0

    # TODO - assuming come in in time order!
    for idx, glf_file in enumerate(glf_files[:-1]):
        # Parse the filename to get the times quicker
        glfname = os.path.basename(glf_file)
        file_time = datetime.strptime(glfname, "log_%Y-%m-%d-%H%M%S.glf").astimezone(
            tz=pytz.timezone("Europe/London")
        )

        nglfname = os.path.basename(glf_files[idx + 1])
        nfile_time = datetime.strptime(nglfname, "log_%Y-%m-%d-%H%M%S.glf").astimezone(
            tz=pytz.timezone("Europe/London")
        )

        if start_date >= file_time and start_date < nfile_time:
            start = True

        if start:
            with GLF(glf_file) as gf:
                for image_rec in gf.images:
                    image_time = image_rec.db_tx_time
                    if image_time > end_date:
                        # Already read past our time
                        skip_remaining = True
                        break

                    if (
                        image_time >= start_date
                        and image_time < end_date
                        and image_rec.header.device_id == sonarid
                    ):
                        # Potentially skip every other frame (Magallannes)
                        img_count += 1

                        if halfrate and img_count % 2 == 0:
                            continue

                        image_data, image_size = gf.extract_image(image_rec)
                        image_np = np.frombuffer(image_data, dtype=np.uint8).reshape(
                            (image_size[1], image_size[0])
                        )

                        yield image_np

                        #frames.append(image_np)

                if skip_remaining:
                    break

    #return frames
    return


def get_glf_time_np(
    glf_path: str,
    start_date: datetime,
    end_date: datetime,
    img_size: Tuple[int, int],
    sonar_id: int,
    crop_height: int,
    halfrate=False,
    cthulhu=False,
) -> Generator[np.array]:
    """ Get frames between particular times. This function is 
    an iterator and yields a single frame, in order to keep
    memory usage down on big time ranges.
    
    Args:
        glf_path (str): a path to a GLF file
        start_date (datetime): the starting datetime.
        end_date (datetime): the ending datetime.
        img_size (Tuple[int, int]): The image size (width then height in pixels).
        sonar_id (int): the sonar id.
        crop_height (int): the height the original image was cropped to.
        halfrate (bool): drop every other frame?
        cthulhu (bool): summon the unspeakable!
    """

    # Find the GLFs and read through till we get to the times we want.
    full_paths = []
    c = None

    if cthulhu:
        # Add lil Cthulthu to all the frames
        c = np.load("cthulhu_fan.npz")["x"]

    for root, _, files in os.walk(glf_path, topdown=False):
        for name in files:
            _, file_extension = os.path.splitext(name)
            if file_extension.lower() == ".glf":
                full_paths.append(os.path.join(root, name))

    print("Found", str(len(full_paths)), "glfs.")

    for img in images_from_glf(full_paths, start_date, end_date, sonar_id, halfrate):
  
        final_img = img[0:crop_height, :]
        final_img = resize(final_img, (img_size[1], img_size[0]), preserve_range=True)

        if cthulhu:
            # Add lil Cthulthu to all the frames
            c = np.load("cthulhu_fan.npz")["x"]
            c = resize(c, final_img.shape, preserve_range=True)
            final_img = np.clip(final_img + c, 0, 255)

        final_img = final_img.astype(float) / 255.0
        assert np.max(final_img) <= 1.0
        yield final_img

    return
