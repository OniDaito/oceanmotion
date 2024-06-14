"""Tests on the blob bounding box."""

import math
import numpy as np
from eval.detection import bbs_in_image

def test_bbs():
    pred_single = np.zeros((512, 1024))
    pred_single[10:20, 100:200] = 1
    pred_single[100:150, 500:600] = 1

    bbs = bbs_in_image(pred_single)

    box_0 = bbs[0]
    assert(math.fabs(box_0.x_min - 100) < 10)
    assert(math.fabs(box_0.x_max - 200) < 10)

    assert(len(bbs) == 2)
