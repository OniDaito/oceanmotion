""" Test our GLF Buffer. """
import time
import os
import pytz
from PIL import Image
from datetime import datetime
from eval.glf import GLFBuffer


def test_glf():
    testdata_dir = os.environ.get("SEALHITS_TESTDATA_DIR", default=".")
    gbuff = GLFBuffer(testdata_dir, 854, 1632, -1, -1, None, None)
    print("GLFBuffer Start / End datetimes:", gbuff.start_date, gbuff.end_date)
    print("GLFBuffer num GLFS:", len(gbuff.dates_glfs))
    img0, date0 = gbuff.__next__()
    img1, date1 = gbuff.__next__()

    pil_img0 = Image.fromarray(img0, "L")
    pil_img0.save("glf_test0.png")

    pil_img1 = Image.fromarray(img1, "L")
    pil_img1.save("glf_test1.png")

    assert(date0 != date1)
    assert(not (img0 == img1).all())

    # Should test a GLF swap
    t0 = time.time()
    
    for i in range(60 * 6 * 4):
        gbuff.__next__()
    
    t1 = time.time()
    print("Buffered time:", t1 - t0)
    print("Image datetimes:", date0, date1)

    sd = datetime.strptime("2023-03-24", "%Y-%m-%d").astimezone(pytz.UTC)
    ed = datetime.strptime("2023-03-25", "%Y-%m-%d").astimezone(pytz.UTC)

    gbuff = GLFBuffer(testdata_dir, 854, 1632, sd, ed)
    print("GLFBuffer Start / End datetimes:", gbuff.start_date, gbuff.end_date)
    print("GLFBuffer num GLFS:", len(gbuff.dates_glfs))
    img0, date0 = gbuff.__next__()
    img1, date1 = gbuff.__next__()

    pil_img0 = Image.fromarray(img0, "L")
    pil_img0.save("glf_test0.png")

    pil_img1 = Image.fromarray(img1, "L")
    pil_img1.save("glf_test1.png")

    assert(date0 != date1)
    assert(not (img0 == img1).all())

