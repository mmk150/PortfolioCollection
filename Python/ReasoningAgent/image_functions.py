from PIL import Image
import numpy as np


def convert2np(PIL_im):
    # Converts Pillow image to Numpy
    # With reference to:
    # https://pillow.readthedocs.io/en/stable/reference/Image.html#
    # https://numpy.org/doc/stable/reference/generated/numpy.reshape.html
    if PIL_im is False:
        return False
    PIL_im = PIL_im.convert("L")
    width, height = PIL_im.size
    flattened = PIL_im.getdata()
    arr = np.asarray(flattened).astype(np.float32)
    arr = np.reshape(arr, newshape=(height, width))
    return arr
