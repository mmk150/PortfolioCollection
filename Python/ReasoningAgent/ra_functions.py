import numpy as np


def ssd_metric(im1, im2):
    # metric for sum of squared differences measure
    im1 = np.copy(im1).astype(np.float32)
    im2 = np.copy(im2).astype(np.float32)
    return np.sum((im1 - im2) ** 2)


def mse_metric(im1, im2):
    # metric for mse(im1,im2)
    N, M = im1.shape[0:2]
    ssd = ssd_metric(im1, im2) / (N * M)
    return ssd


def func_id(im1):
    # the identity function
    return np.copy(im1).astype(np.float32)


def func_rotn(im1, n):
    # Rotates image by 90 degrees, n times
    # https://numpy.org/doc/stable/reference/generated/numpy.rot90.html
    res = np.copy(im1).astype(np.float32)
    res = np.rot90(res, n, axes=(1, 0))
    return res


def func_flipv(im1):
    # flips image vertically
    # https://numpy.org/doc/stable/reference/generated/numpy.flip.html
    res = np.copy(im1).astype(np.float32)
    res = np.flip(res, axis=0)
    return res


def func_fliph(im1):
    # flips image horizontally
    # https://numpy.org/doc/stable/reference/generated/numpy.flip.html
    res = np.copy(im1).astype(np.float32)
    res = np.flip(res, axis=1)
    return res


def naive_diff(im1, im2):
    # naively computes im2-im1 as a last resort function for tricky questions
    diff = np.copy(im2) - np.copy(im1)
    return diff


def get_ops():
    # https://numpy.org/doc/stable/reference/routines.bitwise.html
    operations = [
        ("or", np.bitwise_or),
        ("xor", np.bitwise_xor),
        ("and", np.bitwise_and),
        ("not", np.bitwise_not),
        ("add2", add_mod2),
    ]

    opdict = {
        "xor": np.bitwise_xor,
        "and": np.bitwise_and,
        "or": np.bitwise_or,
        "not": np.bitwise_not,
        "add2": add_mod2,
    }
    return operations, opdict


def cvt2bin(arr):
    # converts an image from 256 grayscale to a binary 0-1 array with array masking
    arr = np.copy(arr).astype(np.float32)
    wmask = arr >= 128
    bmask = arr < 128
    arr[wmask] = 0
    arr[bmask] = 1
    arr = arr.astype(np.int8)
    return arr


def add_mod2(arr1, arr2):
    # with reference to https://numpy.org/doc/stable/reference/generated/numpy.mod.html
    res = arr1 + arr2
    res = np.mod(res, 2)
    return res


def int_pixel_ratio(im1, im2):
    # https://numpy.org/doc/stable/reference/generated/numpy.where.html
    # calculates the "intersection pixel ratio", a metric taken from Joyner et al
    im1 = np.copy(im1)
    im2 = np.copy(im2)
    wpixels = im1 > 128
    bpixels = im1 <= 128
    im1[wpixels] = 0
    im1[bpixels] = 1

    s1 = np.sum(bpixels)
    s1 = s1 / im1.size

    wpixels = im2 > 128
    bpixels = im2 <= 128
    im2[wpixels] = 0
    im2[bpixels] = 1
    s2 = np.sum(bpixels)
    s2 = s2 / im2.size

    s1 = max(0.00001, s1)
    s2 = max(0.00001, s2)

    intersection = np.where(im1 == im2, 1, 0)

    int_blank = intersection > 128
    int_black = intersection <= 128

    s3 = np.sum(int_black) / im1.size

    diff = 1 / s1 - 1 / s2

    res = s3 * diff
    return res


def calc_wpr(im):
    # reference to numpy docs for masked arrays https://numpy.org/doc/stable/reference/maskedarray.generic.html
    # Calculates the proportionate amount of white pixels
    temp = np.copy(im).astype(np.float32)
    mask_white = temp > 20
    mask_black = temp < 20
    temp[mask_white] = 1
    temp[mask_black] = 0

    res = np.sum(temp)
    res = res / temp.size
    return res


def dark_pixel_ratio(im_0, im_1):
    # calculates the proportionate amount of dark pixels
    wmask = im_0 > 128
    bmask = im_0 <= 128

    wcount = np.sum(wmask)
    bcount = np.sum(bmask)

    im_left_val = bcount / (wcount + bcount)

    wmask = im_1 > 128
    bmask = im_1 <= 128

    wcount = np.sum(wmask)
    bcount = np.sum(bmask)

    im_right_val = bcount / (wcount + bcount)

    res = im_right_val - im_left_val

    return res


def equation_solver(a, b, c):
    # an attempt to solve bin_op(a,b)=c
    operations, opdict = get_ops()
    res = []
    for op in operations:
        if np.sum(np.abs(c - op[1](a, b))) < 800:
            res.append((0, 1, 2, op[0], np.sum(np.abs(c - op[1](a, b)))))
        elif np.sum(np.abs(c - op[1](b, a))) < 800:
            res.append((1, 0, 2, op[0], np.sum(np.abs(c - op[1](b, a)))))
    if len(res) == 0:
        res = False
    else:
        res.sort(key=lambda x: x[4])
        return res[0][0:4]
    return res
