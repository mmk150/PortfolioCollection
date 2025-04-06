import numpy as np
from metrics import ssd

from numba import jit

# https://numba.pydata.org/numba-doc/dev/user/5minguide.html

import scipy.ndimage


class NaiveApproach:
    def __init__(self, im1, im2, window_size, max_delta):
        self.im1 = im1
        self.im2 = im2
        self.size = im1.size

        self.window_size = window_size
        self.max_delta = max_delta

        self.dispL2R = None
        self.dispR2L = None
        self.av = None

        self.accuracy = dict()

    def getImages(self):
        im1 = np.copy(self.im1)
        im2 = np.copy(self.im2)
        return im1, im2

    def getAccuracy(self, metric="ssd"):
        acc_dict = self.accuracy
        acc = acc_dict[metric]
        return acc

    def getDisparity(self):
        left_to_right = self.dispL2R
        right_to_left = self.dispR2L
        average = self.av
        return left_to_right, right_to_left, average

    def run(self):
        self.compute_disparity()

    def compute_disparity(self):
        window_size = self.window_size
        max_delta = self.max_delta
        im1, im2 = self.getImages()
        kernel = np.ones(shape=(window_size, window_size))
        dispL2R, dispR2L = compute_disparr_conv(im1, im2, max_delta, window_size, kernel)

        self.dispL2R = dispL2R
        self.dispR2L = dispR2L

    def compute_accuracy(self, ground, metric="ssd"):
        dispL2R, dispR2L, av = self.getDisparity()
        im1, im2 = self.getImages()

        if metric == "ssd":
            acc = ssd(ground, dispL2R)
            acc = acc / (im1.size)
        else:
            acc = -1

        acc_dict = self.accuracy
        acc_dict.update({metric: acc})


def compute_disparr_conv(im1, im2, max_delta, window_size, kernel):
    # with reference to: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.convolve.html
    padding = int(window_size / 2)
    padded1 = np.pad(im1, padding, mode="reflect")
    padded2 = np.pad(im2, padding, mode="reflect")

    disp_arrL = np.zeros(shape=padded1.shape).astype(np.float32)
    temp_minL = np.zeros(shape=padded1.shape).astype(np.float32)
    temp_minL += 9999999

    disp_arrR = np.copy(disp_arrL).astype(np.float32)
    temp_minR = np.copy(temp_minL).astype(np.float32)

    for dx in range(max_delta):
        ssd_calcsL = shift_ssd(dx, padded1, padded2)
        ssd_calcsL = scipy.ndimage.convolve(ssd_calcsL, kernel, mode="reflect")
        disp_arrL, temp_minL = disp_update(dx, ssd_calcsL, disp_arrL, temp_minL)

        ssd_calcsR = shift_ssd(dx, padded2, padded1)
        ssd_calcsR = scipy.ndimage.convolve(ssd_calcsR, kernel, mode="reflect")
        disp_arrR, temp_minR = disp_update(dx, ssd_calcsR, disp_arrR, temp_minR)
    disp_arrL = padding_strip(disp_arrL, padding)
    disp_arrR = padding_strip(disp_arrR, padding)
    return disp_arrL, disp_arrR


def padding_strip(arr, padding):
    dim1, dim2 = arr.shape
    stripped_arr = arr[padding : dim1 - padding, padding : dim2 - padding]
    return stripped_arr


@jit(nopython=True)
def shift_ssd(dx, arr1, arr2):
    # calculates the left shifted and right shifted versions of arr2 and arr1 respectively
    # then goes through the arraywise ssd calculation
    arr1 = np.copy(arr1)
    arr2 = np.copy(arr2)
    left_shift = arr2[:, : arr2.shape[1] - dx]
    right_shift = arr1[:, dx:]
    diff = right_shift - left_shift
    diff = diff * diff
    return diff


@jit(nopython=True)
def disp_update(dx, ssd_calcs, disp_arr, minny):
    # this updates the disparity values and minimums based on ssd_calcs, the minimums, and the dx val
    # used np.where since numba is allergic to indexing via masked arrays apparently
    temp1 = np.copy(minny)
    temp2 = np.copy(disp_arr)
    minny[:, dx:] = np.where(ssd_calcs < temp1[:, dx:], ssd_calcs, temp1[:, dx:])
    disp_arr[:, dx:] = np.where(ssd_calcs < temp1[:, dx:], dx, temp2[:, dx:])
    return disp_arr, minny
