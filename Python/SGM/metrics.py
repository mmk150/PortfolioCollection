import numpy as np
from numba import jit


def ssd(im1, im2):
    # generic ssd defined as \sum_{i} (x_i-y_i)**2, just mean_squared_err with no normalization essentially
    # NB: This only gets used in mean_squared_err, the SSD calcs done elsewhere in the project are most often computed through convolutions
    # calculated arraywise with numpy
    return np.sum((im1 - im2) ** 2)


def mean_squared_err(arr1, arr2):
    # https://en.wikipedia.org/wiki/Mean_squared_error
    # mean square error (ssd/NM)
    ssd_res = ssd(arr1, arr2)
    N = arr1.shape[0]
    M = arr1.shape[1]
    return ssd_res / (N * M)


def birchfield_tomasi(im1, im2, normed=True, sumup=True):
    # definitions taken from https://en.wikipedia.org/wiki/Birchfield%E2%80%93Tomasi_dissimilarity
    # https://users.cs.duke.edu/~tomasi/papers/tomasi/tomasiIjcv99.pdf
    im1 = np.copy(im1).astype(np.float64)
    im2 = np.copy(im2).astype(np.float64)

    expanded1, expanded2 = interpolate(im1, im2)
    rows, cols = im1.shape

    dr = np.zeros(shape=im1.shape)
    dl = np.zeros(shape=im1.shape)
    dist = np.zeros(shape=im1.shape)
    temp1 = np.zeros(shape=(rows, 2))
    temp1[:, 0] = im1[:, 0]
    temp1[:, 1] = im1[:, 1]

    temp2 = np.zeros(shape=(rows, 2))
    temp2[:, 0] = im1[:, 0]
    temp2[:, 1] = im1[:, 1]

    # dx=0 case
    res1 = np.abs(expanded2[:, 0:2] - temp1)
    dr[:, 0] = np.min(res1, axis=1)
    res2 = np.abs(expanded1[:, 0:2] - temp2)
    dl[:, 0] = np.min(res2, axis=1)

    for dx in range(1, cols):
        res1 = np.abs(expanded2[:, 2 * dx - 1 : 2 * (dx + 1)])
        iminr = np.min(res1, axis=1)
        imaxr = np.max(res1, axis=1)

        res2 = np.abs(expanded1[:, 2 * dx - 1 : 2 * (dx + 1)])

        diff1 = im1[:, dx] - imaxr
        diff2 = iminr - im1[:, dx]
        dist[:, dx] += np.where(diff1 > diff2, diff1, diff2)
    dist = np.where(dist > 0, dist, 0)
    if normed:
        dist = dist / im1.size
    if sumup:
        dist = np.sum(dist)
    return dist


@jit(nopython=True)
def interpolate(im1, im2):
    # calculates the interpolateolated image value pixelwise
    # https://en.wikipedia.org/wiki/Birchfield%E2%80%93Tomasi_dissimilarity
    # with reference to the expand function from a previous project
    rows, cols = im1.shape
    expanded1 = np.zeros(shape=(rows, cols * 2))
    expanded2 = np.zeros(shape=(rows, cols * 2))
    bigrows, bigcols = expanded1.shape

    for i in range(0, bigrows):
        for j in range(0, bigcols - 1):
            if j % 2 == 0:
                expanded1[i, j] = im1[i, j // 2]
                expanded2[i, j] = im2[i, j // 2]
            else:
                expanded1[i, j] = im1[i, (j - 1) // 2] + im1[i, (j + 1) // 2]
                expanded1[i, j] /= 2
                expanded2[i, j] = im2[i, (j - 1) // 2] + im2[i, (j + 1) // 2]
                expanded2[i, j] /= 2
    return expanded1, expanded2


@jit(nopython=True)
def census_transform(padded1, padded2, padding, ksize=3):
    # with reference to :
    # https://realpython.com/python-bitwise-operators/
    # https://en.wikipedia.org/wiki/Census_transform
    # Implementation of CT in pytorch : https://github.com/mlaves/census-transform-pytorch/blob/master/census_transform.py
    rows, cols = padded1.shape

    res1 = np.zeros(padded1.shape)
    res2 = np.zeros(padded1.shape)

    for y in range(padding, rows):
        for x in range(padding, cols):
            pixel_1 = padded1[y, x]
            pixel_2 = padded2[y, x]

            window1 = padded1[y - padding : y + padding + 1, x - padding : x + padding + 1]
            window2 = padded2[y - padding : y + padding + 1, x - padding : x + padding + 1]

            window1 = window1.flatten()
            window2 = window2.flatten()

            bitsleft = 0
            bitsright = 0

            for i in range(window1.shape[0]):
                val_left = window1[i]
                val_right = window2[i]
                bitsleft <<= 1
                bitsright <<= 1

                if val_left < pixel_1:
                    bitsleft = bitsleft | 1
                else:
                    bitsleft = bitsleft | 0

                if val_right < pixel_2:
                    bitsright = bitsright | 1
                else:
                    bitsright = bitsright | 0
            res1[y, x] += bitsleft
            res2[y, x] += bitsright
    return res1, res2
