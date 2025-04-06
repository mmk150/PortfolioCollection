import numpy as np
from metrics import census_transform
from numba import jit
from images import post_process


# this whole file coded with reference to the following:
# https://www.mathworks.com/help/visionhdl/ug/stereoscopic-disparity.html
# https://engineering.purdue.edu/kak/Tutorials/SemiGlobalMatching.pdf


class SGBMApproach:
    def __init__(self, im1, im2, max_delta, penalty1=4, penalty2=60):
        self.im1 = im1
        self.im2 = im2

        self.max_delta = max_delta

        self.penalty1 = penalty1
        self.penalty2 = penalty2

        self.energyL2R = None
        self.energyR2L = None

        self.aggL2R = None
        self.aggR2L = None

        self.rawdispL2R = None
        self.rawdispR2L = None
        self.final_dispL2R = None
        self.final_dispR2L = None

    def getImages(self):
        im1 = np.copy(self.im1)
        im2 = np.copy(self.im2)
        return im1, im2

    def getInitial(self):
        energyL2R = self.energyL2R
        energyR2L = self.energyR2L
        return energyL2R, energyR2L

    def getAgg(self):
        aggL2R = self.aggL2R
        aggR2L = self.aggR2L
        return aggL2R, aggR2L

    def getRawDisp(self):
        rawdispL2R = self.rawdispL2R
        rawdispR2L = self.rawdispR2L
        return rawdispL2R, rawdispR2L

    def getFinalDisp(self):
        finalL2R = self.final_dispL2R
        finalR2L = self.final_dispR2L
        return finalL2R, finalR2L

    def run(self, both=False):
        self.computeEn(boolflag=both)
        self.aggregation(boolflag=both)
        self.computeDisparity(boolflag=both)
        self.postProcess(boolflag=both)

    def computeEn(self, boolflag, ksize=7):
        # birchfield tomasi not supported, didn't pan out
        # boolflag: if false we skip the right-to-left disparity calcs
        # ksize=kernel of Census transform window which just equated to padding I guess with how I did it
        im1, im2 = self.getImages()
        max_del = self.max_delta
        rows, cols = im1.shape

        padding = ksize // 2
        self.ksize = ksize
        self.padding = padding

        padded1 = np.pad(im1, padding)
        padded2 = np.pad(im2, padding)

        cvals1, cvals2 = census_transform(padded1, padded2, padding, ksize=ksize)

        # dimcheck
        if cvals1.shape != im1.shape:
            rows, cols = im1.shape
            cvals1 = cvals1[padding : rows + padding, padding : cols + padding]
            cvals2 = cvals2[padding : rows + padding, padding : cols + padding]

        energyL, energyR = initialize_energy(cvals1, cvals2, max_del, doBoth=boolflag)

        self.energyL2R = energyL
        self.energyR2L = energyR
        return True

    def aggregation(self, boolflag=False):
        # calls aggregation step
        # boolflag: if false we skip the right-to-left disparity calcs
        energyL2R, energyR2L = self.getInitial()

        penalty1 = self.penalty1
        penalty2 = self.penalty2

        agg_L2R, agg_R2L = aggregation_step(energyL2R, energyR2L, penalty1, penalty2, doBoth=boolflag)

        self.aggL2R = agg_L2R
        self.aggR2L = agg_R2L

    def computeDisparity(self, boolflag=False):
        # argmin along disp axis
        # boolflag: if false we skip the right-to-left disparity calcs
        aggL2R, aggR2L = self.getAgg()
        dispL2R = np.argmin(aggL2R, axis=2)
        self.rawdispL2R = dispL2R
        self.rawdispR2L = aggR2L
        if boolflag:
            dispR2L = np.argmin(aggR2L, axis=2)
            self.rawdispR2L = dispR2L

    def postProcess(self, boolflag=False):
        # boolflag: if false we skip the right-to-left disparity calcs
        dispL2R, dispR2L = self.getRawDisp()
        disp = dispL2R
        disp = post_process(disp)
        self.final_dispL2R = disp
        self.final_dispR2L = dispR2L
        if boolflag:
            dispR = post_process(dispR2L)
            self.final_dispR2L = dispR


def initialize_energy(left_ct_vals, right_ct_vals, max_delta, doBoth=False):
    # with reference to https://numpy.org/doc/stable/reference/generated/numpy.roll.html
    # as in pt1 its more efficient to left and right shift arrays appropriately

    # left_ct_vals: census transform values for left image
    # right_ct_vals: census transform values for right image
    # max_delta: upper bound for disparity
    # doBoth: boolflag for doing L2R and R2L. If false we just do L2R
    rows, cols = left_ct_vals.shape

    left_energy = np.zeros(shape=(rows, cols, max_delta)).astype(np.float32)
    right_energy = np.zeros(shape=(rows, cols, max_delta)).astype(np.float32)

    for dx in range(max_delta):
        # -dx will leftshit, dx will rightshift
        shifted_rcensus = np.roll(right_ct_vals, dx, axis=1)
        res1 = ct_en_per_disp(left_ct_vals, shifted_rcensus)
        res1 = res1.astype(np.float32)
        left_energy[:, :, dx] = res1

        if doBoth:
            shifted_lcensus = np.roll(left_ct_vals, -dx, axis=1)
            res2 = ct_en_per_disp(right_ct_vals, shifted_lcensus)
            res2 = res2.astype(np.float32)
            right_energy[:, :, dx] = res2

    return left_energy, right_energy


@jit(nopython=True)
def ct_en_per_disp(ct_vals, shifted_arr):
    # hamming distance essentially
    # ct_vals: census transform values
    # shifted_arr: dx-shifted array
    rows, cols = ct_vals.shape
    temp1 = np.zeros(shape=(rows, cols))
    for y in range(rows):
        for x in range(cols):
            shift_val = int(shifted_arr[y, x])
            unshifted_val = int(ct_vals[y, x])
            temp1[y, x] += hamm_dist(shift_val, unshifted_val)
    return temp1


@jit(nopython=True)
def hamm_dist(num1, num2):
    ## with inspiration from https://stackoverflow.com/a/77014674
    # looping through this bitwise & will just zero out the numbers significant bits right to left.
    # since "bitwise_xors" is the bitwise XOR of our census value and the array value of interest
    # this leaves us with https://en.wikipedia.org/wiki/Hamming_distance for the census transform pixel to pixel dist
    # num1: int
    # num2: int
    count = 0
    bitwise_xors = num1 ^ num2
    while bitwise_xors != 0:
        bitwise_xors = bitwise_xors & (bitwise_xors - 1)
        count += 1.0
    return count


def aggregation_step(ct_distancesL, ct_distancesR, penalty1, penalty2, doBoth=False):
    # calc lines approaching from various directions and sum them all
    # we're only going to do two of the main NSEW directions since this code is already slow when written in Python
    # Converting this to handle everything properly with NumPy would likely make it a bit faster
    # But these operations are inherently going to be somewhat expensive, algorithmic complexity wise

    # Done with reference to Purdue SGM lecture pdf by Avinash Kak:
    # https://engineering.purdue.edu/kak/Tutorials/SemiGlobalMatching.pdf
    # And mathworks: https://www.mathworks.com/help/visionhdl/ug/stereoscopic-disparity.html

    # ct_distancesL: initialized from census transform step
    # ct_distancesR: initialized from census transform step
    # penalty1: small error penalty
    # penalty2: large error penalty
    # doBoth: if false we skip the right-to-left disparity calcs
    dir_list = get_compass_directions(ct_distancesL.shape)

    # ns
    ns_lines1 = dir_list[0]
    ns_lines2 = dir_list[1]
    # ew
    ew_lines1 = dir_list[2]
    ew_lines2 = dir_list[3]

    n = lines_calc(ns_lines1, ct_distancesL, penalty1, penalty2, typed="N")
    s = lines_calc(ns_lines2, ct_distancesL, penalty1, penalty2, typed="S")
    e = lines_calc(ew_lines1, ct_distancesL, penalty1, penalty2, typed="E")
    w = lines_calc(ew_lines2, ct_distancesL, penalty1, penalty2, typed="W")

    everythingL = np.zeros(shape=ct_distancesL.shape).astype(np.float32)
    everythingL += n
    everythingL += s
    everythingL += e
    everythingL += w

    everythingR = np.zeros(shape=ct_distancesL.shape).astype(np.float32)
    if doBoth:
        n = lines_calc(ns_lines1, ct_distancesR, penalty1, penalty2, typed="N")
        s = lines_calc(ns_lines2, ct_distancesR, penalty1, penalty2, typed="S")
        e = lines_calc(ew_lines1, ct_distancesR, penalty1, penalty2, typed="E")
        w = lines_calc(ew_lines2, ct_distancesR, penalty1, penalty2, typed="W")

        everythingR += n
        everythingR += s
        everythingR += e
        everythingR += w
    return everythingL, everythingR


@jit(nopython=True)
def lines_calc(lines_arr, ct_distances, penalty1, penalty2, typed):
    # One must remember that traversing "towards North" or "West" is the opposite orientation of the image
    # and associated arrays...

    # Done with reference to Purdue SGM lecture pdf by Avinash Kak:
    # https://engineering.purdue.edu/kak/Tutorials/SemiGlobalMatching.pdf
    # And mathworks: https://www.mathworks.com/help/visionhdl/ug/stereoscopic-disparity.html

    # arr:empty array (image dimensions X max_delta) size of lines for a given type, will be filled and returned
    # ct_distances: census-transform hamming distances previously computed (image dimensions x max_delta)
    # penalty1: small error penalty
    # penalty2: large error penalty
    # typed: type of line/direction we're traveling to

    rows, cols, max_del = ct_distances.shape

    if typed == "N":
        for x in range(cols):
            vecs = ct_distances[:, x, :]
            lines = []
            # South to North is reverse orientation
            for y in range(vecs.shape[0] - 1, -1, -1):
                lines.append(vecs[y, :])
            res_vals = optimize(lines, penalty1, penalty2, typed)
            lines_arr[:, x, :] += res_vals
    elif typed == "S":
        for x in range(cols):
            vecs = ct_distances[:, x, :]
            lines = []
            for y in range(vecs.shape[0]):
                lines.append(vecs[y, :])
            res_vals = optimize(lines, penalty1, penalty2, typed)
            lines_arr[:, x, :] += res_vals
    elif typed == "W":
        for y in range(rows):
            vecs = ct_distances[y, :, :]
            lines = []
            # East to West is reverse orientation
            for x in range(vecs.shape[0] - 1, -1, -1):
                lines.append(vecs[x, :])
            res_vals = optimize(lines, penalty1, penalty2, typed)
            lines_arr[y, :, :] += res_vals
    elif typed == "E":
        for y in range(rows):
            lines = []
            vecs = ct_distances[y, :, :]
            for x in range(vecs.shape[0]):
                lines.append(vecs[x, :])
            res_vals = optimize(lines, penalty1, penalty2, typed)
            lines_arr[y, :, :] += res_vals
    lines_arr = lines_arr.astype(np.float32)
    return lines_arr


@jit(nopython=True)
def get_compass_directions(targetshape):
    # targetshape: shape these arrays need to be in
    n = np.zeros(shape=targetshape).astype(np.float32)
    s = np.zeros(shape=targetshape).astype(np.float32)
    e = np.zeros(shape=targetshape).astype(np.float32)
    w = np.zeros(shape=targetshape).astype(np.float32)
    compass_list = [n, s, e, w]
    return compass_list


@jit(nopython=True)
def optimize(lines, penalty1, penalty2, typed):
    # this calculates the shortest possible cost of traversing along a given cardinal direction
    # (i.e. North to South, East to West) for a given input curves and set of penalties
    # Done with reference to the actual paper, and the Purdue SGM lecture pdf by Avinash Kak:
    # https://engineering.purdue.edu/kak/Tutorials/SemiGlobalMatching.pdf
    # And mathworks: https://www.mathworks.com/help/visionhdl/ug/stereoscopic-disparity.html

    # lines: list of lines
    # penalty1: small error penalty
    # penalty2: large error penalty
    # typed: type of lines

    if typed == "N" or typed == "S" or typed == "W" or typed == "E":
        linelength = len(lines)
        max_del = len(lines[0])
        temp = np.zeros(shape=(linelength, max_del)).astype(np.float32)
        i = 0

        # these are all lines of the same length
        for li in lines:
            temp[i, :] += li
            i += 1

        optimal_vals = np.zeros(shape=(linelength, max_del)).astype(np.float32)
        optimal_vals[0, :] += temp[0, :]

        steps = 1

        while steps < linelength:
            last = steps - 1
            curr = steps

            curr_step = temp[curr, :]
            prev_step = optimal_vals[last, :]

            minimum_delta = apply_flatrate_and_minimize(curr_step, prev_step, penalty1, penalty2)

            optimal_vals[curr, :] += minimum_delta
            steps += 1
        optimal_vals = optimal_vals.astype(np.float32)
        if typed == "W" or typed == "N":
            # flip 180 degrees to match original orientation of image
            optimal_vals = optimal_vals[::-1, :]
        return optimal_vals


@jit(nopython=True)
def apply_flatrate_and_minimize(curr_step, prev_step, penalty1, penalty2):
    # this handles all of the actual optimizing over the disparity axis
    # first we apply penalties appropriately
    # add in the values from the previous step
    # then we compute the min over the disparity axis
    # add value of current step in and return

    # curr_step: step we're on at 'time t' (as an analogy)
    # prev_step: step we were on at 'time t-1' (as an analogy)
    # penalty1: cost for small disparity delta
    # penalty2: cost for large disparity delta
    z = curr_step.shape[0]
    minny = np.zeros(shape=(z, z))
    for disp_i in range(z):
        for disp_j in range(z):
            if abs(disp_i - disp_j) == 0:
                flatrate = 0
            if abs(disp_i - disp_j) == 1:
                flatrate = penalty1
            if abs(disp_i - disp_j) > 1:
                flatrate = penalty2
            minny[disp_i, disp_j] += flatrate + prev_step[disp_i]
    res = np.zeros(shape=z)
    for x in range(z):
        abs_min_x = 999999
        for y in range(z):
            if minny[y, x] <= abs_min_x:
                abs_min_x = minny[y, x]
        res[x] = abs_min_x
    res = res + curr_step
    return res
