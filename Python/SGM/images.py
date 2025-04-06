import numpy as np
import cv2

# This file contained more functionality in other versions of the project, but I've stripped it down to upload it into the generic portfolio repo.


def post_process(disp_im):
    # Filtered with medianBlur to clean it up a bit
    # https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html
    temp = np.copy(disp_im)
    temp = temp.astype(np.uint8)
    kernel = 3
    post = cv2.medianBlur(temp, kernel)
    return post
