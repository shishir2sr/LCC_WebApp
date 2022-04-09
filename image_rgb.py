import cv2
import glob
import pandas as pd
import os
import numpy as np

# select the path
path = '5/5 (42).jpg'


def imgAvgRgb(imag):

    img = imag
    x, y, c = img.shape

    non_zero = len(img[np.nonzero(img)])

    avgr = 0
    avgg = 0
    avgb = 0
    for i in range(0, x):
        for j in range(0, y):
            b, g, r = img[i, j]
            avgr = avgr + r
            avgg = avgg + g
            avgb = avgb + b

    # Here exactly divided by non zero elements. without zere value.
    avgr = avgr / non_zero
    avgg = avgg / non_zero
    avgb = avgb / non_zero
    return avgr, avgg, avgb
