import numpy as np
import pandas as pd
from scipy import ndimage


def hpf_gauss(img, kernel_width):
    """Gaussian high-pass filter"""
    lowpass = ndimage.gaussian_filter(img, kernel_width)
    img -= lowpass
    return img


# class CTHF(dimensions=1):
#    """
#    Compensated top-hat filter
#    """
#
#    def __init__(self, dimensions):
#        """
#        Args:
#            dimensions: int
#                Apply on profiles (1D) or maps (2D)
#        """
#        self.dim = dimensions
#
#    def apply():


def CTHF_map(rad_obj, obj_posx, obj_posy, mapp, alpha):
    """
    Compensated top-hat filter on 2D map.
    Remove long wavelength from ISW signal from flat sky maps.
    alpha: float
        best values are 0.6-0.7 (DOI: 10.1088/0004-637X/786/2/110)
        
    """
    rad_filter = alpha * rad_obj
    extend = np.sqrt(2)
    rad_filter_sqrt2 = np.ceil(extend * rad_filter).astype(int)
    print("rad_filter", rad_filter, rad_filter_sqrt2)
    # annulus thickness normalised against ith void radius
    delta_eta = extend / args["Nbins"]

    # distance of every pixel to centre
    pix_x = pix_y = np.arange(-rad_filter_sqrt2, rad_filter_sqrt2)
    pix_xx, pix_yy = np.meshgrid(pix_x, pix_y)
    pix_dist = np.sqrt(pix_xx ** 2 + pix_yy ** 2) / rad_filter

    # eta gives the annulus to which the pixel belongs
    eta = (pix_dist / delta_eta).astype(int)
    pix_xx = pix_xx[eta < args["Nbins"]]
    pix_yy = pix_yy[eta < args["Nbins"]]
    pix_dist = pix_dist[eta < args["Nbins"]]
    eta = eta[eta < args["Nbins"]]

    annulus_count = [list(eta).count(ee) for ee in np.unique(eta)]
    annulus_buffer = list(np.zeros(args["Nbins"] - len(np.unique(eta))))
    annulus_count = np.asarray(annulus_count + annulus_buffer)
    annulus_value = np.zeros(args["Nbins"])
    for pp in range(len(eta)):
        annulus_value[eta[pp]] += mapp[
            obj_posy + pix_xx[pp], obj_posx + pix_yy[pp],
        ]

    # Mean value in 0 -> rad_filter
    white_hat = np.mean(annulus_value[: np.ceil(1 / delta_eta).astype(int)])

    # Mean value in rad_filter -> sqrt(2)*rad_filter
    black_hat = np.mean(annulus_value[np.ceil(1 / delta_eta).astype(int) :])

    print("CFT", white_hat, black_hat)

    return white_hat - black_hat


def CTHF_profile(profiles, extend, Nbins):
    """
    Compensated top-hat filter applied to profiles
    alpha: float
        best values are 0.6-0.7 (DOI: 10.1088/0004-637X/786/2/110)
    """
    assert extend > np.sqrt(2)

    delta_eta = extend / Nbins

    # Mean value in 0 -> rad_filter
    middle = np.ceil(1 / delta_eta).astype(int)
    white_hat = np.mean(profiles[:middle])

    # Mean value in rad_filter -> sqrt(2)*rad_filter
    maxi = np.ceil(np.sqrt(2) / delta_eta).astype(int)
    black_hat = np.mean(profiles[middle:maxi])

    # filtered_profile = np.convolve([1, 2, 3], [0, 1, 0.5])
    return white_hat - black_hat, white_hat, black_hat, middle, maxi
