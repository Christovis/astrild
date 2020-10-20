import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.ndimage.filters import convolve

from astropy import units as un
from lenstools import ConvergenceMap


class Filters:
    def gaussian_filter(
        img: np.ndarray, theta: float, theta_i: float,
    ) -> np.ndarray:
        """
        Gaussian low-pass filter

        Args:
            img: partial sky-map
            theta: edge-lenght of field-of-view [arcmin]
            theta_i: smoothing kernel width [arcmin]
        """
        img = ConvergenceMap(data=img, angle=theta*un.deg)
        img = img.smooth(scale_angle=theta_i * un.arcmin, kind="gaussian",)
        return img.data

    def gaussian_low_pass_filter(img, kernel_width):
        """
        Gaussian low-pass filter
        
        Args:
            img: partial sky-map
            theta: edge-lenght of field-of-view [arcmin]
            theta_i: smoothing kernel width [arcmin]
        """
        lowpass = ndimage.gaussian_filter(
            img,
            sigma=kernel_width,
            order=0,
            output=np.float64,
            mode='nearest',
        )
        return lowpass

    def gaussian_high_pass_filter(img, kernel_width):
        """
        Gaussian high-pass filter
        
        Args:
            img: partial sky-map
            theta: edge-lenght of field-of-view [arcmin]
            theta_i: smoothing kernel width [arcmin]
        """
        lowpass = ndimage.gaussian_filter(img, kernel_width)
        img -= lowpass
        return img

    def gaussian_third_derivative_filter(
        img: np.ndarray, sigma: float, direction: int,
    ) -> np.ndarray:
        """
        Omni-directional third derivative gaussian kernel (also called DGD3 filter),
        useful for extracting dipole signal, based on
        DOI: 10.3847/2041-8213/ab0bfe
        arXiv: 1812.04241

        Args:
            img: partial sky-map
            sigma: ideally it should be the width of halo, R200. units are [pix]
        """
        gauss_1 = ndimage.gaussian_filter(
            img, sigma*0.5, order=3*direction, output=np.float64, mode='nearest'
        )
        gauss_2 = ndimage.gaussian_filter(
            img, sigma*1.0, order=3*direction, output=np.float64, mode='nearest'
        )
        gauss_3 = ndimage.gaussian_filter(
            img, sigma*2.0, order=3*direction, output=np.float64, mode='nearest'
        )
        return gauss_1 - gauss_2 + gauss_3

    def gaussian_compensated_filter(
        img: np.ndarray, theta: float, theta_i:float, theta_o: float,
    ) -> np.ndarray:
        """
        Compensted gaussian filter as define in arxiv:1907.06657 (Eq. 16)

        Args:
            img: partial sky-map
            theta: edge length in degrees of field-of-view
            theta_i: Inner radius of CG-filter
            theta_o: Outer radius of CG-filter
        """
        def U(theta: np.ndarray, theta_i: float, theta_o: float) -> np.ndarray:
            """ Filter Function """
            x = theta / theta_i
            x_o = theta_o / theta_i
            gt = (((np.pi * theta_i**2.)**(-1.)) * (np.exp(-1. * x**2.))) - \
                (((np.pi * theta_o**2.)**(-1.)) * (1. - np.exp(-1. * x_o**2.)))
            gt[theta_o < theta] = 0
            return gt

        # pixel width in degrees
        pw = theta / img.shape[0]
        # convert filter specifications from arcmin to pixel-width units
        theta_i = (theta_i / 60. ) / pw
        theta_o = (theta_o / 60. ) / pw
        theta_o_int = np.ceil(theta_o).astype('int') #round up theta_o to be safe

        #next we need to define a grid to map the filter onto, which we will use for the convolution
        a = b = 0
        y, x = np.ogrid[a-theta_o_int:a+theta_o_int, b-theta_o_int:b+theta_o_int]
        weights = np.sqrt(x*x + y*y)

        # pass the grid into the filter function
        filt_trunc_gauss = U(weights, theta_i, theta_o)
       
        #convolve the filter grid and the data grid
        img_filtered = convolve(img, filt_trunc_gauss)
        
        return img_filtered



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


    def cthf_map(rad_obj, obj_posx, obj_posy, mapp, alpha):
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


    def cthf_profile(profiles, extend, Nbins):
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
