import time

import pandas as pd
import numpy as np
from typing import Union, Optional

from scipy import signal
from scipy import ndimage
from scipy.ndimage.filters import convolve
from sklearn.decomposition import PCA, MiniBatchDictionaryLearning
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.feature_extraction.image import reconstruct_from_patches_2d

from astropy import units as un
from lenstools import ConvergenceMap


class Filters:
    """
    Collection of filters for lensing convergence
    (physical notation: \kappa, code notation: kappa_2) and
    temperature fluctuation (physical notation: \delta T, code notation: isw_rs)
    fields.

    Methods:
        aperture_photometry:
        dictionary_learning:
        pca:
        gaussian:
        gaussian_high_pass:
        gaussian_field:
        gaussian_compensated:
        apodization:
        gaussian_third_derivative:
        tophat_compensated:
        fwhm_to_sigma:
        sigma_to_fwhm:
    """

    def aperture_photometry(
        img: np.ndarray,
        theta: un.quantity.Quantity,
        alpha: un.quantity.Quantity,
    ) -> np.ndarray:
        """
        Take the average value of the pixels belonging to the disk and
        substract from this average the average value of pixels in a
        ring with bounds [alpha, alpha*sqrt(2)]. This is useful for
        kinematic Sunyaev-Zel’dovich effec analyzes when long wave-modes
        are involved, e.g. due to CMB.
        arxiv: 1607.02139, Sec. III.B

        Args:
            img: flat 2D SkyArray-image
            theta: opening angle of image (fov), [some angular distance]
            alpha: aperture, for kSz it can be 1.8 arcmin

        Returns:
        """
        # get distances of pixels in units of pixels
        _npix = len(img)
        x1edge = np.linspace(1, _npix, _npix) - _npix / 2 - 0.5
        x, y = np.meshgrid(x1edge, x1edge)
        pixel_dist = np.sqrt(x ** 2 + y ** 2)
        # convert aperture unit to pixels
        pix_per_deg = _npix / theta.to(un.deg).value
        alpha_pix = np.ceil(alpha.to(un.deg).value * pix_per_deg).astype("int")
        # get pixels in ring
        indx_ring = np.logical_and(
            alpha_pix < pixel_dist, pixel_dist < alpha_pix*np.sqrt(2),
        )
        img -= np.mean(img[indx_ring])
        return img


    def dictionary_learning(
        clean_data: np.ndarray,
        noisy_data: np.ndarray,
        ntiles: int,
        n_components: int = 5,
    ) -> np.ndarray:
        """
        Args:
        """
        from time import time

        t0 = time()
        # extract reference patches from clean data
        npix = clean_data.shape[0]
        patch_npix = int(clean_data.shape[0] / ntiles)
        data = extract_patches_2d(clean_data, (patch_npix, patch_npix))
        dt = time() - t0
        data = data.reshape(data.shape[0], -1)
        data -= np.mean(data, axis=0)
        data /= np.std(data, axis=0)
        # learn the dictionary from reference patches
        dico = MiniBatchDictionaryLearning(
            n_components=100,
            alpha=0.1,
            n_iter=500,
            # batch_size=3,
            # fit_algorithm='cd',
            # random_state=rng,
            # positive_dict=True,
            # positive_code=True,
        ).fit(data)
        dt = time() - t0
        components = dico.components_
        # extract reference patches from noisy data
        data = extract_patches_2d(noisy_data, (patch_npix, patch_npix))
        data = data.reshape(data.shape[0], -1)
        intercept = np.mean(data, axis=0)
        data -= intercept
        kwargs = {"transform_n_nonzero_coefs": 2}
        dico.set_params(transform_algorithm="omp", **kwargs)
        code = dico.transform(data)
        patches = np.dot(code, components)
        patches += intercept
        patches = patches.reshape(len(data), *(patch_npix, patch_npix))
        cleaned_img = reconstruct_from_patches_2d(patches, (npix, npix))
        dt = time() - t0
        return cleaned_img


    def pca(tiles: np.ndarray, n_components: int = 5) -> np.ndarray:
        """
        CMB fore- and background seperation. Important for detection
        of e.g. dipoles on deltaT map. Note that PCA perfoms just as good as
        FastICA and GMCA.
        Implementation according to results in arxiv:2010.02907

        Args:
            tiles:
                3D
        """
        ntiles = len(tiles)
        npix = tiles[0].shape[0]
        tiles = (tiles).reshape(ntiles, npix ** 2)
        # ensure data is mean-centred
        for idx in range(ntiles):
            tiles[idx] -= np.mean(tiles[idx])
        # compute principle components
        pca = PCA(n_components=n_components, whiten=True).fit(tiles)
        # reconstruct independent signals based on orthogonal components
        components = pca.transform(tiles)
        cleaned_tiles = pca.inverse_transform(components)
        return cleaned_tiles.reshape(ntiles, npix, npix)


    def apodization(
        img: np.ndarray,
        theta: float,
        theta_i: Optional[float] = None,
        r200: Optional[un.quantity.Quantity] = None,
        suppress_radius: Optional[float] = None,
    ) -> np.ndarray:
        """
        Suppress image values beyond theta_i from the centre of the image.
        This avoids edge effects, due to sudden drops, which will become
        apparent in FFT procedues.
        
        Args:
            img: flat 2D SkyArray-image
            theta: opening angle of image [deg]
            theta_i: not used

        Returns:
        """
        def exponential() -> np.ndarray:
            x1edge = np.linspace(1, _npix, _npix) - _npix / 2 - 0.5
            x, y = np.meshgrid(x1edge, x1edge)
            dist = np.sqrt(x ** 2 + y ** 2)
            suppress_radius = suppression_R * R_200c
            return np.exp(-(dist / suppress_radius) ** 3)
        
        _npix = len(img)
        window = np.outer(signal.hann(_npix), signal.hann(_npix))
        return img * window


    def gaussian(
        img: np.ndarray,
        theta: un.quantity.Quantity,
        theta_i: Optional[un.quantity.Quantity] = None,
        fwhm_i: Optional[un.quantity.Quantity] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Gaussian filter for low-pass filter to get rid of long-wavelenths
        (e.g. CMB when interested in ISW-RS signals).
        Note: It is based on lenstools.ConvergenceMap.smooth function,
            but it has nothing in particular todo with convergence maps,
            and can be used for any other map, e.g. isw_rs.

        Args:
            img: partial sky-map
            theta: edge-lenght of field-of-view [deg]
            theta_i: sigma of gaussian used to indicated
                smoothing kernel width, [arcmin]
            fwhm_i: Full-Width-Half-Maximum (fwhm) of gaussian used to indicated
                smoothing kernel width, [arcmin]

        Returns:
        """
        img = ConvergenceMap(data=img, angle=theta.to(un.deg))
        fwhm_i = fwhm_i.to(un.arcmin).value
        if theta_i is None and fwhm_i is None:
            raise ValueError(f"Either theta_i or fwhm_i must be set for smoothing scale.")
        elif theta_i is None:
            sigma_i = Filters.fwhm_to_sigma(fwhm_i)
        else:
            sigma_i = theta_i

        sigma_i *= un.arcmin
        if len(img.data) < 500:
            # for image with less than 500^2 images real-space is faster
            img = img.smooth(
                scale_angle=sigma_i, kind="gaussian", **kwargs
            )
        else:
            # for larger images FFT is optimal
            img = img.smooth(
                scale_angle=sigma_i, kind="gaussianFFT", **kwargs
            )
        return img.data


    def gaussian_high_pass(
        img: np.ndarray,
        theta: un.quantity.Quantity,
        theta_i: Optional[un.quantity.Quantity] = None,
        fwhm_i: Optional[un.quantity.Quantity] = None,
    ) -> np.ndarray:
        """
        Gaussian high-pass filter
        
        Args:
            img: partial sky-map
            theta: edge-lenght of field-of-view [deg]
            theta_i: sigma of gaussian used to indicated
                smoothing kernel width, [arcmin]
            fwhm_i: Full-Width-Half-Maximum (fwhm) of gaussian used to indicated
                smoothing kernel width, [arcmin]
        
        Returns:
        """
        lowpass = Filters.gaussian(img, theta, theta_i, fwhm_i)
        highpass = img - lowpass
        return highpass


    def sigma_to_fwhm(sigma: float) -> float:
        return sigma * (2*np.sqrt(2*np.log(2)))


    def fwhm_to_sigma(fwhm: float) -> float:
        return fwhm / (2*np.sqrt(2*np.log(2)))


    def gaussian_third_derivative_convolution(
        img: np.ndarray,
        theta: un.quantity.Quantity,
        theta_i: un.quantity.Quantity,
        direction: Union[int, np.ndarray] = 1,
    ) -> np.ndarray:
        """
        Omni-directional third derivative gaussian kernel (also called DGD3 filter),
        useful for extracting dipole signal, based on
        DOI: 10.3847/2041-8213/ab0bfe
        arXiv: 1812.04241

        Args:
            img: partial sky-map
            theta: [deg]
            theta_i: [arcmin]
            sigma: ideally it should be the width of halo, R200. units are [pix]
        
        Returns:
        """
        # Compute the smoothing scale in pixel units
        theta_i_pix = np.ceil(
            img.shape[0] * theta_i.to(un.deg).value / theta.to(un.deg).value
        ).astype("int")
        gauss_1 = ndimage.gaussian_filter(
            img,
            sigma=theta_i_pix * 0.5,
            order=3 * direction,
            output=np.float64,
            mode="nearest",
        )
        gauss_2 = ndimage.gaussian_filter(
            img,
            sigma=theta_i_pix * 1.0,
            order=3 * direction,
            output=np.float64,
            mode="nearest",
        )
        gauss_3 = ndimage.gaussian_filter(
            img,
            sigma=theta_i_pix * 2.0,
            order=3 * direction,
            output=np.float64,
            mode="nearest",
        )
        return gauss_1 - gauss_2 + gauss_3

    def gaussian_third_derivative(
        img: np.ndarray,
        theta: un.quantity.Quantity,
        theta_i: un.quantity.Quantity,
        direction: int,
    ) -> np.ndarray:
        """
        Omni-directional 3-order derivative gaussian kernel
        (also called DGD1 filter), useful for extracting dipole signal.
        Based on:
        DOI: 10.3847/2041-8213/ab0bfe
        arXiv: 1812.04241

        Args:
            img: partial sky-map
            theta: opening angle of image (fov), [some angular distance]
            theta_i: [some angular distance]
            sigma: ideally it should be the width of halo, R200. units are [pix]
        """

        def _create_dgd3(
            dist: np.ndarray, theta_fov: float, theta_i: int, axis: int
        ) -> np.ndarray:
            """ Filters Function """
            gauss = (
                Filters.gaussian_field(dist, theta_i * 0.5)
                - Filters.gaussian_field(dist, theta_i)
                + Filters.gaussian_field(dist, theta_i * 2.0)
            )
            d1_gauss = np.gradient(
                gauss, theta_fov / len(dist), axis=axis, edge_order=2
            )
            d2_gauss = np.gradient(
                d1_gauss, theta_fov / len(dist), axis=axis, edge_order=2
            )
            d3_gauss = np.gradient(
                d2_gauss, theta_fov / len(dist), axis=axis, edge_order=2
            )
            return d3_gauss

        _npix = len(img)
        x1edge = np.linspace(1, _npix, _npix) - _npix / 2 - 0.5
        x, y = np.meshgrid(x1edge, x1edge)
        dist = np.sqrt(x ** 2 + y ** 2)
        theta_fov_deg = theta.to(un.deg).value * len(dist) / _npix
        theta_i_pix = np.ceil(
            _npix * theta_i.to(un.deg).value / theta.to(un.deg).value
        ).astype("int")
        window = _create_dgd3(dist, theta_fov_deg, theta_i_pix, direction)
        return np.multiply(window, img)
        

    def gaussian_first_derivative(
        img: np.ndarray,
        theta: un.quantity.Quantity,
        theta_i: un.quantity.Quantity,
        direction: int,
    ) -> np.ndarray:
        """
        Omni-directional 1-order derivative gaussian kernel
        (also called DGD1 filter), useful for extracting dipole signal.
        Based on:
        DOI: 10.3847/2041-8213/ab0bfe
        arXiv: 1812.04241

        Args:
            img: partial sky-map
            theta: opening angle of image (fov), [some angular distance]
            theta_i: [some angular distance]
            sigma: ideally it should be the width of halo, R200. units are [pix]
        """

        def _create_dgd3(
            dist: np.ndarray, theta_fov: float, theta_i: int, axis: int
        ) -> np.ndarray:
            """ Filters Function """
            d1_gauss = np.gradient(
                Filters.gaussian_field(dist, theta_i * 0.5),
                theta_fov / len(dist),
                axis=axis,
                edge_order=2,
            )
            return d1_gauss

        _npix = len(img)
        x1edge = np.linspace(1, _npix, _npix) - _npix / 2 - 0.5
        x, y = np.meshgrid(x1edge, x1edge)
        dist = np.sqrt(x ** 2 + y ** 2)
        theta_fov_deg = theta.to(un.deg).value * len(dist) / _npix
        theta_i_pix = np.ceil(
            _npix * theta_i.to(un.deg).value / theta.to(un.deg).value
        ).astype("int")
        window = _create_dgd3(dist, theta_fov_deg, theta_i_pix, direction)
        return np.multiply(window, img)
   

    def gaussian_field(theta: np.ndarray, sigma: int) -> np.ndarray:
        """
        Args:
            theta: 2D array containing distances to image centre of each pixel.
            sigma: width of Gaussian.
        Returns:
        """
        return np.exp(-theta ** 2 / (2 * sigma ** 2)) / (
            2 * np.pi * sigma ** 2
        )


    def gaussian_compensated(
        img: np.ndarray,
        theta: un.quantity.Quantity,
        theta_i: un.quantity.Quantity,
        theta_o: un.quantity.Quantity,
    ) -> np.ndarray:
        """
        Compensted gaussian filter as define in arxiv:1907.06657 (Eq. 16)

        Args:
            img: partial sky-map
            theta: edge length in degrees of field-of-view
            theta_i: Inner radius of CG-filter
            theta_o: Outer radius of CG-filter
        """

        def _create_cg(
            theta: np.ndarray, theta_i: float, theta_o: float
        ) -> np.ndarray:
            """ Filters Function """
            x = theta / theta_i
            x_o = theta_o / theta_i
            gt = (np.exp(-x ** 2.0) / (np.pi * theta_i ** 2.0)) - (
                (1.0 - np.exp(-x_o ** 2.0)) / (np.pi * theta_o ** 2.0)
            )
            gt[theta_o < theta] = 0
            return gt

        # pixel width in degrees
        pw = theta.to(un.deg).value / img.shape[0]
        # convert filter specifications from arcmin to pixel-width units
        theta_i = theta_i.to(un.deg).value / pw
        theta_o = theta_o.to(un.deg).value / pw
        theta_o_int = np.ceil(theta_o).astype(
            "int"
        )  # round up theta_o to be safe
        # define grid to map the filter onto, which we will use for the convolution
        a = b = 0
        y, x = np.ogrid[
            a - theta_o_int : a + theta_o_int, b - theta_o_int : b + theta_o_int
        ]
        dist = np.sqrt(x ** 2 + y ** 2)
        # pass the grid into the filter function
        window = _create_cg(dist, theta_i, theta_o)
        return convolve(img, window)

    def tophat_compensated(rad_obj, obj_posx, obj_posy, mapp, alpha):
        """
        Compensated top-hat filter on 2D map.
        Remove long wavelength from ISW signal from flat sky maps.
        alpha: float
            best values are 0.6-0.7 (DOI: 10.1088/0004-637X/786/2/110)
            
        """
        rad_filter = alpha * rad_obj
        extend = np.sqrt(2)
        rad_filter_sqrt2 = np.ceil(extend * rad_filter).astype(int)
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
                obj_posy + pix_xx[pp], obj_posx + pix_yy[pp]
            ]

        # Mean value in 0 -> rad_filter
        white_hat = np.mean(annulus_value[: np.ceil(1 / delta_eta).astype(int)])

        # Mean value in rad_filter -> sqrt(2)*rad_filter
        black_hat = np.mean(annulus_value[np.ceil(1 / delta_eta).astype(int) :])

        return white_hat - black_hat


# def cthf_profile(profiles, extend, Nbins):
#    """
#    Compensated top-hat filter applied to profiles
#    alpha: float
#        best values are 0.6-0.7 (DOI: 10.1088/0004-637X/786/2/110)
#    """
#    assert extend > np.sqrt(2)
#
#    delta_eta = extend / Nbins
#
#    # Mean value in 0 -> rad_filter
#    middle = np.ceil(1 / delta_eta).astype(int)
#    white_hat = np.mean(profiles[:middle])
#
#    # Mean value in rad_filter -> sqrt(2)*rad_filter
#    maxi = np.ceil(np.sqrt(2) / delta_eta).astype(int)
#    black_hat = np.mean(profiles[middle:maxi])
#
#    # filtered_profile = np.convolve([1, 2, 3], [0, 1, 0.5])
#    return white_hat - black_hat, white_hat, black_hat, middle, maxi
