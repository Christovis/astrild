import pytest
from pathlib import Path

import pandas as pd
import numpy as np
import numpy.testing as npt
from scipy import signal

from astropy import units as un
from astropy.constants import sigma_T, m_p, c

from astrild.rays.utils import Filters
from astrild.rays.skys import SkyArray
from astrild.rays.skys import SkyUtils

dir_test_base = Path(__file__).parent.parent.parent.absolute()
dir_test_data = dir_test_base / "test_data"
file_test_skymap = dir_test_data / "Ray_maps_zrange_0.08_0.90.h5"
halo_dic = {
    "r200_deg": np.array([0.05]),
    "r200_pix": np.array([50]),
    "m200": np.array([7e13]),
    "c_NFW": np.array([2.]),
    "Dc": np.array([1050]),
    "theta1_pix": np.array([200]),
    "theta2_pix": np.array([200]),
    "theta1_tv": np.array([200]),
    "theta2_tv": np.array([200]),
}
extent = 20
theta = halo_dic["r200_deg"][0]*extent

@pytest.fixture(name="img", scope="module")
def test__analytic_Halo_dT_signal_to_SkyArray():
    dT_map = SkyUtils.analytic_Halo_signal_to_SkyArray(
        halo_idx=np.array([0]),
        halo_cat=halo_dic,
        extent=extent,
        direction=[0],
        suppress=True,
        suppression_R=10,
        npix=400,
        signal="dT",
    )
    return dT_map

def test__gaussian(img):
    fwhm_i = 10 * un.arcmin
    fimg = Filters.gaussian(img, theta=theta*un.deg, fwhm_i=fwhm_i)
    npt.assert_almost_equal(fimg.max() * 1e8, 1.665952, decimal=5)

def test__gaussian_high_pass(img):
    fwhm_i = 5 * un.arcmin
    fimg = Filters.gaussian(img, theta=theta*un.deg, fwhm_i=fwhm_i)
    npt.assert_almost_equal(fimg.max() * 1e8, 1.901196, decimal=5)

def test__sigma_to_fwhm():
    sigma = 1 / (2*np.sqrt(2*np.log(2)))
    fwhm = Filters.sigma_to_fwhm(sigma)
    assert fwhm == 1

def test__fwhm_to_sigma():
    fwhm = (2*np.sqrt(2*np.log(2)))
    sigma = Filters.fwhm_to_sigma(fwhm)
    assert sigma == 1

def test__gaussian_third_derivative(img):
    # test dipole filter along 0-direction
    fimg = Filters.gaussian_third_derivative(
        img, theta=theta*un.deg, theta_i=halo_dic["r200_deg"]*un.deg, direction=0,
    )
    fimg_x_slice = fimg[:, len(fimg)//2]
    fimg_y_slice = fimg[len(fimg)//2, :]
    assert fimg_x_slice.max() == 0.
    npt.assert_almost_equal(fimg_y_slice.max() * 1e7, 1.713281, decimal=5)
    # test dipole filter along 1-direction
    fimg = Filters.gaussian_third_derivative(
        img, theta=theta*un.deg, theta_i=halo_dic["r200_deg"]*un.deg, direction=1,
    )
    fimg_x_slice = fimg[:, len(fimg)//2]
    fimg_y_slice = fimg[len(fimg)//2, :]
    assert fimg_x_slice.max() == 0.
    npt.assert_almost_equal(fimg_y_slice.max() * 1e7, 8.210071, decimal=5)

