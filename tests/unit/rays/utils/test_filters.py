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

@pytest.fixture(name="img", scope="module")
def test__analytic_Halo_dT_signal_to_SkyArray():
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
    dT_map = SkyUtils.analytic_Halo_signal_to_SkyArray(
        halo_idx=np.array([0]),
        halo_cat=halo_dic,
        extent=20,
        direction=[0],
        suppress=True,
        suppression_R=10,
        npix=400,
        signal="dT",
    )
    return dT_map

def test__gaussian(img):
    theta = 10. * un.deg
    fwhm_i = 10 * un.arcmin
    fimg = Filters.gaussian(img, theta=theta, fwhm_i=fwhm_i)
    npt.assert_almost_equal(fimg.max(), 2.0699e-08, decimal=10)


