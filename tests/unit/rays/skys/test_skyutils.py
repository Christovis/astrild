import pytest
from pathlib import Path

import pandas as pd
import numpy as np
import numpy.testing as npt
from scipy import signal

from astropy import units as un
from astropy.constants import sigma_T, m_p, c

from astrild.simulation import Simulation
from astrild.rays import SkyMap
from astrild.rays.skys.sky_utils import SkyUtils, SkyNumbaUtils
from astrild import io as IO

c_light = c.to("km/s").value

dir_test_base = Path(__file__).parent.parent.parent.absolute()
dir_test_data = dir_test_base / "test_data"
file_test_skymap = dir_test_data / "Ray_maps_zrange_0.08_0.90.h5"
default_gsn_dsc = {"std": 0.4, "ngal": 40., "rnd_seed": 34077}


@pytest.fixture(name="map_df", scope="module")
def create_mock_skymap_dataframe():
    map_dic = {
        "shear_x": [c_light**2] * 10,
        "deflt_x": [c_light**2] * 10,
        "kappa_2": [c_light**2] * 10,
        "isw_rs": [c_light**3] * 10,
    }
    return pd.DataFrame(map_dic)

@pytest.fixture(name="map_np", scope="module")
def create_mock_skymap_ndarray():
    return np.outer(
        signal.general_gaussian(100, p=1, sig=10),
        signal.general_gaussian(100, p=1, sig=10),
    )


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
    map_np = SkyUtils.analytic_Halo_signal_to_SkyArray(
        halo_idx=np.array([0]),
        halo_cat=halo_dic,
        extent=20,
        direction=[0],
        suppress=True,
        suppression_R=10,
        npix=400,
        signal="dT",
    )
    assert np.unravel_index(map_np.argmax(), map_np.shape) == (200, 167)
    npt.assert_almost_equal(map_np.min(), -2.0699e-08, decimal=10)
    npt.assert_almost_equal(map_np.mean(), 2.4732e-11, decimal=12)
    npt.assert_almost_equal(map_np.max(), 2.0699e-08, decimal=10)

def test__analytic_Halo_alpha_signal_to_SkyArray():
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
    map_np = SkyUtils.analytic_Halo_signal_to_SkyArray(
        halo_idx=np.array([0]),
        halo_cat=halo_dic,
        extent=20,
        direction=[0],
        suppress=True,
        suppression_R=10,
        npix=400,
        signal="alpha",
    )
    assert np.unravel_index(map_np.argmax(), map_np.shape) == (200, 233)
    npt.assert_almost_equal(map_np.min(), -3.1027e-05, decimal=8)
    npt.assert_almost_equal(map_np.mean(), -3.7073e-08, decimal=11)
    npt.assert_almost_equal(map_np.max(), 3.1027e-05, decimal=8)

def test__convert_code_to_phy_units_of_shear(map_df):
    map_df = SkyUtils.convert_code_to_phy_units("shear_x", map_df)
    assert map_df.shear_x.values[0] == 1.0

def test__convert_code_to_phy_units_of_deflt(map_df):
    map_df = SkyUtils.convert_code_to_phy_units("deflt_x", map_df)
    assert map_df.deflt_x.values[0] == 1.0

def test__convert_code_to_phy_units_of_kappa(map_df):
    map_df = SkyUtils.convert_code_to_phy_units("kappa_2", map_df)
    assert map_df.kappa_2.values[0] == 1.0

def test__convert_code_to_phy_units_of_iswrs(map_df):
    map_df = SkyUtils.convert_code_to_phy_units("isw_rs", map_df)
    assert map_df.isw_rs.values[0] == 1.0

def test__convert_convergence_to_deflection_ctypes(map_np):
    alpha_1, alpha_2 = SkyUtils.convert_convergence_to_deflection_ctypes(
        map_np, 100, 10 * un.deg,
    )
    npt.assert_almost_equal(
        alpha_1.min(), -0.0146571, decimal=6,
    )
    npt.assert_almost_equal(
        alpha_1.mean(), 6.49e-05, decimal=6,
    )
    npt.assert_almost_equal(
        alpha_1.max(), 0.0151880, decimal=6,
    )
