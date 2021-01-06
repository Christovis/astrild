import pytest
from pathlib import Path

import pandas as pd
import numpy as np
import numpy.testing as npt

from astropy import units as un
from astropy.constants import sigma_T, m_p, c

from wys_ars.simulation import Simulation
from wys_ars.rays import SkyMap
from wys_ars.rays.skys.sky_utils import SkyUtils, SkyNumbaUtils
from wys_ars import io as IO

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
    map_np = np.ones((100, 100))
    map_np[40:60, 40:60] = 10
    return map_np


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
    print(np.argmax(alpha_1, axis=1), alpha_1.shape)
    assert np.argmax(alpha_1, axis=1) == 50
    assert np.argmin(alpha_1) == (50, 50)

