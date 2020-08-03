import pytest
from pathlib import Path

import numpy as np
import numpy.testing as npt

from wys_ars.simulation import Simulation
from wys_ars.rays import SkyMap
from wys_ars import io as IO

dir_test_base = Path(__file__).parent.parent.parent.absolute()
dir_test_data = dir_test_base / "test_data"
file_test_skymap = dir_test_data / "Ray_maps_zrange_0.08_0.90.h5"
default_gsn_dsc = {"std": 0.4, "ngal": 40., "rnd_seed": 34077}

@pytest.fixture(name="skymap", scope="module")
def test__class_initialization_with_unit_conversion():
    skymap = SkyMap.from_file(
        npix=2048,
        theta=10.,
        quantity="kappa_2",
        dir_in=dir_test_data,
        map_file=str(file_test_skymap),
        convert_unit=True,
    )
    npt.assert_almost_equal(
        np.min(skymap.data["orig"]), -0.02992, decimal=5,
    )
    return skymap

def test__class_initialization_without_unit_conversion():
    skymap = SkyMap.from_file(
        npix=2048,
        theta=10.,
        quantity="kappa_2",
        dir_in=dir_test_data,
        map_file=str(file_test_skymap),
        convert_unit=False,
    )
    npt.assert_almost_equal(
        np.min(skymap.data["orig"]), -2689263265.931, decimal=2,
    )

def test__galaxy_shape_noise_creation(skymap):
    skymap.create_galaxy_shape_noise(**default_gsn_dsc)
    npt.assert_almost_equal(
        np.min(skymap.data["gsn"]), -0.8097179, decimal=6,
    )
    _old_gsn = np.min(skymap.data["gsn"])
    
    skymap.create_galaxy_shape_noise(**default_gsn_dsc)
    assert np.min(skymap.data["gsn"]) == _old_gsn
    
    default_gsn_dsc["rnd_seed"] = 61809
    skymap.create_galaxy_shape_noise(**default_gsn_dsc)
    assert np.min(skymap.data["gsn"]) != _old_gsn

def test__galaxy_shape_noise_addition(skymap):
    skymap.add_galaxy_shape_noise(on="orig")
    npt.assert_almost_equal(
        np.min(skymap.data["orig_gsn"]), -0.797137513, decimal=2,
    )

    
def test__skymap_smoothing(skymap):
    skymap.smoothing(2.5, on="orig")
    npt.assert_almost_equal(
        np.min(skymap.data["orig_smooth"]), -0.022843, decimal=5,
    )
    skymap.smoothing(2.5, on="orig_gsn")
    npt.assert_almost_equal(
        np.min(skymap.data["orig_gsn_smooth"]), -0.032335, decimal=5,
    )
