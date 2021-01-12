import pytest
from pathlib import Path

import pandas as pd
import numpy as np
import numpy.testing as npt

import camb

from wys_ars.rays.skys import SkyHealpix
from wys_ars.rays.skys import SkyArray

dir_src = Path(__file__).parent.parent.absolute()
rockstar_df = pd.read_hdf(dir_src / "../../test_data/rockstar_in_lc.h5", key="df")


@pytest.fixture(scope="session", name="Cl")
def create_cmb_angular_power_spectrum():
    pars = camb.CAMBparams()
    pars.set_cosmology(
        H0=67.74,
        ombh2=0.022,
        mnu=0.0,
        omk=0,
        tau=0.06,
    )
    pars.InitPower.set_params(As=2e-9, ns=0.965, r=0)
    pars.set_for_lmax(1000, lens_potential_accuracy=0)
    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(
        pars,
        raw_cl=True,   
        CMB_unit=None,
        spectra=('unlensed_scalar', 'lensed_scalar',),
    )
    return powers["unlensed_scalar"][:,0]


def test__from_Cl_array(Cl):
    skycmb = SkyHealpix.from_Cl_array(
        Cl,
        quantity="CMB",
        nside=1024,
        lmax=3000,
        rnd_seed=1111,
    )
    assert list(skycmb.data.keys()) == ["orig"]
    assert skycmb.nside == 1024
    assert skycmb.npix == 12582912
    assert skycmb.opening_angle == 41253
    assert skycmb.quantity == "CMB"
    npt.assert_almost_equal(
        skycmb.data["orig"].mean(), -1.7366862812254835e-12, decimal=17,
    )

