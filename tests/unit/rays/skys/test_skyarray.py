from pathlib import Path
import pytest

import pandas as pd
import numpy as np

from astrild.rays.skys import SkyArray
from astrild.rays.dipole_finder import Dipoles

dir_src = Path(__file__).parent.parent.absolute()
rockstar_df = pd.read_hdf(dir_src / "../../test_data/rockstar_in_lc.h5", key="df")


@pytest.fixture(scope="session", name="halo")
def load_single_halo():
    return rockstar_df.iloc[0].squeeze()

@pytest.fixture(scope="session", name="halos")
def load_multiple_halos():
    return rockstar_df

def test__skyarray_from_halo_series_dT(halo):
    sky_dt = SkyArray.from_halo_series(
        halo,
        npix=int(2 * halo.r200_pix * 10) + 1,
        extent=10,
        direction=[0, 1],
        suppress=True,
        suppression_R=10,
        to="dT",
    )
    assert sky_dt.data["orig"].min() == pytest.approx(-1.7028239210299853e-07, rel=1e-5)
    assert sky_dt.data["orig"].mean() == pytest.approx(-1.9386409608471563e-25, rel=1e-15)
    assert sky_dt.data["orig"].max() == pytest.approx(1.7028239210299855e-07, rel=1e-5)

def test__skyarray_from_halo_series_alpha(halo):
    sky_alpha = SkyArray.from_halo_series(
        halo,
        npix=int(2 * halo.r200_pix * 10) + 1,
        extent=10,
        direction=[0, 1],
        suppress=True,
        suppression_R=10,
        to="alpha",
    )
    assert sky_alpha.data["orig"].min() == pytest.approx(-9.02262751486356e-05, rel=1e-3)
    assert sky_alpha.data["orig"].mean() == pytest.approx(1.647689725443215e-21, rel=1e-15)
    assert sky_alpha.data["orig"].max() == pytest.approx(9.022627514863563e-05, rel=1e-13)
