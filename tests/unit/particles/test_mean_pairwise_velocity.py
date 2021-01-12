import pytest
from pathlib import Path

import pandas as pd
import numpy as np
import numpy.testing as npt
from scipy import signal

from astropy import units as un
from astropy.constants import sigma_T, m_p, c
from astropy.cosmology import LambdaCDM

from wys_ars.particles.hutils.mean_pairwise_velocity import make_rsep, mean_pv_from_tv

dist_bins = np.linspace(0, 50, 40)
particle_nr = 2000

@pytest.fixture(name="pos", scope="module")
def create_positions():
    pos = np.zeros((particle_nr, 3), dtype=float)
    pos[:, 0] = np.linspace(-10, 10, 2000)
    pos[:1000, 1] = -5
    pos[1000:, 1] = np.linspace(5, 50, 1000)  # 5
    pos[:, 2] = 500
    return pos 

@pytest.fixture(name="tvel", scope="module")
def create_transverse_velocity():
    tvel = np.zeros((particle_nr, 2), dtype=float)
    tvel[:1000, 1] = 100
    tvel[1000:, 1] = -100
    return tvel


def test__make_rsep():
    binnr = len(dist_bins)
    binwidth = np.diff(dist_bins)[0]
    rsep = make_rsep(binnr, binwidth)
    assert len(rsep) == 40
    assert len(np.unique(np.diff(rsep)[0])) == 1
    npt.assert_almost_equal(np.diff(rsep)[0], 1.282051282, decimal=8)
    npt.assert_almost_equal(rsep[0], 0.64102564, decimal=8)
    npt.assert_almost_equal(rsep[-1], 50.64102564, decimal=8)

def test__mean_pv_from_tv(pos, tvel):
    rsep, vij = mean_pv_from_tv(
        pos_cart=pos,
        vel_ang=tvel,
        bins=dist_bins,
        multithreading=False,
    )
    assert len(vij) == 40
    npt.assert_almost_equal(vij[0], -9.98742453e-02, decimal=8)
    npt.assert_almost_equal(vij[-1], -1.80198033658e+02, decimal=8)
