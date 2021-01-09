import pytest
from pathlib import Path

import pandas as pd
import numpy as np
import numpy.testing as npt
from scipy import signal

from astropy import units as un
from astropy.constants import sigma_T, m_p, c
from astropy.cosmology import LambdaCDM

from wys_ars.utils import *
from wys_ars import io as IO


@pytest.fixture(name="cosmo", scope="module")
def create_mock_skymap_ndarray():
    return LambdaCDM(H0=67.74, Om0=0.3089, Ode0=0.6911)


class TestDistanceTransformations():
    def test__Dc_to_Da(self):
        assert 1. == Dc_to_Da(2, 1)

    def test__Dc_to_redshift(self, cosmo):
        z = Dc_to_redshift(1000, cosmo, un.Mpc)
        npt.assert_almost_equal(z, 0.2397254714674, decimal=11)

    def test__radius_to_angsize(self):
        theta = radius_to_angsize(radius=10, Da=1000, arcmin=True)
        npt.assert_almost_equal(theta, 34.377467707849, decimal=11)
        theta = radius_to_angsize(radius=10, Da=1000, arcmin=False)
        npt.assert_almost_equal(theta, 0.01, decimal=11)


class TestCoordinateTransformations():
    def test__rad2arcmin(self):
        assert 180*60 == rad2arcmin(np.pi)

    def test__arcmin2rad(self):
        assert np.pi == arcmin2rad(180*60)

    def test__get_cart2sph_jacobian(self):
        jac = get_cart2sph_jacobian(th=10, ph=10)
        np.testing.assert_allclose(
            jac,
            np.array([
                [0.45647263, 0.70404103, 0.54402111],
                [0.29595897, 0.45647263, -0.83907153],
                [-0.83907153, 0.54402111, -0.],
            ]),
            rtol=1e-07,
        )

    def test__get_sph2cart_jacobian(self):
        jac = get_sph2cart_jacobian(th=10, ph=10)
        np.testing.assert_allclose(
            jac,
            np.array([
                [0.45647263,  0.29595897, -0.83907153],
                [0.70404103, 0.45647263, 0.54402111],
                [0.54402111, -0.83907153, -0.],
            ]),
            rtol=1e-07,
        )

    def test__convert_vec_sph2cart(self):
        vec = convert_vec_sph2cart(th=10, ph=10, vij_sph=np.array([1, 1, 1]))
        np.testing.assert_allclose(
            vec, np.array([1.70453477, -0.08663993, -0.29505042]), rtol=1e-07,
        )

    def test__convert_vec_cart2sph(self):
        vec = convert_vec_cart2sph(th=10, ph=10, vij_cart=np.array([1, 1, 1]))
        np.testing.assert_allclose(
            vec, np.array([-0.08663993, 1.70453477, -0.29505042]), rtol=1e-07,
        )

    def test__transform_box_to_lc_cart_coords(self):
        pos = np.array([[1, 1, 1], [1, 1, 1]])
        vec = transform_box_to_lc_cart_coords(pos, boxsize=100, boxdist=100)
        print(vec)
        np.testing.assert_array_equal(
            vec, np.array([[-49, -49, 101], [-49, -49, 101]]),# rtol=1e-07,
        )
