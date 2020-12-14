import os, time
from typing import Tuple, Callable, Union, Optional, List
import ctypes as ct
from pathlib import Path
from functools import partial

import pandas as pd
import numpy as np
import numba as nb

from astropy import units as un
from astropy.constants import sigma_T, m_p, c

sigma_T = sigma_T.to(un.Mpc**2).value #[Mpc^2]
m_p = m_p.to(un.M_sun).value #[M_sun]
c_light = c.to("km/s").value
T_cmb = 2.7251 #[K]
Gcm2 = 4.785E-20 # G/c^2 (Mpc/M_sun)

c_lib_path = Path(__file__).parent.absolute()

class SkyNumbaUtils:
    @nb.jit(nopython=True)
    def convert_convergence_to_deflection_numba(
        kappa: np.ndarray,
        npix: int,
        opening_angle: float,
        padding_factor: int = 4,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Args:
            kappa:
                Convergence map
            opening_angle:
                Edge length of field-of-view [angular distance]
            npix:
                Number of pixels along edge of field-of-view

        Returns:
            alpha1,2:
                deflection angle in units of opening_angle
        """
        xlen, ylen = kappa.shape
        xpad, ypad = xlen * padFac, ylen * padFac
        # Array of 2D dimensionless coordinates
        xsgrid, ysgrid = _make_r_coor(opening_angle, xlen)
        x = np.zeros((xlen, xlen, 2))
        x[:, :, 0] = xsgrid
        x[:, :, 1] = ysgrid
        Lx = x[-1, 0, 0] - x[0, 0, 0]
        Ly = x[0, -1, 1] - x[0, 0, 1]
        # round to power of 2 to speed up FFT
        xpad = np.int(2 ** (np.ceil(np.log2(xpad))))
        ypad = np.int(2 ** (np.ceil(np.log2(ypad))))
        kappa_ft = np.fft.fft2(kappa, s=[xpad, ypad])
        Lxpad = Lx * xpad / xlen
        Lypad = Ly * ypad / ylen
        # make a k-space grid
        kxgrid, kygrid = np.meshgrid(
            np.fft.fftfreq(xpad), np.fft.fftfreq(ypad), indexing="ij"
        )
        kxgrid *= 2 * np.pi * xpad / Lxpad
        kygrid *= 2 * np.pi * ypad / Lypad
        alphaX_kfac = 2j * kxgrid / (kxgrid ** 2 + kygrid ** 2)
        alphaY_kfac = 2j * kygrid / (kxgrid ** 2 + kygrid ** 2)
        # [0,0] component mucked up by dividing by k^2
        alphaX_kfac[0, 0], alphaY_kfac[0, 0] = 0, 0
        alphaX_ft = alphaX_kfac * kappa_ft
        alphaY_ft = alphaY_kfac * kappa_ft
        alphaX = np.fft.ifft2(alphaX_ft)[:xlen, :ylen]
        alphaY = np.fft.ifft2(alphaY_ft)[:xlen, :ylen]
        alpha = np.zeros(x.shape)
        alpha[:, :, 0] = alphaX
        alpha[:, :, 1] = alphaY
        return -alpha  # minus sign correction


class SkyUtils:
    def analytic_Halo_signal_to_SkyArray(
        halo_idx: np.array,
        halo_cat: dict,
        extent: int,
        direction: list,
        suppress: bool,
        suppression_R: float,
        npix: int,
    ) -> np.ndarray:
        map_array = np.zeros((npix, npix))
        #dt_func = partial(
        #    SkyUtils.NFW_temperature_perturbation_map,
        #    extent=extent,
        #    direction=direction,
        #    suppress=suppress,
        #    suppression_R=suppression_R,
        #)
        for idx in halo_idx:
            map_halo = SkyUtils.NFW_deflection_angle_map(
                halo_cat["r200_deg"][idx],
                halo_cat["m200"][idx],
                halo_cat["c_NFW"][idx],
                angu_diam_dist = halo_cat["rad_dist"][idx] * 0.6774,
                npix = int(2 * halo_cat["r200_pix"][idx] * extent) + 1,
                extent=extent,
                direction=direction,
                suppress=suppress,
                suppression_R=suppression_R,
            )
            map_array = SkyUtils.add_patch_to_map(
                map_array,
                map_halo,
                (halo_cat["theta1_pix"][idx], halo_cat["theta2_pix"][idx]),
            )
        return map_array
    

    def add_patch_to_map(
        limg: np.ndarray, simg: np.ndarray, cen_pix: tuple,
    ) -> tuple:
        """
        Add small image (simg) onto large image (limg) such that simg does
        not shoot over the boundary of limg.

        Args:
        limg, simg: large-image to which small-image will be added.
        cen_pix: (x,y)-coordinates of the centre of the small image. The small
            image needs to have an uneven number of pixels.

        Returns:
        simg: Small image that is guaranteed to not exceed the boundary of
            the large image
        x,y-lim: pixel coordinates for the large image
        """
        #assert limg.flags['C_CONTIGUOUS']
        #assert simg.flags['C_CONTIGUOUS']
        rad = int(len(simg)/2)
        xedges = np.arange(cen_pix[0] - rad, cen_pix[0] + rad + 1)
        yedges = np.arange(cen_pix[1] - rad, cen_pix[1] + rad + 1)
        x_pix, y_pix = np.meshgrid(xedges, yedges)
        mask = np.logical_and(
            np.logical_and(0 <= x_pix, x_pix < len(limg)),
            np.logical_and(0 <= y_pix, y_pix < len(limg)),
        )
        x_bool = np.sum(mask, axis=0) > 0
        y_bool = np.sum(mask, axis=1) > 0
        simg = simg[mask].reshape((np.sum(y_bool), np.sum(x_bool)))
        xlim = np.array([xedges[x_bool].min(), xedges[x_bool].max()+1]).astype(int)
        ylim = np.array([yedges[y_bool].min(), yedges[y_bool].max()+1]).astype(int)
        limg[ylim[0]:ylim[1], xlim[0]:xlim[1]] += simg
        return limg

    def NFW_temperature_perturbation_map(
        theta_200c: float,
        M_200c: float,
        c_200c: float,
        vel: Union[list, tuple, np.ndarray],
        angu_diam_dist: Optional[float] = None,
        npix: int = 100,
        extent: float = 1,
        direction: List[int] = [0, 1],
        suppress: bool = False,
        suppression_R: float = 1,
    ) -> np.ndarray:
        """
        The Rees-Sciama / Birkinshaw-Gull / moving cluster of galaxies effect.

        Args:
            vel: transverse to the line-of-sight velocity, [km/sec]

        Returns:
            Temperature perturbation map, \Delta T / T_CMB
        """
        dt_map = np.zeros((npix, npix))
        for direc in direction:
            alpha_map = SkyUtils.NFW_deflection_angle_map(
                theta_200c,
                M_200c,
                c_200c,
                angu_diam_dist,
                npix,
                extent,
                [direc],
                suppress,
                suppression_R,
            )
            dt_map += - alpha_map * vel[direc] / c_light
        return dt_map

    def NFW_deflection_angle_map(
        theta_200c: float,
        M_200c: float,
        c_200c: float,
        angu_diam_dist: Optional[float] = None,
        npix: int = 100,
        extent: float = 1,
        direction: List[int] = [0],
        suppress: bool = False,
        suppression_R: float = 1,
    ) -> np.ndarray: 
        """
        Calculate the deflection angle of a halo with NFW profile using method
        in described Sec. 3.2 in Baxter et al 2015 (1412.7521).

        Note:
            In this application it can be assumed that s_{SL}/s_{S}=1. Furthermore,
            we can deglect vec{theta}/norm{theta} as it will be multiplied by the
            same in the integral of Eq. 9 in Yasini et a. 2018 (1812.04241).
        
        Args:
            theta_200c: radius, [deg]
            M_200c: mass, [Msun]
            c_200c: concentration, [-]
            extent: The size of the map from which the trans-vel is calculated
                in units of R200 of the associated halo.
            suppress:
            suppression_R:
            angu_diam_dist: angular diameter distance, [Mpc]
            direction: 0=(along x-axis), 1=(along y-axis)

        Returns:
            Deflection angle map.
        """
        assert np.sum(direction) <= 1, "Only 0 and 1 are valid direction indications."
        R_200c = np.tan(theta_200c * np.pi / 180) * angu_diam_dist # [Mpc]

        edges = np.linspace(0, 2*R_200c*extent, npix) - R_200c * extent
        thetax, thetay = np.meshgrid(edges, edges)
        R = np.sqrt(thetax ** 2 + thetay ** 2) # distances to pixels
        # Eq. 8
        A = M_200c * c_200c ** 2 / (np.log(1 + c_200c) - c_200c / (1 + c_200c)) / 4. / np.pi
        # constant in Eq. 6
        C = 16 * np.pi * Gcm2 * A / c_200c / R_200c

        R_s = R_200c / c_200c
        x = R / R_s
        x = x.astype(np.complex)
        # Eq. 7
        f = np.true_divide(1, x) * (
            np.log(x / 2) + 2 / np.sqrt(1 - x ** 2) * \
            np.arctanh(np.sqrt(np.true_divide(1 - x, 1 + x)))
        )
        alpha_map = np.zeros((npix, npix)).astype(np.complex)
        for direc in direction:
            # Eq. 6
            if direc == 0:
                thetax_hat = np.true_divide(thetax, R)
                alpha_map += C * thetax_hat * f
            elif direc == 1:
                thetay_hat = np.true_divide(thetay, R)
                alpha_map += C * thetay_hat * f
        alpha_map = np.nan_to_num(alpha_map, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        if suppress:  # suppress alpha at large radii
            suppress_radius = suppression_R * R_200c
            alpha_map *= np.exp(-(R / suppress_radius) ** 3)
        alpha_map = alpha_map.real
        alpha_map[abs(alpha_map) > 100] = 0.  # avoid unphysical results
        return alpha_map

    def convert_code_to_phy_units(
        quantity: str, map_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Convert from RayRamses code units to physical units.
        """
        if quantity in ["shear_x", "shear_y", "deflt_x", "deflt_y", "kappa_2"]:
            map_df.loc[:, [quantity]] /= c_light ** 2
        elif quantity in ["isw_rs"]:
            map_df.loc[:, [quantity]] /= c_light ** 3
        return map_df

    def convert_deflection_to_shear(
        alpha1: np.ndarray, alpha2: np.ndarray, npix: int, opening_angle: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Args:
            alpha1,2:
            opening_angle:
                Edge length of field-of-view [angular distance]
            npix:
                Number of pixels along edge of field-of-view

        Returns:
            gamma1,2: shear map
        """
        # TODO
        al11 = 1 - np.gradient(alpha1, coord, axis=0)
        al12 = -np.gradient(alpha1, coord, axis=1)
        al21 = -np.gradient(alpha2, coord, axis=0)
        al22 = 1 - np.gradient(alpha2, coord, axis=1)
        shear1 = 0.5 * (al11 - al22)
        shear2 = 0.5 * (al21 + al12)
        return shear1, shear2


    def convert_convergence_to_deflection_ctypes(
        kappa: np.ndarray, npix: int, opening_angle: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Args:
            kappa:
                Convergence map
            opening_angle:
                Edge length of field-of-view [angular distance]
            npix:
                Number of pixels along edge of field-of-view

        Returns:
            alpha1,2:
                deflection angle in units of opening_angle
        """
        alpha1, alpha2 = _call_alphas(
            kappa, npix, (opening_angle).to(un.rad).value
        )
        return alpha1, alpha2


def _make_r_coor(
    opening_angle: float, npix: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ Returns pixel coordinates """
    ds = opening_angle / npix
    x1edge = (
        np.linspace(0, opening_angle - ds, npix)
        - opening_angle / 2.0
        + ds / 2.0
    )
    x2, x1 = np.meshgrid(x1edge, x1edge)
    return x1, x2, x1edge


gls = ct.CDLL(c_lib_path / "lib_so_cgls/libglsg.so")
gls.kappa0_to_alphas.argtypes = [
    np.ctypeslib.ndpointer(dtype=ct.c_double),
    ct.c_int,
    ct.c_double,
    np.ctypeslib.ndpointer(dtype=ct.c_double),
    np.ctypeslib.ndpointer(dtype=ct.c_double),
]
gls.kappa0_to_alphas.restype = ct.c_void_p
def _call_alphas(
    kappa: np.ndarray, npix: int, opening_angle: float
) -> Tuple[np.ndarray, np.ndarray]:
    _kappa = np.array(kappa, dtype=ct.c_double)
    alpha1 = np.array(np.zeros((npix, npix)), dtype=ct.c_double)
    alpha2 = np.array(np.zeros((npix, npix)), dtype=ct.c_double)
    gls.kappa0_to_alphas(_kappa, npix, opening_angle, alpha1, alpha2)
    del _kappa
    return alpha1, alpha2


gls.kappa0_to_phi.argtypes = [
    np.ctypeslib.ndpointer(dtype=ct.c_double),
    ct.c_int,
    ct.c_double,
    np.ctypeslib.ndpointer(dtype=ct.c_double),
]
gls.kappa0_to_phi.restype = ct.c_void_p
def _call_cal_phi(
    kappa: np.ndarray, npix: int, opening_angle: float
) -> np.ndarray:
    _kappa = np.array(kappa, dtype=ct.c_double)
    phi = np.array(np.zeros((npix, npix)), dtype=ct.c_double)
    gls.kappa0_to_phi(_kappa, npix, opening_angle, phi)
    return phi
