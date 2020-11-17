import os, time
from typing import Tuple
import ctypes as ct
from pathlib import Path

import pandas as pd
import numpy as np
import numba as nb

from astropy import units as un
from astropy.constants import c as c_light

c_lib_path = Path(__file__).parent.absolute()


class SkyUtils:
    def convert_code_to_phy_units(
        quantity: str, map_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Convert from RayRamses code units to physical units.
        """
        if quantity in ["shear_x", "shear_y", "deflt_x", "deflt_y", "kappa_2"]:
            map_df.loc[:, [quantity]] /= c_light.to('km/s').value**2
        elif quantity in ["isw_rs"]:
            map_df.loc[:, [quantity]] /= c_light.to('km/s').value**3
        return map_df
    
    def convert_deflection_to_shear(
        alpha1: np.ndarray,
        alpha2: np.ndarray,
        npix: int,
        opening_angle: float,
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
        #TODO
        al11 = 1 - np.gradient(alpha1, coord, axis=0)
        al12 = - np.gradient(alpha1, coord, axis=1)
        al21 = - np.gradient(alpha2, coord, axis=0)
        al22 = 1 - np.gradient(alpha2, coord, axis=1)
        shear1 = 0.5*(al11 - al22)
        shear2 = 0.5*(al21 + al12)
        return shear1, shear2

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
        xpad, ypad = xlen*padFac, ylen*padFac
        # Array of 2D dimensionless coordinates
        xsgrid, ysgrid = _make_r_coor(opening_angle, xlen)
        x = np.zeros((xlen, xlen, 2))
        x[:,:,0] = xsgrid
        x[:,:,1] = ysgrid
        Lx = x[-1,0,0] - x[0,0,0]
        Ly = x[0,-1,1] - x[0,0,1]
        # round to power of 2 to speed up FFT
        xpad = np.int(2**(np.ceil(np.log2(xpad))))
        ypad = np.int(2**(np.ceil(np.log2(ypad))))
        kappa_ft = np.fft.fft2(kappa, s=[xpad,ypad])
        Lxpad = Lx * xpad/xlen
        Lypad = Ly * ypad/ylen
        # make a k-space grid
        kxgrid, kygrid = np.meshgrid(
            np.fft.fftfreq(xpad), np.fft.fftfreq(ypad), indexing='ij',
        )
        kxgrid *= 2*np.pi*xpad/Lxpad
        kygrid *= 2*np.pi*ypad/Lypad
        alphaX_kfac = 2j * kxgrid / (kxgrid**2 + kygrid**2)  
        alphaY_kfac = 2j * kygrid / (kxgrid**2 + kygrid**2)
        # [0,0] component mucked up by dividing by k^2
        alphaX_kfac[0,0], alphaY_kfac[0,0] = 0,0
        alphaX_ft = alphaX_kfac * kappa_ft
        alphaY_ft = alphaY_kfac * kappa_ft
        alphaX = np.fft.ifft2(alphaX_ft)[:xlen,:ylen]
        alphaY = np.fft.ifft2(alphaY_ft)[:xlen,:ylen] 
        alpha = np.zeros(x.shape)
        alpha[:,:,0] = alphaX
        alpha[:,:,1] = alphaY
        return -alpha # worry aboutminus sign? Seems to make it work :-)


    def convert_convergence_to_deflection_cython(
        kappa: np.ndarray,
        npix: int,
        opening_angle: float,
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
        alpha1, alpha2 = _call_alphas(kappa, npix, (opening_angle).to(un.rad).value)
        return alpha1, alpha2

def _make_r_coor(
    opening_angle: float, npix: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ Returns pixel coordinates """
    ds = opening_angle / npix
    x1edge = np.linspace(0, opening_angle-ds, npix) - opening_angle/2. + ds/2.
    x2, x1 = np.meshgrid(x1edge, x1edge)
    return x1, x2, x1edge

gls = ct.CDLL(c_lib_path / "lib_so_cgls/libglsg.so")
gls.kappa0_to_alphas.argtypes = [
    np.ctypeslib.ndpointer(dtype = ct.c_double),
    ct.c_int,ct.c_double,
    np.ctypeslib.ndpointer(dtype = ct.c_double),
    np.ctypeslib.ndpointer(dtype = ct.c_double),
]
gls.kappa0_to_alphas.restype  = ct.c_void_p
def _call_alphas(
    kappa: np.ndarray, npix: int, opening_angle: float,
) -> Tuple[np.ndarray, np.ndarray]:
    _kappa = np.array(kappa, dtype=ct.c_double)
    alpha1 = np.array(np.zeros((npix,npix)), dtype=ct.c_double)
    alpha2 = np.array(np.zeros((npix,npix)), dtype=ct.c_double)
    gls.kappa0_to_alphas(_kappa, npix, opening_angle, alpha1, alpha2)
    return alpha1, alpha2

gls.kappa0_to_phi.argtypes = [
    np.ctypeslib.ndpointer(dtype = ct.c_double),
    ct.c_int,ct.c_double,
    np.ctypeslib.ndpointer(dtype = ct.c_double),
]
gls.kappa0_to_phi.restype  = ct.c_void_p
def _call_cal_phi(kappa: np.ndarray, npix: int, opening_angle: float) -> np.ndarray:
    _kappa = np.array(kappa, dtype=ct.c_double)
    phi = np.array(np.zeros((npix, npix)), dtype=ct.c_double)
    gls.kappa0_to_phi(_kappa, npix, opening_angle, phi)
    return phi
