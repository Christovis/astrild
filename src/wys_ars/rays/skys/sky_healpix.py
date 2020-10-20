import os, sys, glob
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd
import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage.filters import convolve

import astropy
from astropy.io import fits
from astropy import units as un
import lenstools
from lenstools import ConvergenceMap
import healpy as hp

from wys_ars.simulation import Simulation
from wys_ars.rays.utils import Filters
from wys_ars.rays.skys.sky_utils import SkyUtils
from wys_ars.rays.skyio import SkyIO
from wys_ars.io import IO

dir_src = Path(__file__).parent.absolute()
default_config_file_ray = dir_src / "configs/ray_snapshot_info.h5"
c_light = 299792.458  # in km/s

class SkyHealpixWarning(BaseException):
    pass


class SkyHealpix:
    """
    The sky-map is constructed through multiple ray-tracing simulations
    run with RayRamses. This class analyzes the 2D map that contains
    the summes pertrurbations of each ray. It can prepare the data for the
    search of voids and peaks.

    Attributes:
        npix:
        theta:
        dirs:

    Methods:
        from_file:
        pdf:
        wl_peak_counts:
        add_galaxy_shape_noise:
        create_galaxy_shape_noise:
        create_cmb:
        create_mask:
        convolution:
    """
    def __init__(
        self,
        npix: int,
        opening_angle: float,
        quantity: str,
        dirs: Dict[str, str],
        mapp: np.ndarray,
        map_file: str,
    ):
        self.map_file = map_file
        self.data = {"orig": mapp}
        self.dirs = dirs
        self.npix = npix
        self.quantity = quantity
        self.opening_angle = opening_angle

    @classmethod
    def from_file(
        cls,
        theta: float,
        quantity: str,
        dir_in: str,
        npix: Optional[int] = None,
        map_file: Optional[str] = None,
        convert_unit: bool = True,
    ) -> "SkyMap":
        """
        Initialize class by reading the skymap data from pandas hdf5 file
        or numpy array.
        The file can be pointed at via map_filename or file_dsc.

        Args:
            map_filename:
                File path with which skymap pd.DataFrame can be loaded.
            file_dsc:
                Dictionary pointing to a file via {path, root, extension}.
                Use when multiple skymaps need to be loaded.
        """
        if map_file is not None:
            if map_file.split(".")[-1] == "h5":
                map_df = pd.read_hdf(map_file, key="df")
                return cls.from_dataframe(
                    npix, theta, quantity, dir_in, map_df, map_file, convert_unit,
                )
            elif map_file.split(".")[-1] == "npy":
                map_array = np.load(map_file)
                return cls.from_array(
                    map_array.shape[0], theta, quantity, dir_in, map_array, map_file
                )
            elif map_file.split(".")[-1] == "fits":
                map_array = ConvergenceMap.load(map_file, format="fits").data
                return cls.from_array(
                    map_array.shape[0], theta, quantity, dir_in, map_array, map_file
                )
        else:
            raise SkyMapWarning('There is no file being pointed at')
    
    @classmethod
    def from_dataframe(
        cls,
        npix: int,
        theta: float,
        quantity: str,
        dir_in: str,
        map_df: pd.DataFrame,
        map_file: str,
        convert_unit: bool = True,
    ) -> "SkyMap":
        """
        Initialize class by reading the skymap data from pandas DataFrame. 

        Args:
            map_filename:
                File path with which skymap pd.DataFrame can be loaded.
            file_dsc:
                Dictionary pointing to a file via {path, root, extension}.
                Use when multiple skymaps need to be loaded.
        """
        if convert_unit:
            map_df = SkyUtils.convert_code_to_phy_units(quantity, map_df)
        map_array = SkyIO.transform_PandasSeries_to_NumpyNdarray(map_df[quantity])
        return cls.from_array(npix, theta, quantity, dir_in, map_array, map_file)
    
    @classmethod
    def from_array(
        cls,
        npix: int,
        theta: float,
        quantity: str,
        dir_in: str,
        map_array: np.array,
        map_file: str,
    ) -> "SkyMap":
        """
        Initialize class by reading the skymap data from np.ndarray.

        Args:
            map_filename:
                File path with which skymap pd.DataFrame can be loaded.
            file_dsc:
                Dictionary pointing to a file via {path, root, extension}.
                Use when multiple skymaps need to be loaded.
        """
        dirs = {"sim": dir_in}
        return cls(npix, theta, quantity, dirs, map_array, map_file)
    
    def create_cmb(
        self,
        filepath_cl: str,
        theta: float = 20.,
        npix: int = 128,
        lmax: int = 3e3,
        rnd_seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Cosmig Microwave Background (CMB) on partial-sky map,
        for which the flat-sky approximation holds (ell > 10).

        Args:
            filepath_cl:
                angular power spectrum of CMB
            theta:
                Edge length of the square field-of-view [deg]
            nside:
                Nr. of pixels per edge of the output full-sky map
            rnd_seed:
                Fix random seed, for reproducability.

        Returns:
            cmb_map:
        """
        cl_cmb = np.load(filepath_cl)
        np.random.seed(rnd_seed)


    def create_mask(self, theta: float, nside: int) -> np.ndarray:
        """
        Args:
            theta:
                Opening-angle
            nside:
        """
        ras = np.array([90-theta/2, 90+theta/2]) * np.pi/180  #60 degree
        decs = np.array([theta/2, 360-theta/2]) * np.pi/180   #60 degree

        npix = hp.nside2npix(nside)
        pixel_theta, pixel_phi = hp.pix2ang(nside, np.arange(npix))
        
        mask = np.ones(npix, dtype=np.bool)
        mask[pixel_theta < ras[0]] = 0
        mask[pixel_theta > ras[1]] = 0
        mask[(decs[0] < pixel_phi) & (pixel_phi < decs[1])] = 0
        self.mask = nmt.mask_apodization(mask, 2.5, apotype="C2")


    def rotate_healpix(
        self,
        skymap: np.ndarray,
        theta: Optional[float] = None,
        phi: Optional[float] = None,
    ) -> np.ndarray:
        """
        """
        if (theta is None) or (phi is None):
            theta = np.random.random()*180
            phi = np.random.random()*180
        
        nside = hp.npix2nside(len(skymap))
        # Get theta, phi for non-rotated map
        t, p = hp.pix2ang(nside, np.arange(hp.nside2npix(nside))) #theta, phi
        # Define a rotator
        r = hp.Rotator(deg=True, rot=[theta,phi])
        # Get theta, phi under rotated co-ordinates
        trot, prot = r(t,p)
        # Interpolate map onto these co-ordinates
        _skymap = hp.get_interp_val(skymap, trot, prot)
        return _skymap
    
    def to_healpix(
        self, ray_nrs: list = None, quantities: list = None, save: bool = True,
    ) -> Union[None, astropy.io.fits.hdu.image.PrimaryHDU]:
        """ """
        if quantities is None:
            quantities = self.map_df.columns.values

        for quantity in quantities:
            map_df = self.code2phy_units(self.map_df, quantity)

            indices = hp.ang2pix(
                self.npix,
                map_df["the_co"].values,
                map_df["phi_co"].values,
                lonlat=False,
            )
            hpxmap = np.zeros(hp.nside2npix(self.npix), dtype=np.float)
            indx = list(
                set(np.arange(hp.nside2npix(self.npix))).symmetric_difference(
                    set(np.unique(indices))
                )
            )
            hpxmap[indx] = hp.UNSEEN
            for i in range(len(map_df[quantity].values)):
                hpxmap[indices[i]] += map_df[quantity].values[i]
