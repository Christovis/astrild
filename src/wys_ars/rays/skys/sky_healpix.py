import os, sys, glob, copy
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
import pymaster as nmt

from wys_ars.simulation import Simulation
from wys_ars.rays.utils import Filters
from wys_ars.rays.skys.sky_namaster import SkyNamaster
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
        create_cmb:
        create_mask:
        convolution:
    """
    def __init__(
        self,
        skymap: np.ndarray,
        opening_angle: float,
        quantity: str,
        dirs: Dict[str, str],
        map_file: Optional[str] = None,
    ):
        self.data = {"orig": skymap}
        self.nside = hp.get_nside(skymap)
        self.npix = hp.nside2npix(self.nside)
        self.opening_angle = opening_angle
        self.quantity = quantity
        self.dirs = dirs
        self.map_file = map_file

    @classmethod
    def from_file(
        cls,
        map_file: str,
        opening_angle: float,
        quantity: str,
        dir_in: str,
        nside: Optional[int] = None,
        convert_unit: bool = True,
    ) -> "SkyHealpix":
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
        assert map_file, SkyMapWarning('There is no file being pointed at')

        file_extension = map_file.split(".")[-1]
        if file_extension == "h5":
            map_df = pd.read_hdf(map_file, key="df")
            return cls.from_dataframe(
                map_df, opening_angle, nside, quantity, dir_in, map_file, convert_unit,
            )
        elif file_extension == "fits":
            map_array = hp.read_map(map_file)
            return cls.from_array(
                map_array, opening_angle, quantity, dir_in, map_file,
            )
    
    @classmethod
    def from_dataframe(
        cls,
        map_df: pd.DataFrame,
        opening_angle: float,
        nside: int,
        quantity: str,
        dir_in: str,
        map_file: str,
        convert_unit: bool = True,
    ) -> "SkyHealpix":
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
        map_array = SkyIO.transform_PandasDataFrame_to_Healpix(map_df, quantity, nside)
        return cls.from_array(
            map_array, opening_angle, quantity, dir_in, map_file,
        )
    
    @classmethod
    def from_array(
        cls,
        map_array: np.array,
        opening_angle: float,
        quantity: str,
        dir_in: str,
        map_file: Optional[str] = None,
    ) -> "SkyHealpix":
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
        map_array = hp.ma(map_array)  # mask out bad values (e.g Nan)
        return cls(map_array, opening_angle, quantity, dirs, map_file)

    def to_namaster(self):
        """
        Transform SkyHealpix to SkyNamaster.
        """
        sky_array = nmt.NmtField(self.data["mask"], [self.data[which]])
        return SkyNamaster.from_array(
            sky_array,
            self.npix,
            self.opening_angle,
            self.quantity,
            self.dirs,
            self.map_file,
        )
    
    def create_cmb(
        self,
        filepath_cl: str,
        lmax: float = 3e3,
        nside: Optional[int] = None,
        rnd_seed: Optional[int] = None,
    ) -> None:
        """
        Cosmig Microwave Background (CMB) on partial-sky map,
        for which the flat-sky approximation holds (ell > 10).

        Args:
            filepath_cl:
                angular power spectrum of CMB
            nside:
                Nr. of pixels per edge of the output full-sky map
            rnd_seed:
                Fix random seed, for reproducability.

        Returns:
            cmb_map:
        """
        if nside is None:
            nside = self.nside
        cl_cmb = np.load(filepath_cl)
        np.random.seed(rnd_seed)
        self.data["cmb"] = hp.sphtfunc.synfast(cmb_tt, nside=nside, lmax=lmax)

    def sum_of_maps(self, map1: str, map2: str) -> None:
        self.data[f"{map1}_{map2}"] = self.data[map1] + self.data[map2]
    
    def add_mask(
        self,
        on: str,
        theta: Optional[float] = None,
    ) -> None:
        """
        """
        if ("mask" not in self.data.keys()) or (theta != self.mask_theta):
            self.create_mask(theta)
        self.data[on+"_mask"] = hp.ma(copy.deepcopy(self.data[on]))
        self.data[on+"_mask"].mask = self.data["mask"]

    def create_mask(self, theta: float,) -> None:
        """
        Mask out unobserved patches of the full-sky.

        Args:
            theta:
                Edge length of the square field-of-view [deg]
            nside:
                Nr. of pixels per edge of the output full-sky map
        """
        print("create_mask", theta)
        # angular positions of all healpix pixels
        pixel_theta, pixel_phi = hp.pix2ang(self.nside, np.arange(self.npix))
        # range of ra and dec of the field-of-view
        ras = np.array([90-theta/2, 90+theta/2]) * np.pi/180  #[rad]
        decs = np.array([theta/2, 360-theta/2]) * np.pi/180   #[rad]
        # create mask
        mask = np.zeros(self.npix, dtype=np.bool)
        mask[pixel_theta < ras[0]] = 1
        mask[pixel_theta > ras[1]] = 1
        mask[(decs[0] < pixel_phi) & (pixel_phi < decs[1])] = 1
        # smooth transition between fov and mask (not available in healpy)
        self.data["mask"] = nmt.mask_apodization(mask, 2.5, apotype="C2")
        self.mask_theta = theta

    def rotate(
        self,
        theta: float,
        phi: float,
        which: str,
        rtn: bool = False,
    ) -> np.ndarray:
        """
        Rotate the full-sky.

        Args:
            theta and phi:
                Rotational angles [deg]
            which:
                Identify which map in self.data should be rotated.
        """
        if (theta is None) or (phi is None):
            theta = np.random.random()*180
            phi = np.random.random()*180
        
        # Get theta, phi for non-rotated map
        t, p = hp.pix2ang(self.nside, np.arange(self.npix))
        # Define a rotator
        r = hp.Rotator(deg=True, rot=[theta,phi])
        # Get theta, phi under rotated co-ordinates
        trot, prot = r(t,p)
        # Interpolate map onto these co-ordinates
        _skymap = copy.deepcopy(self.data[which])
        _skymap = hp.get_interp_val(_skymap, trot, prot)
        if rtn is True:
            return _skymap
        else:
            self.data[which] = _skymap
