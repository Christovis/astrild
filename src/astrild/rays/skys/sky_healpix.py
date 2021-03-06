import os, sys, glob, copy
import argparse
from pathlib import Path
from typing import Dict, Optional, Union, Type

import pandas as pd
import numpy as np
from functools import partial
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage.filters import convolve

import astropy
from astropy.io import fits
from astropy import units as un
import lenstools
from lenstools import ConvergenceMap
import healpy as hp
#import pymaster as nmt

from astrild.simulation import Simulation
from astrild.rays.utils import Filters
from astrild.rays.skys.sky_array import SkyArray
from astrild.rays.skys.sky_namaster import SkyNamaster
from astrild.rays.skys.sky_utils import SkyUtils
from astrild.rays.skyio import SkyIO
from astrild.io import IO

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
        add_field:
        add_galaxy_shape_noise:
        create_galaxy_shape_noise:
        add_cmb:
        create_cmb:
        add_mask:
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
        self._nside = hp.get_nside(skymap)
        self._npix = hp.nside2npix(self.nside)
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
        file_extension = map_file.split(".")[-1]
        assert file_extension in ["h5", "fits", "npy"], SkyHealpixWarning(
            f"The file formart {file_extension} is not supported."
        )
        if file_extension == "h5":
            map_df = pd.read_hdf(map_file, key="df")
            return cls.from_sky_dataframe(
                map_df,
                opening_angle,
                nside,
                quantity,
                dir_in,
                map_file,
                convert_unit,
            )
        elif file_extension == "fits":
            map_array = hp.read_map(map_file)
            return cls.from_sky_array(
                map_array, opening_angle, quantity, dir_in, map_file
            )
        elif file_extension == "npy":
            map_array = np.load(map_file)
            return cls.from_sky_array(
                map_array, opening_angle, quantity, dir_in, map_file
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
        map_array = SkyIO.transform_PandasDataFrame_to_Healpix(
            map_df, quantity, nside
        )
        return cls.from_sky_array(
            map_array, opening_angle, quantity, dir_in, map_file
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


    @classmethod
    def from_Cl_file(
        cls,
        cl_file: str,
        quantity: str,
        dir_in: str,
        nside: int,
        lmax: int = 3000,
        opening_angle: int = 41253,
        key: Optional[str] = None, 
        rnd_seed: Optional[int] = None,
    ) -> "SkyHealpix":
        """
        Args:
            cl_file:
            quantity:
            dir_in:
            nside:
            lmax:
            opening_angle:
            key:
        """
        np.random.seed(rnd_seed)
        file_extension = cl_file.split(".")[-1]
        assert file_extension in ["npz", "npy"], SkyHealpixWarning(
            f"The file formart {file_extension} is not supported."
        )
        if file_extension == "npy":
            cl_array = np.load(cl_file)
        elif file_extension == "npz":
            cl_array = np.load(cl_file)[key]
        return cls.from_Cl_array(
            cl_array, quantity, nside, lmax, opening_angle, dir_in, cl_file
        )


    @classmethod
    def from_Cl_array(
        cls,
        cl_array: np.array,
        quantity: str,
        nside: int,
        lmax: int,
        opening_angle: int = 41253,
        dir_in: Optional[str] = None,
        cl_file: Optional[str] = None,
        rnd_seed: Optional[int] = None,
    ) -> "SkyHealpix":
        """
        Args:
            opening_angle:
                [deg^2]
        """
        np.random.seed(rnd_seed)
        dirs = {"sim": dir_in}
        map_array = hp.sphtfunc.synfast(cl_array, nside=nside, lmax=lmax)
        return cls(map_array, opening_angle, quantity, dirs, cl_file)


    @property
    def nside(self):
        return self._nside

    @property
    def npix(self):
        return self._npix

    
    def to_skyarray(
        self,
        npix: int,
        opening_angle: float,
        of: str = "orig",
    ) -> Type[SkyArray]:
        """
        Args:
            lonra: , [deg]
            latra: , [deg]
        """
        lonra = latra = [0, opening_angle]
        cart_proj = hp.projector.CartesianProj(
            lonra=lonra,
            latra=latra,
            xsize=npix,
            ysize=npix,
        )
        map_array = cart_proj.projmap(
            self.data[of],
            rot=(0., 0.),
            vec2pix_func=partial(hp.vec2pix, self._nside),
        )
        return SkyArray.from_array(
            map_array,
            opening_angle=opening_angle,
            quantity=self.quantity,
            dir_in=self.dirs["sim"],
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
            nside = self._nside
        cl_cmb = np.load(filepath_cl)
        np.random.seed(rnd_seed)
        self.data["cmb"] = hp.sphtfunc.synfast(cmb_tt, nside=nside, lmax=lmax)


    def sum_of_maps(self, map1: str, map2: str) -> None:
        self.data[f"{map1}_{map2}"] = self.data[map1] + self.data[map2]


    def arithmetic_operation_with(
        self, skymap: np.array, on: str, operation: np
    ) -> None:
        """
        Use Numpy function to perform one of the basic arithmetic operations:
            add, substract, multiply, divide
        of two Healpix fields.

        Args:
            skymap:
            on:
            operation:
        """
        _pixel_idx = np.arange(self._npix)
        _unmasked_pixel_idx = _pixel_idx[~self.data[on].mask]
        self.data[on].data[_unmasked_pixel_idx] = operation(
            self.data[on].data[_unmasked_pixel_idx], skymap[_unmasked_pixel_idx]
        )


    def add_mask(self, on: str, theta: Optional[float] = None) -> None:
        """
        """
        if ("mask" not in self.data.keys()) or (theta != self.mask_theta):
            self.create_mask(theta)
        self.data[on + "_mask"] = hp.ma(copy.deepcopy(self.data[on]))
        self.data[on + "_mask"].mask = self.data["mask"]


    def create_mask(self, theta: float) -> None:
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
        _pixel_index = np.arange(self._npix)
        pixel_theta, pixel_phi = hp.pix2ang(self._nside, _pixel_index)
        # range of ra and dec of the field-of-view
        ras = np.array([90 - theta / 2, 90 + theta / 2]) * np.pi / 180  # [rad]
        decs = np.array([theta / 2, 360 - theta / 2]) * np.pi / 180  # [rad]
        # create mask
        mask = np.zeros(self._npix, dtype=np.bool)
        mask[pixel_theta < ras[0]] = 1
        mask[pixel_theta > ras[1]] = 1
        mask[(decs[0] < pixel_phi) & (pixel_phi < decs[1])] = 1
        self.data["mask"] = mask
        self.mask_theta = theta

    
    def rotate(
        self, theta: float, phi: float, which: str, rtn: bool = False
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
            theta = np.random.random() * 180
            phi = np.random.random() * 180

        # Get theta, phi for non-rotated map
        _pixel_index = np.arange(self._npix)
        t, p = hp.pix2ang(self._nside, _pixel_index)
        # Define a rotator
        r = hp.Rotator(deg=True, rot=[theta, phi])
        # Get theta, phi under rotated co-ordinates
        trot, prot = r(t, p)
        # Interpolate map onto these co-ordinates
        _skymap = copy.deepcopy(self.data[which])
        _skymap = hp.get_interp_val(_skymap, trot, prot)
        if rtn is True:
            return _skymap
        else:
            self.data[which] = _skymap
    
    
    #def to_skynamaster(self, apodization_width: float = 2.5):
    #    """
    #    Transform SkyHealpix to SkyNamaster.
    #    """
    #    _skyarray = copy.deepcopy(self.data["orig"])
    #    _skyarray[~self.data["mask"]] = hp.UNSEEN
    #    _mask = copy.deepcopy(self.data["mask"]) * 1
    #    #_mask = nmt.mask_apodization(_mask, apodization_width, apotype="C2")
    #    #print("------------->", _mask.shape)
    #    #self.data["nm"] = nmt.NmtField(_mask, [_skyarray])
    #    # return SkyNamaster.from_array(
    #    #    sky_field,
    #    #    self._npix,
    #    #    self.opening_angle,
    #    #    self.quantity,
    #    #    self.dirs,
    #    #    self.map_file,
    #    # )
