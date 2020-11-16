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
import healpy as hp
import pymaster as nmt

from wys_ars.rays.utils import Filters
from wys_ars.rays.skys.sky_utils import SkyUtils
from wys_ars.rays.skyio import SkyIO
from wys_ars.io import IO

dir_src = Path(__file__).parent.absolute()
default_config_file_ray = dir_src / "configs/ray_snapshot_info.h5"

class SkyNamasterWarning(BaseException):
    pass


class SkyNamaster:
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
    """
    def __init__(
        self,
        skyfield: np.ndarray,
        opening_angle: float,
        quantity: str,
        dirs: Dict[str, str],
        map_file: Optional[str] = None,
    ):
        self.data = {"orig": skyfield}
        self._nside = hp.get_nside(skyfield)
        self._npix = hp.nside2npix(self._nside)
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
        assert map_file, SkyNamasterWarning('There is no file being pointed at')

        file_extension = map_file.split(".")[-1]
        if file_extension == "h5":
            map_df = pd.read_hdf(map_file, key="df")
            return cls.from_dataframe(
                map_df, opening_angle, quantity, dir_in, map_file, convert_unit,
            )
        elif file_extension == "fits":
            map_array = hp.read_map(map_file)
            return cls.from_array(
                map_array, opening_angle, quantity, dir_in, map_file
            )
        elif file_extension == "npy":
            map_array = np.load(map_file)
            return cls.from_array(
                map_array, opening_angle, quantity, dir_in, map_file
            )
    
    @classmethod
    def from_dataframe(
        cls,
        map_df: pd.DataFrame,
        opening_angle: float,
        quantity: str,
        dir_in: str,
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
        map_array = hp.ma(map_array)  # mask out bad values (e.g Nan)
        return cls(map_array, opening_angle, quantity, dirs, map_file)

    def to_namaster(self):
        """
        Transform SkyHealpix to SkyNamaster.
        """
        f_0 = nmt.NmtField(self.data["mask"], [self.data[which]])

    def resize(
        self,
        npix,
        of: Optional[str] = None,
        img: Optional[np.ndarray] = None,
        rtn: bool = False,
    ) -> Union[np.ndarray, None]:    
        if of:
            img = copy.deepcopy(self.data[of])
        img = transform.resize(image, (npix, npix), anti_aliasing=True,)
        if rtn:                                          
            return img                                   
        else:                                            
            self.data[of] = img

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
        nside: Optional[int] = None,
    ) -> None:
        if "mask" not in self.data.keys():
            self.create_mask(theta, nside)
        self.data[on] = hp.ma(self.data[on])
        self.data[on].mask = self.data["mask"]

