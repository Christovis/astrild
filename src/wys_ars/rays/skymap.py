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

from wys_ars.simulation import Simulation
from wys_ars.rays.utils import Filters
from wys_ars.rays.skys import SkyArray
from wys_ars.rays.skys import SkyHealpix
from wys_ars.rays.skyio import SkyIO
from wys_ars.io import IO

dir_src = Path(__file__).parent.absolute()
default_config_file_ray = dir_src / "configs/ray_snapshot_info.h5"

class SkyMapWarning(BaseException):
    pass


class SkyMap:
    """
    The sky-map is constructed through multiple ray-tracing simulations
    run with RayRamses. This class analyzes the 2D map that contains
    the summes pertrurbations of each ray. It can prepare the data for the
    search of voids and peaks.

    Methods:
        from_file:
        from_dataframe:
        from_array:
    """

    def from_file(
        npix: int,
        theta: float,
        quantity: str,
        dir_in: str,
        map_file: Optional[str] = None,
        convert_unit: bool = True,
        sky_type: str = "array",
    ) -> Union[SkyArray, SkyHealpix]:
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
            sky_type:
                Indicate what format the sky should have, as it makes different
                operations are accessible: [healpix, array]
        """
        assert sky_type in ["array", "healpix"], "The declared 'sky_type' is not known"
        assert map_file, SkyMapWarning('There is no file being pointed at')

        file_extension = map_file.split(".")[-1]
        if sky_type == "array":
            if file_extension == "h5":
                map_df = pd.read_hdf(map_file, key="df")
                skymap = SkyArray.from_dataframe(
                    map_df, npix, theta, quantity, dir_in, map_file, convert_unit,
                )
        
            elif file_extension in ["npy", "fits"]: # only possible for SkyArray
                    if file_extension == "npy":
                        map_array = np.load(map_file)
                    elif file_extension == "fits":
                        map_array = ConvergenceMap.load(map_file, format="fits").data
                    skymap = SkyArray.from_array(
                        map_array, npix, theta, quantity, dir_in, map_file,
                    )
        elif sky_type == "healpix":
            skymap = SkyHealpix.from_file(map_file, npix, theta, quantity, dir_in)
        return skymap
 
    def from_dataframe(
        npix: int,
        theta: float,
        quantity: str,
        dir_in: str,
        map_df: pd.DataFrame,
        map_file: str,
        convert_unit: bool = True,
        sky_type: str = "array",
    ) -> Union[SkyArray, SkyHealpix]:
        """
        Initialize class by reading the skymap data from pandas DataFrame. 

        Args:
            map_filename:
                File path with which skymap pd.DataFrame can be loaded.
            file_dsc:
                Dictionary pointing to a file via {path, root, extension}.
                Use when multiple skymaps need to be loaded.
            sky_type:
                Indicate what format the sky should have, as it makes different
                operations are accessible: [healpix, array]
        """
        assert sky_type in ["array", "healpix"], "The declared 'sky_type' is not known"
        if sky_type == "array":
            skymap = SkyArray.from_dataframe(
                map_df, npix, theta, quantity, dir_in, map_file, convert_unit,
            )
        elif sky_type == "healpix":
            skymap = SkyHealpix.from_dataframe(
                npix, theta, quantity, dir_in, map_df, map_file, convert_unit,
            )
        return skymap
 
    def from_array(
        map_array: np.array,
        npix: int,
        theta: float,
        quantity: str,
        dir_in: str,
        map_file: Optional[str]=None,
        sky_type: str = "array",
    ) -> Union[SkyArray, SkyHealpix]:
        """
        Initialize class by reading the skymap data from np.ndarray.

        Args:
            map_filename:
                File path with which skymap pd.DataFrame can be loaded.
            file_dsc:
                Dictionary pointing to a file via {path, root, extension}.
                Use when multiple skymaps need to be loaded.
        """
        if sky_type == "array":
            skymap = SkyArray.from_array(
                map_array, npix, theta, quantity, dir_in, map_file,
            )
        elif sky_type == "healpix":
            skymap = SkyHealpix.from_array(
                map_array, npix, theta, quantity, dir_in, map_file,
            )
        return skymap
    
    def _get_files(self, ray_nrs) -> None:
        """
        Get filenames of save skymaps based on file describtion.
        Args:
            ray_nrs:
                Numbers of ray-tracing snapshots
        """
        if ray_nrs is None:
            self.files["map"] = glob.glob(
                f"{self.dirs['sim']}{self.file_dsc['root']}*"
                + f"{self.file_dsc['extension']}"
            )
        else:
            for ray_nr in ray_nrs:
                self.files["map"].append(
                    self.dirs["sim"] + "Ray_maps_output%05d.h5" % ray_nr
                )
    
