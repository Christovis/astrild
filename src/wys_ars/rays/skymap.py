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


class SkyMap(SkyType):
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
    """
    def __init__(
        self,
        skymap: Union[SkyArray, SkyHealpix],
        npix: int,
        opening_angle: float,
        quantity: str,
        dirs: Dict[str, str],
        map_file: str,
    ):
        self.data = skymap
        self.npix = npix
        self.opening_angle = opening_angle
        self.quantity = quantity
        self.dirs = dirs
        self.map_file = map_file

    @classmethod
    def from_file(
        cls,
        theta: float,
        quantity: str,
        dir_in: str,
        npix: Optional[int] = None,
        map_file: Optional[str] = None,
        convert_unit: bool = True,
        sky_type: str = "array",
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
            sky_type:
                Indicate what format the sky should have, as it makes different
                operations are accessible: [healpix, array]
        """
        assert sky_type in ["array", "healpix"], "The declared 'sky_type' is not known"
        assert map_file, SkyMapWarning('There is no file being pointed at')

        file_extension = map_file.split(".")[-1]
        if file_extension == "h5":
            map_df = pd.read_hdf(map_file, key="df")
            if sky_type == "array":
                skymap = SkyArray.from_dataframe(quantity, map_df, convert_unit)
            elif sky_type == "healpix":
                skymap = SkyHealpix.from_dataframe(
                    npix, theta, quantity, dir_in, map_df, map_file, convert_unit,
                )
        
        elif file_extension in ["npy", "fits"]: # only possible for SkyArray
            if file_extension == "npy":
                map_array = np.load(map_file)
            elif file_extension == "fits":
                map_array = ConvergenceMap.load(map_file, format="fits").data
            skymap = SkyArray.from_array(
                npix, theta, quantity, dir_in, map_array, map_file
            )
        
        dirs = {"sim": dir_in}
        return cls(skymap, npix, theta, quantity, dirs, map_file)
    
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
        sky_type: str = "array",
    ) -> "SkyMap":
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
            skymap = SkyArray.from_dataframe(quantity, map_df, convert_unit)
        elif sky_type == "healpix":
            skymap = SkyHealpix.from_dataframe(
                npix, theta, quantity, dir_in, map_df, map_file, convert_unit,
            )
        dirs = {"sim": dir_in}
        return cls(skymap, npix, theta, quantity, dirs, map_file)
    
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
        skymap = SkyArray.from_array(map_array)
        dirs = {"sim": dir_in}
        return cls(skymap, npix, theta, quantity, dirs, map_file)
    
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
    
    def to_file(
        self,
        dir_out: str,
        on: str = "orig_gsn_smooth",
        extension: str = "npy",
    ) -> None:
        """
        """
        if on == "orig_gsn":
            self.data[on] = self.add_galaxy_shape_noise()
        self.dirs["out"] = dir_out
        filename = self._create_filename(
            self.map_file, self.quantity, on, extension=extension
        )
        IO.save_skymap(self.data[on], dir_out + filename)

    def _create_filename(
        self, file_in: str, quantity: str, on: str, extension: str,
    ) -> str:
        """
        Args:
        """
        quantity = quantity.replace("_", "")
        file_out = file_in.split("/")[-1].replace("Ray", quantity)
        file_out = file_out.replace(".h5", f"_lt.{extension}")
        if ("_lc" not in file_in) and ("zrange" not in file_in):
            file_out = file_out.split("_")
            box_string = [
                string for string in file_in.split("/") if "test" in string
            ][0]
            idx, string = [
                (idx, "%s_" % box_string + string)
                for idx, string in enumerate(file_out)
                if "output" in string
            ][0]
            file_out[idx] = string
            file_out = "_".join(file_out)
        _file = file_out.split(".")[:-1] + [on] #.append(on)
        _file = _file + [extension]
        file_out = ".".join(_file)
        return file_out
    
