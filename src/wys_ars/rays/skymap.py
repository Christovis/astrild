import os, sys, glob
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd
import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage.filters import convolve

import healpy as hp
import astropy
from astropy.io import fits
from astropy import units as un
import lenstools
from lenstools import ConvergenceMap

from wys_ars.simulation import Simulation
from wys_ars.rays.utils import filters as Filters
from wys_ars import io as IO

dir_src = Path(__file__).parent.absolute()
default_config_file_ray = dir_src / "configs/ray_snapshot_info.h5"
c_light = 299792.458  # in km/s

class SkyMapWarning(BaseException):
    pass


class SkyMap:
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
        smoothing:
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
            map_df = cls._convert_code_to_phy_units(cls, quantity, map_df)
        map_array = IO.transform_PandasSeries_to_NumpyNdarray(map_df[quantity])
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
    
    def _convert_code_to_phy_units(
        self, quantity: str, map_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Convert from RayRamses code units to physical units.
        """
        if quantity in ["shear_x", "shear_y", "deflt_x", "deflt_y", "kappa_2"]:
            map_df.loc[:, [quantity]] /= c_light ** 2
        elif quantity in ["isw_rs"]:
            map_df.loc[:, [quantity]] /= c_light ** 3
        return map_df
   
    #@property
    def pdf(self, nbins: int, of: str="orig") -> dict:
        _pdf = {}
        _pdf["values"], _pdf["bins"] = np.histogram(
            self.data[of], bins=nbins, density=True,
        )
        return _pdf

    #@property
    def wl_peak_counts(
        self,
        nbins: int,
        field_conversion: str,
        of: str="orig",
        limits: Optional[tuple] = None,
    ) -> pd.DataFrame:
        if field_conversion == "normalize":
            _map = self.data[of] - np.mean(self.skymap.data[of])
        else:
            _map = self.data[of]

        if limits is None:
            lower_bound = np.percentile(self.data[of], 5)  #np.min(self.data[of])
            upper_bound = np.percentile(self.data[of], 95)  #np.max(self.data[of])
        else:
            lower_bound = min(limits)
            upper_bound = max(limits)

        map_bins = np.arange(
            lower_bound, upper_bound, (upper_bound - lower_bound) / nbins,
        )
        _map = ConvergenceMap(data=_map, angle=self.opening_angle*un.deg)
        _kappa, _pos = _map.locatePeaks(map_bins)
        
        _hist, _kappa = np.histogram(_kappa, bins=nbins, density=False)
        _kappa = (_kappa[1:] + _kappa[:-1]) / 2
        peak_counts_dic = {"kappa": _kappa, "counts": _hist}
        peak_counts_df = pd.DataFrame(data=peak_counts_dic)
        return peak_counts_df

    def smoothing(
        self,
        filter_dsc: dict,
        on: str,
    ) -> None:
        """
        Gaussian smoothing. This should be done after adding GSN to the map.
        Args:
            smoothing:
                Kernel width of the smoothing length-scale in [arcmin]
        """
        if self.quantity == "kappa_2":
            if on == "orig_gsn":
                _map = self.add_galaxy_shape_noise()
            elif on == "orig":
                _map = self.data["orig"]

            for fil, args in filter_dsc.items():
                if fil == "gaussian":
                    self.smoothing_length = args["theta_i"]
                    _map = Filters.gaussian_filter(
                        _map, self.opening_angle, **args
                    )
                    self.data[on + "_gfilter"] = _map
                if fil == "gaussian_truncated":
                    self.smoothing_length = args["theta_i"]
                    _map = Filters.compensated_gaussian_filter(
                        _map, self.opening_angle, **args
                    )
                    self.data[on + "_gtfilter"] = _map
        else:
            raise SkyMapWarning("Not yet implemented")

    def create_galaxy_shape_noise(
        self, std: float, ngal: float, rnd_seed: Optional[int] = None,
    ) -> None:
        """
        Galaxy Shape Noise (GSN)
        e.g.: https://arxiv.org/pdf/1907.06657.pdf

        Args:
            std:
                dispersion of source galaxy intrinsic ellipticity, 0.4 for LSST
            ngal:
                number density of galaxies, 40 for LSST; [arcmin^2]
            rnd_seed:
                Fix random seed, for reproducability.
        Returns:
            gsn_map:
                self.npix x self.npix np.array containing the GSN
        """
        theta_pix = 60 * self.opening_angle / self.npix
        std_pix = 0.007 #np.sqrt(std ** 2 / (2*theta_pix*ngal))
        if rnd_seed is None:
            self.data["gsn"] = np.random.normal(
                loc=0, scale=std_pix, size=[self.npix, self.npix],
            )
        else:
            rg = np.random.Generator(np.random.PCG64(rnd_seed))
            self.data["gsn"] = rg.normal(
                loc=0, scale=std_pix, size=[self.npix, self.npix],
            )
        print(f"The GSN map sigma is {np.std(self.data['gsn'])}", std_pix)

    def add_galaxy_shape_noise(self, on: str = "orig") -> np.ndarray:
        """
        Add GSN on top of skymap.
        
        Args:
            std:
                dispersion of source galaxy intrinsic ellipticity, 0.4 for LSST
            ngal:
                number density of galaxies, 40 for LSST; [arcmin^2]
            rnd_seed:
                Fix random seed, for reproducability.
        """
        if self.quantity == "kappa_2":
            self.data["orig_gsn"] = self.data["orig"] + self.data["gsn"]
            return self.data["orig_gsn"]
        else:
            raise SkyMapWarning(f"GSN should not be added to {self.quantity}")
    
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
    
    def _array_to_fits(self, map_out: np.ndarray) -> astropy.io.fits:
        """ Convert maps that in .npy format into .fits format """
        # Convert .npy to .fits
        data = fits.PrimaryHDU()
        data.header["ANGLE"] = self.opening_angle # [deg]
        data.data = map_out
        return data
   
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
