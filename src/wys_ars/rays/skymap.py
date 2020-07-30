import os, sys, glob
import random
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd
import numpy as np
from scipy.interpolate import RectBivariateSpline

import healpy as hp
import astropy
from astropy.io import fits
from astropy import units as un
import lenstools
from lenstools import ConvergenceMap

from wys_ars.simulation import Simulation
from wys_ars.rays.voids import TunnelsFinder
from wys_ars.power_spectra import PowerSpectrum

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
        find_tunnel_voids:
        add_galaxy_shape_noise:
        create_galaxy_shape_noise:
        smoothing:
    """
    def __init__(
        self,
        npix: int,
        theta: float,
        quantity: str,
        dirs: Dict[str, str],
        mapp: np.ndarray,
        map_file: str,
    ):
        self.map_file = map_file
        self.mapp = mapp
        self.dirs = dirs
        self.npix = npix
        self.quantity = quantity
        self.opening_angle = theta

    @classmethod
    def from_file(
        cls,
        npix: int,
        theta: float,
        quantity: str,
        dir_in: str,
        dir_out: str,
        map_file: Optional[str] = None,
        #file_dsc: Optional[Dict[str, str, str]] = None,
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
                    npix, theta, quantity, dir_in, dir_out, map_df, map_file, convert_unit,
                )
            elif map_file.split(".")[-1] == "npy":
                map_array = np.load(map_file)
                dirs = {"sim": dir_in, "out": dir_out}
                return cls.SkyMap(npix, theta, quantity, dirs, map_array, map_file)
        #elif file_dsc is not None:
        #    #TODO -> better for a supergroup
        #    raise SkyMapWarning("Not implemented yet")
        else:
            raise SkyMapWarning('There is no file being pointed at')
    
    @classmethod
    def from_dataframe(
        cls,
        npix: int,
        theta: float,
        quantity: str,
        dir_in: str,
        dir_out: str,
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
        map_df = cls._convert_code_to_phy_units(cls, quantity, map_df)
        map_array = cls._h5_to_array(cls, map_df[quantity])
        return cls.from_array(npix, theta, quantity, dir_in, dir_out, map_array, map_file)
    
    @classmethod
    def from_array(
        cls,
        npix: int,
        theta: float,
        quantity: str,
        dir_in: str,
        dir_out: str,
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
        dirs = {"sim": dir_in, "out": dir_out}
        return cls(npix, theta, quantity, dirs, map_array, map_file)
    
    def _h5_to_array(self, map_series: pd.Series,) -> np.ndarray:
        """ Convert pd.Series to np.ndarray """
        zip_array = sorted(zip(map_series.index.values, map_series.values,))
        ids = np.array([i for (i, j) in zip_array])
        values = np.array([j for (i, j) in zip_array])
        npix = int(np.sqrt(len(map_series.values)))
        map_out = np.zeros([npix, npix])
        k = 0
        for j in range(npix):
            for i in range(npix):
                map_out[j, i] = values[k]
                k += 1
        return map_out

    def find_tunnel_voids(
        self,
        dir_in: str,
        dir_out: str,
        file_dsc: Dict[str, str] = {"root": "kappa2_maps_zrange", "extension": "fits"},
        sigmas: List[float] = [0.0, 3.0],
        kernel_width: Optional[float] = None,
        field_conversion: Optional[str] = None,
    ) -> None:
        """
        Args:
            file_dsc:
            dir_in:
            dir_out:
            sigmas: significance of void signal
            kernel_width:
                smoothing kernel;[deg]
        """
        # find files
        sim = Simulation(dir_in, None, file_dsc, None)
        _file_path = sim.files[file_dsc["root"]][0]
        print(f"Finding tunnels in -> {_file_path}")

        if kernel_width is None:
            if isinstance(self, kernel_width):
                kernel_width = self.kernel_width
            else:
                raise SkyMapWarning("No smoothing length defined.")
        
        first = True
        tunnels = TunnelsFinder(
            _file_path, self.opening_angle, self.npix, kernel_width
        )
        tunnels.find_peaks(field_conversion)  # more intuitive to put sigma in this fct?
        for sigma in sigmas:
            tunnels.find_voids(dir_out, sigma)

            if first is True:
                voids_df_sum = tunnels.voids
                peaks_df_sum = tunnels.filtered_peaks
                first = False
            else:
                voids_df_sum = voids_df_sum.append(tunnels.voids, ignore_index=True)
                peaks_df_sum = peaks_df_sum.append(tunnels.filtered_peaks, ignore_index=True)

        file_name = ''.join(_file_path.split("/")[-1].split(".")[:-1])
        voids_df_sum.to_hdf(f"{dir_out}voids_{file_name}.h5", key="df")
        peaks_df_sum.to_hdf(f"{dir_out}peaks_{file_name}.h5", key="df")

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

            if save:
                file_out = self.dirs["out"] + ff.split("/")[-1].replace(
                    ".h5", "_%s_hp.fits" % quantity
                )
                self._save_map(file_out, map_out)
            else:
                self.data = hpxmap

    def smoothing(
        self,
        kernel_width: float,
    ) -> Union[None, lenstools.image.convergence.ConvergenceMap]:
        """
        Gaussian smoothing. This should be done after adding GSN to the map.
        Args:
            smoothing:
                Kernel width of the smoothing length-scale in [arcmin]
        """
        self.kernel_width = kernel_width
        _mapp = self._array_to_fits(self.mapp)
        file_out = self._create_filename(self.map_file, f"tmp", extension="fits")
        self._save_map(file_out, _mapp)
        _mapp = ConvergenceMap.load(file_out, format="fits")
        self.mapp = _mapp.smooth(scale_angle=kernel_width * un.arcmin, kind="gaussian",)

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
        std_pix = np.sqrt(std ** 2 / 2 * 1.0 / theta_pix ** 2 * 1.0 / ngal)
        if rnd_seed is None:
            self.gsn_map = np.random.normal(
                loc=0, scale=std_pix, size=[self.npix, self.npix],
            )
        else:
            rg = np.random.Generator(np.random.PCG64(rnd_seed))
            self.gsn_map = rg.normal(
                loc=0, scale=std_pix, size=[self.npix, self.npix],
            )

    def add_galaxy_shape_noise(
        self, std: float, ngal: float, rnd_seed: Optional[int] = None,
    ) -> None:
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
        self.create_galaxy_shape_noise(std, ngal, rnd_seed)
        self.mapp += self.gsn_map
    
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
   
    def to_file(self, method: str, extension: str) -> None:
        """
        """
        file_out = self._create_filename(
            self.map_file, self.quantity, extension=extension
        )
        print("Save in:\n   %s" % file_out)
        self.mapp.save(file_out, format=extension)

    def _create_filename(
        self, file_in: str, quantity: str, extension: str,
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
        return self.dirs["out"] + file_out
    
    def _save_map(
        self,
        file_out: str,
        map_out: Union[
            np.ndarray,
            astropy.io.fits.hdu.image.PrimaryHDU,
            lenstools.image.convergence.ConvergenceMap,
        ],
    ) -> None:
        """
        Args:
        """
        # remove old
        if os.path.exists(file_out):
            os.remove(file_out)
        # create new
        if isinstance(map_out, np.ndarray):
            np.save(file_out, map_out)
        elif isinstance(map_out, astropy.io.fits.hdu.image.PrimaryHDU):
            map_out.writeto(file_out)
        elif isinstance(map_out, lenstools.image.convergence.ConvergenceMap):
            map_out.save(file_out)
