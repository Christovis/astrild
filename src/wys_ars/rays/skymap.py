import os, sys, glob
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

dir_src = Path(__file__).parent.absolute()
default_config_file_ray = dir_src / "configs/ray_snapshot_info.h5"
c_light = 299792.458  # in km/s

class SkyMap(BaseException):
    pass


class SkyMap:
    """
    The sky-map is constructed through multiple ray-tracing simulations
    (e.g. run with RayRamses). This class analyzes the 2D map that contains
    the summes pertrurbations of each ray. It can prepare the data for the
    search of voids and peaks.

    Attributes:
    Methods:
    """
    def __init__(
        self,
        file_map: str,
        file_dsc: Dict[str, str],
        dir_in: str,
        dir_out: str,
        npix: int,
        theta: float,
    ):
        """
        Args:
            dir_sim: simulation directory
            dir_out: directory where results will be stored
            npix: number of pixels per edge
            theta: opening angle; [degrees]
        """
        if file_map is None:
            self.files = {"map": []}
        else:
            self.files = {"map": [file_map]}
        self.dirs = {"sim": dir_in, "out": dir_out}
        self.npix = npix
        self.opening_angle = theta
        self.file_dsc = file_dsc

    def find_tunnel_voids(
        self,
        dir_in: str,
        dir_out: str,
        file_dsc: Dict[str, str] = {"root": "kappa2_maps_zrange", "extension": "fits"},
        sigmas: List[float] = [0.0, 3.0],
        kernel_width: float = 2.5,
    ) -> None:
        """
        Args:
            file_dsc:
            dir_in:
            dir_out:
            sigmas: significance of void signal
            kernel_width: ;[deg]

        Returns:
        """
        sim = Simulation(dir_in, None, file_dsc, None)
        _file_path = sim.files[file_dsc["root"]][0]

        print(f"Finding tunnels in -> {_file_path}")
        
        first = True
        for sigma in sigmas:
            # find tunnels
            tunnels = TunnelsFinder(
                _file_path, self.opening_angle, self.npix, kernel_width
            )
            tunnels.find_peaks()  # more intuitive to put sigma in this fct?
            tunnels.find_voids(dir_out, sigma)

            if first is True:
                voids_df_sum = tunnels.voids
                peaks_df_sum = tunnels.peaks
                first = False
            else:
                voids_df_sum = voids_df_sum.append(tunnels.voids, ignore_index=True)
                peaks_df_sum = peaks_df_sum.append(tunnels.peaks, ignore_index=True)

        file_name = ''.join(_file_path.split("/")[-1].split(".")[:-1])
        voids_df_sum.to_hdf(f"{dir_out}voids_{file_name}.h5", key="df")
        peaks_df_sum.to_hdf(f"{dir_out}peaks_{file_name}.h5", key="df")

    def to_healpix(
        self, ray_nrs: list = None, quantities: list = None, save: bool = True,
    ) -> Union[None, astropy.io.fits.hdu.image.PrimaryHDU]:
        """ """
        if len(self.files["map"]) == 0:
            self._get_files(ray_nrs)

        for ff in self.files["map"]:
            map_df = self._read_map(ff)
            if quantities is None:
                quantities = map_df.columns.values

            for quantity in quantities:
                map_df = self._code2phy_units(map_df, quantity)

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

    def to_lenstools(
        self,
        ray_nrs: Optional[list] = None,
        quantities: Optional[list] = None,
        smoothing: Optional[float] = None,  # [arcmin]
        gsn_dsc: Optional[Dict[str, float]] = None,
        extension: str = "fits",
        save: bool = True,
    ) -> Union[None, lenstools.image.convergence.ConvergenceMap]:
        """ Save ray-tracing maps as lenstools.ConvergenceMap """
        if len(self.files["map"]) == 0:
            self._get_files(ray_nrs)
        
        for fname in self.files["map"]:
            print(f"Convert: {fname}")
            map_df = self._read_map(fname)
            if quantities is None:
                quantities = map_df.columns.values

            for quantity in quantities:
                map_df = self._code2phy_units(map_df, quantity)
                map_out = self._h5_to_array(map_df[quantity])

                if gsn_dsc and isinstance(map_out, np.ndarray):
                    print("Add galaxy shape noise")
                    map_out += self.galaxy_shape_noise(**gsn_dsc)

                map_out = self._array_to_fits(map_out)
                file_out = self._create_filename(fname, f"tmp", None, extension="fits")
                self._save_map(file_out, map_out)
                map_out = ConvergenceMap.load(file_out, format="fits")

                if smoothing:
                    print(f"Smoothing of convergence map with {smoothing} [arcmin]")
                    map_out = map_out.smooth(
                        scale_angle=smoothing * un.arcmin, kind="gaussian",
                    )

                if save:
                    file_out = self._create_filename(fname, quantity, None, extension=extension)
                    print("Save in:\n   %s" % file_out)
                    #self._save_map(file_out, map_out)
                    map_out.save(file_out, format=extension)

        if not save:
            return map_out

    def galaxy_shape_noise(self, std: float, ngal: float,) -> np.ndarray:
        """
        Galaxy Shape Noise (GSN)
        e.g.: https://arxiv.org/pdf/1907.06657.pdf

        Args:
            std:
                dispersion of source galaxy intrinsic ellipticity, 0.4 for LSST
            ngal:
                number density of galaxies, 40 for LSST; [arcmin^2]
        Returns:
            gsn_map:
                self.npix x self.npix np.array containing the GSN
        """
        theta_pix = 60 * self.opening_angle / self.npix
        std_pix = np.sqrt(std ** 2 / 2 * 1.0 / theta_pix ** 2 * 1.0 / ngal)

        gsn_map = np.random.normal(loc=0, scale=std_pix, size=[self.npix, self.npix],)
        return gsn_map
    
    def _read_map(self, file_map: str) -> pd.DataFrame:
        return pd.read_hdf(file_map, key="df")

    def _code2phy_units(self, map_df: pd.DataFrame, quantity: str) -> pd.DataFrame:
        """ Convert from code units to physical units """
        if quantity in ["deflt_x", "deflt_y", "kappa_2"]:
            map_df.loc[:, [quantity]] /= c_light ** 2
        elif quantity in ["isw_rs"]:
            map_df.loc[:, [quantity]] /= c_light ** 3
        return map_df

    def _get_files(self, ray_nrs):
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
    
    def _h5_to_array(self, map_series: pd.Series,) -> np.ndarray:
        """ Convert pd.hf to np.ndarray """
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

    def _array_to_fits(self, map_out: np.ndarray) -> astropy.io.fits:
        """ Convert maps that in .npy format into .fits format """
        # Convert .npy to .fits
        data = fits.PrimaryHDU()
        data.header["ANGLE"] = self.opening_angle # [deg]
        data.data = map_out
        return data
    
    def _create_filename(
        self, file_in: str, quantity: str, dir_root: str, extension: str,
    ) -> str:
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
    ):
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
