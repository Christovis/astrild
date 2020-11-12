import os, sys, glob, copy
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
from importlib import import_module

import pandas as pd
import numpy as np
from scipy.interpolate import RectBivariateSpline
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.feature_extraction.image import reconstruct_from_patches_2d

import astropy
from astropy.io import fits
from astropy import units as un
import lenstools
from lenstools import ConvergenceMap
import pymaster as nmt

from wys_ars.simulation import Simulation
from wys_ars.rays.utils import Filters
from wys_ars.rays.skys.sky_utils import SkyUtils
from wys_ars.rays.skyio import SkyIO
from wys_ars.io import IO

dir_src = Path(__file__).parent.parent.absolute()
default_config_file_ray = dir_src / "configs/ray_snapshot_info.h5"

class SkyArrayWarning(BaseException):
    pass


class SkyArray:
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
        zoom:
        division:
        merge:
    """
    def __init__(
        self,
        skymap: np.ndarray,
        opening_angle: float,
        quantity: str,
        dirs: Dict[str, str],
        map_file: Optional[str] = None,
    ):
        print("------------------", skymap.shape)
        self.data = {"orig": skymap}
        self._npix = skymap.shape[0]
        self._opening_angle = opening_angle
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
        npix: Optional[int] = None,
        convert_unit: bool = True,
    ) -> "SkyArray":
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
        assert map_file, "There is no file being pointed at"

        file_extension = map_file.split(".")[-1]
        if file_extension == "h5":
            map_df = pd.read_hdf(map_file, key="df")
            return cls.from_dataframe(
                map_df, opening_angle, quantity, dir_in, map_file, npix, convert_unit,
            )
        elif file_extension in ["npy", "fits"]:
            if file_extension == "npy":
                map_array = np.load(map_file)
            elif file_extension == "fits":
                map_array = ConvergenceMap.load(map_file, format="fits").data
            return cls.from_array(
                map_array, opening_angle, quantity, dir_in, map_file,
            )
    
    @classmethod
    def from_dataframe(
        cls,
        map_df: pd.DataFrame,
        opening_angle: float,
        quantity: str,
        dir_in: str,
        map_file: str,
        npix: Optional[int] = None,
        convert_unit: bool = True,
    ) -> "SkyArray":
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
    ) -> "SkyArray":
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
        return cls(map_array, opening_angle, quantity, dirs, map_file)
   
    @property
    def npix(self) -> int:
        return self._npix
    
    @property
    def opening_angle(self) -> float:
        return self.opening_angle
    
    def pdf(self, nbins: int, of: str="orig") -> dict:
        _pdf = {}
        _pdf["values"], _pdf["bins"] = np.histogram(
            self.data[of], bins=nbins, density=True,
        )
        return _pdf

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
        _map = ConvergenceMap(data=_map, angle=self._opening_angle*un.deg)
        _kappa, _pos = _map.locatePeaks(map_bins)
        
        _hist, _kappa = np.histogram(_kappa, bins=nbins, density=False)
        _kappa = (_kappa[1:] + _kappa[:-1]) / 2
        peak_counts_dic = {"kappa": _kappa, "counts": _hist}
        peak_counts_df = pd.DataFrame(data=peak_counts_dic)
        return peak_counts_df

    def crop(
        self,
        xlimit: Union[tuple, list],
        ylimit: Union[tuple, list],
        of: Optional[str] = None,
        img: Optional[np.ndarray] = None,
        rtn: bool = False,
    ) -> np.ndarray:
        """
        Zoom into sky_array map.

        Args:
            xlimit and ylimit:
                Boundaries of zoom. If given in ints units are pixels,
                if floats percentages are used.
            of:
                skymap image identifier.
        """
        if of:
            img = copy.deepcopy(self.data[of])
        npix = img.shape[0]
        xlimit = np.asarray(xlimit)
        ylimit = np.asarray(ylimit)
        if isinstance(xlimit[0], float):
            xlimit = (npix * xlimit/100).astype(int)
            ylimit = (npix * ylimit/100).astype(int)
        zoom = img[int(xlimit[0]):int(xlimit[1]), int(ylimit[0]):int(ylimit[1])]
        if rtn:
            return zoom
        else:
            print(f"Image crop to x={xlimit} and y={ylimit}.")
            self.data[of] = zoom
            self._opening_angle = np.int(
                self._opening_angle * abs(np.diff(xlimit)) / self._npix
            )
            self._npix = zoom.shape[0]

    def division(
        self,
        ntiles: int,
        of: Optional[str] = None,
        img: Optional[np.ndarray] = None,
        rtn: bool = False,
    ) -> Union[List[np.ndarray], None]:
        """
        Divide image into tiles.
        Should use sklearn.feature_extraction.image.extract_patches_2d

        Args:
            ntiles:
                Nr. of tiles per edge (as to be in 2^n).
        """
        if of:
            img = copy.deepcopy(self.data[of])
        npix = img.shape[0]
        edges = list(np.arange(0, npix, npix/ntiles)) + [npix]
        edges = np.array(
            [edges[idx:idx+2] for idx in range(len(edges)-1)]
        ).astype(int)
        tiles = []
        for xlim in edges:
            for ylim in edges:
                tiles.append(self.crop(xlim, ylim, img=img, rtn=True))
        print(f"The image is divided into {len(tiles)} tiles.")
        tiles = np.asarray(tiles)
        if rtn:
            return tiles
        else:
            self.tiles = tiles

    def merge(
        self,
        tiles: np.ndarray,
        rtn: bool = False,
    ) -> Union[np.ndarray, None]:
        """
        Merge tiles created with self.division.
        Should use sklearn.feature_extraction.image.reconstruct_from_patches_2d

        Args:
            tiles:
                3D
        """
        ntiles = len(tiles)
        nrows = int(np.sqrt(ntiles))
        _parts = np.arange(0, ntiles+nrows, nrows)
        row_tile_idx = [(_parts[ii], _parts[ii+1]) for ii in range(nrows)]
        row_tiles = []
        for idx in range(nrows):
            start = row_tile_idx[idx][0]
            end = row_tile_idx[idx][1]
            row_tiles.append(np.hstack((tiles[start:end])))
        row_tiles = np.asarray(row_tiles)
        img = np.vstack((row_tiles))
        if rtn:
            return img

    def filter(
        self,
        filter_dsc: dict,
        on: Optional[str] = None,
        sky_array: Optional[np.ndarray] = None,
        rtn: bool = False,
    ) -> Union[np.ndarray, None]:
        """
        Convolve kernel (filter_dsc) over skymap image (on)
        Args:
            filter_dsc:
                Kernel description.
            on:
                skymap image identifier.
        """
        # load rays/utils/filters.py package for dynamic function call
        module = import_module("wys_ars.rays.utils")
        if on == "orig_gsn":
            _map = copy.deepcopy(self.add_galaxy_shape_noise())
        elif sky_array is None:
            _map = copy.deepcopy(self.data[on])
        else:
            _map = copy.deepcopy(sky_array)

        map_name = [on]
        for filter_name, args in filter_dsc.items():
            abbrev = args["abbrev"]
            del args["abbrev"]
            self.smoothing_length = args["theta_i"]

            clas = getattr(module, "Filters")
            fct = getattr(clas, filter_name)
            _map = fct(_map, self._opening_angle, **args)
            map_name.append(abbrev)
        if rtn:
            return _map
        else:
            self.data[("_").join(map_name)] = _map

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
        theta_pix = 60 * self._opening_angle / self._npix
        std_pix = 0.007 #np.sqrt(std ** 2 / (2*theta_pix*ngal))
        if rnd_seed is None:
            self.data["gsn"] = np.random.normal(
                loc=0, scale=std_pix, size=[self._npix, self._npix],
            )
        else:
            rg = np.random.Generator(np.random.PCG64(rnd_seed))
            self.data["gsn"] = rg.normal(
                loc=0, scale=std_pix, size=[self._npix, self._npix],
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
            raise SkyArrayWarning(f"GSN should not be added to {self.quantity}")

    def create_cmb(
        self,
        filepath_cl: str,
        lmax: int = 3e3,
        rnd_seed: Optional[int] = None,
        rtn: bool = False,
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
        if rnd_seed:
            np.random.seed(rnd_seed)
        Nx = Ny = self._npix
        Lx = Ly = self._opening_angle * np.pi / 180.

        cl_tt_cmb = np.load(filepath_cl)[1]
        cmb = nmt.synfast_flat(
            Nx, Ny, Lx, Ly,
            cls=np.array([cl_tt_cmb]),
            spin_arr=np.array([0])
        )[0]
        if rtn:
            return cmb
        else:
            self.data["cmb"] = cmb

    def add_cmb(
        self,
        filepath_cl: str,
        on: str = "orig",
        lmax: Optional[float] = None,
        rnd_seed: Optional[int] = None,
        rtn: bool = False,
        overwrite: bool = True,
    ) -> np.ndarray:
        """
        Args:
        """
        if self.quantity == "isw_rs":
            if "cmb" not in self.data.keys():
                self.create_cmb(filepath_cl, lmax, rnd_seed)
            _map = self.data[on] + self.data["cmb"]
            if rtn:
                return _map
            else:
                if overwrite:
                    self.data[on] = _map
                else:
                    self.data[f"{on}_cmb"] = _map
        else:
            raise SkyArrayWarning(f"CMB should not be added to {self.quantity}")
    
    def convert_convergence_to_deflection(
        self,
        on: Optional[str] = None,
        sky_array: Optional[np.ndarray] = None,
        rtn: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
        assert self.quantity in ["kappa_1", "kappa_2"], (
            "Deflection angle can only be calculated from the kappa map"
        )
        if sky_array is None:
            _map = copy.deepcopy(self.data[on])
        else:
            _map = copy.deepcopy(sky_array)
        alpha_1, alpha_2 = SkyUtils.convert_convergence_to_deflection(
            _map, self._npix, self._opening_angle
        )
        if rtn:
            return alpha_1, alpha_2
        else:
            self.data["defltx"] = alpha_2
            self.data["deflty"] = alpha_1
