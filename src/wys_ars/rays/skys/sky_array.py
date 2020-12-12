import os, sys, glob, copy
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, TypeVar
from importlib import import_module

import pandas as pd
import numpy as np
import numba as nb
from skimage import transform

from joblib import Parallel, delayed
from multiprocessing import cpu_count

import astropy
from astropy.constants import sigma_T, m_p, c
from astropy.io import fits
from astropy import units as un
import lenstools
from lenstools import ConvergenceMap
#import pymaster as nmt

from wys_ars.rays.utils import Filters
from wys_ars.rays.skys.sky_utils import SkyUtils, SkyNumbaUtils
from wys_ars.rays.skyio import SkyIO
from wys_ars.io import IO

#import matplotlib as mpl
#mpl.use('Agg')
#import matplotlib.pyplot as plt

dir_src = Path(__file__).parent.parent.absolute()
default_config_file_ray = dir_src / "configs/ray_snapshot_info.h5"

sigma_T = sigma_T.to(un.Mpc**2).value #[Mpc^2]
m_p = m_p.to(un.M_sun).value #[M_sun]
c_light = c.to("km/s").value
T_cmb = 2.7251 #[K]
Gcm2 = 4.785E-20 # G/c^2 (Mpc/M_sun)

# store available nr. of cpus for parallel computation
ncpus_available = cpu_count()

class SkyArrayWarning(BaseException):
    pass


class SkyArray:
    """
    The sky-map is constructed through multiple ray-tracing simulations
    run with RayRamses. This class analyzes the 2D map that contains
    the summes pertrurbations of each ray. It can prepare the data for the
    search of voids and peaks.

    Attributes:
        skymap:
        opening_angle: [deg]
        quantity:
        dirs:
        map_file:

    Methods:
        from_file:
        from_dataframe:
        from_array:
        from_halo_to_deflection_angle_map:
        from_halo_to_temperature_perturbation_map:
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
        convert_convergence_to_deflection:
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
            opening_angle: [deg]
        """
        assert map_file, "There is no file being pointed at"

        file_extension = map_file.split(".")[-1]
        if file_extension == "h5":
            map_df = pd.read_hdf(map_file, key="df")
            return cls.from_dataframe(
                map_df,
                opening_angle,
                quantity,
                dir_in,
                map_file,
                npix,
                convert_unit,
            )
        elif file_extension in ["npy", "fits"]:
            if file_extension == "npy":
                map_array = np.load(map_file)
            elif file_extension == "fits":
                map_array = ConvergenceMap.load(map_file, format="fits").data
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
        npix: Optional[int] = None,
        convert_unit: bool = True,
    ) -> "SkyArray":
        """
        Initialize class by reading the skymap data from pandas DataFrame. 

        Args:
            map_filename:
                File path with which skymap pd.DataFrame can be loaded.
            opening_angle: [deg]
        """
        if convert_unit:
            map_df = SkyUtils.convert_code_to_phy_units(quantity, map_df)
        map_array = SkyIO.transform_RayRamsesOutput_to_NumpyNdarray(
            map_df[quantity].values
        )
        return cls.from_array(
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
    ) -> "SkyArray":
        """
        Initialize class by reading the skymap data from np.ndarray.

        Args:
            map_filename:
                File path with which skymap pd.DataFrame can be loaded.
            opening_angle: [deg]
        """
        assert map_array.shape[0] == map_array.shape[1]
        dirs = {"sim": dir_in}
        return cls(map_array, opening_angle, quantity, dirs, map_file)
    

    @classmethod
    def from_halo_to_deflection_angle_map(
        cls,
        theta_200c: float,
        M_200c: float,
        c_200c: float,
        angu_diam_dist: Optional[float] = None,
        npix: int = 100,
        extent: float = 1,
        direction: List[int] = [0, 1],
        suppress: bool = False,
        suppression_R: float = 1,
    ) -> "SkyArray":
        """
        Calculate the deflection angle of a halo with NFW profile using method
        in described Sec. 3.2 in Baxter et al 2015 (1412.7521).

        Note:
            In this application it can be assumed that s_{SL}/s_{S}=1. Furthermore,
            we can deglect vec{theta}/norm{theta} as it will be multiplied by the
            same in the integral of Eq. 9 in Yasini et a. 2018 (1812.04241).
        
        Args:
            theta_200c: radius, [deg]
            M_200c: mass, [Msun]
            c_200c: concentration, [-]
            extent: The size of the map from which the trans-vel is calculated
                in units of R200 of the associated halo.
            suppress:
            suppression_R:
            angu_diam_dist: angular diameter distance, [Mpc]
            direction: 0=(along x-axis), 1=(along y-axis), if 0 and 1 are given
                the sum of both maps is returned.
        """
        alpha_map = SkyUtils.NFW_deflection_angle_map(
            theta_200c,
            M_200c,
            c_200c,
            angu_diam_dist,
            npix,
            extent,
            direction,
            suppress,
            suppression_R,
        )
        opening_angle = 2 * theta_200c * extent
        if 1 in direction and 0 in direction:
            quantity = "alpha"
        elif 0 in direction:
            quantity = "alphax"
        else:
            quantity = "alphay"
        return cls(alpha_map, opening_angle, quantity, dirs=None, map_file=None)
    

    @classmethod
    def from_halo_to_temperature_perturbation_map(
        cls,
        theta_200c: float,
        M_200c: float,
        c_200c: float,
        vel: Union[list, tuple, np.ndarray],
        angu_diam_dist: Optional[float] = None,
        extent: float = 1,
        direction: List[int] = [0, 1],
        suppress: bool = False,
        suppression_R: float = 1,
        npix: int = 100,
        ncpus: int = 1,
    ) -> "SkyArray":
        """
        The Rees-Sciama / Birkinshaw-Gull / moving cluster of galaxies effect.

        Args:
            vel: transverse to the line-of-sight velocity, [km/sec]

        Returns:
            Temperature perturbation map, \Delta T / T_CMB
        """
        dt_map = SkyUtils.NFW_deflection_angle_map(
            theta_200c,
            M_200c,
            c_200c,
            angu_diam_dist,
            npix,
            extent,
            direction,
            suppress,
            suppression_R,
        )
        opening_angle = 2 * theta_200c * extent
        if 1 in direction and 0 in direction:
            quantity = "isw_rs"
        elif 0 in direction:
            quantity = "isw_rs_x"
        else:
            quantity = "isw_rs_y"
        return cls(dt_map, opening_angle, quantity, dirs=None, map_file=None)
    

    @classmethod
    def from_halo_catalogue_to_temperature_perturbation_map(
        cls,
        halo_cat: pd.DataFrame,
        extent: float = 1,
        direction: List[int] = [0, 1],
        suppress: bool = False,
        suppression_R: float = 1,
        npix: int = 8192,
        opening_angle: float = 20.,
        ncpus: int = 1,
    ) -> "SkyArray":
        """
        The Rees-Sciama / Birkinshaw-Gull / moving cluster of galaxies effect.

        Args:
            vel: transverse to the line-of-sight velocity, [km/sec]

        Returns:
            Temperature perturbation map, \Delta T / T_CMB
        """
        map_array = np.zeros((npix, npix))
        
        halo_dict = halo_cat[[
            "r200_deg",
            "r200_pix",
            "m200",
            "c_NFW",
            "rad_dist",
            "theta1_pix",
            "theta2_pix",
            "theta1_vel",
            "theta2_vel",
        ]].to_dict(orient='list')
        halo_idx = range(len(halo_dict["m200"]))
        
        if ncpus == 1:
            map_array = SkyUtils.analytic_Halo_signal_to_SkyArray(
                halo_idx, halo_dict, map_array, extent, direction, suppress, suppression_R,
            )
        else:
            halo_idx_batches = np.array_split(halo_idx, ncpus)
            map_sub_arrays = Parallel(n_jobs=ncpus, require='sharedmem')(
                delayed(SkyUtils.analytic_Halo_signal_to_SkyArray)(
                    halo_idx_batch,
                    halo_dict,
                    map_array,
                    extent,
                    direction,
                    suppress,
                    suppression_R,
                ) for halo_idx_batch in halo_idx_batches
            )
            map_array = sum(map_sub_arrays)
            print(type(map_array), map_array.shape, map_array.min(), map_array.max())

        if 1 in direction and 0 in direction:
            quantity = "isw_rs"
        elif 0 in direction:
            quantity = "isw_rs_x"
        else:
            quantity = "isw_rs_y"
        return cls(map_array, opening_angle, quantity, dirs=None, map_file=None)
  
    
    @property
    def ncpus(self):
        return self._ncpus


    @ncpus.setter
    def ncpus(self, val: int):
        if (ncpus == 0) or (ncpus < -1):
            raise ValueError(
                f"ncpus={ncpus} is not valid. Please enter a value " +\
                ">0 for ncpus or -1 to use all available cores."
            )
        elif ncpus == -1:
            self._ncpus = ncpus_available
        else:
            self._ncpus = val


    @property
    def npix(self) -> int:
        return self._npix

    @property
    def opening_angle(self) -> float:
        return self._opening_angle


    def pdf(self, nbins: int, of: str = "orig") -> dict:
        _pdf = {}
        _pdf["values"], _pdf["bins"] = np.histogram(
            self.data[of], bins=nbins, density=True
        )
        return _pdf

    def wl_peak_counts(
        self,
        nbins: int,
        field_conversion: str,
        of: str = "orig",
        limits: Optional[tuple] = None,
    ) -> pd.DataFrame:
        """
        Signal peak counts. This is used commonly used in weak-lensing,
        but it doesn't need to stop there...
        """
        if field_conversion == "normalize":
            _map = self.data[of] - np.mean(self.skymap.data[of])
        else:
            _map = self.data[of]

        if limits is None:
            lower_bound = np.percentile(
                self.data[of], 5
            )  # np.min(self.data[of])
            upper_bound = np.percentile(
                self.data[of], 95
            )  # np.max(self.data[of])
        else:
            lower_bound = min(limits)
            upper_bound = max(limits)

        map_bins = np.arange(
            lower_bound, upper_bound, (upper_bound - lower_bound) / nbins
        )
        _map = ConvergenceMap(data=_map, angle=self._opening_angle * un.deg)
        _kappa, _pos = _map.locatePeaks(map_bins)

        _hist, _kappa = np.histogram(_kappa, bins=nbins, density=False)
        _kappa = (_kappa[1:] + _kappa[:-1]) / 2
        peak_counts_dic = {"kappa": _kappa, "counts": _hist}
        peak_counts_df = pd.DataFrame(data=peak_counts_dic)
        return peak_counts_df


    def resize(
        self,
        npix,
        of: Optional[str] = None,
        img: Optional[np.ndarray] = None,
        rtn: bool = False,
        orig_data: str = None,
    ) -> Union[np.ndarray, None]:
        """
        Lower the nr. of pixels of image. Useful for tests.
        
        Args:
            npix: the new pixel nr. per edge of the image
            of: skymap image identifier.
            orig_data: What to do with data of the image to be processed:
                e.g. no, shallow, or deep copy
        """
        img = self._manage_img_data(img, orig_data)
        img = transform.resize(img, (npix, npix), anti_aliasing=True)
        if rtn:
            return img
        else:
            self.data[of] = img

    
    def crop(
        self,
        xlimit: Union[Tuple, List, np.array],
        ylimit: Union[Tuple, List, np.array],
        of: Optional[str] = None,
        img: Optional[np.ndarray] = None,
        rtn: bool = False,
        orig_data: str = None,
    ) -> Union[np.ndarray, None]:
        """
        Zoom into sky_array map.

        Args:
            x,ylimit: Boundaries of zoom. If given in ints units are pixels,
                if floats percentages are used.
            of: skymap image identifier.
            orig_data: What to do with data of the image to be processed:
                e.g. no, shallow, or deep copy
        """
        if of:
            assert of in list(self.data.keys()), "Map does not exist."
            img = self.data[of]
        
        img = self._manage_img_data(img, orig_data)
        xlimit = np.asarray(xlimit)
        ylimit = np.asarray(ylimit)
        assert np.diff(xlimit) == np.diff(ylimit), SkyArrayWarning("The whole class is currently designed for square images.")
        if isinstance(xlimit[0], float):
            _npix = img.shape[0]
            xlimit = (_npix * xlimit / 100).astype(int)
            ylimit = (_npix * ylimit / 100).astype(int)
        zoom = img[xlimit[0] : xlimit[1], ylimit[0] : ylimit[1]]
        if rtn:
            return zoom
        else:
            print(f"Image crop to x={xlimit} and y={ylimit}.")
            self.data[of] = zoom
            self._opening_angle = (
                self._opening_angle * abs(np.diff(xlimit)) / self._npix
            )
            self._npix = zoom.shape[0]


    def division(
        self,
        ntiles: int,
        of: Optional[str] = None,
        img: Optional[np.ndarray] = None,
        rtn: bool = False,
        orig_data: str = None,
    ) -> Union[List[np.ndarray], None]:
        """
        Divide image into tiles.
        Should use sklearn.feature_extraction.image.extract_patches_2d

        Args:
            ntiles: Nr. of tiles per edge (as to be in 2^n).
            orig_data: What to do with data of the image to be processed:
                e.g. no, shallow, or deep copy
        """
        img = self._manage_img_data(img, orig_data)
        npix = img.shape[0]
        edges = list(np.arange(0, npix, npix / ntiles)) + [npix]
        edges = np.array(
            [edges[idx : idx + 2] for idx in range(len(edges) - 1)]
        ).astype(int)
        tiles = []
        for xlim in edges:
            for ylim in edges:
                tiles.append(self.crop(xlim, ylim, img=img, rtn=True))
        print(
            f"The image is divided into {len(tiles)} tiles, " +\
            f"each with {tiles[0].shape[0]}^2 pixels."
        )
        tiles = np.asarray(tiles)
        if rtn:
            return tiles
        else:
            self.tiles = tiles
            self._tile_npix = tiles[0].shape[0]
            self._tile_opening_angle = self._opening_angle * self._tile_npix / self._npix


    def merge(
        self, tiles: np.ndarray, rtn: bool = False
    ) -> Union[np.ndarray, None]:
        """
        Merge tiles created with self.division.
        Should use sklearn.feature_extraction.image.reconstruct_from_patches_2d

        Args:
            tiles: 3D
        """
        ntiles = len(tiles)
        nrows = int(np.sqrt(ntiles))
        _parts = np.arange(0, ntiles + nrows, nrows)
        row_tile_idx = [(_parts[ii], _parts[ii + 1]) for ii in range(nrows)]
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
        img: Optional[np.ndarray] = None,
        rtn: bool = False,
        orig_data: str = None,
    ) -> Union[np.ndarray, None]:
        """
        Apply kernel (filter_dsc) over skymap image (on).
        
        Args:
            filter_dsc: Kernel description.
                Note that theta_i should be given in astropy.units!
            on: skymap image identifier.
            orig_data: What to do with data of the image to be processed:
                e.g. no, shallow, or deep copy
        """
        if on:
            assert on in list(self.data.keys()), "Map does not exist."
            img = self.data[on]
            map_name = [on]
        else:
            map_name = [""]
        
        # load rays/utils/filters.py package for dynamic function call
        module = import_module("wys_ars.rays.utils")
        img = self._manage_img_data(img, orig_data)

        for filter_name, args in filter_dsc.items():
            if rtn is False:
                abbrev = args["abbrev"]
                del args["abbrev"]
                map_name.append(abbrev)

            clas = getattr(module, "Filters")
            fct = getattr(clas, filter_name)
            img = fct(img, self._opening_angle * un.deg, **args)
        if rtn:
            return img
        else:
            self.data[("_").join(map_name)] = img


    def create_galaxy_shape_noise(
        self, std: float, ngal: float, rnd_seed: Optional[int] = None
    ) -> None:
        """
        Galaxy Shape Noise (GSN), e.g.: arxiv:1907.06657

        Args:
            std: dispersion of source galaxy intrinsic ellipticity, 0.4 for LSST
            ngal: Nr. density of galaxies, 40 for LSST; [arcmin^2]
            rnd_seed: Fix random seed, for reproducability.
        Returns:
            gsn_map:
                self.npix x self.npix np.array containing the GSN
        """
        theta_pix = 60 * self._opening_angle / self._npix
        std_pix = 0.007  # np.sqrt(std ** 2 / (2*theta_pix*ngal))
        if rnd_seed is None:
            self.data["gsn"] = np.random.normal(
                loc=0, scale=std_pix, size=[self._npix, self._npix]
            )
        else:
            rg = np.random.Generator(np.random.PCG64(rnd_seed))
            self.data["gsn"] = rg.normal(
                loc=0, scale=std_pix, size=[self._npix, self._npix]
            )
        print(f"The GSN map sigma is {np.std(self.data['gsn'])}", std_pix)


    def add_galaxy_shape_noise(self, on: str = "orig") -> np.ndarray:
        """
        Add GSN on top of skymap.
        
        Args:
            std: dispersion of source galaxy intrinsic ellipticity, 0.4 for LSST
            ngal: Nr. density of galaxies, 40 for LSST; [arcmin^2]
            rnd_seed: Fix random seed, for reproducability.
        """
        if "kappa" in self.quantity:
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
            filepath_cl: angular power spectrum of CMB
            theta: Edge length of the square field-of-view [deg]
            nside: Nr. of pixels per edge of the output full-sky map
            rnd_seed: Fix random seed, for reproducability.

        Returns:
            cmb_map:
        """
        if rnd_seed:
            np.random.seed(rnd_seed)
        Nx = Ny = self._npix
        Lx = Ly = self._opening_angle * np.pi / 180.0

        cl_tt_cmb = np.load(filepath_cl)[1]
        #cmb = nmt.synfast_flat(
        #    Nx, Ny, Lx, Ly, cls=np.array([cl_tt_cmb]), spin_arr=np.array([0])
        #)[0]
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
        if "isw" in self.quantity:
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
        img: Optional[np.ndarray] = None,
        npix: Optional[int] = None,
        opening_angle: Optional[float] = None,
        rtn: bool = True,
        orig_data: str = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Args:
            on: String to indicate which map in self.data should be used.
            img: 2D convergence map.
            rtn: Bool to indicate whether to attach result to class object or return.
            orig_data: What to do with data of the image to be processed:
                e.g. no, shallow, or deep copy

        Returns:
            alpha_1,2: 2D deflection angle map [rad]
        """
        #TODO handle tiles/multiple images
        assert self.quantity in [
            "kappa_1",
            "kappa_2",
        ], "Deflection angle can only be calculated from the kappa map"
        img = self._manage_img_data(img, orig_data)

        if npix is None: npix = self._npix
        if opening_angle is None: opening_angle = self._opening_angle

        alpha_1, alpha_2 = SkyUtils.convert_convergence_to_deflection_ctypes(
            img, npix, opening_angle * un.deg
        )
        if rtn:
            return alpha_2, alpha_1
        else:
            self.data["defltx"] = alpha_2
            self.data["deflty"] = alpha_1


    def convert_deflection_to_shear(
        self,
        on: Optional[str] = None,
        img: Optional[np.ndarray] = None,
        rtn: bool = False,
        orig_data: str = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Args:
            on: String to indicate which map in self.data should be used.
            img: 2D convergence map.
            rtn: Bool to indicate whether to attach result to class object or return.
            orig_data: What to do with data of the image to be processed:
                e.g. no, shallow, or deep copy

        Returns:
            gamma_1,2: 2D shear map [-]
        """
        assert self.quantity in [
            "alpha"
        ], "Shear can only be calculated from the deflection angle map"
        img = self._manage_img_data(img, orig_data)
        gamma_1, gamma_2 = SkyUtils.convert_deflection_to_shear(
            img, self._npix, self._opening_angle * un.deg
        )
        if rtn:
            return gamma_2, gamma_1
        else:
            self.data["gammax"] = gamma_2
            self.data["gammay"] = gamma_1
    

    @staticmethod
    def _manage_img_data(
        img: np.ndarray,
        orig_data: str = None,
    ) -> np.ndarray:
        """
        Handle the memory location of the image to be processed.

        Args:
            img: memory pointer of the image data.
            orig_data: action key word.

        Returns:
            cimg: pointer to (new) memory location.
        """
        if orig_data == "shallow":
            cimg = copy.copy(img)
        elif orig_data == "deep":
            cimg = copy.deepcopy(img)
        else:
            cimg = img
        return cimg
