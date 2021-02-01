import time, copy
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type, Union

import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy import integrate
from sklearn.neighbors import NearestNeighbors

from joblib import Parallel, delayed
import multiprocessing

from astropy import units as un
from astropy.constants import c as c_light
from lenstools import ConvergenceMap

from astrild.io import IO
from astrild.rays.skys import SkyArray
from astrild.rays.utils import object_selection

default_filter_dipole_identification = {
    "gaussian_high_pass": {"abbrev": "hpf", "theta_i": 1},
    "gaussian_third_derivative": {"abbrev": "3df", "direction": 1},
    "gaussian_low_pass": {"abbrev": "lpf", "theta_i": 1},
}
default_filter_dipole_vel_tx = {
    "gaussian_high_pass": {"fwhm_i": 5 * un.arcmin},  # [arcmin]
    "gaussian_third_derivative": {"theta_i": 1, "direction": 1},
    "apodization": {"theta_i": None},
}
default_filter_dipole_vel_ty = {
    "gaussian_high_pass": {"fwhm_i": 5 * un.arcmin},  # [arcmin]
    "gaussian_third_derivative": {"theta_i": 1, "direction": 0},
    "apodization": {"theta_i": None},
}

# store available nr. of cpus for parallel computation
ncpus_available = multiprocessing.cpu_count()


class DipolesWarning(BaseException):
    """Base class for other exceptions"""

    pass


class Dipoles:
    """
    Class to find and analyse dipol signals.

    Attributes:
        dipoles:
            pd.DataFrame of all identified dipoles on the map.

    Methods:
        from_file:
        from_dataframe:
        from_sky:
        _filter:  <- remove this function
        _get_convergence_thresholds:
        _signal_to_noise_ratio:
        _remove_peaks_crossing_edge:
        _get_index_and_distance:
        find_nearest:
        find_nearest_in_box:
        find_nearest_in_snap:
        _get_index_and_distance:
        get_transverse_velocities_from_sky:
        get_single_transverse_velocity_from_sky:
    """

    def __init__(self, dipoles: pd.DataFrame):
        self.data = dipoles
        self._ncpus = 1

    @classmethod
    def from_sky(
        cls,
        skymap: Type[SkyArray],
        on: str,
        bin_dsc: dict,
        kernel_width: float = 5,
        direction: int = 1,
        filters: bool = True,
    ) -> "Dipoles":
        """
        Find peaks on the dipole signal map. It is assumed that the convergence maps
        were created with astrild.rays.visuals.map and filter with:
            I) high-pass II) DGD3 III) low-pass gaussian filters.

        Args:
            kernel_width: Smoothing kernel with [arcmin]
        Returns:
        """
        if filters is True:
            skymap = cls._filter(skymap, kernel_width, direction)

        thresholds = cls._get_convergence_thresholds(
            sky_array=skymap.data[bin_dsc["on"]], nbins=bin_dsc["nbins"]
        )

        _map = ConvergenceMap(
            data=skymap.data[on], angle=skymap.opening_angle * un.deg
        )
        deltaT, pos_deg = _map.locatePeaks(thresholds)
        deltaT, pos_deg = cls._remove_peaks_crossing_edge(
            skymap.npix, skymap.opening_angle, kernel_width, deltaT, pos_deg
        )
        assert len(deltaT) != 0, "No peaks"
        peak_dir = {
            "deltaT": deltaT,
            "x_deg": pos_deg[:, 0],
            "y_deg": pos_deg[:, 1],
        }

        # find significance of peaks
        peak_dir["snr"] = cls._signal_to_noise_ratio(
            peak_dir["deltaT"], _map.data
        )
        peak_dir["x_pix"] = np.rint(
            peak_dir["x_deg"] * skymap.npix / skymap.opening_angle
        ).astype(int)
        peak_dir["y_pix"] = np.rint(
            peak_dir["y_deg"] * skymap.npix / skymap.opening_angle
        ).astype(int)
        peak_df = pd.DataFrame(data=peak_dir)
        # attrs is experimental and may change without warning.
        peak_df.attrs["map_file"] = skymap.map_file
        peak_df.attrs["filters"] = filters
        peak_df.attrs["kernel_width"] = kernel_width
        return cls.from_dataframe(peak_df)

    @classmethod
    def from_file(cls, filename_dip: str) -> "Dipoles":
        """ Create class from pd.DataFrame file """
        df = pd.read_hdf(filename_dip, key="df")
        return cls.from_dataframe(df)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> "Dipoles":
        """ Create class from pd.DataFrame file """
        return cls(df)

    @classmethod
    def _filter(
        cls,
        skymap: Type[SkyArray],
        kernel_width: float,
        filter_dsc_x: dict = default_filter_dipole_identification,
    ) -> Type[SkyArray]:
        """
        Prepre skymap for dipole detection.
        Note: This works only for skymap.data["orig"]
        """
        # TODO: update to new code
        # prepare map for dipole detection
        _map = skymap.filter(filter_dsc, on="orig", rtn=True)
        skymap.data["orig_hpf_3df_lpf"] = skymap.filter(
            filter_dsc, sky_array=np.abs(_map), rtn=True
        )
        return skymap

    @classmethod
    def _get_convergence_thresholds(
        cls, sky_array: np.ndarray, nbins: int = 100
    ) -> np.array:
        """
        Define thresholds for lenstools to find peaks on convergence map.
        Important to do this on un-smoothed and with no-gsn skymap.
        """
        bins = np.arange(
            np.min(sky_array),
            np.max(sky_array) * 1.1,
            (np.max(sky_array) * 1.1 - np.min(sky_array)) / nbins,
        )
        return bins

    @classmethod
    def _signal_to_noise_ratio(
        cls,
        peak_values: np.ndarray,
        map_values: np.ndarray,
        sigma: Optional[float] = None,
    ) -> np.ndarray:
        """
        Assess signifance of peaks and remove peaks suffereing edge effects

        Args:
        """
        _kappa_std = np.std(map_values)
        snr = peak_values / _kappa_std
        return snr

    @classmethod
    def _remove_peaks_crossing_edge(
        cls,
        npix: int,
        opening_angle: float,
        kernel_width: float,
        sigma: np.ndarray,
        pos: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Args:
            sigma:
            pos: Peak x,y-positions [deg]

        Returns:
            sigma and pos
        """
        pixlen = opening_angle / npix  # [deg]
        bufferlen = np.ceil(
            kernel_width / (60 * pixlen)
        )  # length of buffer zone
        # convert degrees to pixel number
        x = pos[:, 0].value * npix / opening_angle
        y = pos[:, 1].value * npix / opening_angle
        indx = np.logical_and(
            np.logical_and(x <= npix - 1 - bufferlen, x >= bufferlen),
            np.logical_and(y <= npix - 1 - bufferlen, y >= bufferlen),
        )
        sigma = sigma[indx]
        pos = pos[indx, :]
        print(
            f"{len(indx)} peaks were within kernel_width of FOV edge"
            + f" and had to be removed. \n{len(sigma)} peaks are left."
        )
        return sigma, pos

    def find_nearest(
        self,
        df2: pd.DataFrame,
        column_labels: dict,
        column_add: Optional[list] = None,
    ) -> None:
        """
        Args:
            df2: DataFrame of object that should be associated with dipoles.
            columns: Identify columns in df2 that should be attached to self.data
        """
        if "box_nr" in self.data.columns.values:
            distances, ids = self.find_nearest_in_box(self.data, df2)
        elif "snap_nr" in self.data.columns.values:
            distances, ids = self.find_nearest_in_snap(self.data, df2)
        else:
            # find corresponding group-halo to dipole signal
            distances, ids = self._get_index_and_distance(self.data, df2)
        self.data[column_labels["id"]] = ids
        self.data[column_labels["dist"]] = distances

        if column_add is not None:
            for key in column_add:
                assert key not in self.data.columns.values
                self.data[key] = [
                    df2[df2["id"] == i][key].values[0]
                    if i != -99999
                    else -99999
                    for i in ids
                ]


    def find_nearest_in_box(
        self, df1: pd.DataFrame, df2: pd.DataFrame
    ) -> tuple:
        """
        This function has the same purpose as self.find_nearest(), but is for
        simulation box.
        
        Args:
            df1, df2: pd.DataFrames of objects that should be associated with 
            each other.
        """
        distances = []
        ids = []
        for box_nr in np.unique(df1["box_nr"].values):
            df1_in_box = df1[df1["box_nr"] == box_nr]
            df2_in_box = df2[df2["box_nr"] == box_nr]
            if "ray_nr" in df1_in_box.columns.values:
                _distances, _indices = self.find_nearest_in_snap(
                    df1_in_box, df2_in_box
                )
            else:
                _distances, _indices = self._get_index_and_distance(
                    df1_in_box, df2_in_box
                )
            distances += _distances
            ids += _indices
        return distances, ids


    def find_nearest_in_snap(
        self, df1: pd.DataFrame, df2: pd.DataFrame
    ) -> tuple:
        """
        This function has the same purpose as self.find_nearest(), but is for
        simulation snapshot.

        Args:
            df1, df2: pd.DataFrames of objects that should be associated with 
            each other.
        """
        distances = []
        ids = []
        for ray_nr in np.unique(df1["ray_nr"].values):
            df1_in_snap = df1[df1["ray_nr"] == ray_nr]
            df2_in_snap = df2[df2["ray_nr"] == ray_nr]
            _distances, _indices = self._get_index_and_distance(
                df1_in_snap, df2_in_snap
            )
            distances += _distances
            ids += _indices
        return distances, ids


    def _get_index_and_distance(
        self, df1: pd.DataFrame, df2: pd.DataFrame
    ) -> tuple:
        """
        Args:
            df1, df2: pd.DataFrames of objects that should be associated with 
            each other.
        
        Returns:
            distances:
            ids:
        """
        if len(df2.index.values) == 0:
            print("There are no Halos in this distance")
            distances = np.ones(len(df1.index.values)) * -99999
            ids = np.ones(len(df1.index.values)) * -99999
        else:
            nbrs = NearestNeighbors(n_neighbors=1, algorithm="ball_tree").fit(
                df2[["x_deg", "y_deg"]].values
            )
            distances, indices = nbrs.kneighbors(df1[["x_deg", "y_deg"]].values)
            distances = distances.T[0]
            ids = df2["id"].values[indices.T[0]].astype(int)
            if len(ids) > len(np.unique(ids)):  # handle dublications
                nan_idx = []
                for u_id in np.unique(ids):
                    _idx = np.where(ids == u_id)[0]
                    min_idx = np.argmin(distances[_idx])
                    nan_idx += list(np.delete(_idx, min_idx))
                ids[nan_idx] = -99999
                distances[nan_idx] = -99999
        return list(distances), list(ids)


    @property
    def ncpus(self):
        return self._ncpus


    @ncpus.setter
    def ncpus(self, val: int):
        if (val == 0) or (val < -1):
            raise ValueError(
                f"ncpus={ncpus} is not valid. Please enter a value " +\
                ">0 for ncpus or -1 to use all available cores."
            )
        elif val == -1:
            self._ncpus = ncpus_available
        else:
            self._ncpus = val
    
    
    def get_transverse_velocities_from_sky(
        self,
        skyarrays: Dict[str, Type[SkyArray]],
        extend: float,
        ncpus: int = 1,
        filter_dsc_x: dict = default_filter_dipole_vel_tx,
        filter_dsc_y: dict = default_filter_dipole_vel_ty,
    ) -> None:
        """
        Measure the transverse dipole/halo velocity (mtv) through the temperature
        perturbation map, by using the dipole signal of a moving potential.

        Args:
            skyarrays: Dictionary must contains SkyArray instances of
                unfiltered isw-rs map [-] and deflection angle map [rad].
                The maps can contain either the individual x,y-components
                of the vector fields or their sum.
            extend: The size of the map from which the trans-vel is calculated
                in units of R200 of the associated halo.
            ncpus: number of cpus to use for parallel processing. if -1 than all
                available cpus are used.
            filter_dsc_x,y: Dictionary of filters applied to cropped map
                in x,y-direction.
        
        Returns:
        """
        assert any("isw_rs" in s for s in list(skyarrays.keys()))
        assert any("alpha" in s for s in list(skyarrays.keys()))
        self._ncpus = ncpus
        
        def copy_or_sort_keys(keys: str, get: str) -> list:
            _kk = [k for k in keys if get in k]
            if len(_kk) == 1:
                keys_f = [_kk[0]] * 2
            else:
                # make sure that the list contains first
                # the _x and then _y component
                keys_f = sorted(_kk)
            return keys_f

        def integration(dip: pd.Series,) -> tuple:
            # get image which will be integrated to find dipole transverse vel
            dT_zoom = [
                self.get_dipole_image(
                    skyarrays[k],
                    (dip.theta1_pix, dip.theta2_pix),
                    dip.r200_pix * extend,
                    dip.r200_deg * extend,
                )
                for k in keys_isw_rs
            ]
            alpha_zoom = [
                self.get_dipole_image(
                    skyarrays[k],
                    (dip.theta1_pix, dip.theta2_pix),
                    dip.r200_pix * extend,
                    dip.r200_deg * extend,
                )
                for k in keys_alpha
            ]
            # centre dT-map on average temperature within R200
            [
                dT_zoom[k].filter(
                    filter_dsc={
                        "aperture_photometry": {"abbrev": "ap", "alpha": dip.r200_deg * un.deg}
                    },
                    on="orig",
                    rtn=False,
                    orig_data="shallow",
                )
                for k in range(len(dT_zoom))
            ]
            # filter images to remove CMB+Noise and enhance dipole signal
            filter_dsc_x["gaussian_third_derivative"]["theta_i"] = (
                1 * dip.r200_deg * un.deg
            )
            filter_dsc_y["gaussian_third_derivative"]["theta_i"] = (
                1 * dip.r200_deg * un.deg
            )
            for idx, filtr in enumerate([filter_dsc_x, filter_dsc_y]):
                dT_zoom[idx] = dT_zoom[idx].filter(
                    filtr, on="orig_ap", rtn=True
                )
                alpha_zoom[idx] = alpha_zoom[idx].filter(
                    filtr, on="orig", rtn=True
                )
            return self.get_single_transverse_velocity_from_sky(
                dT_zoom[0], dT_zoom[1], alpha_zoom[0], alpha_zoom[1],
            )
       
        keys_isw_rs = copy_or_sort_keys(list(skyarrays.keys()), "isw_rs")
        keys_alpha = copy_or_sort_keys(list(skyarrays.keys()), "alpha")
        
        _array_of_failures = np.ones(len(self.data.index)) * -99999

        if "theta1_mtvel" in self.data.columns.values:
            _x_vel = self.data["theta1_mtvel"].values
            _y_vel = self.data["theta2_mtvel"].values
            self.data = self.data.drop(["theta1_mtvel", "theta2_mtvel"], axis=1)
        else:
            _x_vel = copy.deepcopy(_array_of_failures)
            _y_vel = copy.deepcopy(_array_of_failures)

        dip_index = self._get_index_of_dip_far_from_edge(
            extend, skyarrays[keys_isw_rs[0]].npix,
        )
        if len(dip_index) == 0:
            self.data["theta1_mtvel"] = _array_of_failures
            self.data["theta2_mtvel"] = _array_of_failures
        
        else:
            print(
                f"Calculate the trans. vel. of {len(dip_index)} dipoles " +\
                f"with {self._ncpus} cpus"
            )
            if self._ncpus == 1:
                _vt = []
                for idx, dip in self.data.iloc[dip_index].iterrows():
                    _vt.append(integration(dip))
            else:
                _vt = Parallel(n_jobs=self._ncpus)(
                    delayed(integration)(dip)
                    for idx, dip in self.data.iloc[dip_index].iterrows()
                )
            
            _vt = np.asarray(_vt).T
            _x_vel[dip_index] = _vt[0]
            _y_vel[dip_index] = _vt[1]
            self.data["theta1_mtvel"] = _x_vel
            self.data["theta2_mtvel"] = _y_vel


    def _get_index_of_dip_far_from_edge(
        self, extend: float, npix: int
    ) -> np.array:
        """
        Args:
            extend:
            npix: Nr. of pixels on edge of SkyMap
        """
        return object_selection.trim_dataframe_of_objects_crossing_edge(
            self.data, extend, npix, key_size="r200_pix", rtn="index"
        )


    @staticmethod
    def get_dipole_image(
        img: Type[SkyArray], cen_pix: tuple, extend_pix: int, extend_deg: float
    ) -> Type[SkyArray]:
        xlim = np.array(
            [cen_pix[1] - extend_pix, cen_pix[1] + extend_pix]
        ).astype(int)
        ylim = np.array(
            [cen_pix[0] - extend_pix, cen_pix[0] + extend_pix]
        ).astype(int)
        img_zoom = SkyArray.from_array(
            img.crop(xlim, ylim, of="orig", rtn=True),
            opening_angle=2 * extend_deg,
            quantity=None,
            dir_in=None,
        )
        return img_zoom

    
    @staticmethod
    def apply_filter_on_single_dipole_image(
        dipole: pd.Series,
        skyarrays: Dict[str, Type[SkyArray]],
        filter_dsc_x: dict = default_filter_dipole_vel_tx,
        filter_dsc_y: dict = default_filter_dipole_vel_ty,
    ) -> tuple:
        """
        Filter set-up as in arxiv:1812.04241

        Args:
            dipole: Dipole properties
            skyarrays: Dictionary must contains SkyArray instances of
                unfiltered isw-rs map [-] and deflection angle map [rad].
                The maps can contain either the individual x,y-components
                of the vector fields or their sum.
            filter_dsc_x,y: Dictionary of filters applied to cropped map
                in x,y-direction.
        Returns:
        """
        filter_dsc_x["gaussian_third_derivative"]["theta_i"] = (
            2 * dipole.r200_deg * un.deg
        )
        filter_dsc_y["gaussian_third_derivative"]["theta_i"] = (
            2 * dipole.r200_deg * un.deg
        )
        keys = list(skyarrays.keys())
        for key in keys: 
            for label, filtr in zip(["_x", "_y"], [filter_dsc_x, filter_dsc_y]):
                skyarrays[key + label] = skyarrays[key].filter(
                    filtr, on="orig", rtn=True
                )
        return skyarrays


    @staticmethod
    def get_single_transverse_velocity_from_sky(
        deltaTx: np.ndarray,
        deltaTy: np.ndarray,
        alphax: np.ndarray,
        alphay: np.ndarray,
    ) -> tuple:
        """
        Velocity vector component from ISWRS and deflection angle map.
        Eq. 9 in arxiv:1812.04241

        Args:
            deltaTx,y:
                Filtered ISW-RS/temp. perturbation maps [-]
            alphax,y:
                Filtered deflection angle maps [rad]
        Returns:
            x,y-components of the transverse velocity [km/sec]
        """

        def _transverse_velocity_component(deltaT, alpha) -> float:
            return -c_light.to("km/s").value * np.sum(deltaT) / np.sum(alpha)

        x_vel = _transverse_velocity_component(deltaTx, alphax)
        y_vel = _transverse_velocity_component(deltaTy, alphay)
        return x_vel, y_vel
