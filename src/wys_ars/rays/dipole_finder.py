import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type

import pandas as pd
import xarray as xr
import numpy as np
from scipy.interpolate import interp1d
from scipy import integrate
from sklearn.neighbors import NearestNeighbors

from astropy import units as un
from lenstools import ConvergenceMap

from wys_ars.rays.utils import object_selection
from wys_ars.simulation import Simulation
from wys_ars.rays.skymap import SkyMap
from wys_ars.rays.utils.filters import Filters
from wys_ars.profiles import profile_2d as Profiles2D
from wys_ars.io import IO

class DipolesWarning(BaseException):
    pass


class Dipoles:
    """
    Class to find and analyse dipol signals.

    Attributes:

    Methods:
    """

    def __init__(
        self,
        dipoles: pd.DataFrame,
    ):
        """
        Read dipols data files.

        Args:
        """
        self.data = dipoles

    @classmethod
    def from_sky(
        cls,
        skymap: Type[SkyMap],
        on: str,
        bin_dsc: dict,
        kernel_width: float=5,
        direction: int=1,
        filters: bool = True,
    ) -> "Dipoles":
        """
        Find peaks on the dipole signal map. It is assumed that the convergence maps
        were created with wys_ars.rays.visuals.map and filter with:
            I) high-pass II) DGD3 III) low-pass gaussian filters.

        Args:
            kernel_width:
                Smoothing kernel with [arcmin]
        Returns:
        """
        if filters is True:
            skymap = cls._filter(skymap, kernel_width, direction)
       
        print("----------------", skymap.data[bin_dsc["on"]].shape)
        thresholds = cls._get_convergence_thresholds(
            sky_array=skymap.data[bin_dsc["on"]],
            nbins=bin_dsc["nbins"],
        )
        
        _map = ConvergenceMap(
            data=skymap.data[on],
            angle=skymap.opening_angle*un.deg
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
        peak_dir["snr"] = cls._signal_to_noise_ratio(peak_dir["deltaT"], _map.data)
        peak_dir["x_pix"] = np.rint(
            peak_dir["x_deg"] * skymap.npix / skymap.opening_angle
        ).astype(int)
        peak_dir["y_pix"] = np.rint(
            peak_dir["y_deg"] * skymap.npix / skymap.opening_angle
        ).astype(int)
        peak_df = pd.DataFrame(data=peak_dir)
        # attrs is experimental and may change without warning.
        peak_df.attrs['map_file'] = skymap.map_file 
        peak_df.attrs['filters'] = filters
        peak_df.attrs['kernel_width'] = kernel_width
        return cls(peak_df)

    @classmethod
    def from_file(
        cls,
        filename_dip: str,
    ) -> "Dipoles":
        """
        Args:
            kernel_width:
                Smoothing kernel with [arcmin]
        Returns:
        """
        peak_df = pd.read_hdf(filename_dip, key="df")
        return cls(peak_df)

    @classmethod
    def _filter(
        cls,
        skymap: Type[SkyMap],
        kernel_width: float,
        direction: int,
    ) -> Type[SkyMap]:
        """
        Prepre skymap for dipole detection.
        Note: This works only for skymap.data["orig"]
        """
        # prepare map for dipole detection
        filter_dsc = {
            "gaussian_high_pass": {
                "abbrev": "ghpf",
                "theta_i": kernel_width,
            },
            "gaussian_third_derivative": {
                "abbrev": "g3df",
                "theta_i": kernel_width,
                "direction": direction,
            }
        }
        _map = skymap.convolution(filter_dsc, on="orig", rtn=True)
        filter_dsc = {
            "gaussian_low_pass": {
                "abbrev": "glpf",
                "theta_i": kernel_width,
            },
        }
        skymap.data["orig_ghpf_g3df_glpf"] = skymap.convolution(
            filter_dsc, sky_array=np.abs(_map), rtn=True,
        )
        return skymap

    @classmethod
    def _get_convergence_thresholds(
        cls,
        sky_array: np.ndarray,
        nbins: int = 100,
    ) -> np.array:
        """
        Define thresholds for lenstools to find peaks on convergence map.
        Important to do this on un-smoothed and with no-gsn skymap.
        """
        bins = np.arange(
            np.min(sky_array), np.max(sky_array)*1.1,
            (np.max(sky_array)*1.1 - np.min(sky_array)) / nbins,
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
        Remove peaks within 1 smoothing length from map edges

        Args:
            sigma:
            pos:
                Peak x,y-positions [deg]

        Returns:
            sigma and pos
        """
        pixlen = opening_angle / npix  #[deg]
        bufferlen = np.ceil(kernel_width / (60 * pixlen))  # length of buffer zone
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
            f"{len(indx)} peaks were within kernel_width of FOV edge" + \
            f" and had to be removed. \n{len(sigma)} peaks are left."
        )
        return sigma, pos


    def find_nearest(
        self, df2: pd.DataFrame, columns: dict,
    ) -> None:
        """
        Method used to e.g. find haloes which cause a dipole.

        Args:
            df2:
        """
        if "box_nr" in self.data.columns.values:
            distances, ids = self.find_nearest_in_box(self.data, df2)
        elif "snap_nr" in self.data.columns.values:
            distances, ids = self.find_nearest_in_snap(self.data, df2)
        else:
            # find corresponding group-halo to dipole signal
            distances, ids = self._get_index_and_distance(self.data, df2)
        self.data[columns["id"]] = ids
        self.data[columns["dist"]] = distances


    def find_nearest_in_box(
        self, df1: pd.DataFrame, df2: pd.DataFrame,
    ) -> tuple:
        distances = []
        ids = []
        for box_nr in np.unique(df1["box_nr"].values):
            df1_in_box = df1[df1["box_nr"] == box_nr]
            df2_in_box = df2[df2["box_nr"] == box_nr]
            if "ray_nr" in df1_in_box.columns.values:
                _distances, _indices = self.find_nearest_in_snap(
                    df1_in_box, df2_in_box,
                )
            else:
                _distances, _indices = self._get_index_and_distance(
                    df1_in_box, df2_in_box,
                )
            distances += _distances
            ids += _indices
        return distances, ids


    def find_nearest_in_snap(
        self, df1: pd.DataFrame, df2: pd.DataFrame,
    ) -> tuple:
        distances = []
        ids = []
        for ray_nr in np.unique(df1["ray_nr"].values):
            df1_in_snap = df1[df1["ray_nr"] == ray_nr]
            df2_in_snap = df2[df2["ray_nr"] == ray_nr]
            _distances, _indices = self._get_index_and_distance(
                df1_in_snap, df2_in_snap,
            )
            distances += _distances
            ids += _indices
        return distances, ids

    def _get_index_and_distance(
        self, df1: pd.DataFrame, df2: pd.DataFrame,
    ) -> tuple:
        if len(df2.index.values) == 0:
            print("There are no Halos in this distance")
            distances = np.ones(len(df1.index.values)) * -99999
            ids = np.ones(len(df1.index.values)) * -99999
        else:
            nbrs = NearestNeighbors(
                n_neighbors=1,
                algorithm='ball_tree'
            ).fit(df2[["x_deg", "y_deg"]].values)
            distances, indices = nbrs.kneighbors(df1[["x_deg", "y_deg"]].values)
            distances = distances.T[0]
            ids = df2["id"].values[indices.T[0]].astype(int)
            if len(ids) > len(np.unique(ids)):
                nan_idx = []
                for u_id in np.unique(ids):
                    _idx = np.where(ids == u_id)[0]
                    min_idx = np.argmin(distances[_idx])
                    nan_idx += list(np.delete(_idx, min_idx))
                ids[nan_idx] = -99999
                distances[nan_idx] = -99999
        return list(distances), list(ids)

    def _get_transverse_velocity(
        self, iswrssky: Type[SkyMap], defltxsky: Type[SkyMap], defltysky: Type[SkyMap],
    ) -> tuple:
        """
        Velocity vector component from ISWRS and deflection angle map.
        Eq. 9 in arxiv:1812.04241
        """
        x_vel = self._get_transverse_velocity_component(
            iswrssky.data["orig_hpf_x3df_apo"], defltxsky.data["orig_x3df_apo"]
        )
        y_vel = self._get_transverse_velocity_component(
            iswrssky.data["orig_hpf_y3df_apo"], defltysky.data["orig_y3df_apo"]
        )
        return x_vel, y_vel

    def _get_transverse_velocity_component(
        self, iswrs: np.ndarray, kappa: np.ndarray,
    ) -> float:
        """
        Velocity vector component from ISWRS and deflection angle map.
        Eq. 9 in arxiv:1812.04241
        """
        c_light = 299792.458  #[km/s]
        return -c_light * np.sum(iswrs) / np.sum(kappa)
