import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type

import pandas as pd
import xarray as xr
import numpy as np
from scipy.interpolate import interp1d
from scipy import integrate
from astropy import units as un
from lenstools import ConvergenceMap

from wys_ars.rays.utils import object_selection
from wys_ars.simulation import Simulation
from wys_ars.rays.skymap import SkyMap
from wys_ars.rays.utils import filters as Filters
from wys_ars.profiles import profile_2d as Profiles2D
from wys_ars import io as IO


class DipoleFinderWarning(BaseException):
    pass


class DipoleFinder:
    """
    Class to find and analyse dipol signals.

    Attributes:
        dataset_file:
        data:
        finder_spec:
        skymap_dsc:

    Methods:
        find_dipoles:
    """

    def __init__(
        self,
        skymap: Type[SkyMap],
        on: str,
        kernel_width: float=5,
        direction: int=1,
    ) -> "Dipols":
        """
        Read dipols data files.

        Args:
            skymap:
                SkyMap object
            on:
                Map identified in the collection of SkyMaps
            sigma:
                DGD3 Filter width (recommended 5 or 200)
        """
        self.skymap = skymap
        self.smoothing_length = kernel_width

        _map = Filters.gaussian_high_pass_filter(
            skymap.data[on], kernel_width
        )
        _map = Filters.gaussian_third_derivative_filter(
            _map, kernel_width, direction
        )
        self.skymap.data["orig_hpf_dgd3f_lpf"] = Filters.gaussian_low_pass_filter(
            np.abs(_map), kernel_width
        )


    def find_dipoles(
        self,
        on: str,
        thresholds_dsc: dict,
    ) -> None:
        """
        Find peaks on the dipole signal map. It is assumed that the convergence maps
        were created with wys_ars.rays.visuals.map and filter with:
            I) high-pass II) DGD3 III) low-pass gaussian filters.

        Args:
        Returns:
        """
        self.on = on
        _map = self.skymap.data[on]
        _map = ConvergenceMap(data=_map, angle=self.skymap.opening_angle*un.deg)
        
        thresholds = self._get_convergence_thresholds(**thresholds_dsc)
        _peaks = {}
        _peaks["sigma"], _peaks["pos"] = _map.locatePeaks(thresholds)
        _peaks["sigma"], _peaks["pos"] = self._remove_peaks_crossing_edge(**_peaks)
        assert len(_peaks["sigma"]) != 0, "No peaks"

        # find significance of peaks
        _peaks["snr"] = self._signal_to_noise_ratio(_peaks["sigma"], _map.data)
        self.peaks = _peaks
        print("--------------------------", len(self.peaks["sigma"]), np.min(self.peaks["sigma"]), np.max(self.peaks["sigma"]))
 

    def _get_convergence_thresholds(
        self, on: str = "orig_hpf_dgd3f", nbins: int = 100,
    ) -> np.array:
        """
        Define thresholds for lenstools to find peaks on convergence map.
        Important to do this on un-smoothed and with no-gsn skymap.
        """
        return np.arange(
            np.min(self.skymap.data[on]),
            np.max(self.skymap.data[on]),
            (np.max(self.skymap.data[on]) - np.min(self.skymap.data[on])) / nbins,
        )


    def _signal_to_noise_ratio(
        self,
        peak_values: np.ndarray,
        map_values: np.ndarray,
        sigma: Optional[float] = None,
    ) -> np.ndarray:
        """
        Assess signifance of peaks and remove peaks suffereing edge effects

        Args:
        """
        _kappa_mean = np.mean(map_values)
        _kappa_std = np.std(map_values)
        snr = (peak_values - _kappa_mean) / _kappa_std
        return snr


    def _remove_peaks_crossing_edge(
        self,
        sigma: np.ndarray,
        pos: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Remove peaks within 1 smoothing length from map edges

        Args:
            sigma:
            pos:

        Returns:
            sigma:
            pos:
        """
        pixlen = self.skymap.opening_angle / self.skymap.npix  #[deg]
        bufferlen = np.ceil(self.smoothing_length / (60 * pixlen))  # length of buffer zone
        # convert degrees to pixel number
        x = pos[:, 0].value * self.skymap.npix/ self.skymap.opening_angle
        y = pos[:, 1].value * self.skymap.npix / self.skymap.opening_angle
        indx = np.logical_and(
            np.logical_and(x <= self.skymap.npix - 1 - bufferlen, x >= bufferlen),
            np.logical_and(y <= self.skymap.npix - 1 - bufferlen, y >= bufferlen),
        )
        sigma = sigma[indx]
        pos = pos[indx, :]
        print(
            f"{len(indx)} peaks were within smoothing_length of FOV edge" + \
            f" and had to be removed. \n{len(sigma)} peaks are left."
        )
        return sigma, pos

