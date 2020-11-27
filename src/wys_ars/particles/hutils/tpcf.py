from typing import List, Optional, Union

import numpy as np
from scipy.stats import binned_statistic
from scipy.optimize import curve_fit, newton

from halotools.mock_observables import tpcf as tpcf_r
from halotools.mock_observables import s_mu_tpcf
from halotools import mock_observables as mo


class TPCF:
    def compute(
        pos: np.ndarray,
        vel: np.ndarray,
        boxsize: float,
        space: str,
        s_range: [tuple, np.array],
        mu_range: Union[tuple, np.array, None],
        nthreads: int = 1,
        los: Optional[int] = None,
    ) -> np.ndarray:
        """
        Args:
            space: [real/configuartion, redshift]
            chi_range: distance range over which to calculate TPCF
            mu_range: angle range, cos(theta), over which to calculate TPCF
        """
        if type(s_range) == tuple:
            nbins = 40
            s_range = np.linspace(min(s_range), max(s_range), nbins)

        if type(mu_range) == tuple:
            nbins = 40
            mu_range = np.sort(
                1. - np.geomspace(min(mu_range), max(mu_range), nbins)
            )

        tpcf = TPCF.tpcf_s(
            pos,
            vel,
            s_range,
            mu_range,
            los,
            boxsize,
            nthreads,
        )
        s_range = (s_range[1:] + s_range[:-1]) / 2.0
        return s_range, mu_range, tpcf

    def tpcf_s(
        pos: np.ndarray,
        vel: np.ndarray,
        chi_range: np.array,
        mu_range: np.array,
        los: int = 2,
        boxsize: float = 500.,
        nthreads: int = 1,
    ) -> np.ndarray:
        """
        TPCF in redshift space

        Args:
        
        Note: halotools tpcf_s_mu assumes the line of sight is always the z direction
        """

        # convert real- into redshift-space coord.
        pos_s = pos.copy()
        pos_s[:, los] += vel[:, los] / 100. #[Mpc/h]

        # enforce periodic boundary condition
        pos_s[:, los] = np.where(
            pos_s[:, los] > boxsize,
            pos_s[:, los] - boxsize,
            pos_s[:, los],
        )
        pos_s[:, los] = np.where(
            pos_s[:, los] < 0.,
            pos_s[:, los] + boxsize,
            pos_s[:, los],
        )

        if(los != 2):
            # rotate coords. to ensure los=2 as it is required by halotools
            pos_s_old = pos_s.copy()
            pos_s[:, 2] = pos_s_old[:, los]
            pos_s[:, los] = pos_s_old[:, 2]

        # compute TPCF in redshift space
        tpcf = s_mu_tpcf(
            sample1=pos_s,
            s_bins=chi_range,
            mu_bins=mu_range,
            period=boxsize,
            estimator="Landy-Szalay",
            num_threads=nthreads,
        )
        
        return tpcf
       
