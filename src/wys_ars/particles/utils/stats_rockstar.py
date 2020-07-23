from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.stats import binned_statistic
from scipy.optimize import curve_fit, newton

from halotools.mock_observables import tpcf
from halotools import mock_observables as mo

from wys_ars.utils.arepo_hdf5_library import read_hdf5


class Rockstar:
    def histograms(
        snapshot: pd.DataFrame,
        nbins: int,
        properties: List[str],
    ) -> Dict[str, np.array]:
        """
        Comput the concentration/mass relation.
     
        Args:
        Returns:
        """
        hist = {}
        for prop, limits in properties.items():
            if isinstance(limits[0], str):
                limits = tuple([float(ii) for ii in limits])
            hist[prop] = np.histogram(
                snapshot[prop].values, bins=nbins, range=limits
            )[0] / len(snapshot.index)
        return hist

    def concentration_mass_rel(
        snapshot: pd.DataFrame,
        limits: tuple = None,
        nbins: int = 20,
        method: str = "prada",
    ) -> Tuple[np.array, np.array]:
        """
        Comput the concentration/mass relation.
     
        Args:
            method:
                nfw:
        Returns:
        """
        if limits is None:
            # TODO double check untis
            limits = (
                np.log10(snapshot["m200c"].min()),  # [Mpc/h]
                np.log10(snapshot["m200c"].max()),  # [Mpc/h]
            )
        elif isinstance(limits[0], str):
            limits = tuple([float(ii) for ii in limits])

        # filter mass range
        snapshot = snapshot[
            (10 ** min(limits) < snapshot["m200c"])
            & (snapshot["m200c"] < 10 ** max(limits))
        ]

        snapshot["c_nfw"] = snapshot["r200c"] / snapshot["Rs"]  # TODO double check

        if len(snapshot.index.values) != 0:
            c_mean, edges, _ = binned_statistic(
                snapshot["m200c"].values,
                values=snapshot["c_nfw"].values,
                statistic="mean",
                bins=np.logspace(min(limits), max(limits), nbins + 1),
            )
            mass_bins = (edges[1:] + edges[:-1]) / 2.0
        else:
            mass_bins = None
            c_mean = None
        return mass_bins, c_mean
