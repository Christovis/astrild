from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.stats import binned_statistic
from scipy.optimize import curve_fit, newton

#from halotools.mock_observables import tpcf
#from halotools import mock_observables as mo


class Rockstar:
    def halo_mass_fct(
        snapshot: pd.DataFrame,
        limits: tuple = (11.78, 16),
        nbins: int = 20,
    ) -> tuple:
        """
        Compute the halo mass function

        Args:
            data:
                halo mass in [M_{\odot}/h]
        """
        bins = np.logspace(min(limits), max(limits), nbins + 1)
        mass = snapshot["m200c"].values
        print("min mass ----------------->", np.min(mass))
        mass = mass[min(limits) < mass]
        # count halos within mass range
        mass_count, edges = np.histogram(mass, bins=bins)
        # calculate mass function
        mass_count = np.cumsum(mass_count[::-1])[::-1]
        mass_bin = (edges[1:] + edges[:-1]) / 2.0
        return mass_bin, mass_count

    def histograms(
        snapshot: pd.DataFrame,
        nbins: int,
        dimesions: int,
        properties: List[str],
        base: Optional[str] = None,
    ) -> Dict[str, np.array]:
        """
        Comput the concentration/mass relation.
     
        Args:
        Returns:
        """
        if dimesions == 1:
            hist = {}
            for prop, limits in properties.items():
                if isinstance(limits[0], str):
                    limits = tuple([float(ii) for ii in limits])
                hist[prop] = np.histogram(
                    snapshot[prop].values, bins=nbins, range=limits, density=True,
                )[0]
        elif dimesions == 2:
            #TODO
            # https://stackoverflow.com/questions/57562613/python-earth-mover-distance-of-2d-arrays
            pass
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

    #def two_point_corr_fct(
    #    snapshot: pd.DataFrame,
    #    limits: tuple = None,
    #    nbins: int = None,
    #    boxsize: float = None,
    #) -> tuple:
    #    if boxsize is None:
    #        boxsize = snapshot.header.boxsize / 1e3  #[Mpc/h]
    #    if limits is None:
    #        limits = (0.3, boxsize / 5)
    #    if nbins is None:
    #        nbins = int(2 / 3 * max(limits))

    #    r = np.geomspace(min(limits), max(limits), nbins)
    #    r_c = 0.5 * (r[1:] + r[:-1])
    #    real_tpcf = mo.tpcf(
    #        snapshot[["x", "y", "z"]].values,  #[Mpc/h]
    #        rbins=r,  #[Mpc/h]
    #        period=boxsize,
    #        estimator="Landy-Szalay",
    #    )
    #    return r_c, real_tpcf
