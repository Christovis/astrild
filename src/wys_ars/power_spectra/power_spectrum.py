import os
from typing import Dict, List, Optional, Tuple, Type, Union
import subprocess

import numpy as np
import pandas as pd

from nbodykit.lab import *

from wys_ars.simulation import Simulation


class PowerSpectrumWarning(BaseException):
    pass


class PowerSpectrum:
    """
    Attributes:
        sim_type:
        simulation:

    Methods:
        compute:
    """

    def __init__(self, sim_type: str, simulation: Type[Simulation]):
        """ """
        self.sim = simulation
        self.sim.type = sim_type

    def compute(
        self,
        quantity: Optional[str] = None,
        snap_nrs: Optional[List[int]] = None,
        file_dsc: Dict[str, str] = {"root": "dtfe", "extension": "npy"},
        dir_in: Optional[str] = None,
        dir_out: Optional[str] = None,
        save: bool = True,
    ) -> Union[None, Dict[str, dict]]:
        """
        Power spectrum of particle quanities.
        
        Args:
            quantity: [rho, phi, dphi/dt, chi, velocity]
        """
        if not dir_in:
            dir_in = self.sim.dirs["sim"]
        if snap_nrs:
            assert set(snap_nrs) < set(self.sim.dir_nrs), PowerSpectrumWarning(
                f"Some of the snapshots {snap_nrs} do not exist" + \
                f"in:\n{self.sim.dir_nrs}"
            )
            _file_paths = self.sim.get_file_paths(
                file_dsc, dir_in, "max"
            )
        else:
            snap_nrs = self.sim.get_file_nrs(file_dsc, dir_in, "max")
            _file_paths = self.sim.get_file_paths(file_dsc, dir_in, "max")

        pk = {"k": {}, "P": {}}
        for snap_nr, file_path in zip(snap_nrs, _file_paths):
            value_map = self._read_data(file_path, quantity)
            if len(value_map.shape) == 3:
                k, Pk = self._power_spectrum_3d(value_map)
            elif len(value_map.shape) == 2:
                k, Pk = self._power_spectrum_2d(value_map)
            else:
                raise PowerSpectrumWarning(
                    f"{len(value_map.shape)}D is not supported :-("
                )
            pk["k"]["snap_%d" % snap_nr] = k
            pk["P"]["snap_%d" % snap_nr] = Pk
        
        if len(_file_paths) > 1:
            # check that wavenumbers of different snapshots are the same
            _columns = list(pk["k"].keys())
            assert np.sum(pk["k"][_columns[0]]) == np.sum(pk["k"][_columns[1]])

        if save:
            self._save_results(file_dsc, quantity, pk)
        else:
            return pk

    def _read_data(self, file_in: str, quantity: Optional[str] = None,) -> np.ndarray:
        """ """
        value_map = np.zeros((self.sim.npar, self.sim.npar, self.sim.npar))
        if ".h5" in file_in:
            fields = pd.read_hdf(file_in, key="df")
            x = (self.sim.npar * fields["x"].values).astype(int)
            y = (self.sim.npar * fields["y"].values).astype(int)
            z = (self.sim.npar * fields["z"].values).astype(int)
            value_map[(x, y, z)] = fields[quantity].values
        elif ".npy" in file_in:
            value_map = np.load(file_in)
            #if len(value_map.shape) == 4:
            #    value_map = self._get_vector_magnitude(value_map)
        return value_map

    def _get_vector_magnitude(self, value_map: np.ndarray) -> np.ndarray:
        """ Compute vector magnitude for 3D array """
        for ii in range(3):
            value_map[:, :, :, ii] = np.square(value_map[:, :, :, ii])
        value_map = np.sum(value_map, axis=3)
        value_map = np.sqrt(value_map)
        assert len(value_map.shape) == 3
        return value_map

    def _power_spectrum_3d(self, value_map: np.ndarray) -> Tuple[np.array, np.array]:
        """
        Compute the 3D power spectrum

        Args:
            value_map:
                three dimensional array containing values of interest
        Returns:
            k:
                wavenumber
            Pk:
                power at each wavenumber
        """
        _k_min = 2 * np.pi / self.sim.boxsize
        _mesh = ArrayMesh(
            value_map,
            Nmesh=self.sim.domain_level,
            compensated=False,
            BoxSize=self.sim.boxsize,
        )
        r = FFTPower(
            _mesh,
            mode="1d",
            # dk=Delta_k,
            kmin=_k_min,
            # kmax=None,
        )
        k = np.array(r.power["k"])  # the k-bins
        Pk = np.array(r.power["power"].real)  # the power spectrum
        Pk_shotnoise = r.power.attrs["shotnoise"]  # shot-noise
        Pk -= Pk_shotnoise
        print("Pk wavenumber ------>", k.min(), k.max())
        return k, Pk

    def _power_spectrum_2d():
        raise PowerSpectrumWarning("This class method is not implemented yet")

    def _save_results(self, file_dsc: Dict[str, str], quantity: str, pk: dict) -> None:
        """ 
        Save results each power spectrum of each simulations snapshot

        Args:
            quantity:
                Quantity of whicht the power spectrum was calculated,
                e.g. divergence velocity, matter, Phi, ...
            pk:
                Simulation power spectra for different snapshots/redshifts.
        """
        _columns = list(pk["k"].keys())
        df = pd.DataFrame(data=pk["P"], index=pk["k"][_columns[0]],)
        file_out = self.sim.dirs["out"] + "pk_%s.h5" % (quantity)
        if os.path.exists(file_out):
            os.remove(file_out)
        print(f"Saving results to -> {file_out}")
        df.to_hdf(file_out, key="df", mode="w")
