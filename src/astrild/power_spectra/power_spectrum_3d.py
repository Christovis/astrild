import os
from typing import Dict, List, Optional, Tuple, Type, Union
import subprocess

import numpy as np
import pandas as pd

from nbodykit.lab import *

from astrild.simulation import Simulation
from astrild.io import IO


class PowerSpectrum3DWarning(BaseException):
    pass


class PowerSpectrum3D:
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
        quantities: List[str],
        file_dsc: List[Dict[str, str]],
        snap_nrs: Optional[List[int]] = None,
        dir_out: Optional[str] = None,
        save: bool = True,
    ) -> Union[None, dict]:
        """
        Power spectrum of particle quanities.
        
        Args:
            quantities: [rho, phi, dphi/dt, chi, velocity, kappa, \Delta T]
            file_dsc: {path: , root: , extention: }
        """
        if snap_nrs:
            assert set(snap_nrs) < set(self.sim.dir_nrs), PowerSpectrumWarning(
                f"Some of the snapshots {snap_nrs} do not exist" + \
                f"in:\n{self.sim.dir_nrs}"
            )
            _file_paths1 = self.sim.get_file_paths(
                file_dsc[0], file_dsc[0]["path"], "max"
            )
            if len(file_dsc) > 1:
                _file_paths2 = self.sim.get_file_paths(
                    file_dsc[1], file_dsc[1]["path"], "max"
                )
        else:
            _file_path = file_dsc[0].pop("path")
            _file_dsc = file_dsc[0]
            snap_nrs = self.sim.get_file_nrs(_file_dsc, _file_path, "max")
            _file_paths1 = self.sim.get_file_paths(_file_dsc, _file_path, "max")
            if len(file_dsc) > 1:
                _file_path = file_dsc[1].pop("path")
                _file_dsc = file_dsc[1]
                _file_paths2 = self.sim.get_file_paths(
                    _file_dsc, _file_path, "max"
                )

        snap_nrs = np.sort(snap_nrs)
        if len(file_dsc) > 1:
            pk = self._cross_power_spectra(quantities, snap_nrs, _file_paths1, _file_paths2)
        else:
            pk = self._auto_power_spectra(quantities, snap_nrs, _file_paths1)
        
        if save:
            self._save_results(quantities, pk)
        else:
            return pk
   
    def _auto_power_spectra(
        self,
        quantity: List[str],
        snap_nrs: np.array,
        _file_paths: List[str]
    ) -> dict:
        pk = {"k": {}, "P": {}}
        print(np.sort(snap_nrs))
        print(_file_paths)
        for snap_nr, file_path in zip(snap_nrs, _file_paths):
            print("---------------------------------")
            print(snap_nr, file_path)
            print("---------------------------------")
            value_map = self._read_data(file_path, quantity)
            if len(value_map.shape) == 3:
                k, Pk = self._power_spectrum_3d(value_map)
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
        
        return pk
    
    def _cross_power_spectra(
        self,
        quantity: List[str],
        snap_nrs: np.array,
        _file_paths1: List[str],
        _file_paths2: List[str],
    ) -> dict:
        pk = {"k": {}, "P": {}}
        for snap_nr, file_path1, file_path2 in zip(snap_nrs, _file_paths1, _file_paths2):
            value_map1 = self._read_data(file_path1, None)
            value_map2 = self._read_data(file_path2, None)
            if len(value_map1.shape) == 3:
                k, Pk = self._power_spectrum_3d(value_map1, value_map2)
            else:
                raise PowerSpectrumWarning(
                    f"{len(value_map.shape)}D is not supported :-("
                )
            pk["k"]["snap_%d" % snap_nr] = k
            pk["P"]["snap_%d" % snap_nr] = Pk
        
        if len(_file_paths1) > 1:
            # check that wavenumbers of different snapshots are the same
            _columns = list(pk["k"].keys())
            assert np.sum(pk["k"][_columns[0]]) == np.sum(pk["k"][_columns[1]])
        
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

    def _power_spectrum_3d(
        self,
        value_map1: np.ndarray,
        value_map2: Optional[np.ndarray] = None,
    ) -> Tuple[np.array, np.array]:
        """
        Compute the 3D auto or cross power spectrum.

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
        if value_map2 is None:
            _mesh1 = ArrayMesh(
                value_map1,
                Nmesh=self.sim.domain_level,
                compensated=False,
                BoxSize=self.sim.boxsize,
            )
            r = FFTPower(
                _mesh1,
                mode="1d",
                # dk=Delta_k,
                kmin=_k_min,
                # kmax=None,
            )
        else:
            _mesh1 = ArrayMesh(
                value_map1,
                Nmesh=self.sim.domain_level,
                compensated=True,
                interlaced=True,
                window='TSC',
                BoxSize=self.sim.boxsize,
            )
            _mesh2 = ArrayMesh(
                value_map2,
                Nmesh=self.sim.domain_level,
                compensated=True,
                interlaced=True,
                window='TSC',
                BoxSize=self.sim.boxsize,
            )
            # apply correction for the window to the mesh
            #_mesh1 = _mesh1.apply(_mesh1.CompensateTSC, kind='circular', mode='complex')
            #_mesh2 = _mesh2.apply(_mesh2.CompensateTSC, kind='circular', mode='complex')
            r = FFTPower(
                first=_mesh1,
                mode="1d",
                second=_mesh2,
                kmin=_k_min,
                #kmax=3.,
            )
        k = np.array(r.power["k"])
        Pk = np.array(r.power["power"].real - r.power.attrs["shotnoise"])
        print("Pk wavenumber ------>", k.min(), k.max())
        return k, Pk

    def _save_results(
        self,
        #file_dsc: List[Dict[str, str]],
        quantity: List[str],
        pk: dict
    ) -> None:
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
        filename = self.sim.dirs["out"] + "pk_%s.h5" % (("_").join(quantity))
        IO._remove_existing_file(filename)
        print(f"Saving results to -> {filename}")
        df.to_hdf(filename, key="df", mode="w")
