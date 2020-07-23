import os
from typing import Dict, List, Optional, Tuple, Type, Union
import subprocess

import numpy as np
import pandas as pd

from nbodykit.lab import *

from wys_ars.simulation import Simulation


class MapTransformWarning(BaseException):
    pass


class MapTransform:
    """
    Attributes:

    Methods:
    """

    def __init__(self, sim_type: str, simulation: Type[Simulation]):
        """ """
        self.sim = simulation
        self.sim.type = sim_type

    def divergence(
        self,
        quantity: Optional[str] = None,
        snap_nrs: Optional[List[int]] = None,
        file_dsc: Dict[str, str] = {"root": "dtfe", "extension": "npy"},
        directory: Optional[str] = None,
        save: bool = True,
    ) -> Union[None, Dict[str, dict]]:
        """
        Compute vector divergence field.
        
        Args:
        Returns:
        """
        if not directory:
            directory = self.sim.dirs["sim"]
        if snap_nrs:
            assert set(snap_nrs) < set(self.sim.dir_nrs), MapTransformWarning(
                f"Some of the snapshots {snap_nrs} do not exist" + \
                f"in:\n{self.sim.dir_nrs}"
            )
            _file_paths = self.sim.get_file_paths(
                file_dsc, directory, "max"
            )
        else:
            snap_nrs = self.sim.get_file_nrs(file_dsc, directory, "max")
            _file_paths = self.sim.get_file_paths(file_dsc, directory, "max")

        for snap_nr, file_path in zip(snap_nrs, _file_paths):
            _value_map = self._read_data(file_path, quantity)
            if len(_value_map.shape) == 4:
                _value_map = self._compute_divergence(_value_map)
            else:
                raise MapTransformWarning(
                    f"{len(_value_map.shape)}D is not supported yet."
                )
            if save:
                self._save_results("div_", file_path, _value_map)
            else:
                return _value_map


    def _read_data(self, file_in: str, quantity: Optional[str] = None,) -> np.ndarray:
        """
        Read pd.DataFrame created by wys_ars.particles.ecosmog.dtfe()

        Args:
        Returns:
        """
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


    def _compute_divergence(self, value_map: np.ndarray) -> np.ndarray:
        """ Compute vector divergence field, nabla^iv_i """
        dx_vx = np.gradient(
                value_map[:,:,:,0], 1/self.sim.npar, axis=0, edge_order=2,
        )
        dy_vy = np.gradient(
                value_map[:,:,:,1], 1/self.sim.npar, axis=1, edge_order=2,
        )
        dz_vz = np.gradient(
                value_map[:,:,:,2], 1/self.sim.npar, axis=2, edge_order=2,
        )
        div_v = (dx_vx + dy_vy + dz_vz)
        return div_v


    def _save_results(self, file_tag: str, file_path: str, value_map: np.ndarray) -> None:
        """ 
        Save results each power spectrum of each simulations snapshot

        Args:
        """
        directory = file_path.split("/")[:-1]
        file_name = file_tag + file_path.split("/")[-1]
        file_out = "/".join(directory) + "/" + file_name
        print(f"Save result in -> {file_out}")
        np.save(file_out, value_map)
