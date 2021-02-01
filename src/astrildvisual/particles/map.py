# Create 2D np.ndarrays of ray-ramses results stored in .h5 files
# through merge.py
import os, sys, glob
import argparse

import numpy as np
import pandas as pd
from scipy.interpolate import griddata

from astrild.simulation import Simulation

c_light = 299792.458  # in km/s


class Maps(Simulation):
    def __init__(
        self,
        boxsize: float = 500.0,
        domain_level: int = 512,
        dir_sim: str = None,
        dir_out: str = None,
        snap_nrs: list = None,
        file_root: str = "Ray_maps_output",
        dir_root: str = None,
    ):
        """
        Input:
            dir_sim: simulation directory
            dir_out: directory where results will be stored
            theta: opening angle
        """
        super().__init__(dir_sim, dir_out, file_root, dir_root)
        self.boxsize = boxsize
        self.npix = domain_level
        self._get_file_nrs(snap_nrs)
        self._get_file_paths(snap_nrs)

    def _read_fields(self, file_map: str) -> pd.DataFrame:
        return pd.read_hdf(file_map, key="df")

    def _save_map(self, file_out: str, map_out):
        file_out = self.dirs["out"] + file_out
        # remove old
        if os.path.exists(file_out):
            os.remove(file_out)
        # create new
        np.save(file_out, map_out)

    def to_array(
        self,
        centre: float = 0.5,
        depth: float = 0.1,
        _snap_nrs: list = None,
        quantities: list = None,
        save: bool = True,
    ) -> np.ndarray:
        """
        Convert pd.hf to np.ndarray

        Input:
        ------
            depth: thickness of slice cut through simulation box
            _snap_nrs:
            quantities:
            save:
        """
        depth = 1 / self.npix * (1 + depth)
        # make 2d mesh and map of slice data
        xxi, yyi = np.mgrid[
            0.0 : 1.0 : complex(0, self.npix), 0.0 : 1.0 : complex(0, self.npix)
        ]
        for file_nr, file_path in zip(self.file_nrs, self.files[self.file_root]):
            if file_nr != 12:
                continue
            print(f"Converting file number {file_nr}")
            fields = self._read_fields(file_path)
            fields = fields.loc[
                (fields["z"] > centre - depth / 2) & (fields["z"] < centre + depth / 2)
            ]

            for quantity in quantities:
                print(f"    Create map of {quantity}")
                if np.min(fields["x"].values) < 0.0:
                    print("Map lifted by %.3f" % np.abs(fields["x"].values))
                    values = fields[quantity].values + np.abs(fields[quantity].values)
                else:
                    values = fields[quantity].values

                value_map = griddata(
                    (fields["x"].values, fields["y"].values),
                    values,
                    (xxi, yyi),
                    method="linear",
                    fill_value=np.mean(values),
                )

                if save:
                    box_name = self.dirs["sim"].split("/")[-1]
                    filename = "%s_map_%s_out%05d.npy" % (quantity, box_name, file_nr)
                    print("save to --->", filename, self.dirs["sim"])
                    self._save_map(self.dirs["out"] + filename, value_map.T)
