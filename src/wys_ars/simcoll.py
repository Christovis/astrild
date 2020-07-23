import os, re, glob
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import yaml
import numpy as np
import pandas as pd
import xarray as xr

from wys_ars.rays.rayramses import RayRamses
from wys_ars.particles.ecosmog import Ecosmog

dir_src = Path(__file__).parent.absolute()
default_file_config = dir_src / "configs/simulation_collection.yaml"
default_config_file_particle = dir_src / "configs/particle_snapshot_info.h5"
default_config_file_ray = dir_src / "configs/ray_snapshot_info.h5"


class SimulationCollectionWarning(BaseException):
    pass


class SimulationCollection:
    """
    Class to handle a collection of Ramses based simulations.

    Attributes:
        config_file:
        config_file_df:

    Methods:
        compress_stats:
        sum_raytracing_snapshots:
    """

    def __init__(
        self,
        config: pd.DataFrame,
        sims: Dict[str, Union[Ecosmog, RayRamses]],
    ):
        """
        Args:
            config:
                Contains info on simulations in the collection and
                their directories & files that contain data to analyze.
            sims:
                Contains the simulations in the collection.
        """
        self.config = config
        self.sim = sims
        self.sim_nrs = np.arange(1, len(list(sims.keys())) + 1)


    @classmethod
    def from_file(
        cls,
        config_file: str = default_file_config,
        config_file_df: str = default_config_file_particle,
    ) -> "SimulationCollection":
        """
        Initialize SimulationCollection from path to config files.

        Args:
            config_file:
            config_file_df:
        """
        with open(config_file) as f:
            sims_args = yaml.load(f, Loader=yaml.FullLoader)
        sims = {}
        for sim_name, sim_args in sims_args.items():
            if sim_args["type"] == "particles":
                sims[sim_name] = Ecosmog(**sim_args["init"])
            elif sim_args["type"] == "rays":
                sims[sim_name] = RayRamses(**sim_args["init"])
            else:
                raise SimulationCollectionWarning(
                    f"{sim_args['type']} have not been simulated :-("
                )
        if not os.path.isfile(config_file_df):
            raise SimulationCollectionWarning(
                "The file 'ray_snapshot_info.h5' does note exist"
            )
        if sim_args["type"] == "particles":
            config = pd.read_hdf(default_config_file_particle, key="df")
        elif sim_args["type"] == "rays":
            config = pd.read_hdf(default_config_file_ray, key="df")
        
        return SimulationCollection(config, sims)

    def _find_common_z(self) -> np.array:
        """
        Find the redshifts that the simulations in the collection have in common.
        """
        z_nrs = self.config.loc[(self.sim_nrs[0],)]["redshift"].values
        for sim_nr in self.sim_nrs:
            z_nrs = np.intersect1d(z_nrs, self.config.loc[(sim_nr,)]["redshift"].values)
        z_nrs = z_nrs[z_nrs < 2.3]
        return z_nrs

    def compress_stats(
        self,
        file_dsc: Dict[str, str],
        dir_out: str,
        snap_nrs: Optional[np.ndarray] = None,
        z_nrs: Optional[List[float]] = None,
        zmatch: bool = False,
        labels: Optional[Dict[str, str]] = {"x": "bin", "y": "value"},
    ) -> None:
        """
        Combine halo statistics of individual simulations which are stored in
        pandas .h5 format into one xarray dataset file.

        Args:
        """
        if zmatch:
            z_nrs = self._find_common_z()

        # initialize arrays that will contain results
        if file_dsc["extension"] == "h5":
            _stats_file = (
                self.sim[list(self.sim.keys())[0]].dirs["sim"]
                + "%s.h5" % file_dsc["root"]
            )
            y_val_box = pd.read_hdf(_stats_file, key="df")
            y_val = np.zeros((len(self.sim_nrs), len(z_nrs), len(y_val_box.index.values)))
            snap_nrs = np.zeros((len(self.sim_nrs), len(z_nrs)))
        else:
            raise SimulationCollectionWarning("File extension not supported")

        # loop over simulations in collection
        for sim_idx, sim_name in enumerate(list(self.sim.keys())):
            sim_df = self.config.loc[(sim_idx + 1,)]
            _stats_file = (
                self.sim[sim_name].dirs["sim"]
                + f"{file_dsc['root']}.{file_dsc['extension']}"
            )
            y_val_box = pd.read_hdf(_stats_file, key="df")
            for z_idx, z_nr in enumerate(z_nrs):
                snap_nr = sim_df.iloc[
                    (sim_df["redshift"] - z_nr).abs().argsort()[:1]
                ].index.values[0]
                snap_nrs[sim_idx, z_idx] = snap_nr
                y_val[sim_idx, z_idx, :] = y_val_box["snap_%d" % snap_nr].values

        x_val = y_val_box.index.values
        ds = xr.Dataset(
            {labels["y"]: (["box", "redshift", labels["x"]], y_val)},
            coords={
                "redshift": z_nrs,
                "box": self.sim_nrs,
                labels["x"]: x_val,
                "snapshot": (["box", "redshift"], snap_nrs),
            },
        )
        self._stats_to_file(ds, file_dsc, dir_out)
    

    def compress_histograms(
        self,
        file_dsc: Dict[str, str],
        dir_out: str,
    ) -> None:
        """
        Args:
        """
        # initialize arrays that will contain results
        if file_dsc["extension"] == "h5":
            _stats_file = (
                self.sim[list(self.sim.keys())[0]].dirs["sim"]
                + "%s.h5" % file_dsc["root"]
            )
            y_val_box = pd.read_hdf(_stats_file, key="df")
            columns = y_val_box.columns.values
            y_val = np.zeros((
                len(self.sim_nrs),
                len(y_val_box.columns.values),
                len(y_val_box.index.values),
            ))
        else:
            raise SimulationCollectionWarning("File extension not supported")

        # loop over simulations in collection
        for sim_idx, sim_name in enumerate(list(self.sim.keys())):
            sim_df = self.config.loc[(sim_idx + 1,)]
            _stats_file = (
                self.sim[sim_name].dirs["sim"]
                + f"{file_dsc['root']}.{file_dsc['extension']}"
            )
            y_val_box = pd.read_hdf(_stats_file, key="df")
            for c_idx, col in enumerate(columns):
                y_val[sim_idx, c_idx, :] = y_val_box[col].values

        x_val = y_val_box.index.values
        ds = xr.Dataset(
            {"count": (["box", "property", "bin"], y_val)},
            coords={
                "box": self.sim_nrs,
                "property": columns,
                "bin": x_val,
            },
        )
        self._stats_to_file(ds, file_dsc, dir_out)

    
    def _stats_to_file(
        self, ds: xr.Dataset, file_dsc: Dict[str, str], dir_out: str
    ) -> None:
        """ Write xr.Dataset to .nc file """
        if not os.path.isdir(dir_out):
            Path(dir_out).mkdir(parents=False, exist_ok=True)
        file_out = dir_out + "%s.nc" % file_dsc["root"]
        print(f"Save in -> {file_out}")
        ds.to_netcdf(file_out)


    def sum_raytracing_snapshots(
        self,
        dir_out: str,
        columns: list,
        columns_z_shift: list,
        integration_range: dict,
        ray_file_root: str = "Ray_maps_output%05d.h5",
        sim_folder_root: str = "box%d",
        z_src: float = None,
        z_src_shift: float = None,
    ) -> None:
        """
        Adds different ray-tracing outputs together. This can give you the
        integrated ray-tracing quantities between arbitrary redshifts along
        the ligh-cone.
        """
        # sim_folder_root = self.dirs["lc"] + sim_folder_root
        box_ray_nrs = self._get_box_and_ray_nrs(integration_range)
        # loop over simulations in collection
        first = True
        for sim_idx, sim_name in enumerate(self.sim.keys()):
            _sim = self.sim[sim_name]
            box_nr = int(re.findall(r"\d+", sim_name)[0])
            if box_nr not in list(box_ray_nrs.keys()):
                continue

            for ray_nr in box_ray_nrs[box_nr]:
                sim_info_df = self.config.loc[(box_nr, ray_nr)]

                ray_file = glob.glob(
                    _sim.dirs["sim"]
                    + f"{_sim.file_dsc['root']}_*{ray_nr}."
                    + f"{_sim.file_dsc['extension']}"
                )[0]
                ray_map_df = pd.read_hdf(ray_file, key="df", mode="r")

                print(
                    "Box Nr. %d; %s; Redshift %.3f"
                    % (box_nr, os.path.basename(ray_file), sim_info_df["redshift"]),
                    len(ray_map_df.index.values),
                )

                if z_src_shift is not None and sim_info_df["redshift"] <= z_src_shift:
                    raise SimulationCollectionWarning(
                        "Redshift shift has not correct data structure"
                    )
                    # what snapshot to use if end of lightcone-box is reached
                    if (ray_box_info_df.name[1] == ray_nrs.min()) and (box_nr < 4):
                        z_next = self.config.loc[(box_nr + 1, 1)]["redshift"]
                    else:
                        z_next = ray_box_info_df.iloc[ii + 1]["redshift"]

                    # Shift redshift of light source
                    # only of kappa but not of iswrs !!!
                    ray_map_df["kappa_2"] = self._translate_redshift(
                        ray_map_df["kappa_2"],
                        sim_info_df["redshift"],
                        z_next,
                        z_src,
                        z_src_shift,
                    )

                if first is True:
                    ray_df_sum = ray_map_df
                    first = False

                else:
                    for column in columns:
                        ray_df_sum[column] = (
                            ray_df_sum[column].values + ray_map_df[column].values
                        )
        self._merged_snapshots_to_file(ray_df_sum, dir_out, integration_range)

    def _get_box_and_ray_nrs(self, integration_range: dict) -> dict:
        """ Get all box and ray-snapshot numbers for selected range """
        if not integration_range["z"]:
            if integration_range["box"][0] == 0:
                print("Integrate over whole light-cone")
                self.complete_lc = True
            elif integration_range["ray"][0] == 0:
                print("Integrate over box", integration_range["box"])
                self.config = self.config[
                    self.config.index.get_level_values(0).isin(integration_range["box"])
                ]
                self.complete_lc = False
        else:
            print("Integrate over redshift-range", integration_range["z"])
            # if merging based on redshift
            z_range = np.asarray(integration_range["z"])
            self.config = self.config[
                (z_range.min() < self.config["redshift"])
                & (self.config["redshift"] < z_range.max())
            ]
            self.complete_lc = False

        box_and_ray_nrs = {}
        for box_nr, ray_nr in self.config.index.values:
            box_and_ray_nrs.setdefault(box_nr, []).append(ray_nr)
        return box_and_ray_nrs

    def _translate_redshift(
        self,
        quantity: str,
        z_near: float,
        z_far: float,
        z_src: float,
        z_src_shift: float,
    ) -> float:
        """
        Shift ray-tracing quantity in redshift.

        Parameters
        ----------
            quantity pandas.DataSeries:
                ray-ramses output quantity
            x_near np.float:
                comoving distance closer to observer
            x_far np.float:
                comoving distance further from observer
            x_src np.float:
                source redshift used in ray-ramses simulation
        """
        x_far = self.cosmology.comoving_distance(z_far).to_value("Mpc")
        x_near = self.cosmology.comoving_distance(z_near).to_value("Mpc")
        x_src = self.cosmology.comoving_distance(z_src).to_value("Mpc")

        if z_far > z_src_shift:
            # if z of next snapshot larger than new source z, set the new source
            # equal to it, so that a distance of 150[Mpc/h] is maintained
            x_src_shift = self.cosmology.comoving_distance(z_far).to_value("Mpc")
        else:
            x_src_shift = self.cosmology.comoving_distance(z_src_shift).to_value("Mpc")

        x_mid = 0.5 * (x_far + x_near)

        quantity_shift = (
            quantity
            * self._kernel_function(x_mid, x_src_shift)
            / self._kernel_function(x_mid, x_src)
        )
        return quantity_shift

    def _kernel_function(self, x: float, x_s: float) -> float:
        """
        Args:
            x np.float:
                comoving distance
            x_s np.float:
                comoving distance to source
        
        Returns:
        """
        g = (x_s - x) * x / x_s
        return g

    def _merged_snapshots_to_file(
        self, ray_df_sum: pd.DataFrame, dir_out: str, integration_range: dict
    ) -> None:
        """
        Write merged ray-tracing pd.DataFrame to .h5 file
        Args:
        Returns:
        """
        if not integration_range["z"]:
            if integration_range["box"][0] == 0:
                fout = dir_out + "Ray_maps_lc.h5"
            elif integration_range["ray"][0] == 0:
                fout = dir_out + "Ray_maps_box%d.h5" % box_nr
        else:
            fout = dir_out + "Ray_maps_zrange_%.2f_%.2f.h5" % (
                self.config["redshift"].values.min(),
                self.config["redshift"].values.max(),
            )
        if not os.path.isdir(dir_out):
            Path(dir_out).mkdir(parents=True, exist_ok=True)
        print("Save in %s" % fout)
        ray_df_sum.to_hdf(fout, key="df", mode="w")
