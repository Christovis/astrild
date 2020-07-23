import os
from gc import collect
from pathlib import Path
from typing import List, Optional, Tuple, Type
from importlib import import_module

import yaml
import numpy as np
import pandas as pd

from wys_ars.particles.ecosmog import Ecosmog
from wys_ars.particles.utils import SubFind
from wys_ars.particles.utils import Rockstar
from wys_ars.utils.arepo_hdf5_library import read_hdf5

dir_src = Path(__file__).parent.absolute()
default_halo_stats_config = dir_src / "configs/halo_stats.yaml"


class HalosWarning(BaseException):
    pass


class Halos:
    """
    Class to manage Rockstar & SubFind halos and get their statistics such as:
        - halo mass fct.
        - two point correlation fct.
        - concentration mass relation
        - pairwise velocity distribution

    Attributes:
        sim_type:
        simulation:

    Methods:
        get_subfind_halo_data:
        get_subfind_stats:
        get_rockstar_halo_data:
        get_rockstar_stats:
    """

    def __init__(self, simulation: Type[Ecosmog]):
        self.sim = simulation
        self.sim.type = "particles"


    def get_subfind_stats(
        self, config_file: str = default_halo_stats_config, save: bool = True,
    ):
        """
        Compute statistics of halos identified with SubFind from one or a
        collection of simulations.

        Args:
            config_file:
                file pointer in which containes info on what statistics to
                compute and their settings.
            save:
                wether to save results to file.
        """
        # load settings (stg)
        with open(config_file) as f:
            statistics = yaml.load(f, Loader=yaml.FullLoader)

        for name in statistics.keys():
            statistics[name]["results"] = {"bins": {}, "values": {}}
        # load particles/utils/stats.py package for dynamic function call
        module = import_module("wys_ars.particles.utils")

        # sort statistics according to required halos resolutions
        stat_names_ord = self._sort_statistics(statistics)

        for snap_nr in self.sim.dir_nrs:
            snapshot = self.get_subfind_halo_data(snap_nr)
            if snapshot is None:
                print(f"No sub- & halos found for snapshot {snap_nr}")
                continue

            resolution = 0
            for stat_name in stat_names_ord:
                if statistics[stat_name]["resolution"] != resolution:
                    resolution = int(statistics[stat_name]["resolution"])
                    snapshot = self.filter_resolved_subfind_halos(snapshot, resolution)
                print(f"     Compute {stat_name}")
                clas = getattr(module, "SubFind")
                fct = getattr(clas, stat_name)
                bins, values = fct(snapshot, **statistics[stat_name]["args"])
                if (bins is not None) and (values is not None):
                    statistics[stat_name]["results"]["bins"]["snap_%d" % snap_nr] = bins
                    statistics[stat_name]["results"]["values"][
                        "snap_%d" % snap_nr
                    ] = values
            collect()
        if save:
            self._save_results("subfind", statistics)
        else:
            self.statistics = statistics


    def get_subfind_halo_data(self, snap_nr: int) -> read_hdf5.snapshot:
        """ """
        snapshot = read_hdf5.snapshot(
            snap_nr,
            self.sim.dirs["sim"],
            part_type_list=["dm"],
            snapbases=["/snap-groupordered_"],
            # check_total_particle_number=True,
            # verbose=True,
        )
        snapshot.group_catalog(
            [
                "Group_M_Crit200",
                "Group_R_Crit200",
                "GroupPos",
                "GroupVel",
                "GroupFirstSub",
                "GroupLenType",
                "SubhaloVmax",
            ]
        )
        if snapshot.cat["n_groups"] == 0:
            snapshot = None
        else:
            snapshot.cat.update(
                {
                    "SubhaloVmax": snapshot.cat["SubhaloVmax"][
                        (snapshot.cat["GroupFirstSub"][:]).astype(np.int64)
                    ]
                }
            )
        return snapshot

    def filter_resolved_subfind_halos(
        self, snapshot: read_hdf5.snapshot, nr_particles: int,
    ) -> read_hdf5.snapshot:
        """
        Filter halos with '> nr_particles' particles
        """
        idx = snapshot.cat["GroupLenType"][:, 1] > nr_particles
        # idx = snapshot.cat["Group_M_Crit200"][:] > \
        #    100*(snapshot.header.massarr[1] * 1e10 / snapshot.header.hubble)
        for key, value in snapshot.cat.items():
            if len(value.shape) == 0:
                continue
            elif len(value.shape) == 1:
                snapshot.cat.update({key: value[idx]})
            elif len(value.shape) == 2:
                snapshot.cat.update({key: value[idx, :]})
            else:
                raise HalosWarning(
                    f"The group data {key} has weird dimensions: {value.shape}."
                )
        return snapshot

    def get_rockstar_stats(
        self,
        snap_nrs: Optional[List[int]] = None,
        config_file: str = default_halo_stats_config,
        save: bool = True,
    ):
        """
        Compute statistics of halos identified with Rockstar from one or a
        collection of simulations.

        rockstar:
            https://bitbucket.org/gfcstanford/rockstar/src/main/
            https://github.com/yt-project/rockstar
            https://www.cosmosim.org/cms/documentation/database-structure/tables/rockstar/

        Args:
            config_file:
                file pointer in which containes info on what statistics to
                compute and their settings.
            save:
                wether to save results to file.
        """
        # load settings (stg)
        with open(config_file) as f:
            statistics = yaml.load(f, Loader=yaml.FullLoader)

        for name in statistics.keys():
            statistics[name]["results"] = {"bins": {}, "values": {}}
        # load particles/utils/stats.py package for dynamic function call
        module = import_module("wys_ars.particles.utils")

        # sort statistics according to required halo resolutions
        stat_names_ord = self._sort_statistics(statistics)
        
        if snap_nrs is None:
            snap_nrs = self.sim.dir_nrs

        for snap_nr in snap_nrs:
            snapshot = self.get_rockstar_halo_data(
                self.sim.files["halos"][str(snap_nr)]
            )
            if len(snapshot.index.values) == 0:
                print(f"No sub- & halos found for snapshot {snap_nr}")
                continue

            resolution = 0
            for stat_name in stat_names_ord:
                if statistics[stat_name]["resolution"] != resolution:
                    resolution = int(statistics[stat_name]["resolution"])
                    snapshot = self.filter_resolved_rockstar_halos(
                        snapshot, resolution
                    )
                print(f"     Compute {stat_name}")
                clas = getattr(module, "Rockstar")
                fct = getattr(clas, stat_name)
                if stat_name != "histograms":
                    bins, values = fct(snapshot, **statistics[stat_name]["args"])
                    if (bins is not None) and (values is not None):
                        statistics[stat_name]["results"]["bins"]["snap_%d" % snap_nr] = bins
                        statistics[stat_name]["results"]["values"][
                            "snap_%d" % snap_nr
                        ] = values
                else:
                    hist = fct(snapshot, **statistics[stat_name]["args"])
                    statistics[stat_name]["results"]["values"]["snap_%d" % snap_nr] = hist
        if save:
            self._save_results("rockstar", statistics)
        else:
            self.statistics = statistics


    def get_rockstar_halo_data(self, files_path: list) -> pd.DataFrame:
        """ """
        # TODO: currently only one directory supported, e.g. 012
        first = True
        for file_path in files_path:
            snapshot_part = pd.read_csv(
                file_path, header=0, skiprows=np.arange(1, 20), delim_whitespace=True,
            )

            if first is True:
                snapshot = snapshot_part
                first = False
            else:
                snapshot = snapshot.append(snapshot_part, ignore_index=True)
        return snapshot


    def filter_resolved_rockstar_halos(
        self, snapshot: pd.DataFrame, nr_particles: int,
    ) -> pd.DataFrame:
        """
        Filter halos with '> nr_particles' particles
        """
        return snapshot[snapshot["num_p"] > nr_particles]


    def _sort_statistics(self, statistics: dict) -> list:
        """ Sort statistics by their required particle resolution """
        resolutions = np.zeros(len(list(statistics.keys())))
        for idx, (_, stg) in enumerate(statistics.items()):
            resolutions[idx] = int(stg["resolution"])
        idxs = np.argsort(resolutions)
        return [list(statistics.keys())[idx] for idx in idxs]


    def _save_results(self, halofinder: str, methods: dict):
        """
        Save results of each statistic of each simulations snapshot 
        for Rockstar and SubFind.
        """
        for method, stg in methods.items():
            if method != "histograms":
                columns = list(stg["results"]["bins"].keys())
                if len(self.sim.dir_nrs) > 1:
                    assert np.sum(stg["results"]["bins"][columns[0]]) == np.sum(
                        stg["results"]["bins"][columns[1]]
                    )
                df = pd.DataFrame(
                    data=stg["results"]["values"], index=stg["results"]["bins"][columns[0]],
                )
                file_out = self.sim.dirs["out"] + halofinder + "_" + method + ".h5"
                if os.path.exists(file_out):
                    os.remove(file_out)
                print(f"Saving results to -> {file_out}")
                df.to_hdf(file_out, key="df", mode="w")
            else:
                for snap_nr, stg_in_snap in stg["results"]["values"].items():
                    data = np.asarray(list(stg_in_snap.values())).T
                    columns = list(stg_in_snap.keys())
                    df = pd.DataFrame(data=data, columns=columns)
                    file_out = f"{self.sim.dirs['out']}{halofinder}_{method}_{snap_nr}.h5"
                    if os.path.exists(file_out):
                        os.remove(file_out)
                    print(f"Saving results to -> {file_out}")
                    df.to_hdf(file_out, key="df", mode="w")



    def _create_filename(self, file_in: str, quantity: str):
        """ Create file-name for merged snapshots"""
        quantity = quantity.replace("_", "")
        file_out = file_in.split("/")[-1].replace("Ray", quantity)
        file_out = file_out.replace(".h5", "_lt.fits")
        if ("_lc" not in file_in) or ("zrange" not in file_in):
            file_out = file_out.split("_")
            box_string = [string for string in file_in.split("/") if "box" in string][0]
            idx, string = [
                (idx, "%s_" % box_string + string)
                for idx, string in enumerate(file_out)
                if "output" in string
            ][0]
            file_out[idx] = string
            file_out = "_".join(file_out)
        return self.sim.dirs["out"] + file_out
