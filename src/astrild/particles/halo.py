import os
from gc import collect
from pathlib import Path
from typing import List, Optional, Tuple, Type, Union
from importlib import import_module

import yaml
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

#from halotools.mock_observables import tpcf_multipole

from astrild.particles.ecosmog import Ecosmog
from astrild.particles.hutils import SubFind
from astrild.particles.hutils import Rockstar
#from astrild.particles.utils import TPCF
from astrild.utils.arepo_hdf5_library_2020 import read_hdf5
from astrild.io import IO

dir_src = Path(__file__).parent.absolute()
default_halo_stats_config = dir_src / "configs/halo_stats.yaml"

dm_particle_mass = 7.98408e10 #[Msun/h]

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
        from_subfind:
        from_rockstar:
        from_dataframe:
        from_file:
        get_subfind_stats:
        get_subfind_tpcf:
        get_rockstar_stats:
        get_rockstar_tpcf:
        filter_resolved_subfind_halos:
        filter_resolved_rockstar_halos:
        _save_results:
        _sort_statistics:
        _create_filename:
    """

    def __init__(
        self,
        halos: Union[read_hdf5.snapshot, pd.DataFrame],
        simulation: Optional[Type[Ecosmog]] = None,
    ):
        self.data = halos
        self.sim = simulation
        if hasattr(self.sim, "files") == False:
            self.halotype = None
        elif "fof" in list(self.sim.files.keys()):
            self.halotype = "Arepo"
        elif "halos" in list(self.sim.files.keys()):
            self.halotype = "Rockstar"

    @classmethod
    def from_subfind(
        cls, snap_nr: int, simulation: Optional[Type[Ecosmog]] = None,
    ) -> "Halos":
        """ """
        snapshot = read_hdf5.snapshot(
            snap_nr,
            simulation.dirs["sim"],
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
                "SubhaloPos",
                "SubhaloVel",
                "SubhaloMass",
                "SubhaloHalfmassRad",
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
        return cls(snapshot, simulation)
    

    @classmethod
    def from_rockstar(
        cls, snap_nr: int, simulation: Optional[Type[Ecosmog]] = None,
    ) -> "Halos":
        """
        Load halo data from Rockstar halo finder into pandas.DataFrame

        Args:
            snap_nr:
            simulation:
        """
        # TODO: currently only one directory supported, e.g. 012
        files_path = simulation.files["halos"][str(snap_nr)]
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
        return cls.from_dataframe(snapshot, simulation)
    
   
    @classmethod
    def from_file(
        cls, filename: str, simulation: Optional[Type[Ecosmog]] = None,
    ) -> "Halos":
        """ """
        df = pd.read_hdf(filename, key="df")
        return cls.from_dataframe(df, simulation)
   

    @classmethod
    def from_dataframe(
        cls, df: pd.DataFrame, simulation: Optional[Type[Ecosmog]] = None,
    ) -> "Halos":
        """ """
        return cls(df, simulation)


    def get_subfind_stats(
        self, config_file: str = default_halo_stats_config, save: bool = True,
    ) -> None:
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
        module = import_module("astrild.particles.hutils")

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

    def filter_resolved_subfind_halos(
        self, snapshot: read_hdf5.snapshot, nr_particles: int,
    ) -> read_hdf5.snapshot:
        """
        Filter halos with '> nr_particles' particles
        
        Args:

        Return:
        """
        min_mass = dm_particle_mass * nr_particles
        mass = snapshot.cat["Group_M_Crit200"][:] * snapshot.header.hubble  # [Msun/h]
        idx_groups = mass > min_mass
        mass = snapshot.cat["SubhaloMass"][:] * snapshot.header.hubble  # [Msun/h]
        idx_subhalos = mass > min_mass
        # idx = snapshot.cat["GroupLenType"][:, 1] > nr_particles
        # idx = snapshot.cat["Group_M_Crit200"][:] > \
        #    100*(snapshot.header.massarr[1] * 1e10 / snapshot.header.hubble)
        return self.filter_subfind_and_fof_halos(snapshot, idx_groups, idx_subhalos)
    
    def filter_nonzero_subfind_halos_size(
        self, snapshot: read_hdf5.snapshot,
    ) -> read_hdf5.snapshot:
        """
        Filter halos with non-zero size

        Args:

        Return:
        """
        rad = snapshot.cat["Group_R_Crit200"][:]  # [ckpc/h]
        idx_groups = rad > 0
        rad = snapshot.cat["SubhaloHalfmassRad"][:]  # [ckpc/h]
        idx_subhalos = rad > 0
        return self.filter_subfind_and_fof_halos(snapshot, idx_groups, idx_subhalos)
    
    def filter_subfind_and_fof_halos(
        self,
        snapshot: read_hdf5.snapshot,
        idx_groups: np.ndarray,
        idx_subhalos: np.ndarray,
    ) -> read_hdf5.snapshot:
        """ Filter sub- and fof-halos by indices """
        for key, value in snapshot.cat.items():
            if "Group" in key:
                idx = idx_groups
            elif ("Subhalo" in key) and (len(snapshot.cat[key]) > len(idx_groups)):
                idx = idx_subhalos
            else:
                HalosWarning(f"The key is {key} is a problem")
                continue

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
   

    #def get_subfind_tpcf(
    #    self,
    #    subfind_type: str,
    #    config: dict,
    #    save: bool = True,
    #) -> None:
    #    """
    #    Compute real- and redshift-space TPCF for halos. This computation is
    #    done using halotools.

    #    https://halotools.readthedocs.io/en/latest/index.html

    #    Args:
    #        subfind_type: ["Group", "Subhalo"]
    #        config:
    #        save:
    #            wether to save results to file.
    #    """
    #    tpcf = {}
    #    for l in config["multipoles"]:
    #        tpcf[str(l)] = {}
    #    multipoles = config["multipoles"]
    #    del config["multipoles"]

    #    for snap_nr in self.sim.dir_nrs:
    #        snapshot = self.get_subfind_halo_data(snap_nr)
    #        
    #        if snapshot is None:
    #            print(f"No sub- & halos found for snapshot {snap_nr}")
    #            continue

    #        snapshot = self.filter_resolved_subfind_halos(snapshot, 100)
    #      
    #        if subfind_type == "group":
    #            halo_pos = snapshot.cat["GroupPos"][:] * \
    #                snapshot.header.hubble / 1e3  #[Mpc/h]
    #            scale_factor = 1 / (1 + snapshot.header.redshift)
    #            print("test a -------", scale_factor)
    #            halo_vel = snapshot.cat["GroupVel"][:] / scale_factor #[km/s]
    #        if subfind_type == "subhalo":
    #            halo_pos = snapshot.cat["SubhaloPos"][:] * \
    #                snapshot.header.hubble / 1e3  #[Mpc/h]
    #            halo_vel = snapshot.cat["SubhaloVel"][:]  #[km/s]

    #        s_bins, mu_range, tpcf_s= TPCF.compute(
    #            pos=halo_pos,
    #            vel=halo_vel,
    #            **config,
    #            multipole=l,
    #        )
    #        for l in multipoles:
    #            _tpcf = tpcf_multipole(tpcf_s, mu_range, order=l)
    #            tpcf[str(l)]["snap_%d" % snap_nr] = _tpcf
    #            print(l, "!!!!!!!!!!!! snap_%d" % snap_nr, _tpcf)
    #    
    #    tpcf["s_bins"] = s_bins
    #    if save:
    #        IO.save_tpcf(
    #            self.sim.dirs['out'],
    #            config,
    #            multipoles,
    #            "subfind",
    #            "_"+subfind_type,
    #            tpcf,
    #        )
    #    else:
    #        self.tpcf = tpcf

    def get_rockstar_stats(
        self,
        config_file: str = default_halo_stats_config,
        snap_nrs: Optional[List[int]] = None,
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
        module = import_module("astrild.particles.hutils")

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


    #def get_rockstar_tpcf(
    #    self,
    #    config: dict,
    #    snap_nrs: Optional[List[int]] = None,
    #    save: bool = True,
    #) -> None:
    #    """
    #    Compute real- and redshift-space TPCF for halos. This computation is
    #    done using halotools.

    #    https://halotools.readthedocs.io/en/latest/index.html

    #    Args:
    #        config:
    #        save:
    #            wether to save results to file.
    #    """
    #    tpcf = {}
    #    for l in config["multipoles"]:
    #        tpcf[str(l)] = {}
    #    multipoles = config["multipoles"]
    #    del config["multipoles"]
    #    
    #    if snap_nrs is None:
    #        snap_nrs = self.sim.dir_nrs

    #    for snap_nr in snap_nrs:
    #        snapshot = self.get_rockstar_halo_data(
    #            self.sim.files["halos"][str(snap_nr)]
    #        )
    #        
    #        if snapshot is None:
    #            print(f"No sub- & halos found for snapshot {snap_nr}")
    #            continue

    #        snapshot = self.filter_resolved_rockstar_halos(snapshot, 100)
    #      
    #        halo_pos = snapshot[["x", "y", "z"]].values  #[Mpc/h]
    #        halo_vel = snapshot[["vx", "vy", "vz"]].values  #[km/s]

    #        s_bins, mu_range, tpcf_s= TPCF.compute(
    #            pos=halo_pos,
    #            vel=halo_vel,
    #            **config,
    #        )
    #        for l in multipoles:
    #            _tpcf = tpcf_multipole(tpcf_s, mu_range, order=l)
    #            tpcf[str(l)]["snap_%d" % snap_nr] = _tpcf
    #    
    #    tpcf["s_bins"] = s_bins

    #    if save:
    #        IO.save_tpcf(
    #            self.sim.dirs['out'],
    #            config,
    #            multipoles,
    #            "rockstar",
    #            "",
    #            tpcf,
    #        )
    #    else:
    #        self.tpcf = tpcf
    

    def filter_resolved_rockstar_halos(
        self, snapshot: pd.DataFrame, nr_particles: int,
    ) -> pd.DataFrame:
        """
        Filter halos with '> nr_particles' particles
        """
        min_mass = dm_particle_mass * nr_particles
        return snapshot[snapshot["m200c"] > min_mass]


    def _sort_statistics(self, statistics: dict) -> List[str]:
        """
        Sort statistics by their required particle resolution
        (low -to-> high).
        """
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
                if "seperate" in list(stg["args"].keys()):
                    compare = np.sum(stg["args"]["seperate"]["compare"])
                    if compare == 2:
                        compare = "11"
                    if compare == 3:
                        compare = "12"
                    if compare == 4:
                        compare = "22"
                else:
                    compare = "00"
                file_out = f"{self.sim.dirs['out']}{halofinder}_{method}_{compare}.h5"
                if os.path.exists(file_out):
                    os.remove(file_out)
                print(f"Saving results to -> {file_out}")
                df.to_hdf(file_out, key="df", mode="w")
            else:
                for snap_nr, stg_in_snap in stg["results"]["values"].items():
                    data = np.asarray(list(stg_in_snap.values())).T
                    columns = list(stg_in_snap.keys())
                    df = pd.DataFrame(data=data, columns=columns)
                    file_out = f"{self.sim.dirs['out']}{halofinder}_{method}" + \
                            "_{snap_nr}.h5"
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


    @staticmethod
    def get_nearest_neighbours(
        df: pd.DataFrame,
        target_id: int,
        dmax: Optional[int] = None,
        extent: Optional[int] = None,
    ) -> tuple:
        """
        Args:
            df: halo DataFrame
            target_id: object id for which to find NNs
            dmax: maximal distance between objects

        Return:
            indices and distances
        """
        pos = df[["theta1_deg", "theta2_deg"]].values
        pos_i = df[df["id"] == target_id][["theta1_deg", "theta2_deg"]].values
        if dmax is None:
            dmax = df[df["id"] == target_id]["r200_deg"].values
        if extent is not None:
            dmax *= extent
        if len(pos_i.shape) == 1:
            pos_i = pos_i[np.newaxis, :]
        btree = BallTree(pos)
        pairs = btree.query_radius(pos_i, dmax, return_distance=True,)
        return pairs[0][0], pairs[1][0]
