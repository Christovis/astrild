import os, sys, glob
import argparse
from struct import *
from pathlib import Path
from typing import List, Dict, Optional, Union, Type

import numpy as np
import pandas as pd

import astropy
from astropy import units as un
from astropy import constants as const
from astropy.cosmology import z_at_value, LambdaCDM

from astrild.simulation import Simulation
from astrild.particles.ecosmog import Ecosmog
from astrild.particles.halo import Halos
from astrild.utils.geometrical_transforms import (
    transform_box_to_lc_cart_coords,
    radial_coordinate_in_lc,
    angular_coordinate_in_lc,
)
from astrild.io import IO

dir_src = Path(__file__).parent.absolute()
default_config_file_df = dir_src / "configs/ray_snapshot_info.h5"


class RayRamsesWarning(BaseException):
    pass


class RayRamses(Simulation):
    """
    Attributes:
        dir_sim: directory of simulation
        dir_out: directory for results/returns
        file_dsc: file describtion for identification
        dir_root:
        opening_angle: light-cone opening angle; [deg^2]
        npix: nr. of cells along the four light-cone edges
        config: contains snapshot infos (e.g. redshifts, comov. dist, ...)

    Methods:
        compress_snapshot:
        sum_snapshots:
        find_halos_in_raytracing_box:
        find_halos_in_raytracing_snapshot:
    """

    def __init__(
        self,
        config: Union[None, pd.DataFrame],
        dir_sim: str,
        dir_out: str,
        file_dsc: dict = {"root": "Ray_maps", "extension": "dat"},
        dir_root: str = None,
        opening_angle: float = 20.0,
        npix: int = 8192,
        cosmo: astropy.cosmology = LambdaCDM,
    ):
        super().__init__(dir_sim, dir_out, file_dsc, dir_root)
        self.opening_angle = opening_angle
        self.npix = npix
        self.config = config
        self.cosmo = cosmo


    def compress_snapshot(
        self,
        fields: list,
        dir_out: str = None,
        convert: bool = False,
        cosmo: astropy.cosmology = LambdaCDM,
        save: bool = True,
    ) -> None:
        """
        Combines the ray-tracing outputs of individual CPUs at a given snapshot
        into one pandas .h5 file.

        Args:
        Returns:
        """
        self.cosmology = cosmo

        self.file_nrs = self.get_file_nrs(
            self.file_dsc, self.dirs["sim"], "min", True
        )
        self.files = {
            self.file_dsc["root"]: self.get_file_paths(
                self.file_dsc, self.dirs["sim"], uniques="min"
            )
        }
        # run through ray-ramses snapshots
        for ray_nr in np.unique(self.file_nrs):
            print("Ray-Nr %d" % ray_nr)
            idx = (np.where(self.file_nrs == ray_nr)[0]).astype(int)
            cpu_files = [
                self.files[self.file_dsc["root"]][idx]
                for idx, rnr in enumerate(self.file_nrs)
                if rnr == ray_nr
            ]
            first = True

            # run through cpu-files for snapshot
            for cpu_file in cpu_files:
                print("    %s" % cpu_file.split("/")[-1])

                ray_df = pd.read_csv(
                    cpu_file,
                    delim_whitespace=True,
                    skipinitialspace=True,
                    names=fields,
                    header=None,
                    lineterminator="\n",
                )

                if convert is True:
                    # convert comoving distance to [Gpc/h]
                    ray_df["chi_co"] /= self.cosmology.H0.value / 100

                    # Correct gamma_1 and gamma_2 to put in the form
                    # that normally appears in the literature
                    ray_df["shear_y"] *= 2.0 * sin(ray_df["the_co"])
                    gamm1_corr = -ray_df["shear_x"] * cos(
                        2.0 * ray_df["phi_co"]
                    ) - ray_df["shear_y"] * sin(2.0 * ray_df["phi_co"])
                    gamm2_corr = -ray_df["shear_x"] * sin(
                        2.0 * ray_df["phi_co"]
                    ) + ray_df["shear_y"] * sin(2.0 * ray_df["phi_co"])
                    ray_df["shear_x"] = gamm1_corr
                    ray_df["shear_y"] = gamm2_corr

                if first:
                    ray_collect_df = ray_df
                    first = False
                else:
                    ray_collect_df = ray_collect_df.append(ray_df)

            ray_collect_df = ray_collect_df.sort_values(
                by=["rayid"], axis=0, ascending=True
            )
            ray_collect_df = ray_collect_df.set_index("rayid")
            file_out = self.dirs["sim"] + "%s_output%05d.h5" % (
                self.file_dsc["root"],
                ray_nr,
            )
            ray_collect_df.to_hdf(file_out, key="df", mode="w")


    def sum_snapshots(
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
        the ligh-cone. The ray-tracing outputs must either have the format of
        RayRamses outputs in pd.DataFrames or np.ndarray images.

        Args:
            dir_out:
            columns:
            columns_z_shift:
            integration_range:
            ray_file_root:
            sim_folder_root:
            z_src:
            z_src_shift:
        """
        file_name = self.dirs["lc"] + "ray_snapshot_info.h5"
        if not os.path.isfile(file_name):
            raise RayRamsesWarning(
                "The file 'ray_snapshot_info.h5' does note exist"
            )
        self.ray_info_df = pd.read_hdf(file_name, key="s")
        sim_folder_root = self.dirs["lc"] + sim_folder_root

        box_ray_nrs = _get_box_and_ray_nrs(integration_range)

        first = True
        for box_nr, ray_nr in box_ray_nrs:
            sim_info_df = self.ray_info_df.loc[(box_nr, ray_nr)]
            self.dirs["sim"] = sim_folder_root % box_nr + "/"
            ray_file = self.dirs["sim"] + ray_file_root % ray_nr
            ray_map_df = pd.read_hdf(ray_file)

            print(
                "Box Nr. %d; %s; Redshift %.3f"
                % (box_nr, os.path.basename(ray_file), sim_info_df["redshift"]),
                len(ray_map_df.index.values),
            )

            if (
                z_src_shift is not None
                and sim_info_df["redshift"] <= z_src_shift
            ):
                raise RayRamsesWarning(
                    "Redshift shift has not correct data structure"
                )
                # what snapshot to use if end of lightcone-box is reached
                if (ray_box_info_df.name[1] == ray_nrs.min()) and (box_nr < 4):
                    z_next = self.ray_info_df.loc[(box_nr + 1, 1)]["redshift"]
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


    def _get_box_and_ray_nrs(self, integration_range: dict) -> np.ndarray:
        """
        Get all box and ray-snapshot numbers for selected range.
        
        Args:
        Returns:
        """
        if not integration_range["z"]:
            if integration_range["box"][0] == 0:
                print("Integrate over whole light-cone")
                self.complete_lc = True
            elif integration_range["ray"][0] == 0:
                print("Integrate over box", integration_range["box"])
                self.ray_info_df = ray_info_df[
                    ray_info_df.index.get_level_values(0).isin(
                        integration_range["box"]
                    )
                ]
                self.complete_lc = False
        else:
            print("Integrate over redshift-range", integration_range["z"])
            # if merging based on redshift
            z_range = np.asarray(integration_range["z"])
            self.ray_info_df = self.ray_info_df[
                (z_range.min() < self.ray_info_df["redshift"])
                & (self.ray_info_df["redshift"] < z_range.max())
            ]
            self.complete_lc = False

        return self.ray_info_df.index.values


    def _translate_redshift(
        self,
        quantity: str,
        z_near: float,
        z_far: float,
        z_src: float,
        z_src_shift: float,
    ) -> float:
        """
        Args:
            quantity pandas.DataSeries:
                ray-ramses output quantity
            x_near np.float:
                comoving distance closer to observer
            x_far np.float:
                comoving distance further from observer
            x_src np.float:
                source redshift used in ray-ramses simulation

        Returns:
        """
        x_far = self.cosmology.comoving_distance(z_far).to_value("Mpc")
        x_near = self.cosmology.comoving_distance(z_near).to_value("Mpc")
        x_src = self.cosmology.comoving_distance(z_src).to_value("Mpc")

        if z_far > z_src_shift:
            # if z of next snapshot larger than new source z, set the new source
            # equal to it, so that a distance of 150[Mpc/h] is maintained
            x_src_shift = self.cosmology.comoving_distance(z_far).to_value(
                "Mpc"
            )
        else:
            x_src_shift = self.cosmology.comoving_distance(
                z_src_shift
            ).to_value("Mpc")

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
        """
        if not integration_range["z"]:
            if integration_range["box"][0] == 0:
                fout = dir_out + "Ray_maps_lc.h5"
                print("Save in %s" % fout)
                ray_df_sum.to_hdf(fout, key="df", mode="w")
            elif integration_range["ray"][0] == 0:
                fout = dir_out + "Ray_maps_box%d.h5" % box_nr
                print("Save in %s" % fout)
                ray_df_sum.to_hdf(fout, key="df", mode="w")
        else:
            fout = dir_out + "Ray_maps_zrange_%.2f_%.2f.h5" % (
                self.ray_info_df["redshift"].values.min(),
                self.ray_info_df["redshift"].values.max(),
            )
            print("Save in %s" % fout)
            ray_df_sum.to_hdf(fout, key="df", mode="w")

    @staticmethod
    def find_subfind_halos_in_raytracing_snapshot(
        ecosmog: Type[Ecosmog],
        box_nr: int,
        snap_nr: int,
        ray_nr: int,
        boxdist: float,
        boxsize: float,
        opening_angle: float,
        snaplimit: tuple,
        hubble: float,
    ) -> Union[None, pd.DataFrame]:
        coeff = hubble / 1e3
        halos = Halos.from_subfind(snap_nr, ecosmog)
        
        if halos.data is None:
            return None
        else:
            halocat = halos.filter_nonzero_subfind_halos_size(halos.data).cat
        
        halocatindex = np.arange(len(halocat["Group_M_Crit200"][:]))
        print(f"There are {len(halocatindex)} halos in box {box_nr} snapshot {snap_nr}")

        # cartesian coord. in light-cone [Mpc/h]
        pos = halocat["GroupPos"][:, :] * coeff
        pos[:, 0] = pos[:, 0] - boxsize / 2
        pos[:, 1] = pos[:, 1] - boxsize / 2
        pos[:, 2] = pos[:, 2] + boxdist
        # radial distance from observer [Mpc/h]
        rad_dist = np.sqrt(pos[:, 0]**2 + pos[:, 1]**2 + pos[:, 2]**2)
        # angular coord [deg]
        theta1_deg = np.arctan(pos[:, 0] / pos[:, 2]) * 180/np.pi
        theta2_deg = np.arctan(pos[:, 1] / pos[:, 2]) * 180/np.pi

        # index of halos in light-cone
        indx = np.where(
            (rad_dist >= np.min(snaplimit))
            & (rad_dist <= np.max(snaplimit))
            & (np.abs(theta1_deg) <= opening_angle / 2)
            & (np.abs(theta2_deg) <= opening_angle / 2)
        )[0]
        print(f"There are {len(indx)} halos in light-cone in box {box_nr} snapshot {snap_nr}")

        # project 3D velocity along line-of-sight in cart. coord.
        pos_norm = np.linalg.norm(pos[indx, :], axis=1)
        vr = (
            (
                halocat["GroupVel"][indx, :] * pos[indx, :] # element-wise dot-product
            ).sum(axis=1) / (pos_norm**2)
        )[:, np.newaxis] * pos[indx, :]
        # project 3D velocity along transverse to line-of-sight direction
        # in cart. coord.
        vt = halocat["GroupVel"][indx, :] - vr
        # small angle approximation
        # -> TODO improve by projecting on spher. coord. unit vectors e_theta and e_phi
        vel_x = vt[:, 0]
        vel_y = vt[:, 1]

        r200_deg = (
            np.arctan(halocat["Group_R_Crit200"][indx] * coeff / rad_dist[indx])
            * 180 / np.pi
        )

        halo_id = [
            int(f"{box_nr}{snap_nr}{ii}")
            for ii in halocatindex[indx].astype(int)
        ]
        halos_dict = {
            "id": halo_id,
            "x": pos[indx, 0],
            "y": pos[indx, 1],
            "z": pos[indx, 2],
            "rad_dist": rad_dist[indx],
            "theta1_deg": theta1_deg[indx] + opening_angle / 2,
            "theta1_pix": _degree_to_pixel(
                theta1_deg[indx] + opening_angle / 2
            ),
            "theta2_deg": theta2_deg[indx] + opening_angle / 2,
            "theta2_pix": _degree_to_pixel(
                theta2_deg[indx] + opening_angle / 2
            ),
            "x_vel": halocat["GroupVel"][indx, 0],
            "y_vel": halocat["GroupVel"][indx, 1],
            "z_vel": halocat["GroupVel"][indx, 2],
            "theta1_tv": vel_x,
            "theta2_tv": vel_y,
            "m200": halocat["Group_M_Crit200"][indx],
            "r200_deg": r200_deg,
            "r200_pix": _degree_to_pixel(r200_deg),
            "ray_nr": [ray_nr + 1] * len(indx),
            "snap_nr": [snap_nr] * len(indx),
        }
        halos_df = pd.DataFrame(data=halos_dict)
        return halos_df


    def find_halos_in_raytracing_box(
        self,
        ecosmog: Type[Ecosmog],
        snapdist: float,
        box_nr: int,
        boxsize: float,
        hubble: float,
    ) -> Union[None, pd.DataFrame]:
        """
        Args:

        Returns:
        """
        boxdist = snapdist[-1]
        first = True
        # run through ray-ramses snapshots
        for ray_nr in np.unique(self.file_nrs)[:-1]:
            snap_nr = (
                ray_nr + len(ecosmog.config.index) - len(self.config.index)
            )
            snaplimit = (snapdist[ray_nr - 1], snapdist[ray_nr])
            
            if "fof" in list(ecosmog.files.keys()):
                halos_df = self.find_subfind_halos_in_raytracing_snapshot(
                    ecosmog,
                    box_nr,
                    snap_nr,
                    ray_nr,
                    boxdist,
                    boxsize,
                    self.opening_angle,
                    self.npix,
                    snaplimit,
                    hubble,
                )
            elif "halos" in list(ecosmog.files.keys()):
                halos_df = self.find_rockstar_halos_in_raytracing_snapshot(
                    ecosmog,
                    box_nr,
                    snap_nr,
                    ray_nr,
                    boxdist,
                    boxsize,
                    self.opening_angle,
                    self.npix,
                    snaplimit,
                )

            if (first is True) and (halos_df is not None):
                halos_df_sum = halos_df
                first = False
            elif first is False:
                halos_df_sum = halos_df_sum.append(halos_df, ignore_index=True)
        return halos_df_sum
    

    @staticmethod
    def find_subfind_halos_in_raytracing_snapshot(
        ecosmog: Type[Ecosmog],
        box_nr: int,
        snap_nr: int,
        ray_nr: int,
        boxdist: float,
        boxsize: float,
        opening_angle: float,
        npix: int,
        snaplimit: tuple,
        hubble: float,
    ) -> Union[None, pd.DataFrame]:
        coeff = hubble / 1e3
        halos = Halos.from_subfind(snap_nr, ecosmog)
        
        if halos.data is None:
            return None
        else:
            halocat = halos.filter_nonzero_subfind_halos_size(halos.data).cat
        
        halocatindex = np.arange(len(halocat["Group_M_Crit200"][:]))
        print(f"There are {len(halocatindex)} halos in box {box_nr} snapshot {snap_nr}")

        # cartesian coord. in light-cone [Mpc/h]
        pos = halocat["GroupPos"][:, :] * coeff
        pos = transform_box_to_lc_cart_coords(pos, boxsize, boxdist)
        rad_dist = radial_coordinate_in_lc(pos)
        theta1_deg, theta2_deg = angular_coordinate_in_lc(pos, unit="deg")

        # index of halos in light-cone
        indx = np.where(
            (rad_dist >= np.min(snaplimit))
            & (rad_dist <= np.max(snaplimit))
            & (np.abs(theta1_deg) <= opening_angle / 2)
            & (np.abs(theta2_deg) <= opening_angle / 2)
        )[0]
        print(f"There are {len(indx)} halos in light-cone in box {box_nr} snapshot {snap_nr}")

        # project 3D velocity along line-of-sight in cart. coord.
        pos_norm = np.linalg.norm(pos[indx, :], axis=1)
        vr = (
            (
                halocat["GroupVel"][indx, :] * pos[indx, :] # element-wise dot-product
            ).sum(axis=1) / (pos_norm**2)
        )[:, np.newaxis] * pos[indx, :]
        # project 3D velocity along transverse to line-of-sight direction
        # in cart. coord.
        vt = halocat["GroupVel"][indx, :] - vr
        # small angle approximation
        # -> TODO improve by projecting on spher. coord. unit vectors e_theta and e_phi
        vel_x = vt[:, 0]
        vel_y = vt[:, 1]

        r200_deg = (
            np.arctan(halocat["Group_R_Crit200"][indx] * coeff / rad_dist[indx])
            * 180 / np.pi
        )

        halo_id = [
            int(f"{box_nr}{snap_nr}{ii}")
            for ii in halocatindex[indx].astype(int)
        ]
        halos_dict = {
            "id": halo_id,
            "x": pos[indx, 0],
            "y": pos[indx, 1],
            "z": pos[indx, 2],
            "rad_dist": rad_dist[indx],
            "theta1_deg": theta1_deg[indx] + opening_angle / 2,
            "theta1_pix": _degree_to_pixel(
                theta1_deg[indx] + opening_angle / 2, opening_angle, npix
            ),
            "theta2_deg": theta2_deg[indx] + opening_angle / 2,
            "theta2_pix": _degree_to_pixel(
                theta2_deg[indx] + opening_angle / 2, opening_angle, npix
            ),
            "x_vel": halocat["GroupVel"][indx, 0],
            "y_vel": halocat["GroupVel"][indx, 1],
            "z_vel": halocat["GroupVel"][indx, 2],
            "theta1_tv": vel_x,
            "theta2_tv": vel_y,
            "m200": halocat["Group_M_Crit200"][indx],
            "r200_deg": r200_deg,
            "r200_pix": _degree_to_pixel(r200_deg, opening_angle, npix),
            "ray_nr": [ray_nr + 1] * len(indx),
            "snap_nr": [snap_nr] * len(indx),
        }
        halos_df = pd.DataFrame(data=halos_dict)
        return halos_df


    @staticmethod
    def find_rockstar_halos_in_raytracing_snapshot(
        ecosmog: Type[Ecosmog],
        box_nr: int,
        snap_nr: int,
        ray_nr: int,
        boxdist: float,
        boxsize: float,
        opening_angle: float,
        npix: int,
        snaplimit: tuple,
    ) -> Union[None, pd.DataFrame]:
        halos = Halos.from_rockstar(snap_nr, ecosmog)
        
        if halos.data is None:
            return None
        else:
            halocat = halos.data
        
        halocatindex = np.arange(len(halocat["m200c"].values))
        print(f"There are {len(halocatindex)} halos in box {box_nr} snapshot {snap_nr}")

        # cartesian coord. in light-cone [Mpc/h]
        pos = halocat[["x", "y", "z"]].values
        pos = transform_box_to_lc_cart_coords(pos, boxsize, boxdist)
        rad_dist = radial_coordinate_in_lc(pos)
        theta1_deg, theta2_deg = angular_coordinate_in_lc(pos, unit="deg")

        # index of halos in light-cone
        indx = np.where(
            (rad_dist >= np.min(snaplimit))
            & (rad_dist <= np.max(snaplimit))
            & (np.abs(theta1_deg) <= opening_angle / 2)
            & (np.abs(theta2_deg) <= opening_angle / 2)
        )[0]
        halocat = halocat.iloc[halocatindex[indx]]
        if len(indx) == 0: return None

        pos = pos[halocatindex[indx], :]
        rad_dist = rad_dist[halocatindex[indx]]
        theta1_deg = theta1_deg[halocatindex[indx]]
        theta2_deg = theta2_deg[halocatindex[indx]]
        print(f"There are {len(halocat.index.values)} halos in light-cone in box {box_nr} snapshot {snap_nr}")

        # get redshift
        redshift = Dc_to_redshift(ecosmog.cosmo, rad_dist * un.Mpc)

        # get angular distance

        # project 3D velocity along line-of-sight in cart. coord.
        pos_norm = np.linalg.norm(pos, axis=1)
        vr = (
            (
                halocat[["vx", "vy", "vz"]].values * pos # element-wise dot-product
            ).sum(axis=1) / (pos_norm**2)
        )[:, np.newaxis] * pos
        # project 3D velocity along transverse to line-of-sight direction
        # in cart. coord.
        vt = halocat[["vx", "vy", "vz"]].values - vr
        # small angle approximation
        # -> TODO improve by projecting on spher. coord. unit vectors e_theta and e_phi

        r200_deg = (
            np.arctan(halocat["r200c"].values / (rad_dist * 1e3))
            * 180 / np.pi
        )

        halo_id = [
            int(f"{box_nr}{snap_nr}{ii}")
            for ii in halocatindex[indx].astype(int)
        ]
        halos_dict = {
            "id": halo_id,
            "x": pos[:, 0],
            "y": pos[:, 1],
            "z": pos[:, 2],
            "Dc": rad_dist,
            "Da": rad_dist / (1 + redshift),
            "redshift": redshift,
            "theta1_deg": theta1_deg + opening_angle / 2,
            "theta1_pix": _degree_to_pixel(theta1_deg + opening_angle/2, opening_angle, npix),
            "theta2_deg": theta2_deg + opening_angle / 2,
            "theta2_pix": _degree_to_pixel(theta2_deg + opening_angle/2, opening_angle, npix),
            "x_vel": halocat["vx"].values,
            "y_vel": halocat["vy"].values,
            "z_vel": halocat["vz"].values,
            "theta1_tv": vt[:, 0],
            "theta2_tv": vt[:, 1],
            "m200": halocat["m200c"].values,
            "c_NFW": halocat["r200c"].values / halocat["Rs"].values,
            "r200_deg": r200_deg,
            "r200_pix": _degree_to_pixel(r200_deg, opening_angle, npix),
            "ray_nr": [ray_nr + 1] * len(pos),
            "snap_nr": [snap_nr] * len(pos),
        }
        halos_df = pd.DataFrame(data=halos_dict)
        return halos_df
        

def _degree_to_pixel(deg: np.ndarray, opening_angle, npix) -> np.ndarray:
    """ Convert degree to pixel position """
    return np.ceil(deg * npix / opening_angle).astype(int)


def Dc_to_redshift(cosmo: astropy.cosmology, Dc: un.quantity.Quantity,) -> np.array:
    """ Return redshift at comoving distance [Mpc] """
    return np.array([z_at_value(cosmo.comoving_distance, dist) for dist in Dc])

