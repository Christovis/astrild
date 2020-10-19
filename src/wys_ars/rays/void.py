import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type

import pandas as pd
import xarray as xr
import numpy as np
from scipy.interpolate import interp1d
from scipy import integrate
from astropy import units as u

from wys_ars.rays.utils import object_selection
from wys_ars.simulation import Simulation
from wys_ars.rays.skymap import SkyMap
from wys_ars.profiles import profile_2d as Profiles2D
from wys_ars import io as IO


class VoidsWarning(BaseException):
    pass


class Voids:
    """
    Class to manage void finders such as tunnels, svf/zobov, and watershed.

    Attributes:
        dataset_file:
        data:
        finder_spec:
        skymap_dsc:

    Methods:
        filter_size:
        filter_sigma:
        filter_snapshot:
        get_profiles:
        get_profile_stats:
    """

    def __init__(
        self,
        dataset_file: str,
        data: pd.DataFrame,
        finder_spec: dict,
        skymap_dsc: dict,
    ):
        self.dataset_file = dataset_file
        self.data = data
        self.finder_spec = finder_spec
        self.skymap_dsc = skymap_dsc
        print(
            f"There are {len(self.data.index)} "
            + f"{self.finder_spec['name']} voids"
        )

    @classmethod
    def from_file(
        cls,
        finder: str, skymap_dsc: dict,
        ffile: Optional[str] = None,
        file_dsc: Optional[dict] = None,
    ) -> "Voids":
        """
        Read void data files.

        Args:
            file_dsc: {path, root, extension}
        """
        if ffile is None:
            sim = Simulation(file_dsc["path"], None, file_dsc.pop("path"), None)
            _file = sim.files[file_dsc["root"]][0]
        else:
            _file = ffile

        if finder == "tunnels":
            data = pd.read_hdf(_file, key="df")
            finder_spec = {
                "name": finder,
                "sigmas": {"name": "sigma", "values": data["sigma"].unique()},
            }
        elif finder == "svf":
            fname = args.obj_dir + "%s/%s/SVF_lc%d.h5" % (
                args.finder,
                args.hod,
                lc_nr,
            )
            data = pd.read_hdf(fname, key="df")
            finder_spec = {
                "name": finder,
                "sigmas": {
                    "name": "void_overlap",
                    "values": data["void_overlap"].unique(),
                },
            }
            if args.nr_radius_bins == 9999:
                nr_radius_bins = (
                    data.groupby(["ray_nr", "void_overlap"])
                    .count()
                    .x_deg.values.max()
                )
            else:
                nr_radius_bins = args.nr_radius_bins
        elif finder == "zobov":
            fname = args.obj_dir + "%s/%s/ZOBOV_lc%d.h5" % (
                args.finder,
                args.hod,
                lc_nr,
            )
            data = pd.read_hdf(fname, key="df")
            finder_spec = { "name": finder, "sigmas": {
                    "name": "void_min_den",
                    "values": np.linspace(
                        data["void_min_den"].min(),
                        data["void_min_den"].max(),
                        3,
                    ),
                },
            }
        elif finder == "wvf":
            data = pd.read_hdf(_file, key="df")
            finder_spec = None
            finder_spec = {
                "name": finder,
            }
        return Voids(_file, data, finder_spec, skymap_dsc)

    def _read_skymap(self, file_in: str) -> np.ndarray:
        """ load convergence maps in numpy format """
        extension = file_in.split(".")[-1]
        if extension == "npy":
            skymap = np.load(file_in)
        elif extension == "npz":
            _tmp = np.load(file_in)
            skymap = _tmp[
                "arr_0"
            ]  # dict-key comes from lenstools.ConvergenceMap
        return skymap


    def get_void_size_fct(
        self,
        nbins: int,
        limits: Optional[tuple] = None,
        save: bool = True,
        mapp: Optional[str] = None,
        dir_out: Optional[str] = None,
    ) -> pd.DataFrame:
        """ """
        for idx, nu in enumerate(np.unique(self.data["sigma"].values)):
            nu_idx = self.data["sigma"].values == nu
            rad_deg = self.data["rad_deg"].values[nu_idx]

            if limits is None:
                lower_bound = np.percentile(rad_deg, 5)
                upper_bound = np.percentile(rad_deg, 95)
            else:
                lower_bound = min(limits[idx])
                upper_bound = max(limits[idx])
            print("------------------------------------", len(rad_deg), lower_bound, upper_bound)

            bins = np.arange(
                lower_bound, upper_bound, (upper_bound - lower_bound) / nbins,
            )

            _hist, _rad = np.histogram(
                self.data["rad_deg"].values, bins=bins, range=limits, density=False,
            )
            _hist = np.cumsum(_hist[::-1])[::-1]
            _rad = (_rad[1:] + _rad[:-1]) / 2.0
            void_counts_dic = {"rad": _rad, "counts": _hist}
            void_counts_df = pd.DataFrame(data=void_counts_dic)
            
            if save:
                filename = f"tunnel_void_counts_zrange_0.08_0.90_{mapp}_nu{int(nu)}.h5"
                IO.save_dataFrame(dir_out, filename, void_counts_df)
        

    def get_profiles(
        self,
        radii_max: float,
        nr_rad_bins: int,
        void_resolution: int = 10,
        skymap_file: Optional[str] = None,
        skymap: Optional[np.ndarray] = None,
        field_conversion: Optional[str] = None,
        dir_out: Optional[str] = None,
        save: bool = False,
    ) -> None:
        """
        Find the profile of voids on wys_ars.rays.SkyMap

        Args:
            radii_max:
                Radii out to which the profile is measured.
            nr_rad_bins:
                nr. of radii at which profile is measured.

        Returns:
        """
        # get skymap data from which to find profiles
        self.field_conversion = field_conversion
        if skymap is None:
            if skymap_file is None:
                _skymap = self._read_skymap(self.skymap_dsc["file"])
            else:
                _skymap = self._read_skymap(skymap_file)
        else:
            _skymap = skymap

        if self.field_conversion is "normalize":
            # center map on zero
            _skymap -= np.mean(_skymap)

        if self.finder_spec["name"] in ["tunnels", "wvf"]:
            self.data = self._trim_edges(
                self.data, radii_max, self.skymap_dsc["npix"],
            )
            self.data = self.data.reset_index()
        # only resolved voids
        self.data = self.data[radii_max*self.data["rad_pix"] > void_resolution]
        self.data = self.data.reset_index()
        print(
            f"Get the profile of {len(self.data.index)} "
            + f"{self.finder_spec['name']} voids"
        )
        self.profiles = Profiles2D.from_map(
            self.data, _skymap, radii_max, nr_rad_bins,
        )
        if save:
            dir_out = '/'.join(self.dataset_file.split("/")[:-1])
            if self.field_conversion is None:
                self.field_conversion = ''
            filename = self.field_conversion + '_' + \
                ''.join(self.dataset_file.split("/")[-1].split(".")[:-1])
            filename = f"{dir_out}/signle_profiles_{filename}.h5"
            print(f"Save profiles in -> {filename}")
            df = pd.DataFrame(
                data=self.profiles["values"].T,
                index=self.profiles["radii"],
                columns=self.data.index.values,
            )
            df.to_hdf(filename, key='df', mode='w')

    def get_profile_stats(
        self,
        cats: Optional[List[str]] = None,
        field_conversion: Optional[str] = None,
        dir_out: Optional[str] = None,
        save: bool = False,
    ) -> None:
        """
        Get profile statistics: mean, 16% and 84% of un- or categorized voids.

        Args:
            cats:
                What categorizations to use.
        """
        if field_conversion:
            self.field_conversion = field_conversion
        
        if cats:
            # initialize result arrays
            cat_per_cats = [len(np.unique(self.data[cat].values)) for cat in cats]
            cats_dict = {cat:np.unique(self.data[cat].values) for idx, cat in enumerate(cats)}
            nr_rad_bins = len(self.profiles["radii"])
            _mean = np.zeros(tuple(cat_per_cats + [nr_rad_bins]))
            _low_err = np.zeros(tuple(cat_per_cats + [nr_rad_bins]))
            _high_err = np.zeros(tuple(cat_per_cats + [nr_rad_bins]))
            _obj_size_min = np.zeros(tuple(cat_per_cats))
            _obj_size_max = np.zeros(tuple(cat_per_cats))
            _obj_in_cat = np.zeros(tuple(cat_per_cats))

            #TODO create dynamic for-loops via recursion
            for ss, sigma in enumerate(cats_dict["sigma"]):
                # filter voids
                _void_in_cat = self.data.loc[self.data["sigma"] == sigma]
                print(
                    f"There are {len(_void_in_cat.index.values)} "
                    + f"profiles for nu=(sigma)"
                )
                # clean-up and averaging profiles
                _mean[ss, :] = Profiles2D.mean_and_interpolate(
                    self.profiles["values"][_void_in_cat.index.values, :],
                    _void_in_cat["rad_pix"].values,
                    self.profiles["radii"].max(),
                    len(self.profiles["radii"]),
                )
                if self.field_conversion == 'tangential_shear':
                    _mean[ss, :] = self._compute_tangential_shear(
                        self.profiles["radii"], _mean[ss, :],
                    )
                # apply bootstrap to get errors
                error = Profiles2D.bootstrapping(
                    self.profiles["values"][_void_in_cat.index.values, :],
                    _mean[ss, :],
                    _void_in_cat,
                    self.skymap_dsc["npix"],
                    self.profiles["radii"].max(),
                    len(self.profiles["radii"]),
                )
                _low_err[ss, :] = error[0, :]
                _high_err[ss, :] = error[1, :]

                # min. & max. void size in bin
                #(* 3600 * u.arcsec * object_dist).to(u.Mpc, u.dimensionless_angles()) / u.Mpc
                _obj_size_min[ss] = _void_in_cat["rad_deg"].min()
                _obj_size_max[ss] = _void_in_cat["rad_deg"].max()
                _obj_in_cat[ss] = len(_void_in_cat.index)
            ds = xr.Dataset(
                {
                    "mean": (list(cats_dict.keys()) + ["radius"], _mean),
                    "lowerr": (list(cats_dict.keys()) + ["radius"], _low_err),
                    "higherr": (list(cats_dict.keys()) + ["radius"], _high_err),
                },
                coords={
                    "sigma": cats_dict["sigma"],
                    "radius": self.profiles["radii"],
                    "size_min": (list(cats_dict.keys()), _obj_size_min),
                    "size_max": (list(cats_dict.keys()), _obj_size_max),
                    "nr_of_obj": (list(cats_dict.keys()), _obj_in_cat),
                },
            )
        else:
            # initialize result arrays
            nr_rad_bins = len(self.profiles["radii"])
            _mean = np.zeros(nr_rad_bins)
            _low_err = np.zeros(nr_rad_bins)
            _high_err = np.zeros(nr_rad_bins)
            _obj_size_min = np.zeros(1)
            _obj_size_max = np.zeros(1)
            _obj_in_cat = np.zeros(1)

            #TODO create dynamic for-loops via recursion
            # filter voids
            _void_in_cat = self.data
            # clean-up and averaging profiles
            _mean[:] = Profiles2D.mean_and_interpolate(
                self.profiles["values"][_void_in_cat.index.values, :],
                _void_in_cat["rad_pix"].values,
                self.profiles["radii"].max(),
                len(self.profiles["radii"]),
            )
            if self.field_conversion == 'tangential_shear':
                _mean[:] = self._compute_tangential_shear(
                    self.profiles["radii"], _mean[:],
                )
            # apply bootstrap to get errors
            error = Profiles2D.bootstrapping(
                self.profiles["values"][_void_in_cat.index.values, :],
                _mean[:],
                _void_in_cat,
                self.skymap_dsc["npix"],
                self.profiles["radii"].max(),
                len(self.profiles["radii"]),
            )
            _low_err[:] = error[0, :]
            _high_err[:] = error[1, :]

            # min. & max. void size in bin
            #(* 3600 * u.arcsec * object_dist).to(u.Mpc, u.dimensionless_angles()) / u.Mpc
            _obj_size_min[0] = _void_in_cat["rad_deg"].min()
            _obj_size_max[0] = _void_in_cat["rad_deg"].max()
            _obj_in_cat[0] = len(_void_in_cat.index)
            
            ds = xr.Dataset(
                {
                    "mean": (["radius"], _mean),
                    "lowerr": (["radius"], _low_err),
                    "higherr": (["radius"], _high_err),
                },
                coords={
                    "radius": self.profiles["radii"],
                    "size_min": _obj_size_min,
                    "size_max": _obj_size_max,
                    "nr_of_obj": _obj_in_cat,
                },
            )

        if save:
            dir_out = '/'.join(self.dataset_file.split("/")[:-1])
            if self.field_conversion is None:
                self.field_conversion = ''
            filename = self.field_conversion + '_' + \
                ''.join(self.dataset_file.split("/")[-1].split(".")[:-1])
            filename = f"{dir_out}/profile_{filename}.nc"
            print(f"Save profile statistics in -> {filename}")
            ds.to_netcdf(filename)

    def _trim_edges(
        self, voids: pd.DataFrame, radii_max: float, npix: int
    ) -> None:
        """
        Remove voids whose radii goes over the simulation boundary.

        Args:
            radii_max: npix:
                Nr. of pixels on edge of SkyMap
        """
        return object_selection.trim_edges(voids, radii_max, npix)

    def filter_snapshot(self, ray_nr: int) -> None:
        """ """
        if hasattr(self, "filtered"):
            voids.filtered = self.filtered[self.filtered["ray_nr"] == ray_nr]
        else:
            voids.filtered = self.data[self.data["ray_nr"] == ray_nr]

    def filter_sigma(self, sigma: float) -> None:
        """ """
        if hasattr(self, "filtered"):
            if self.finder is "svf":
                self.filtered = self.filtered[
                    voids.filtered["void_overlap"] == sigma
                ]
            elif self.finder is "tunnels":
                self.filtered = self.filtered[self.filtered["sigma"] == sigma]
            elif self.finder is "zobov":
                self.filtered = voids.filtered[
                    voids.filtered["halo_den"] == sigma
                ]
        else:
            if self.finder is "svf":
                self.filtered = self.data[voids.data["void_overlap"] == sigma]
            elif self.finder is "tunnels":
                self.filtered = self.data[self.data["sigma"] == sigma]
            elif self.finder is "zobov":
                self.filtered = voids.data[voids.data["halo_den"] == sigma]

    def filter_size(self, size_bin, ret: bool = True) -> None:
        """ """
        if ret:
            if hasattr(self, "filtered"):
                return self.filtered[self.filtered["radius_bin"] == size_bin]
            else:
                return self.data[self.data["radius_bin"] == size_bin]
        else:
            if hasattr(self, "filtered"):
                voids.filtered = self.filtered[
                    self.filtered["radius_bin"] == size_bin
                ]
            else:
                voids.filtered = self.data[self.data["radius_bin"] == size_bin]

    def select_type(self, void_type: str, tracers, args):
        """ """
        if void_type == "minimal":
            self.data = object_selection.minimal_voids(
                self.data, tracers.data, args
            )
            
    def categorize_type(self):
        """ Categorize voids into [minimal, ..., ...] """
        # TODO
        raise VoidsWarning("Not implemented yet")

    def categorize_sizes(
        self, bins: int, min_obj_nr: int
    ) -> tuple:
        """
        Group objects according to their size.

        Args:
            bins:
                Nr. of size cateories.
            min_obj_nr:
                Minimum nr. of objects can from a category.
        """
        self.data = object_selection.categorize_sizes(
            self.data, "log", bins, min_obj_nr
        )

    def _compute_tangential_shear(
        self, rad: np.array, prof: np.array,
    ) -> np.array:
        """
        Compute the tangential shear profile for a given convergence profile.

        Args:
        """
        def _integrand(r):
            return 2 * np.pi * r * kappa_r(r)

        kappa_r = interp1d(rad, prof, fill_value='extrapolate')
        shear = np.zeros(len(rad))
        for l in range(len(rad)):
            _val = integrate.quad(_integrand, 0, rad[l])[0] 
            shear[l] = _val / (np.pi * rad[l]**2) - prof[l]
        return shear


def init_voids(ff, args):
    """
    Args:
        ff : str
            Path to void binary file
    Returns:
        voids : dic
    """
    # read void binary file
    h, dI, data = halo.readHaloData(ff)
    voids = {
        "radius": {
            "deg": data[:, 1],
            "pix": np.rint(
                data[:, 1] * args["Npix"] / args["field_width"]
            ).astype(int),
        },
        "pos_x": {
            "deg": data[:, 2],
            "pix": np.rint(
                data[:, 2] * args["Npix"] / args["field_width"]
            ).astype(int),
        },
        "pos_y": {
            "deg": data[:, 3],
            "pix": np.rint(
                data[:, 3] * args["Npix"] / args["field_width"]
            ).astype(int),
        },
    }
    return voids
