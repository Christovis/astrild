import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type

import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import interp1d
from scipy import integrate
from scipy.spatial import cKDTree
from astropy import units as u

from astrild.rays.utils import object_selection
from astrild.simulation import Simulation
from astrild.rays.skymap import SkyMap
from astrild.profiles import profile_2d as Profiles2D


class PeaksWarning(BaseException):
    pass


class Peaks:
    """
    Peaks in convergance map

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
        finder: str,
        skymap_dsc: dict,
        _file: Optional[str] = None,
        file_dsc: Optional[dict] = None,
    ) -> "Peaks":
        """
        Read void data files.

        Args:
            file_dsc: {path, root, extension}
        """
        if _file is None:
            sim = Simulation(file_dsc["path"], None, file_dsc.pop("path"), None)
            _file = sim.files[file_dsc["root"]][0]

        if finder == "tunnels":
            data = pd.read_hdf(_file, key="df")
            finder_spec = {
                "name": finder,
                "sigmas": {"name": "sigma", "values": data["sigma"].unique()},
            }
        return Peaks(_file, data, finder_spec, skymap_dsc)

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

    def get_profiles(
        self,
        radii_max: float,
        nr_rad_bins: int,
        skymap_file: Optional[str] = None,
        skymap: Optional[np.ndarray] = None,
        save: bool = False,
        field_conversion: Optional[str] = None,
        dir_out: Optional[str] = None,
    ) -> None:
        """
        Find the profile of peaks on astrild.rays.SkyMap

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

        if self.finder_spec["name"] == "tunnels":
            self.data = self._trim_edges(
                self.data, radii_max, self.skymap_dsc["npix"]
            )
            self.data = self.data.reset_index()
        print(
            f"Get the profile of {len(self.data.index)} "
            + f"{self.finder_spec['name']} voids"
        )
        self.profiles = Profiles2D.from_map(
            self.data, _skymap, radii_max, nr_rad_bins, field_conversion
        )

    def get_profile_stats(
        self,
        cats: List[str],
        field_conversion: Optional[str] = None,
        dir_out: str = None,
        save: bool = False,
    ) -> None:
        """
        Get profile statistics: mean, 16% and 84% of un- or categorized voids.

        Args:
            cats:
                What categorizations to use.
        """
        if (
            field_conversion is not None
            and self.field_conversion is not None
            and field_conversion != self.field_conversion
        ):
            raise VoidsWarning("Contradictory field convergence")
        elif field_conversion is not None:
            self.field_conversion = field_conversion

        # initialize result arrays
        cat_per_cats = [len(np.unique(self.data[cat].values)) for cat in cats]
        cats_dict = {
            cat: np.unique(self.data[cat].values)
            for idx, cat in enumerate(cats)
        }
        nr_rad_bins = len(self.profiles["radii"])
        _mean = np.zeros(tuple(cat_per_cats + [nr_rad_bins]))
        _low_err = np.zeros(tuple(cat_per_cats + [nr_rad_bins]))
        _high_err = np.zeros(tuple(cat_per_cats + [nr_rad_bins]))
        _obj_size_min = np.zeros(tuple(cat_per_cats))
        _obj_size_max = np.zeros(tuple(cat_per_cats))
        _obj_in_cat = np.zeros(tuple(cat_per_cats))

        # TODO create dynamic for-loops via recursion
        for ss, sigma in enumerate(cats_dict["sigma"]):
            # filter voids
            _void_in_cat = self.data.loc[self.data["sigma"] == sigma]
            # clean-up and averaging profiles
            _mean[ss, :] = Profiles2D.mean_and_interpolate(
                self.profiles["values"][_void_in_cat.index.values, :],
                _void_in_cat["rad_pix"].values,
                self.profiles["radii"].max(),
                len(self.profiles["radii"]),
            )
            if self.field_conversion == "tangential_shear":
                _mean[ss, :] = self._compute_tangential_shear(
                    self.profiles["radii"], _mean[ss, :]
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
            # (* 3600 * u.arcsec * object_dist).to(u.Mpc, u.dimensionless_angles()) / u.Mpc
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
        if save:
            dir_out = "/".join(self.dataset_file.split("/")[:-1])
            if self.field_conversion is None:
                self.field_conversion = ""
            file_name = (
                self.field_conversion
                + "_"
                + "".join(self.dataset_file.split("/")[-1].split(".")[:-1])
            )
            print(
                f"Save profile statistics in -> {dir_out}/profile_{file_name}.nc"
            )
            ds.to_netcdf(f"{dir_out}/profile_{file_name}.nc")

    def _trim_edges(
        self, voids: pd.DataFrame, radii_max: float, npix: int
    ) -> None:
        """
        Remove voids whose radii goes over the simulation boundary.

        Args:
            radii_max:
            npix:
                Nr. of pixels on edge of SkyMap
        """
        return object_selection.trim_edges(voids, radii_max, npix)

    def filter_sigma(self, sigma: float) -> None:
        """ """
        if hasattr(self, "filtered"):
            if self.finder is "tunnels":
                self.filtered = self.filtered[self.filtered["sigma"] == sigma]
        else:
            if self.finder is "tunnels":
                self.filtered = self.data[self.data["sigma"] == sigma]

    def categorize_sizes(self, bins: int, min_obj_nr: int) -> tuple:
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

    def categorize_sizes(self, bins: int, min_obj_nr: int) -> tuple:
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


def set_radii(
    peaks: pd.DataFrame, voids: pd.DataFrame, npix, opening_angle
) -> pd.DataFrame:
    """
    Args:
    Returns:
    """
    # convert from pd.DataFrame to np.ndarray
    peaks_pos = peaks[["x_deg", "y_deg"]].values

    # convert from dict to np.ndarray
    if voids.__class__ is dict:
        voids_pos = np.concatenate(
            (
                voids["pos_x"]["deg"].reshape((len(voids["pos_x"]["deg"]), 1)),
                voids["pos_y"]["deg"].reshape((len(voids["pos_y"]["deg"]), 1)),
            ),
            axis=1,
        )
    elif voids.__class__ is pd.core.frame.DataFrame:
        voids_pos = np.concatenate(
            (
                voids["x_deg"].values.reshape((len(voids["x_deg"].values), 1)),
                voids["y_deg"].values.reshape((len(voids["y_deg"].values), 1)),
            ),
            axis=1,
        )

    # build tree of voids
    tree = cKDTree(voids_pos)
    # find nearest void for each peak
    distances, edges = tree.query(peaks_pos, k=1)

    peaks["rad_deg"] = pd.Series(distances)
    peaks["rad_pix"] = peaks["rad_deg"].apply(
        lambda x: np.rint(x * (npix / opening_angle)).astype(int)
    )
    return peaks


def load_txt_add_pix(fname, args):
    """
    Args:
        data : np.str
            String of fiel to load
    Returns:
        peaks : pd.DataFrame
    """
    peaks = pd.read_csv(
        fname, delim_whitespace=True, names=["x_deg", "y_deg", "nu"]
    )

    peaks["x_pix"] = peaks["x_deg"].apply(
        lambda x: np.rint(x * (args["Npix"] / args["field_width"])).astype(int)
    )
    peaks["y_pix"] = peaks["y_deg"].apply(
        lambda x: np.rint(x * (args["Npix"] / args["field_width"])).astype(int)
    )
    return peaks


if __name__ == "__main__":
    Peaks()
