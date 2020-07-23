import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from wys_ars.rays.rayramses import RayRamses
from wys_ars.rays.utils.object_selection import trim_edges
from wys_ars.rays.utils import object_selection

class PeaksWarning(BaseException):
    pass


class Peaks:
    """
    Peaks in convergance map
    """

    def __init__(self, sim_type, simulation: Type[RayRamses]):
        self.sim = simulation
        self.sim.type = sim_type

        self.filename = filename
        peak_files = glob.glob(
            args.featureindir + "peaks_in_%s_nu?p?.h5" % args.quantity_object
        )

    def read(self, args):
        """
        Args:
            data : np.str
                String of fiel to load
        Returns:
            peaks : pd.DataFrame
        """
        peaks = pd.read_hdf(self.filename, "peaks")
        # dataframe to array
        peaks = peaks[peaks.columns.values].values

        # array to dic
        self.data = {}
        self.data["RAD"] = {"deg": peaks[:, 5], "pix": np.rint(peaks[:, 6]).astype(int)}
        self.data["POS"] = {
            "deg": np.array([peaks[:, 0], peaks[:, 1]]).T,
            "pix": np.array(
                [np.rint(peaks[:, 3]).astype(int), np.rint(peaks[:, 4]).astype(int)]
            ).T,
        }

    def trim_edges(self, args):

        self.data = trim_edges(self.data, args)

    def group_size(self, args):

        obj_bin_indx, bin_indx = object_selection.group_size(self.data, args)

        self.data["rad_bin"] = obj_bin_indx
        self.bin_indx = bin_indx


def load_txt_add_pix(fname, args):
    """
    Args:
        data : np.str
            String of fiel to load
    Returns:
        peaks : pd.DataFrame
    """
    peaks = pd.read_csv(fname, delim_whitespace=True, names=["x_deg", "y_deg", "nu"])

    peaks["x_pix"] = peaks["x_deg"].apply(
        lambda x: np.rint(x * (args["Npix"] / args["field_width"])).astype(int)
    )
    peaks["y_pix"] = peaks["y_deg"].apply(
        lambda x: np.rint(x * (args["Npix"] / args["field_width"])).astype(int)
    )
    return peaks


def set_radii(peaks: pd.DataFrame, voids: dict, npix: int, opening_angle: float) -> pd.DataFrame:
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


if __name__ == "__main__":
    Peaks()
