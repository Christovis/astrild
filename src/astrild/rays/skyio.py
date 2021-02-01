import os, sys, glob
from typing import Dict, List, Optional, Union

import pandas as pd
import numpy as np
import numba as nb

import healpy as hp
import astropy

from astrild.io import IO


class SkyIO:
    def transform_PandasDataFrame_to_Healpix(
        df: pd.DataFrame, quantity: str, nside: int
    ) -> np.ndarray:
        """
        Project data stored in pd.DataFrame to create HealPix map.
        """
        # find pixel id corresponding to a angular coordinate for a given NSIDE
        df["pix_id"] = hp.ang2pix(
            nside, df["the_co"].values, df["phi_co"].values
        )
        # get mean of data values lying in same pixel
        df = df.groupby(["pix_id"]).mean()
        hpmap = np.zeros(hp.nside2npix(nside), dtype=np.float)
        hpmap[df.index.values] = df.loc[df.index.values][quantity]
        hpmap[hpmap == 0.0] = hp.UNSEEN
        return hpmap

    @nb.jit(nopython=True)
    def transform_RayRamsesOutput_to_NumpyNdarray(
        values: np.array
    ) -> np.ndarray:
        """
        """
        _index = np.arange(len(values))
        _zip_array = sorted(zip(_index, values))
        _values = np.array([j for (i, j) in _zip_array])
        _npix = int(np.sqrt(len(values)))
        array = np.zeros((_npix, _npix))
        k = 0
        for j in range(_npix):
            for i in range(_npix):
                array[j, i] = _values[k]
                k += 1
        return array

    def transform_NumpyNdarray_to_PandasSeries(
        array: np.ndarray
    ) -> pd.DataFrame:
        """
        """
        # TODO

    def to_file(
        dir_out: str, on: str = "orig_gsn_smooth", extension: str = "npy"
    ) -> None:
        """
        """
        if on == "orig_gsn":
            self.data[on] = self.add_galaxy_shape_noise()
        self.dirs["out"] = dir_out
        filename = self._create_filename(
            self.map_file, self.quantity, on, extension=extension
        )
        IO.save_skymap(self.data[on], dir_out + filename)

    def _create_filename(
        file_in: str, quantity: str, on: str, extension: str
    ) -> str:
        """  
        Args:
        """
        quantity = quantity.replace("_", "")
        file_out = file_in.split("/")[-1].replace("Ray", quantity)
        file_out = file_out.replace(".h5", f"_lt.{extension}")
        if ("_lc" not in file_in) and ("zrange" not in file_in):
            file_out = file_out.split("_")
            box_string = [
                string for string in file_in.split("/") if "test" in string
            ][0]
            idx, string = [
                (idx, "%s_" % box_string + string)
                for idx, string in enumerate(file_out)
                if "output" in string
            ][0]
            file_out[idx] = string
            file_out = "_".join(file_out)
        _file = file_out.split(".")[:-1] + [on]  # .append(on)
        _file = _file + [extension]
        file_out = ".".join(_file)
        return file_out

    def _array_to_fits(self, map_out: np.ndarray) -> astropy.io.fits:
        """ Convert maps that in .npy format into .fits format """
        # Convert .npy to .fits
        data = fits.PrimaryHDU()
        data.header["ANGLE"] = self.opening_angle  # [deg]
        data.data = map_out
        return data
