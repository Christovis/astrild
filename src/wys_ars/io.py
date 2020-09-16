import os, sys, glob
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import xarray as xr
import astropy

def transform_PandasSeries_to_NumpyNdarray(
    series: pd.Series) -> np.ndarray:
    """
    Used for wys_ars.rays.SkyMap
    """
    _zip_array = sorted(zip(series.index.values, series.values,))
    _values = np.array([j for (i, j) in _zip_array])
    _npix = int(np.sqrt(len(series.values)))
    array = np.zeros([_npix, _npix])
    k = 0
    for j in range(_npix):
        for i in range(_npix):
            array[j, i] = _values[k]
            k += 1
    return array

def save_skymap(
    data: Union[np.ndarray, astropy.io.fits.hdu.image.PrimaryHDU],
    filename: str,
) -> None:
    """
    Write wys_ars.rays.SkyMap.data to a file.
    """
    _remove_existing_file(filename)
    print("Save in:\n   %s" % filename)
    if isinstance(data, np.ndarray):
        np.save(filename, data)
    elif isinstance(data, astropy.io.fits.hdu.image.PrimaryHDU):
        data.writeto(filename)

def _remove_existing_file(filename) -> None:
    if os.path.exists(filename):
        os.remove(filename)

def save_DataFrame(direct: str, filename: str) -> None:
    print("Save to -> ", direct + filename)
    df.to_hdf(direct + filename, key="df", mode="w",)
