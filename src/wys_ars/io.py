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

def save_dataFrame(direct: str, filename: str, df: pd.DataFrame) -> None:
    file_path = direct + filename
    if os.path.exists(file_path):
        os.remove(file_path)
    print("Save to -> ", file_path)
    df.to_hdf(file_path, key="df", mode="w",)

def save_tpcf(
    dir_out: str,
    config: dict,
    multipoles: list,
    halofinder: str,
    object_type: str,
    tpcf: dict,
) -> None:
    """
    Save reults of wys_ars.particles.halo.get_tpcf
    """
    for l in multipoles:
        # create pd.DataFrame
        dic = {"s": tpcf["s_bins"],}
        for key, result in tpcf[str(l)].items():
            dic[key] = result
        df = pd.DataFrame.from_dict(dic)
        
        # write to hdf
        compare = "00"  # indicating that the TPCF was calculate without categorizing halos
        filename = f"{halofinder}{object_type}_tpcf_s_{l}_{compare}.h5"
        save_dataFrame(dir_out, filename, df)
