import os, sys, glob
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import xarray as xr
import astropy


class IO:
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
