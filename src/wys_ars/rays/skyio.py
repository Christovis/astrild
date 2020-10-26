import os, sys, glob
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

import healpy as hp

from wys_ars.io import IO

class SkyIO:
    def transform_PandasDataFrame_to_Healpix(
        df: pd.DataFrame, quantity: str, nside: int = 1024*4,
    ) -> np.ndarray:
        """
        Used for wys_ars.rays.SkyMap
        Used for angular power spectrum calculation
        """
        npix = hp.nside2npix(nside)
        # Go from HEALPix coordinates to indices                                         
        indices = hp.ang2pix(                                                            
            nside,                                                                       
            df["the_co"].values,#*180/np.pi,                                        
            df["phi_co"].values,#*180/np.pi,                                        
        )                                                                                
        # Initiate the map and fill it with the values                                   
        hpmap = np.ones(npix, dtype=np.float)*hp.UNSEEN                           
        for i in range(nsources):                                                        
            hpmap[indices[i]] = df[quantity].values[i]
        return hpmap

    def transform_PandasSeries_to_NumpyNdarray(
        series: pd.Series
    ) -> np.ndarray:
        """
        Used for any analysis of skymap features.
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
    
    def transform_NumpyNdarray_to_PandasSeries(
        array: np.ndarray
    ) -> pd.DataFrame:
        """
        Used for any analysis of skymap features.
        """
        #TODO

    def to_file(
        dir_out: str,
        on: str = "orig_gsn_smooth",
        extension: str = "npy",
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
        file_in: str, quantity: str, on: str, extension: str,
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
        _file = file_out.split(".")[:-1] + [on] #.append(on)
        _file = _file + [extension]
        file_out = ".".join(_file) 
        return file_out
