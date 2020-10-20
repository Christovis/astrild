import os, sys, glob
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

import healpy as hp

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
        Used for wys_ars.rays.SkyMap
        Used for all operations regarding convergence peaks and voids
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
