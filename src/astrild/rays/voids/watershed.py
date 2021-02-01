import os, sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type

import pandas as pd
import numpy as np
from scipy.spatial import cKDTree

from astropy.io import fits
from astropy import units as un
from lenstools import ConvergenceMap

# from astrild.rays import peak as Peaks
from astrild.rays.skymap import SkyMap
from astrild.rays.voids.tunnels import halo
from astrild.rays.voids.tunnels import textFile

this_file_path = Path(__file__).parent.absolute()


class WatershedFinderWarning(BaseException):
    pass


class WatershedFinder:
    """
    The tunnels void finder by Marius Cautun.
    Source: arxiv:1710.01730

    Attributes:
        skymap:
            SkyMap object
    Methods:
        find_peaks:
            Identify traces which are used to find voids.
        find_voids:
            Identify voids.
    """

    def __init__(self, skymap: Type[SkyMap]):
        self.skymap = skymap

    def to_file(self, dir_out: str) -> None:
        filename = self._create_filename(
            obj="voids", dir_out=dir_out, on=self.on
        )
        print("Save voids in -> ", filename)
        self.voids_df.to_hdf(filename, key="df")
        filename = self._create_filename(
            obj="peaks", dir_out=dir_out, on=self.on
        )
        self.peaks_df.to_hdf(filename, key="df")
        # IO.save_peaks_and_voids(filename)

    def _create_filename(self, obj: str, dir_out: str, on: str) -> str:
        _filename = self.skymap._create_filename(
            self.skymap.map_file, self.skymap.quantity, on, extension="_"
        )
        # _filename = ''.join(_filename.split("/")[-1].split(".")[:-1])
        _filename = "".join(_filename.split(".")[:-1])
        return f"{dir_out}/{obj}_{_filename}.h5"


def _bin2df(file_in: str, npix: int, opening_angle: float) -> pd.DataFrame:
    """
    Convert watershed results which are stored in binary format into
    python pandas.DataFrame
    Args:
    """
    # read void binary file
    h, dI, data = halo.readHaloData(file_in)
    void_dir = {
        "x_deg": data[:, 4] / 60,
        "x_pix": np.rint(data[:, 4] * npix / (60 * opening_angle)).astype(int),
        "y_deg": data[:, 3] / 60,
        "y_pix": np.rint(data[:, 3] * npix / (60 * opening_angle)).astype(int),
        "rad_deg": data[:, 1] / 60,
        "rad_pix": np.rint(data[:, 1] * npix / (60 * opening_angle)).astype(
            int
        ),
    }

    print(
        "x_pix",
        np.min(void_dir["x_pix"]),
        np.max(void_dir["x_pix"]),
        np.min(void_dir["x_deg"]),
        np.max(void_dir["x_deg"]),
    )
    print(
        "y_pix",
        np.min(void_dir["y_pix"]),
        np.max(void_dir["y_pix"]),
        np.min(void_dir["y_deg"]),
        np.max(void_dir["y_deg"]),
    )
    print(
        "rad_pix",
        np.min(void_dir["rad_pix"]),
        np.max(void_dir["rad_pix"]),
        np.min(void_dir["rad_deg"]),
        np.max(void_dir["rad_deg"]),
    )
    print(len(void_dir["x_pix"]))

    void_df = pd.DataFrame(data=void_dir)
    return void_df


def _npy2fits(mapfile, mapfile_fits, field_width) -> None:
    """
    Convert maps that are stored in .npy into .fits format
    (applies only to 'flat' projection maps).

    Args:
        path : str
        field_width : np.float
            given in degrees
    """
    # Convert .npy to .fits
    data = fits.PrimaryHDU()
    data.header["ANGLE"] = field_width  # [deg]

    # load convergence maps in numpy format
    extension = mapfile.split(".")[-1]
    if extension == "npy":
        data.data = np.load(mapfile)
    elif extension == "npz":
        _tmp = np.load(mapfile)
        data.data = _tmp[
            "arr_0"
        ]  # dict-key comes from lenstools.ConvergenceMap
    if os.path.exists(mapfile_fits):
        os.remove(mapfile_fits)
    data.writeto(mapfile_fits)


def _peaks2npy(val, pos, outfile) -> None:
    """
    Save to .npy file
    Args:
    """
    val = np.array(val)
    pos = np.array(pos)
    wl_peaks = np.zeros((val.shape[0], 3))
    wl_peaks[:, :2] = pos
    wl_peaks[:, 2] = val
    np.save(outfile, wl_peaks)


def _peaks2txt(val, pos, outfile) -> None:
    """
    Save to .txt file
    Args:
    """
    val = np.asarray(val)
    pos = np.asarray(pos)
    with open(outfile, "w") as txt_file:
        for ii in range(len(val)):
            txt_file.write(
                "%.4f %.4f %.4f\n" % (pos[ii, 0], pos[ii, 1], val[ii])
            )
    txt_file.close()


def _dataframe2txt(df, outfile) -> None:
    """
    Args:
    """
    # pd.dataframe to np.ndarray
    array = df[df.columns.values].values.shape
    with open(outfile, "w") as txt_file:
        for ii in range(len(val)):
            txt_file.write(
                "%.4f %.4f %.4f %.4f %.4f %.4f %.4f\n"
                % (
                    array[ii, 2],  # nu
                    array[ii, 0],  # x-pos in deg
                    array[ii, 1],  # y-pos in deg
                    array[ii, 3],  # x-pos in pix
                    array[ii, 4],  # y-pos in pix
                    array[ii, 5],  # radius in deg
                    array[ii, 6],  # radius in pix
                )
            )
    txt_file.close()


def _txt2bin(filein, fileout, field_width) -> None:
    """
    Args:
    """
    fileinname = sys.argv[0].rsplit("/")[-1]  # TODO: change to os.path
    programOptionsDesc = fileinname + " " + " ".join(sys.argv[1:])
    # box size of convergence map
    box = [0.0, field_width, 0.0, field_width, 0.0, field_width]  # [arcmin]

    # Binary parameters for Marius Cautun code
    # column in file:   none   0  1  2  3
    dataDescription = """ID   x   y   kappa"""
    mpcUnit = 1.0
    massUnit = 1.0
    massColumn = 2
    noIntColumns = 1
    IntegerColumns = "0:1"
    noFloatColumns = 3
    FloatColumns = "1:4"

    # read the input data
    noDescLines, noColumns, noRows = textFile.getTextFileProperties(filein)
    desc, data = textFile.readTextFile(
        filein, noDescLines, noRows, noColumns, np.float32, VERBOSE=True
    )

    # define the output data
    data.shape = -1, noColumns
    noHalos = data.shape[0]
    dataIntegers = np.zeros((noHalos, noIntColumns), np.int32)
    # an ID for each void; the same as in the input file
    dataIntegers[:, 0] = np.arange(noHalos)
    dataFloats = np.zeros((noHalos, noFloatColumns), np.float32)
    dataFloats[:, :] = data[:, :]

    # set-up the header of .bin file
    header = halo.HaloHeader()
    header.noHalos = noHalos
    header.noColumnsIntegers = noIntColumns
    header.noColumnsFloats = noFloatColumns
    header.noColumns = header.noColumnsIntegers + header.noColumnsFloats
    header.mpcUnit = mpcUnit
    header.positionColumns[:] = [0, 1, 2]
    header.box[:] = box[:]
    header.massUnit = massUnit
    header.massColumn = massColumn
    names = halo.getColumnNames(dataDescription)
    header.columnNamesIntegers = eval("names[" + IntegerColumns + "]")
    header.columnNamesFloats = eval("names[" + FloatColumns + "]")
    header.AddProgramCommands(programOptionsDesc)
    # write the data to a file
    halo.writeHaloData(fileout, header, dataIntegers, dataFloats)
