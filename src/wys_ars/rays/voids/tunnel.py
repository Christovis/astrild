import os, sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type

import pandas as pd
import numpy as np
from scipy.spatial import cKDTree

from astropy.io import fits
from astropy import units as un
from lenstools import ConvergenceMap

#from wys_ars.rays import peak as Peaks
from wys_ars.rays.skymap import SkyMap
from wys_ars.rays.voids.tunnels import halo
from wys_ars.rays.voids.tunnels import textFile
from wys_ars.io import IO

this_file_path = Path(__file__).parent.absolute()

class TunnelsFinderWarning(BaseException):
    pass


class TunnelsFinder:
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

    def find_peaks(
        self,
        on: str,
        field_conversion: str,
        thresholds_dsc: dict,
        snr_sigma: Optional[float] = None,
        save: bool = False,
    ) -> None:
        """
        Find peaks on convergence map. It is assumed that the convergence maps
        were created with wys_ars.rays.visuals.map and have appropriate
        smoothing and galaxy shape noise.

        Args:
        Returns:
        """
        self.on = on
        if field_conversion == "normalize":
            _map = self.skymap.data[on] - np.mean(self.skymap.data[on])
        else:
            _map = self.skymap.data[on]

        thresholds = self._get_convergence_thresholds(**thresholds_dsc)

        _map = ConvergenceMap(data=_map, angle=self.skymap.opening_angle*un.deg)
        _peaks = {}
        _peaks["kappa"], _peaks["pos"] = _map.locatePeaks(thresholds)
        _peaks["kappa"], _peaks["pos"] = self._remove_peaks_crossing_edge(**_peaks)
        assert len(_peaks["kappa"]) != 0, "No peaks"

        # find significance of peaks
        _peaks["snr"] = self._signal_to_noise_ratio(_peaks["kappa"], _map.data, snr_sigma)
        self.peaks = _peaks
        if save:
            #IO.save()
            pass

    def _get_convergence_thresholds(
        self, on: str = "orig", nbins: int = 100,
    ) -> np.array:
        """
        Define thresholds for lenstools to find peaks on convergence map.
        Important to do this on un-smoothed and with no-gsn skymap.
        """
        return np.arange(
            np.min(self.skymap.data[on]),
            np.max(self.skymap.data[on]),
            (np.max(self.skymap.data[on]) - np.min(self.skymap.data[on])) / nbins,
        )

    def _signal_to_noise_ratio(
        self,
        peak_values: np.ndarray,
        map_values: np.ndarray,
        sigma: Optional[float] = None,
    ) -> np.ndarray:
        """
        Assess signifance of peaks and remove peaks suffereing edge effects

        Args:
        """
        if sigma is None:
            #_kappa_mean = np.mean(map_values)
            #snr = (peak_values - _kappa_mean) / _kappa_std
            _kappa_std = np.std(map_values)
            snr = peak_values / _kappa_std
        else:
            snr = peak_values / sigma
        return snr

    def _remove_peaks_crossing_edge(
        self,
        kappa: np.ndarray,
        pos: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Remove peaks within 1 smoothing length from map edges
        
        Args:
            kappa:
            pos:

        Returns:
            kappa:
            pos:
        """
        pixlen = self.skymap.opening_angle / self.skymap.npix  #[deg]
        bufferlen = np.ceil(self.skymap.smoothing_length / (60 * pixlen))  # length of buffer zone
        # convert degrees to pixel number
        x = pos[:, 0].value * self.skymap.npix/ self.skymap.opening_angle
        y = pos[:, 1].value * self.skymap.npix / self.skymap.opening_angle
        indx = np.logical_and(
            np.logical_and(x <= self.skymap.npix - 1 - bufferlen, x >= bufferlen),
            np.logical_and(y <= self.skymap.npix - 1 - bufferlen, y >= bufferlen),
        )
        kappa = kappa[indx]
        pos = pos[indx, :]
        print(
            f"{len(indx)} peaks were within smoothing_length of FOV edge" + \
            f" and had to be removed. \n{len(kappa)} peaks are left."
        )
        return kappa, pos


    def find_voids(
        self, snrs: List[float], dir_temp: Optional[str] = None, rtn: bool = False,
    ) -> None:
        """
        analyze convergence map for different peak heights
        Args:
            snr: signifance of void traces, in this case peaks on convergence map.
            dir_temp: directory for temporary file storage. Files will be deleted after,
                voids were found.
        Returns:
            Creates pd.DataFrame for voids and peaks
        """
        if dir_temp:
            dir_out = dir_temp
        else:
            dir_out = "/".join(self.skymap.map_file.split("/")[:-1])
        first = True
        for snr in snrs:
            print("\n Start analyzes for nu=%f" % snr)
            snr_label = str(snr).replace(".", "p")[:]

            # filter sigma
            idx = self.peaks["snr"] > snr
            pos_tmp = self.peaks["pos"][idx, :]
            nu_tmp = self.peaks["snr"][idx]

            # prepare .txt file for Marius's void finder
            file_peaks_txt = f"{dir_out}/peaks_in_kappa2_nu{snr_label}.txt"
            _peaks2txt(nu_tmp, pos_tmp, file_peaks_txt)

            # store peaks in pd.DataFrame
            peak_dir = {
                "x_deg": pos_tmp[:, 0],
                "x_pix": np.rint(
                    pos_tmp[:, 0] * self.skymap.npix / self.skymap.opening_angle
                ).astype(int),
                "y_deg": pos_tmp[:, 1],
                "y_pix": np.rint(
                    pos_tmp[:, 1] * self.skymap.npix / self.skymap.opening_angle
                ).astype(int),
                "sigma": snr,
            }

            # prepare .bin file for Marius's void finder
            file_peaks_bin = file_peaks_txt.replace(".txt", ".bin")
            _txt2bin(file_peaks_txt, file_peaks_bin, self.skymap.opening_angle)

            file_voids_bin = f"{dir_out}/voids_in_kappa2_nu{snr_label}.bin"
            os.system(
                f"{this_file_path}/tunnels/void_finder_spherical_2D " + \
                f"{file_peaks_bin} {file_voids_bin} -l 0. -a 0.2 -x 0 -y 1;" # overlapping
                #f"{file_peaks_bin} {file_voids_bin} -l 1. -a 0.2 -x 0 -y 1;" # non-oberlap.
            )

            # read void results and prepare pd.DataFrame for storage
            self.voids = _bin2df(
                file_voids_bin, snr, self.skymap.npix, self.skymap.opening_angle
            )
            os.remove(file_voids_bin)
            os.remove(file_peaks_bin)
            os.remove(file_peaks_txt)

            # adding radii to peaks
            peaks_df = pd.DataFrame(data=peak_dir)
            self.filtered_peaks = self.set_peak_radii(
                peaks_df, self.voids, self.skymap.npix, self.skymap.opening_angle
            )

            if first is True:
                voids_df_sum = self.voids
                peaks_df_sum = self.filtered_peaks
                first = False
            else:
                voids_df_sum = voids_df_sum.append(
                    self.voids, ignore_index=True
                )
                peaks_df_sum = peaks_df_sum.append(
                    self.filtered_peaks, ignore_index=True
                )
        self.peaks_orig_df = pd.DataFrame(data=self.peaks["snr"])
        if rtn:
            return peaks_df_sum, voids_df_sum
        else:
            self.peaks_df = peaks_df_sum
            self.voids_df = voids_df_sum


    def set_peak_radii(
        self,
        peaks: pd.DataFrame,
        voids: pd.DataFrame,
        npix,
        opening_angle,
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
            lambda x: np.rint(x * (
                npix / opening_angle
            )).astype(int)
        )
        return peaks

    def to_file(self, dir_out: str) -> None:
        filename = self._create_filename(obj="voids", dir_out=dir_out, on=self.on)
        print("Save voids in -> ", filename)
        self.voids_df.to_hdf(filename, key="df")
        filename = self._create_filename(obj="peaks", dir_out=dir_out, on=self.on)
        self.peaks_orig_df.to_hdf(filename, key="df")
        #IO.save_peaks_and_voids(filename)

    def _create_filename(self, obj: str, dir_out: str, on: str) -> str:
        _filename = self.skymap._create_filename(
            self.skymap.map_file, self.skymap.quantity, on, extension="_"
        )
        #_filename = ''.join(_filename.split("/")[-1].split(".")[:-1])
        _filename = ''.join(_filename.split(".")[:-1])
        return f"{dir_out}/{obj}_{_filename}.h5"


def _bin2df(file_in, sigma, npix, opening_angle) -> pd.DataFrame:
    """
    Convert Marius tunnels results which are stored in binary format into
    python pandas.DataFrame
    Args:
    """
    # read void binary file
    h, dI, data = halo.readHaloData(file_in)
    void_dir = {
        "x_deg": data[:, 2],
        "x_pix": np.rint(data[:, 2] * npix / opening_angle).astype(int),
        "y_deg": data[:, 3],
        "y_pix": np.rint(data[:, 3] * npix / opening_angle).astype(int),
        "rad_deg": data[:, 1],
        "rad_pix": np.rint(data[:, 1] * npix / opening_angle).astype(int),
    }
    void_df = pd.DataFrame(data=void_dir)
    void_df["sigma"] = sigma

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
        data.data = _tmp["arr_0"]  #dict-key comes from lenstools.ConvergenceMap
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
            txt_file.write("%.4f %.4f %.4f\n" % (pos[ii, 0], pos[ii, 1], val[ii]))
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
