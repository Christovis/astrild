# Functions to handle power spectra
import os, sys, glob
from typing import Dict, List, Optional, Tuple, Type, Union

import pandas as pd
import numpy as np
from scipy import integrate

import astropy
import astropy.units as u
from astropy.cosmology import LambdaCDM
from nbodykit.lab import *

from wys_ars.simulation import Simulation
from .power_spectrum_3d import PowerSpectrum3D


class PowMesWarning(BaseException):
    pass


class PowMes(PowerSpectrum3D):
    """
    POWMES: Accurate estimators of power spectra N-body simulations
    Source: https://arxiv.org/pdf/0811.0313.pdf

    Methods:
        read_pk_file:
        read_pk:
    """

    def __init__(self, sim_type: str, simulation: Type[Simulation]):
        super().__init__(sim_type, simulation)


    #def compute(self):
    #    raise PowMesWarning("This method is not yet implemented for PowMes.")


    def read_file(
        self, infile: str, boxsize: float, npix: int
    ) -> Tuple[List[float], List[float]]:
        """
        Args:
        Returns:
            k: wavenumber
            Pk: power
        """
        ik = np.loadtxt(
            infile, usecols=[0], skiprows=0, unpack=True, dtype=np.float32
        )
        P_z00 = np.loadtxt(
            infile, usecols=[3], skiprows=0, unpack=True, dtype=np.float32
        )
        W_z00 = np.loadtxt(
            infile, usecols=[4], skiprows=0, unpack=True, dtype=np.float32
        )
        k = ik * 2 * np.pi / boxsize
        # Pk = (P_z00 - W_z00/npix**3)*boxsize**3
        Pk = P_z00 * boxsize ** 3
        return k, Pk


    def csv_to_h5(
        self,
        snap_nrs: Optional[List[int]] = None,
        file_dsc: Dict[str, str] = {"root": "dtfe", "extension": "npy"},
        directory: Optional[str] = None,
        save: bool = True,
    ) -> Union[None, Dict[str, dict]]:
        """
        Args:
        Returns:
        """
        if not directory:
            directory = self.sim.dirs["sim"]

        if snap_nrs:
            assert set(snap_nrs) < set(self.sim.dir_nrs), PowMesWarning(
                f"Some of the snapshots {snap_nrs} do not exist" + \
                f"in:\n{self.sim.dir_nrs}"
            )
            _file_paths = self.sim.get_file_paths(
                file_dsc, directory, uniques="max"
            )
        else:
            snap_nrs = self.sim.get_file_nrs(file_dsc, directory, uniques="max", sort=True)
            _file_paths = self.sim.get_file_paths(file_dsc, directory, uniques="max")

        print(snap_nrs, _file_paths)

        pk = {"k": {}, "P": {}}
        for snap_nr, file_path in zip(snap_nrs, _file_paths):
            k, Pk = self.read_file(file_path, self.sim.boxsize, self.sim.npar)
            print('-----------', snap_nr, Pk[-10:])
            pk["k"]["snap_%d" % snap_nr] = k
            pk["P"]["snap_%d" % snap_nr] = Pk

        if save:
            self._save_results(file_dsc, "matter", pk)
        else:
            return pk


def align_lin_nonlin(lin, nonlin, k):
    return lin[0] - np.mean(nonlin[(1e-2 < k) & (k < 1e-1)])


def run(snap_nrs, params_b3, quantities):
    """
    """
    for param_b3 in params_b3:
        indir = "/cosma7/data/dp004/dc-beck3/3_Proca/cvg_b3_000001_with_cbf/"

        for snapnr in snap_nrs:
            print("Reading data of snapshot %d" % snapnr)
            # load snapshot
            infile = indir + "grav_%05d.h5" % snapnr
            fields = pd.read_hdf(infile, key="df")

            # Pk settings
            boxsize = 200  # [Mpc/h]
            grid_size = 256  # default number of mesh cells per coordinate axis
            # Delta_k  = 1.0e-2   # size of k bins  (where k is the wave vector in Fourier Space)
            k_min = 2 * np.pi / boxsize  # smallest k value

            # value map
            x = (grid_size * fields["x"].values).astype(int)
            y = (grid_size * fields["y"].values).astype(int)
            z = (grid_size * fields["z"].values).astype(int)
            print("box-size", x.min(), x.max())

            pk_dict = {}
            for quant in quantities:
                value_map = np.zeros((grid_size, grid_size, grid_size))

                if quant in ["di_sf", "di_lp_sf", "lp2_sf"]:
                    value_map[(x, y, z)] = fields["sf"].values
                    # partial derivative 1
                    di_sf, dj_sf, dk_sf = np.gradient(
                        value_map, boxsize / grid_size, edge_order=2
                    )
                    if quant is "di_sf":
                        value_map = di_sf
                        value_map[abs(value_map) > 5e5] = 0.0
                        label = "di_sf"
                    elif quant in ["di_lp_sf", "lp2_sf"]:
                        # partial derivative 2
                        di_di_sf = np.gradient(
                            di_sf, boxsize / grid_size, axis=0, edge_order=2
                        )
                        dj_dj_sf = np.gradient(
                            dj_sf, boxsize / grid_size, axis=1, edge_order=2
                        )
                        dk_dk_sf = np.gradient(
                            dk_sf, boxsize / grid_size, axis=2, edge_order=2
                        )
                        lp_sf = di_di_sf + dj_dj_sf + dk_dk_sf
                        # partial derivative 3
                        di_lp_sf = np.gradient(
                            lp_sf, boxsize / grid_size, axis=0, edge_order=2
                        )
                        if quant is "di_lp_sf":
                            value_map = di_lp_sf
                            value_map[0:5, :] = value_map[5:10, :]
                            value_map[-6:-1, :] = value_map[-10:-5, :]
                            value_map[abs(value_map) > 5e5] = 0.0
                            label = "di_lp_sf"
                        elif quant is "lp2_sf":
                            # partial derivative 3
                            di_lp_sf, dj_lp_sf, dk_lp_sf = np.gradient(
                                lp_sf, boxsize / grid_size, edge_order=2
                            )
                            di_di_lp_sf = np.gradient(
                                di_lp_sf,
                                boxsize / grid_size,
                                axis=0,
                                edge_order=2,
                            )
                            dj_dj_lp_sf = np.gradient(
                                dj_lp_sf,
                                boxsize / grid_size,
                                axis=1,
                                edge_order=2,
                            )
                            dk_dk_lp_sf = np.gradient(
                                dk_lp_sf,
                                boxsize / grid_size,
                                axis=2,
                                edge_order=2,
                            )
                            lp2_sf = di_di_lp_sf + dj_dj_lp_sf + dk_dk_lp_sf
                            value_map = lp2_sf
                            value_map[0:5, :] = value_map[5:10, :]
                            value_map[-6:-1, :] = value_map[-10:-5, :]
                            value_map[abs(value_map) > 5e5] = 0.0
                            label = "lp2_sf"
                elif quant in ["_cbf"]:

                    if quant in ["lp2_"]:
                        value_map[(x, y, z)] = fields[quant].values
                        label = quant
                    if quant in ["lp_cbf"]:
                        value_map[(x, y, z)] = fields[quant].values
                        # partial derivative 1
                        di_sf, dj_sf, dk_sf = integrate.quad(
                            value_map, boxsize / grid_size, edge_order=2
                        )

                else:
                    raise "Error"
                print("Power-spectrum of %s" % label)

                # power-spectrum of density-fluctuations
                mesh = ArrayMesh(
                    value_map,
                    Nmesh=grid_size,
                    compensated=False,
                    BoxSize=boxsize,
                )
                r = FFTPower(
                    mesh,
                    mode="1d",
                    # dk=Delta_k,
                    kmin=k_min,
                )
                k = np.array(r.power["k"])  # the k-bins
                Pk = np.array(r.power["power"].real)  # the power spectrum
                Pk_shotnoise = r.power.attrs["shotnoise"]  # shot-noise
                Pk -= Pk_shotnoise
                pk_dict["Pk_%s" % label] = Pk
                print("PkPkPk", np.mean(Pk))

            pk_dict["k"] = k
            pk_df = pd.DataFrame(pk_dict)
            pk_df.to_hdf(
                indir + "pk_extradof_%05d.h5" % snapnr, key="df", mode="w"
            )


if __name__ == "__main__":
    snap_nrs = [3]  # snapshots
    b3 = ["000001"]  # cvG model parameter
    quantities = [
        "sf",
        "di_sf",
        "lp2_sf",
        "di_lp_sf",
        "lp_cbf1",
        "lp2_cbf1",
        "lp2_cbf2",
        "lp2_cbf3",
    ]  # cvg fields

    run(snap_nrs, b3, quantities)
