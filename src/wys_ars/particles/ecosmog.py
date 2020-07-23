import os, re, sys, glob
import time
from struct import *
from typing import List, Dict, Optional, Union

import numpy as np
import pandas as pd

import astropy
from astropy import units as u
from astropy import constants as const

# from astropy.cosmology import LambdaCDM, cvG

from nbodykit.lab import *

from wys_ars.particles.utils import DTFE
from wys_ars.simulation import Simulation

t1 = time.clock()


class EcosmogWarning(BaseException):
    pass


class Ecosmog(Simulation):
    """
    Attributes:
        dir_sim: directory of simulation
        dir_out: directory for results/returns
        file_dsc: file describtion for identification
        dir_root:
        boxsize: simulation boxsize; [Mpc/h]
        domain_level: nr. of cells on domain level grid

    Methods:
        to_gadget
        dtfe
        compress_snapshot
    """

    def __init__(
        self,
        dir_sim: str,
        dir_out: str = None,
        file_dsc: dict = {"root": "snap_012", "extension": None},
        dir_root: str = "snapdir",
        boxsize: float = 500.0,
        domain_level: int = 512,
    ):
        super().__init__(dir_sim, dir_out, file_dsc, dir_root)
        self.boxsize = boxsize
        self.domain_level = domain_level
        self.npar = domain_level


    def to_gadget(self) -> None:
        EcosmogWarning("Not implemented yet")


    def dtfe(
        self,
        snap_nrs: Optional[List[int]] = None,
        file_root: str = "snap_",
        quantity: str = "velocity",
        file_dsc: dict = {"root": "snap_012.", "extension": None},
    ) -> None:
        """
        Return density and velocity estimates on a fixed (n x n)-grid using the
        Delaunay Tessellation Field Estimator.
        User Guide:
            www.astro.rug.nl/~voronoi/DTFE/download/DTFE_user_guide_1.0.pdf

        Args:
            snap_nrs:
                Snapshot numbers to analyze
            file_root:
                Root of files to use
            quantity:
                What quantity to process and return
            file_dsc:
                File describtion of files to use for analyzes

        Returns:
        """
        if quantity == "velocity":
            _file_extension = ".vel"
        if quantity == "density":
            _file_extension = ".den"

        # filter snapshot numbers
        if snap_nrs:
            assert set(snap_nrs) < set(self.dir_nrs), EcosmogWarning(
                f"Some of the snapshots {snap_nrs} do not exist" + \
                f"in: {self.dir_nrs}."
            )
            idx = np.array(
                [1 if val in snap_nrs else 0 for ii, val in enumerate(self.dir_nrs)],
                dtype=bool,
            )
            _dir_nrs = self.dir_nrs[idx]
            _dir_paths = np.array(self.dirs[self.dir_root])[idx]
        else:
            _dir_nrs = self.dir_nrs
            _dir_paths = self.dirs[self.dir_root]

        for snap_nr, snap_dir in zip(_dir_nrs, _dir_paths):
            file_in = snap_dir + "/" + file_dsc["root"] + "%i"
            file_out = self.dirs["sim"] + "dtfe_%05d" % snap_nr
            DTFE.estimate_field(
                quantity, file_in, file_out, self.domain_level,
            )
            file_in = file_out + _file_extension
            directory = file_in.split("/")[:-1]
            file_name = f"{quantity}_" + file_in.split("/")[-1].split(".")[0] + ".npy"
            file_out = "/".join(directory) + "/" + file_name
            value_map = DTFE.binary_to_array(
                file_in, file_out, save=True, remove_binary=True
            )


    def compress_snapshot(
        self,
        amr_levels: tuple,
        domain_level: int,
        fields: list,
        snap_nr: list,
        file_root: str = "grav",  # TODO maybe combine this with __init__.file_dsc
        dir_out: str = None,
        save: bool = True,
    ) -> Union[None, pd.DataFrame]:
        """
        Reading Ecosmogs ./poisson/output_poisson.f90 files
        and transcribing them into pandas.to_hdf data structures.

        Note: This only works for simulations without AMR!

        Args:
            snapshot:
            fields:
            file_root:

        Returns:
        """
        self.levelmin = min(amr_levels)
        self.levelmax = max(amr_levels)
        _dimfac = 2 ** self.dimensions

        for snap_nr, snap_dir in zip(self.dir_nrs, self.dirs[self.dir_root]):
            if snap_nr != 12:
                continue
            _snap_files = glob.glob(
                snap_dir + "/%s_%05d.out?????" % (file_root, snap_nr)
            )
            idx = [int(fil.split(".")[-1][-5:]) for fil in _snap_files]
            _snap_files = [_snap_files[ii] for ii in np.argsort(idx)]
            print("There are %d files to be transcribed" % len(_snap_files))

            # stored quantities to convert data format of
            datlis = [[] for t in range(len(fields))]

            # run through output files of snapshot
            _count = 0
            Ngrid = 0

            for snap_file in _snap_files:
                with open(snap_file, "rb") as f:
                    content = f.read()
                print("Reading " + snap_file.split("/")[-1])

                # header info
                pmin = 0
                pmax = 48  # lines of header
                info = unpack("i" * 3 * 4, content[pmin:pmax])
                [ncpu, ndim, _nlevelmax, nboundary] = [
                    info[1],
                    info[4],
                    info[7],
                    info[10],
                ]

                # run through resolution levels
                for ilevel in range(self.levelmin, self.levelmax + 1):
                    # run through nboundary+ncpu
                    for ibound in range(1, nboundary + ncpu + 1):
                        pmin0 = pmax
                        pmax0 = pmin0 + 4 * 3 * 2
                        info = unpack("i" * 3 * 2, content[pmin0:pmax0])
                        [currlevel, ncache] = [info[1], info[4]]
                        Ngrid += ncache * _dimfac

                        if ncache == 0:
                            # if no data in that boundary/cpu
                            pmax = pmax0
                            continue

                        # simulation dimensions
                        for dim in range(1, _dimfac + 1):
                            j = 0

                            # run through parameters
                            for N in range(1, len(fields) + 1):
                                pmin = pmax0 + (8 * N - 4) + (N - 1) * 8 * ncache
                                pmax = pmin + ncache * 8
                                info = unpack("d" * ncache, content[pmin:pmax])
                                # datlis[j].append(float(info[0]))
                                for floatelem in info:
                                    datlis[j].append(floatelem)
                                j += 1

                            pmax0 = pmax + 4
                        pmax = pmax0
                f.close()
                _count += 1

            print(f"Transposing: {time.clock() - t1} sec")
            datlis = np.transpose(datlis)
            print(f"Mapping to tuple: {time.clock() - t1} sec")
            datlis = map(tuple, datlis)
            print(f"Finding set of values: {time.clock() - t1} sec")
            datlis = set(datlis)
            print(f"Mapping back to array: {time.clock() - t1} sec")
            datlis = np.asarray([list(dat) for dat in datlis])
            datlis = np.transpose(datlis)

            data_dic = {}
            for idx, field in enumerate(fields):
                data_dic[field] = datlis[idx]
            data_df = pd.DataFrame(data_dic)

            if save:
                filename = file_root.split("_")[0] + "_out%05d.h5" % snap_nr
                data_df.to_hdf(
                    self.dirs["sim"] + filename, key="df", mode="w",
                )
            else:
                return data_df
