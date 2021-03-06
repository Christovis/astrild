import os, re, glob
import logging
from pathlib import Path
from typing import Dict, List, Optional, Type, Union

import numpy as np


class SimulationWarning(BaseException):
    pass


class Simulation:
    """
    Super-class to manage Ramses based simulations codes such as:
    ECOSMOG, Ray-Ramses, GRAMSES

    Attributes:
        dir_sim:
        dir_out:
        file_dsc:
        dir_root:

    Methods:
        get_file_nrs:
        get_file_paths:
        get_dir_nrs:
        get_dir_paths:
    """

    def __init__(
        self, dir_sim: str, dir_out: str, file_dsc: Dict[str, str], dir_root: str,
    ):
        if dir_out is None:
            dir_out = dir_sim
        self.dirs = {"sim": dir_sim, "out": dir_out}
        self.name = self._get_simname_from_dirsim(dir_sim)
        self.file_dsc = file_dsc
        
        if dir_root is None:
            self.dir_root = "sim"
        elif dir_root is not None:
            self.dir_root = dir_root
            self.dir_nrs = self.get_dir_nrs(sort=True)
            self.dirs[dir_root] = self.get_dir_paths(None, dir_root)
        
        if self.file_dsc["root"] is None:
            self.file_nrs = None
            self.files = {}
        elif self.file_dsc["root"] is not None:
            self.file_nrs = self.get_file_nrs(self.file_dsc, self.dirs["sim"], "max", True)
            self.files = {
                self.file_dsc["root"]: self.get_file_paths(
                    self.file_dsc, self.dirs["sim"], "max"
                )
            }
        self.dimensions = 3


    def _get_simname_from_dirsim(self, dir_sim: str) -> str:
        """
        Set the directory name inside which the simulation is as the
        simulation name.
        """
        return [elem for elem in dir_sim.split('/') if len(elem) > 0][-1]


    def _get_all_files(
        self, file_dsc: Dict[str, str], directory: Optional[str] = None,
    ) -> List[str]:
        """ """
        if directory is None: directory = self.dirs["sim"]
        template = f"{directory}/{file_dsc['root']}_*{file_dsc['extension']}"
        files = glob.glob(template)
        return files


    def get_file_nrs(
        self,
        file_dsc: dict,
        directory: Optional[str] = None,
        uniques: Union[None, str] = "max",
        sort: bool = False,
    ) -> Union[np.array, Dict[int, np.array]]:
        """
        Args:
            file_dsc:
            directory:
            uniques: If the filenames matching to file_dsc have multiple numbers
                which have to be distinguished, this argument provides an indication
                which of the numbers to pick as file id.
            sort:

        Returns:
        """
        if directory is None: directory = self.dirs["sim"]
        _files = self._get_all_files(file_dsc, directory)

        if len(_files) == 0:
            _files = self._get_all_files(file_dsc, self.dirs[self.dir_root][0])

        file_ids = np.array(
            [re.findall(r"\d+", fil.split("/")[-1]) for fil in _files]
        ).astype(int)
        if len(file_ids.shape) == 2:
            file_ids = file_ids.T
            var = np.array([len(np.unique(column)) for column in file_ids])
            if uniques == "max":
                file_ids = file_ids[np.argmax(var)]
            elif uniques == "min":
                file_ids = file_ids[np.argmin(var)]
        if sort:
            file_ids = np.sort(file_ids)
        return file_ids


    def get_file_paths(
        self,
        file_dsc: Optional[dict] = None,
        directory: Optional[str] = None,
        uniques: Union[None, str] = "max",
    ) -> Union[List[str], Dict[int, List[str]]]:
        """
        Collect paths to files defined by sim-config/file_dsc.
        By default they are searched first in the simulations root directory,
        if nothing found search in sim-config/dir_root.
        """
        if directory is None: directory = self.dirs["sim"]
        files = {}
        files[file_dsc["root"]] = self._get_all_files(file_dsc, directory)
        if len(files[file_dsc["root"]]) == 0:
            files[file_dsc["root"]] = {}
            for _dir_nr, _dir in zip(self.dir_nrs, self.dirs[self.dir_root]):
                _file_paths = self._get_all_files(file_dsc, _dir)
                _file_ids = self.get_file_nrs(file_dsc, _dir, uniques, sort=False)
                _idxs = np.argsort(_file_ids)
                files[file_dsc["root"]][str(_dir_nr)] = [
                    _file_paths[idx] for idx in _idxs
                ]
        elif len(files[file_dsc["root"]]) > 1:
            file_ids = self.get_file_nrs(file_dsc, directory, uniques, sort=False)
            idxs = np.argsort(file_ids)
            files[file_dsc["root"]] = [files[file_dsc["root"]][idx] for idx in idxs]
        return list(files.values())[0]


    def _get_all_paths(self, dir_root: Optional[str] = None,) -> List[str]:
        """
        Args:
        Returns:
        """
        if dir_root is None: dir_root = self.dir_root
        # list all designations starting with dir_root
        dirs = glob.glob(self.dirs["sim"] + dir_root + "_*")
        # filter out directories
        dirs = [path for path in dirs if "." not in path]
        return dirs


    def get_dir_nrs(
        self, dir_root: Optional[str]=None, sort: bool=True,
    ) -> np.array:
        """
        Args:
        Returns:
        """
        if dir_root is None: dir_root = self.dir_root
        _dirs = self._get_all_paths(dir_root)
        dir_ids = np.array(
            [int(re.findall(r"\d+", dire.split("/")[-1])[0]) for dire in _dirs]
        )
        if sort:
            dir_ids = np.sort(dir_ids)
        return dir_ids


    def get_dir_paths(self, dir_ids: list, dir_root: str) -> List[str]:
        """
        Args:
        Returns:
        """
        if dir_root is None: dir_root = self.dir_root
        dirs = []
        if dir_ids is None:
            dirs = self._get_all_paths(dir_root)
            dir_ids = self.get_dir_nrs(dir_root, sort=False)
            idxs = np.argsort(dir_ids)
            dirs = [dirs[idx] for idx in idxs]
        else:
            if "_" not in dir_root:
                _dir_root = dir_root + "_%03d"
            for dir_id in dir_ids:
                _dir = self.dirs["sim"] + _dir_root % dir_id + "/"
                assert os.path.isdir(_dir), SimulationWarning(
                    f"The directory '{_dir}' does not exist :-("
                )
                dirs.append(_dir)
        return dirs


    @staticmethod
    def remove_files(files: list) -> None:
        [Path(f).unlink() for f in files]
