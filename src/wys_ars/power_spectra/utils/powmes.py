import os
import subprocess
from pathlib import Path

dir_src = Path(__file__).parent.absolute()
default_powmes_config = dir_src / "configs/powmes.yaml"


class PowMes:
    """ The Delaunay Tessellation Field Estimator Code """

    def compute_pk(
        self, file_in: str, file_out: str, dir_sim: str, domain_level: int = 512,
    ):
        subprocess.call(f"./DTFE {file_in} {file_out} -g {domain_level} -p -f velocity")

    def binary_to_array(self, file_in: str):
        header, value = readDensityData(file_in)
        value.shape = header.gridSize[0], header.gridSize[1], header.gridSize[2]
        return value
