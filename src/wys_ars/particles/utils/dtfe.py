import os
from typing import Union

import numpy as np

from wys_ars.particles.utils.density import readDensityData


class EcosmogWarning(BaseException):
    pass


class DTFE:
    """
    The Delaunay Tessellation Field Estimator Code

    Methods:
        estimate_field
        binary_to_array
    """

    def estimate_field(
        quantity: str, file_in: str, file_out: str, domain_level: int = 512,
    ) -> None:
        """
        Args:
            quantity:
            file_in:
            file_out:
            domain_level:
        """
        os.system(
            f"/cosma/home/dp004/dc-beck3/3_Proca/dtfe/" + \
            f"DTFE {file_in} {file_out} -g {domain_level} -p -f {quantity}"
        )

    def binary_to_array(
        file_in: str, file_out: str, save: bool = True, remove_binary: bool = True,
    ) -> Union[None, np.ndarray]:
        """
        Args:
            file_in:
            file_out:
            save:
            remove_binary:

        Returns:
            value:
                - velocity field at domain level resolution in
                gadget units; [km/s * sqrt(a)]
                - density field at domain level resolution
        """
        header, value = readDensityData(file_in)
        if ".vel" in file_in:
            value.shape = header.gridSize[0], header.gridSize[1], header.gridSize[2], 3
        elif ".den" in file_in:
            value.shape = header.gridSize[0], header.gridSize[1], header.gridSize[2]

        if remove_binary:
            os.remove(file_in)

        if save:
            np.save(file_out, value)
        else:
            return value
