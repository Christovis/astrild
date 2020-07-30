import os
from typing import List, Union

import numpy as np

from wys_ars.particles.utils.density import readDensityData

dtfe_field_dimensions = {
    "density": 1,
    "density_a": 1,
    "velocity": 3,
    "velocity_a": 3,
    "divergence": 1,
    "divergence_a": 1,
    "vorticity": 3,
    "vorticity_a": 3,
    "shear": 5,
    "shear_a": 5,
    "gradient": 9,
    "gradient_a": 9,
}


class DTFEWarning(BaseException):
    pass


class DTFE:
    """
    The Delaunay Tessellation Field Estimator Code

    Methods:
        estimate_field
        binary_to_array
    """

    def estimate_field(
        quantities: List[str], file_in: str, file_out: str, domain_level: int = 512,
    ) -> None:
        """
        Args:
            quantity:
                e.g. [density, velocity, divergence, vorticity]
            file_in:
            file_out:
            domain_level:
        """
        quantities = " ".join(quantities)
        os.system(
            f"/cosma/home/dp004/dc-beck3/3_Proca/dtfe/" + \
            f"DTFE {file_in} {file_out} -g {domain_level} -p -f {quantities}"
        )

    def binary_to_array(
        quantity: str, file_in: str, file_out: str, save: bool = True, remove_binary: bool = True,
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
        if dtfe_field_dimensions[quantity] == 3:
            value.shape = header.gridSize[0], header.gridSize[1], header.gridSize[2], 3
        elif dtfe_field_dimensions[quantity] == 1:
            value.shape = header.gridSize[0], header.gridSize[1], header.gridSize[2]
        else:
            raise DTFEWarning(
                f'Dont know how to handle {dtfe_field_dimensions[quantity]} dimensions'
            )

        if remove_binary:
            os.remove(file_in)

        if save:
            np.save(file_out, value)
        else:
            return value
