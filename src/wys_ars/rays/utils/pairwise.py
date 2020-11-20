import numpy as np
import numba as nb

def spherical_to_cartesian_coord(pos_vec: np.array) -> np.array:
    """ Change [r, theta, phi] to [x, y, z] """
    return np.array([
        pos_vec[0]*np.sin(pos_vec[1])*np.cos(pos_vec[2]),
        pos_vec[0]*np.sin(pos_vec[1])*np.sin(pos_vec[2]),
        pos_vec[0]*np.cos(pos_vec[1]),
    ])

def pairwise_distance(obj1_pos: np.array, obj2_pos: np.array) -> float:
    """  """
    return np.sqrt(
        (obj1_pos[0] - obj2_pos[0])**2 + \
        (obj1_pos[1] - obj2_pos[1])**2 + \
        (obj1_pos[2] - obj2_pos[2])**2 + \
    )

def transverse_geometrical_factor(pos_i: np.ndarray, pos_j: np.ndarray,) -> np.ndarray:
    """
    Eq. 5 in arxiv:1812.04241

    Args:
        pos_i,j: object spherical coordinates (r, theta, phi)
            in units of [cMpc/h, deg, deg]
    """
    bij = 2*


def mean_pairwise_vels_from_transverse_vels(
    tv_vec: np.ndarray, pos_vec: np.ndarray,
) -> np.ndarray:
    """
    Eq. 6 in arxiv:1812.04241

    Args:
        t_vec: transverse to the line-of-sight velocity (theta_vel, phi_vel)
            in units of [km/sec, km/sec]
        pos_vec: object spherical coordinates (r, theta, phi)
            in units of [cMpc/h, deg, deg]
    Returns:
    .. math::
        v_{ij}(r)
    """

    for obj1_tv, obj1_pos in zip(tv_vec, pos_vec):
        pairwise_distance(obj1_pos - pos_vec)

        tv_diff = obj1_tv - pos_vec
        transverse_geometrical_factor(obj1_pos, pos_vec)
        
