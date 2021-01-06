import numpy as np
import numba as nb

from astropy import units as un
from astropy.constants import c, sigma_T, m_p
from astropy.cosmology import z_at_value

# ------------------------
# Distance Transformations
# ------------------------
def Dc_to_Da(Dc: float, redshift: float) -> float:
    """
    Calculate the angular diameter distance (Da) from 
    comoving distance (Dc) and redshift (redshift).
    """
    return Dc/(1 + redshift)

def Dc_to_redshift(Dc: float, cosmo, units=un.Mpc) -> float:
    """ Calculate the redshift from comoving distance (D_c) """
    return z_at_value(cosmo.comoving_distance, Dc * units)

def radius_to_angsize(radius: float, Da: float, arcmin: bool=True) -> float:
    """
    Calculate the angular radius (theta) of the halo from its radius and 
    angular diameter distance (Da).

    Note: radius and D_a must have the same units

    Args:
    Returns:
        if arcmin == True: return the value in arcmin
    """
    ang_size = np.true_divide(radius, D_a)
    if arcmin: ang_size = rad2arcmin(ang_size)
    return ang_size


# ------------------------
# Coordinate Transformations
# ------------------------
def rad2arcmin(angle: float) -> float:
    """Convert radians to arcmins"""
    return np.rad2deg(angle)*60

def arcmin2rad(angle: float) -> float:
    """Convert arcmins to radians"""
    return np.deg2rad(angle/60)

def get_cart2sph_jacobian(th: float, ph: float) -> np.ndarray:
    """
    Calculate the transformation matrix (jacobian) for spherical to
    cartesian coordinates at line of sight (th, ph) [radians] (th is the
    polar angle with respect to z and ph is the azimuthal angle w.r.t. x).

    Args:
    Returns:
    Example:
        th_rad = np.deg2rad(df['th'].values)
        ph_rad = np.deg2rad(df['ph'].values)
        v_cart = np.array([df['vx'],df['vy'],df['vz']])
        thph_grid = np.array([th_rad,ph_rad])
        J_cart2sph = cart2sph2(th_rad,ph_rad)
        v_cart2sph = np.einsum('ij...,i...->j...',J_cart2sph,v_cart)
    """
    row1 = np.stack((np.sin(th) * np.cos(ph), np.cos(th) * np.cos(ph), -np.sin(ph)))
    row2 = np.stack((np.sin(th) * np.sin(ph), np.cos(th) * np.sin(ph), np.cos(ph)))
    row3 = np.stack((np.cos(th), -np.sin(th), 0.0 * np.cos(th)))
    return np.squeeze(np.stack((row1, row2, row3)))

def get_sph2cart_jacobian(th: float, ph: float) -> np.ndarray:
    """
    Calculate the transformation matrix (jacobian) for spherical to
    cartesian coordinates.

    Args:
    Returns:
    Example: see cart2sph2()
    """
    row1 = np.stack((np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)))
    row2 = np.stack((np.cos(th) * np.cos(ph), np.cos(th) * np.sin(ph), -np.sin(th)))
    row3 = np.stack((-np.sin(ph), np.cos(ph), 0.0 * np.cos(th)))
    return np.squeeze(np.stack((row1, row2, row3)))

def convert_vec_sph2cart(th: np.ndarray, ph: np.ndarray, vij_sph: np.array) -> tuple:
    """
    Calculate the cartesian velocity components from the spherical ones.
    Args:
        th: theta angle in spherical coordinates.
        ph: phi angle in spherical coordinates.
        vij_sph: vector in spherical coorindates, [v_r, v_th, v_ph]
    Returns:
        transformed velocity field and defined as:
            v_x , v_y, v_z
    """
    # find the spherical to cartesian coords transformation matrix
    J = get_sph2cart_jacobian(th, ph)
    vij_cart = np.einsum('ij...,i...->j...', J, vij_sph.T).T
    return vij_cart

def convert_vec_cart2sph(th: float, ph: float, vij_cart: np.array) -> tuple:
    """
    Calculate the spherical velocity components from the cartesian ones.
    Args:
        th: theta angle in spherical coordinates.
        ph: phi angle in spherical coordinates.
        vij_cart: vector in spherical coorindates, [v_x, v_y, v_z]
    Returns:
        transformed velocity field and defined as:
            v_r (radial), v_th (co-latitude), v_ph (longitude)
    """
    # find the cartesian to spherical coords transformation matrix
    J = get_cart2sph_jacobian(th, ph)
    vij_sph = np.einsum('ij...,i...->j...', J, vij_cart.T).T
    return vij_sph

def transform_box_to_lc_cart_coords(
    pos: np.ndarray, boxsize: float, boxdist: float,
) -> np.ndarray:
    """ Translate objects position in box to light-cone """
    # cartesian coord. in light-cone [Mpc/h]
    pos[:, 0] = pos[:, 0] - boxsize / 2
    pos[:, 1] = pos[:, 1] - boxsize / 2
    pos[:, 2] = pos[:, 2] + boxdist
    return pos

def radial_coordinate_in_lc(pos: np.ndarray) -> np.array:
    """ Radial distance from observer, units of pos[:,:] """
    return np.sqrt(pos[:, 0]**2 + pos[:, 1]**2 + pos[:, 2]**2)

def angular_coordinate_in_lc(pos: np.ndarray, unit="deg") -> tuple:
    """ Angular coordinates with respect to z-axis, [rad/deg] """
    theta1 = np.arctan(pos[:, 0] / pos[:, 2])
    theta2 = np.arctan(pos[:, 1] / pos[:, 2])
    if unit == "deg":
        theta1 *= 180/np.pi
        theta2 *= 180/np.pi
    elif unit == "rad":
        pass
    return theta1, theta2
