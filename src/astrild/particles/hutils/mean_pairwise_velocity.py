from typing import Optional

import numpy as np
import numba as nb
import itertools
from concurrent import futures

from astrild.utils.geometrical_transforms import (
    get_cart2sph_jacobian,
    get_sph2cart_jacobian,
    convert_vec_sph2cart,
    convert_vec_cart2sph,
    angular_coordinate_in_lc,
)

def mean_pv_from_tv(
    pos_cart: np.ndarray,
    vel_ang: np.ndarray,
    bins: np.ndarray,
    theta1: Optional[np.ndarray] = None,
    theta2: Optional[np.ndarray] = None,
    multithreading: bool = True,
    Nthreads=nb.config.NUMBA_NUM_THREADS,
):
    """
    Computes the mean pairwise velocity as estimated the transverse to the
    line-of-sight (LOS) velocity components of the objects.
    Based on Yasini et al. 2018 (arxiv:1812.04241).
    
    Args:
        pos_cart: 2D array with the cartesian coordinates of the object, [Mpc/h]
        vel_ang: 2D array with RA and DEC vel. of the object, [km/sec]
        bins: 1D array containing the distances [Mpc/h] of the histogram edges
        theta(1,2)_deg: Spherical coordinates, where
            theta1 = azimuth angle/longitude/phi/RA and
            theta2 = inclination angle/90°-polar angle/theta=90°-theta2/Latitude/DEC
            of the object, [degrees]
        multithreading: sets if you want to run in different threads for
        Nthreads: or if you want a gigantic for loop.
    
    Returns:
        rsep: distances [Mpc/h] of the histogram centres
        pest: mean pairwise velocity estimate
    """
    assert len(pos_cart) <= 50000, "This routine is not optimized to handle larger data-sets"
    
    if theta1 is None:
        theta1_rad, theta2_rad = angular_coordinate_in_lc(pos_cart, unit="rad")
        # shift origin to corner
        theta1_rad += (10 * np.pi / 180)
        theta2_rad += (10 * np.pi / 180)
    elif np.max(theta1) > 2*np.pi:
        theta1_rad = np.deg2rad(theta1)
        theta2_rad = np.deg2rad(theta2)
    else:
        theta1_rad = theta1
        theta2_rad = theta2

    binnr = len(bins)
    binwidth = np.diff(bins)[0]
    nr_of_obj = len(theta1_rad)

    vel_sph = np.hstack((np.zeros((len(theta1_rad), 1)), vel_ang))
    vel_cart = convert_vec_sph2cart(theta2_rad, theta1_rad, vel_sph)

    # allocate memory for results for each object
    noms = np.array([np.zeros(binnr) for j in range(nr_of_obj - 1)])
    denoms = np.array([np.zeros(binnr) for j in range(nr_of_obj - 1)])
    rows = range(nr_of_obj - 1)  # iterate over (n-1) rows for i!=j...
    
    # duplicate for each object except for itself
    poss_cart_x = itertools.repeat(pos_cart[:, 0], nr_of_obj - 1)
    poss_cart_y = itertools.repeat(pos_cart[:, 1], nr_of_obj - 1)
    poss_cart_z = itertools.repeat(pos_cart[:, 2], nr_of_obj - 1)
    vels_cart_x = itertools.repeat(vel_cart[:, 0], nr_of_obj - 1)
    vels_cart_y = itertools.repeat(vel_cart[:, 1], nr_of_obj - 1)
    vels_cart_z = itertools.repeat(vel_cart[:, 2], nr_of_obj - 1)
    binnrs = itertools.repeat(binnr, nr_of_obj - 1)
    binwidths = itertools.repeat(binwidth, nr_of_obj - 1)

    if multithreading:
        print("Running in %i threads..." % Nthreads)
        with futures.ThreadPoolExecutor(Nthreads) as ex:
            ex.map(
                pairwise_one_row,
                rows,
                poss_cart_x,
                poss_cart_y,
                poss_cart_z,
                vels_cart_x,
                vels_cart_y,
                vels_cart_z,
                binnrs,
                binwidths,
                noms,
                denoms,
            )
    else:
        print("Running on only one thread.")
        list(map(
            pairwise_one_row,
            rows,
            poss_cart_x,
            poss_cart_y,
            poss_cart_z,
            vels_cart_x,
            vels_cart_y,
            vels_cart_z,
            binnrs,
            binwidths,
            noms,
            denoms,
        ))
    
    nom = np.array(noms).sum(axis=0)
    denom = np.array(denoms).sum(axis=0)
    pest = nom[denom > 0] / denom[denom > 0]
    rsep = make_rsep(binnr, binwidth)
    return rsep, pest


@nb.jit(nopython=True, nogil=True)
def pairwise_one_row(
    i,
    pos_cart_x: np.ndarray,
    pos_cart_y: np.ndarray,
    pos_cart_z: np.ndarray,
    tv_cart_x: np.ndarray,
    tv_cart_y: np.ndarray,
    tv_cart_z: np.ndarray,
    binnr: int,
    binwidth: int,
    nom: np.ndarray,
    denom: np.ndarray,
):
    """
    Computation of mean pairwise velocity for one object.
    Based on Yasini et al. 2018 (arxiv:1812.04241).

    Args:
        i: for which object index to compute mean pairwise velocity estimate
        pos_cart_(x,y,z): 3D cartesian coordinates of each object
        tv_cart_(x,y,z): 3D cartesian vector of transverse velocity of each object
        binnr: number of bins in histogram for the estimator
        binwidth: bin size, [Mpc/h]
        nom: 1D array to collect results of nominator of Eq. 6 in Yasini
        denom: 1D array to collect results of denominator of Eq. 6 in Yasini
    """
    pos_i = np.array([pos_cart_x[i], pos_cart_y[i], pos_cart_z[i]])
    pos_i_unit = pos_i / np.linalg.norm(pos_i)
    tv_cart_i = np.array([tv_cart_x[i], tv_cart_y[i], tv_cart_z[i]])

    for j in range(i + 1, len(pos_cart_x)):
        # get pairwise distance between obj_i and obj_j
        pos_j = np.array([pos_cart_x[j], pos_cart_y[j], pos_cart_z[j]])
        pos_ij = pos_i - pos_j
        pos_ij_norm = np.linalg.norm(pos_ij)
        # get bin
        binval_ij = int(pos_ij_norm / binwidth)
        
        if binval_ij < binnr:
            # get pairwise transverse velocity
            tv_cart_j = np.array([tv_cart_x[j], tv_cart_y[j], tv_cart_z[j]])
            tv_ij = tv_cart_i - tv_cart_j
            # get geometrical parameter
            pos_j_unit = pos_j / np.linalg.norm(pos_j)
            pos_ij_unit = pos_ij / pos_ij_norm
            dotprod_i = np.dot(pos_ij_unit, pos_i_unit)
            dotprod_j = np.dot(pos_ij_unit, pos_j_unit)
            q_ij = 0.5 * (2.*pos_ij_unit - pos_i_unit*dotprod_i - pos_j_unit*dotprod_j)
            # result
            nom[binval_ij] += np.dot(tv_ij, q_ij)
            denom[binval_ij] += q_ij[0]**2 + q_ij[1]**2 + q_ij[2]**2


def make_rsep(binnr: int, binwidth: float) -> np.ndarray:
    """
    Generates the x axis of the histogram.
    Bin positions are halfway of the step.
    In the following diagram, "x" marks the rsep histogram axis.
    the "|" mark the edge of each bin.

    Notice that if a point falls between 0 and 1 int(r_sep/binsz) it
    is asigned to the first bin, and labeled rsep_bin = 0.5
    If a point falls between 1 and 2 it is labeled rsep_bin = 1.5 and so
    forth.
    |  x  |  x  |  x  |  x  |
    0     1     2     3     4  -> rsep
      0.5   1.5   2.5   3.5    -> rsep_bins
    
    Args:
        binsz: bin size
        nrbin: number of bins
    """
    return np.linspace(0, (binnr - 1) * binwidth, binnr) + binwidth / 2.0


def make_rsep_uneven_bins(bin_edges: np.ndarray) -> np.ndarray:
    """
    Generates x axis of the histogram as in make_rsep
    bin_edges is a numpy array with the edges of the bins.
    """
    return (bin_edges[1:] + bin_edges[:-1]) / 2.0
