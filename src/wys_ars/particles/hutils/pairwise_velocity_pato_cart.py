from typing import Optional

import numpy as np
import numba as nb
import itertools
from concurrent import futures

from wys_ars.utils.geometrical_transforms import (
    get_cart2sph_jacobian,
    get_sph2cart_jacobian,
    convert_vec_sph2cart,
    convert_vec_cart2sph,
    angular_coordinate_in_lc,
)

def mean_pv(
    pos_cart: np.ndarray,
    vel_cart: np.ndarray,
    bins: np.ndarray,
):
    """
    Produces the ksz curve for givne arguments.
    
    Args:
        Dc: radial/comoving distance, [Mpc]
        pos_ang: 2D array with RA and DEC vel. of the object, [km/sec]
        vel_ang: 2D array with RA and DEC vel. of the object, [km/sec]
        binsz: bin size, [Mpc]
        nrbin: Nr. of separation bins
    
    Returns:
    """
    assert len(pos_cart) <= 50000, "This routine is not optimized to handle larger data-sets"
    
    binnr = len(bins)
    binwidth = np.diff(bins)[0]
    nr_of_obj = len(pos_cart)

    # allocate memory for results for each object
    vij = np.array([np.zeros(binnr) for j in range(nr_of_obj - 1)])
    counts = np.array([np.zeros(binnr) for j in range(nr_of_obj - 1)])
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
        vij,
        counts,
    ))
    
    vij = np.array(vij)
    counts = np.array(counts)
    vij = vij.sum(axis=0)
    counts = counts.sum(axis=0)
    assert not np.any(counts == 0)
    pest = vij / counts
    rsep = make_rsep(binnr, binwidth)
    print(pest.shape, rsep.shape)
    return rsep, pest


@nb.jit(nopython=True, nogil=True)
def pairwise_one_row(
    i,
    pos_cart_x: np.ndarray,
    pos_cart_y: np.ndarray,
    pos_cart_z: np.ndarray,
    v_cart_x: np.ndarray,
    v_cart_y: np.ndarray,
    v_cart_z: np.ndarray,
    binnr: int,
    binwidth: int,
    nom: np.ndarray,
    counts: np.ndarray,
):
    """
    Args:
        i: for which object index to compute mean pairwise velocity estimate
        binnr: number of separation binszs for the estimator
        binwidth: bin size, [Mpc]
        nom: numpy array of size binnr
        denom: idem
    """
    pos_i = np.array([pos_cart_x[i], pos_cart_y[i], pos_cart_z[i]])
    v_cart_i = np.array([v_cart_x[i], v_cart_y[i], v_cart_z[i]])

    for j in range(i + 1, len(pos_cart_x)):
        # get pairwise distance between obj_i and obj_j
        pos_j = np.array([pos_cart_x[j], pos_cart_y[j], pos_cart_z[j]])
        pos_ij = pos_i - pos_j
        pos_ij_norm = np.linalg.norm(pos_ij)
        # get bin
        binval_ij = int(pos_ij_norm / binwidth)
        
        #assert binval_ij_1 == binval_ij
        if binval_ij < binnr:
            v_cart_j = np.array([v_cart_x[j], v_cart_y[j], v_cart_z[j]])
            v_ij = v_cart_i - v_cart_j
            nom[binval_ij] += np.linalg.norm(v_ij)
            counts[binval_ij] += 1


def make_rsep_uneven_bins(bin_edges: np.ndarray) -> np.ndarray:
    """
    Generates x axis of the histogram as in make_rsep
    bin_edges is a numpy array with the edges of the bins.
    """
    return (bin_edges[1:] + bin_edges[:-1]) / 2.0


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
