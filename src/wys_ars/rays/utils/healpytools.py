import sys
import numpy as np
import pandas as pd
import healpy as hp
from scipy.interpolate import RectBivariateSpline

# sys.path.insert(0, './lib/')
# import lm_cfuncs as cf

# convolution with Gaussian
# healpy.sphtfunc.smoothing


def create_map(theta, phi, value, nside, npix, nsources):
    """

    """
    # Go from HEALPix coordinates to indices
    indices = hp.ang2pix(nside, theta[:], phi[:])  # *180/np.pi,  # *180/np.pi,

    # Initiate the map and fill it with the values
    hpmap = np.ones(npix, dtype=np.float) * hp.UNSEEN
    for i in range(nsources):
        hpmap[indices[i]] = value[i]

    return hpmap


# Rotation
def rotate_map(hmap, rot_theta, rot_phi, nside):
    """
    Take hmap (a healpix map array) and return another healpix map array 
    which is ordered such that it has been rotated in (theta, phi) by the 
    amounts given.

    Usage:
    -----
        hpxmap = hpu.rotate_map(hpxmap, 90, 180, nside)
    """
    nside = hp.npix2nside(len(hmap))
    # Get theta, phi for non-rotated map
    t, p = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))  # theta, phi
    # Define a rotator
    r = hp.Rotator(deg=True, rot=[rot_phi, rot_theta])
    # Get theta, phi under rotated co-ordinates
    trot, prot = r(t, p)
    # Interpolate map onto these co-ordinates
    rot_map = hp.get_interp_val(hmap, trot, prot)
    return rot_map


def lensing(cmb, lp1, lp2, deflt_1, deflt_2, nsources, nside):
    """
    lp1 lp2: from ray-ramses theta_co phi_co
    """
    # Mapping light rays from image plane to source plan
    [sp1, sp2] = [lp1 - deflt_1, lp2 - deflt_2]  # [arcsec]
    cmb_lensed = hp.get_interp_val(cmb, sp1, sp2)
    return cmb_lensed
