from typing import Dict, List, Optional, Type, Union

import pandas as pd
import xarray as xr
import numpy as np
from scipy.spatial import cKDTree
from scipy.interpolate import RectBivariateSpline


def from_map(
    objects: pd.DataFrame,
    skymap: np.ndarray,
    extend: float,
    nr_profile_bins: int,
) -> dict:
    """
    Generate profiles for a list of objects [peaks, voids].

    Requires the:
        - convergance map from which the voids are generated,
        - void radii,
        - the cut off (extend(float) - in units of void radii),
        - annulus thickiness (units of pixel number),
        - void centers (x and y seperately)

    Args:
        objects:
        skymap:
            The 2D array that represents the partial sky map (e.g. convergance).
        extend:
        nr_profile_bins:
        field_conversion:
            E.g. if you want to scale the profile by its mean.

    Returns:
        profiles:
            Dictionary of profiles for each oject in objects.
    """
    # round up to make sure all values are included
    profile_radii = np.ceil(objects["rad_pix"].values * extend).astype(int)
    # annulus thickness normalised against ith void radius
    delta_eta = extend / nr_profile_bins

    profiles = {"values": []}
    # Run throug objects
    for idx, row in objects.iterrows():
        profile = profiling(
            int(row["rad_pix"]),
            (int(row["x_pix"]), int(row["y_pix"])),
            skymap,
            delta_eta,
            extend,
            nr_profile_bins,
        )
        profiles["values"].append(profile["values"])
    profiles["values"] = np.asarray(profiles["values"])
    profiles["radii"] = profile["radii"]
    return profiles

def to_file(self, ray_z, ray_nrs, finder: str, sigmas, profiles_radii, args) -> None:
    """
    Save file to .nc file using xarray.Dataset.

    Args:
    """
    ds = xr.Dataset(
        {
            "mean": (["ray_nr", "sigma", "size_bin", "radius"], self.mean),
            "lowerr": (["ray_nr", "sigma", "size_bin", "radius"], self.lowerr),
            "higherr": (["ray_nr", "sigma", "size_bin", "radius"], self.higherr),
        },
        coords={
            "redshift": ray_z,
            "ray_nr": ray_nrs,
            "sigma": sigmas,
            "radius": profiles_radii,
            "size_min": (["ray_nr", "sigma", "size_bin"], self.obj_size_min),
            "size_max": (["ray_nr", "sigma", "size_bin"], self.obj_size_max),
            "nr_of_obj": (["ray_nr", "sigma", "size_bin"], self.obj_in_prof),
        },
    )
    if finder is "svf":
        ds.to_netcdf(
            args.out_dir
            + "%s/%s/SVF_lc%d_isw_profiles.nc" % (args.finder, args.hod, lc_nrs[ll])
        )
    elif finder is "tunnels":
        ds.to_netcdf(args.out_dir + "profiles_kappa2_lc.nc")


def profiling(
    obj_radius: int,
    obj_pos: tuple,
    mapp: np.ndarray,
    delta_eta: float,
    extend: float,
    nr_profile_bins: int,
) -> Dict[str, np.array]:
    """
    Profile of single object on a 2D map.
    
    Args:
        obj_radius:
            Radius of object in units of pixels.
        obj_pos:
            (x,y)-position of object in units of pixels.
        mapp:
        delta_eta:
            The annulus thickness normalised against ith void radius.
        extend:
        nr_profile_bins:
    """
    prof_radius = np.ceil(obj_radius * extend).astype(int)
    # distance of every pixel to centre
    pix_x = pix_y = np.arange(-prof_radius, prof_radius)
    pix_xx, pix_yy = np.meshgrid(pix_x, pix_y)
    pix_dist = np.sqrt(pix_xx ** 2 + pix_yy ** 2) / obj_radius

    # eta gives the annulus to which the pixel belongs
    eta = (pix_dist / delta_eta).astype(int)
    pix_xx = pix_xx[eta < nr_profile_bins]
    pix_yy = pix_yy[eta < nr_profile_bins]
    eta = eta[eta < nr_profile_bins]

    annulus_count = [list(eta).count(ee) for ee in np.unique(eta)]
    # Make sure annulus_count has nr_profile_bins
    annulus_buffer = list(np.zeros(nr_profile_bins - len(np.unique(eta))))
    annulus_count = np.asarray(annulus_count + annulus_buffer)
    annulus_value = np.zeros(nr_profile_bins)
    # Run through shells
    for pp in range(len(eta)):
        annulus_value[eta[pp]] += mapp[
            obj_pos[1] + pix_xx[pp], obj_pos[0] + pix_yy[pp],
        ]

    profile = {}
    # Radial bin distance
    r_2 = np.linspace(0, extend, nr_profile_bins + 1)
    r = 0.5 * (r_2[1:] + r_2[:-1])
    profile["radii"] = r
    # Radial bin values
    profile["values"] = annulus_value / annulus_count
    return profile


def mean_profile(profiles, object_radius_pn, extend, thickness, n_bins):
    """
    Find mean profiles of quantity from array of kappa profiles from tunnels.txt file
    """
    object_radius_upper = np.ceil(object_radius_pn * extend).astype(int)
    num_objects = len(object_radius_pn)
    # n = void_radius_upp/thickness
    radial_bins_num = np.ceil(object_radius_pn / thickness).astype(int)

    r = []
    for i in range(len(kappa_profiles)):
        binEdges = np.linspace(0, extend, len(kappa_profiles[i]))
        r_mid = (binEdges[1:] + binEdges[:-1]) / 2
        r.append(r_mid)

    # n_bins = 11
    bins = np.linspace(0, extend, n_bins)
    counts = []
    kappa_i = np.zeros((n_bins))
    N = np.zeros((n_bins))

    for i in range(num_objects):
        # Return the indices of the bins to which each annulus belongs.
        counts.append(np.digitize(r[i], bins))

    # Run through each object
    for i in range(num_objects):
        # Run through each data point in a particular object
        for j in range(len(counts[i])):
            # subtract one because np.digitize does not follow python indexing convention
            bin_i = counts[i][j] - 1
            annulus_kappa_value = kappa_profiles[i][j]
            kappa_i[bin_i] += annulus_kappa_value * radial_bins_num[i] ** 2
            N[bin_i] += 1

    # now need to turn kappa_i into mean kappa values
    kappas = kappa_i / (np.sum(radial_bins_num ** 2))

    return kappas


def interpolate(
    profile: np.array, objects_rad: float, extend: float, nr_rad_bins: int,
) -> np.array:
    """
    Interpolate profiles to have same radii bins.

    Args:
        profile: np.array of 2D np.ndarray containing radii and values
    """
    r = np.linspace(0, extend, nr_rad_bins)
    nans = np.argwhere(np.isnan(profile))

    if len(nans) > 0:
        for i in range(len(nans)):
            profile[nans[i, 0], nans[i, 1]] = 0

        for i in range(len(nans)):
            sort = profile[nans[i, 0]] != 0
            profile[nans[i, 0]] = np.interp(r, r[sort], profile[nans[i, 0]][sort])

    elif len(nans) == 0 and len(np.where(profile == 0)[0]) != 0:
        zeros_min = np.array(np.where(profile == 0)).T[0, 0]
        zeros = np.array(np.where(profile == 0)).T

        for i in range(len(profile) - zeros_min):
            sort = profile[zeros[i, 0]] != 0
            profile[zeros[i, 0]] = np.interp(r, r[sort], profile[zeros[i, 0]][sort])

    return profile


def mean_and_interpolate(
    profile, objects_rad: float, extend: float, nr_rad_bins: int,
) -> np.array:
    """
    Get the mean profile of all the profiles of objects, which are weighted
    by their size.

    Args:
        profile: np.array of 2D np.ndarray containing radii and values
    """
    r = np.linspace(0, extend, nr_rad_bins)
    nans = np.argwhere(np.isnan(profile))

    if len(nans) > 0:
        nans_min = nans[0, 0]

        for i in range(len(nans)):
            profile[nans[i, 0], nans[i, 1]] = 0

        for i in range(len(nans)):
            sort = profile[nans[i, 0]] != 0
            profile[nans[i, 0]] = np.interp(r, r[sort], profile[nans[i, 0]][sort])

    elif len(nans) == 0 and len(np.where(profile == 0)[0]) != 0:
        zeros_min = np.array(np.where(profile == 0)).T[0, 0]
        zeros = np.array(np.where(profile == 0)).T

        for i in range(len(profile) - zeros_min):
            sort = profile[zeros[i, 0]] != 0
            profile[zeros[i, 0]] = np.interp(r, r[sort], profile[zeros[i, 0]][sort])

    mean_profile = np.average(profile, axis=0, weights=objects_rad ** 2)
    return mean_profile


def _blockshaped(arr, nrows, ncols):
    """
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.
    """
    h, w = arr.shape
    return (
        arr.reshape(h // nrows, nrows, -1, ncols)
        .swapaxes(1, 2)
        .reshape(-1, nrows, ncols)
    )


def bootstrapping(
    profiles: np.ndarray,
    mean_profile: np.ndarray,
    objects: pd.DataFrame,
    npix: int,
    extend: float,
    nr_rad_bins: int,
) -> None:
    """
    Find bootstrap errors on profiles, given as the 16th and 84th percentiles

    Args:
        profiles: list
            object profiles
    """
    num_objects = len(profiles)

    # at each void center, place a tag (i+1) which indicates
    # which void kappa profile to call
    mask = np.zeros((npix, npix))
    for i in range(num_objects):
        mask[objects["x_pix"].values[i], objects["y_pix"].values[i]] = i + 1

    # split mask into blocks
    w = 256
    blocks = _blockshaped(mask, w, w)

    # turn blocks into lists now that i have the useful information
    block_i = []
    for i in range(len(blocks)):
        x_i, y_i = np.where(blocks[i] != 0)
        block_i.append(blocks[i][x_i, y_i])

    b_i = []
    profile_indices = []
    b_l = 100  # number of bootstrap realisations
    # Run through bootstraps
    for j in range(b_l):

        profile_indices_i = []
        # Run through mask blocks
        for i in range(len(blocks)):
            rand_int = np.random.randint(0, len(blocks))
            b = block_i[rand_int]
            profile_indices_i.append(b)
        profile_indices.append(profile_indices_i)

    mean_profile_list = np.zeros((b_l, nr_rad_bins))
    # Run through bootstraps
    for j in range(b_l):
        # subtract one to turn tags into python indices
        k_i = np.concatenate(profile_indices[j]) - 1
        num_profiles = len(k_i)
        profiles_boot = []
        radii = []

        for l in range(num_profiles):
            profiles_boot.append(profiles[int(k_i[l])])
            radii.append(objects["rad_pix"].values[int(k_i[l])])
        # now compute mean profile
        profiles_boot = np.array(profiles_boot)
        radii = np.array(radii)
        sort = np.flip(np.argsort(radii), 0)
        if len(sort) > 0:
            profile_boot_mean = mean_and_interpolate(
                profiles_boot[sort], radii[sort], extend, nr_rad_bins,
            )
            mean_profile_list[j] = profile_boot_mean

    error = np.zeros((nr_rad_bins, 2))
    for i in range(nr_rad_bins):
        error[i] = [
            np.percentile(mean_profile_list.T[i], 16),  # lower bound
            np.percentile(mean_profile_list.T[i], 84),  # higher bound
        ]

    e = np.array([mean_profile - error.T[0], error.T[1] - mean_profile])
    e = np.squeeze(e)  # remove single dimensional entries
    return e


#def __init__(self, voids: pd.DataFrame, z_nr, args):
#    self.mean = np.zeros(
#        (z_nr, len(voids.finder_sigmas), args.nr_radius_bins, args.nr_profile_bins)
#    )
#    self.lowerr = np.zeros(
#        (z_nr, len(voids.finder_sigmas), args.nr_radius_bins, args.nr_profile_bins)
#    )
#    self.higherr = np.zeros(
#        (z_nr, len(voids.finder_sigmas), args.nr_radius_bins, args.nr_profile_bins)
#    )
#    self.obj_size_min = np.zeros(
#        (z_nr, len(voids.finder_sigmas), args.nr_radius_bins,)
#    )
#    self.obj_size_max = np.zeros(
#        (z_nr, len(voids.finder_sigmas), args.nr_radius_bins,)
#    )
#    self.obj_in_prof = np.zeros(
#        (z_nr, len(voids.finder_sigmas), args.nr_radius_bins,)
#    )

