import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree
from scipy.interpolate import RectBivariateSpline
import time


class Profiles3D:
    def get_one_profile(
        self, halo_idx: int, quantity: str = "mass", nbins: int = 20,
    ):
        """
        Get density profile of halo with id halo_idx
        Args:
            halo_idx:
            The halo
            quantity:
                [count, mass, temperature, velocity dispersion,
                pairwise radial velocity dispersion,]
            nbins: number of bins in the radial direction

        Returns:
                bin_radii; radial bin centers.
                bin_densities: density in randial bins
        """
        # particles that belong to halo halo_idx
        coordinates_in_halo = self.coordinates[
            self.cum_N_particles[halo_idx] : self.cum_N_particles[halo_idx]
            + self.N_particles[halo_idx]
        ]

        # particle distances w.r.t halo centre
        rel_par_pos = (
            np.linalg.norm((coordinates_in_halo - self.halo_pos[halo_idx]), axis=1)
            / self.r200c[halo_idx]
        )

        if "velo" in quantity:
            # particle velocity w.r.t halo
            rel_par_vel = (
                np.linalg.norm((coordinates_in_halo - self.halo_pos[halo_idx]), axis=1)
                / self.r200c[halo_idx]
            )
            bin_radii, bin_densities = prof.from_particle_data(
                rel_par_pos, rel_par_vel, 0, quantity, nbins,
            )
        else:
            bin_radii, bin_densities = prof.from_particle_data(
                rel_par_pos, 0, self.Mpart, quantity, nbins,
            )

        return bin_radii, bin_densities

    def from_particle_data(pos, vel, Mpart, quantity, nbins):
        """
        Get profile of halo
        Args:
            quantity: str
                One of [count, mass, temperature, velocity dispersion,
                pairwise radial velocity dispersion,]
        """
        # create radii bins
        min_rad = 0.05
        max_rad = 1
        nbins = 20
        bins = np.logspace(np.log10(min_rad), np.log10(max_rad), nbins + 1, base=10.0)
        bin_radii = 0.5 * (bins[1:] + bins[:-1])

        # volumne enclosed by each radii bin, normalized by r200
        bin_volumes = 4.0 / 3.0 * np.pi * (bins[1:] ** 3 - bins[:-1] ** 3)

        # only for dm
        number_particles = np.histogram(pos, bins=bins)[0]
        bin_masses = number_particles * Mpart  # [Msun]
        bin_value = bin_masses / bin_volumes

        return bin_radii, bin_value


def profiling(prof_radius, obj_radius, obj_posx, obj_posy, mapp, delta_eta, args):
    """Profile of single object on a 2D scalar map"""
    # distance of every pixel to centre
    pix_x = pix_y = np.arange(-prof_radius, prof_radius)
    pix_xx, pix_yy = np.meshgrid(pix_x, pix_y)
    pix_dist = np.sqrt(pix_xx ** 2 + pix_yy ** 2) / obj_radius

    # eta gives the annulus to which the pixel belongs
    eta = (pix_dist / delta_eta).astype(int)
    pix_xx = pix_xx[eta < args.nr_profile_bins]
    pix_yy = pix_yy[eta < args.nr_profile_bins]
    eta = eta[eta < args.nr_profile_bins]

    annulus_count = [list(eta).count(ee) for ee in np.unique(eta)]
    # Make sure annulus_count has nr_profile_bins
    annulus_buffer = list(np.zeros(args.nr_profile_bins - len(np.unique(eta))))
    annulus_count = np.asarray(annulus_count + annulus_buffer)
    annulus_value = np.zeros(args.nr_profile_bins)
    # Run through shells
    for pp in range(len(eta)):
        annulus_value[eta[pp]] += mapp[
            obj_posy + pix_xx[pp], obj_posx + pix_yy[pp],
        ]

    profile = {}
    # Radial bin values
    profile["values"] = annulus_value / annulus_count
    # Radial bin distance
    r_2 = np.linspace(0, args.extend, args.nr_profile_bins + 1)
    r = 0.5 * (r_2[1:] + r_2[:-1])
    profile["radii"] = r

    return profile


def mean_profile(profiles, object_radius_pn, extent, thickness, n_bins):
    """
    Find mean profiles of quantity from array of kappa profiles from tunnels.txt file
    """
    object_radius_upper = np.ceil(object_radius_pn * extent).astype(int)
    num_objects = len(object_radius_pn)
    # n = void_radius_upp/thickness
    radial_bins_num = np.ceil(object_radius_pn / thickness).astype(int)

    r = []
    for i in range(len(kappa_profiles)):
        binEdges = np.linspace(0, extent, len(kappa_profiles[i]))
        r_mid = (binEdges[1:] + binEdges[:-1]) / 2
        r.append(r_mid)

    # n_bins = 11
    bins = np.linspace(0, extent, n_bins)
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


def interpolate(profile, objects_rad, args):
    """
    Interpolate profiles to have same radii bins.

    Args:
        profile: np.array of 2D np.ndarray containing radii and values
    """
    r = np.linspace(0, args.extend, args.nr_profile_bins)
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

    return profile


def mean_and_interpolate(profile, objects_rad, args):
    """
    Get the mean profile of all the profiles of objects

    Args:
        profile: np.array of 2D np.ndarray containing radii and values
    """
    r = np.linspace(0, args.extend, args.nr_profile_bins)
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


def bootstrapping(profiles, mean_profile, objects, args):
    """
    Find bootstrap errors on profiles, given as the 16th and 84th percentiles

    Args:
        profiles: list
            object profiles
    """
    num_objects = len(profiles)

    # at each void center, place a tag (i+1) which indicates
    # which void kappa profile to call
    mask = np.zeros((args.Npix, args.Npix))
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

    mean_profile_list = np.zeros((b_l, args.nr_profile_bins))
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
                profiles_boot[sort], radii[sort], args
            )
            mean_profile_list[j] = profile_boot_mean

    error = np.zeros((args.nr_profile_bins, 2))
    for i in range(args.nr_profile_bins):
        error[i] = [
            np.percentile(mean_profile_list.T[i], 16),  # lower bound
            np.percentile(mean_profile_list.T[i], 84),  # higher bound
        ]

    e = np.array([mean_profile - error.T[0], error.T[1] - mean_profile])
    e = np.squeeze(e)  # remove single dimensional entries
    return e
