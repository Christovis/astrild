from typing import List, Optional, Tuple, Type

import pandas as pd
import numpy as np


def categorize_sizes(
    objects: pd.DataFrame,
    binning_method: str,
    nr_size_cats: int,
    min_obj_nr: int,
) -> pd.DataFrame:
    """
    Group objects according to their angular size into Nbins different bins.

    Args:
    """
    # units of radii bins
    if binning_method is "log":
        obj_size = np.log10(objects["rad_deg"].values)
    else:
        obj_size = objects["rad_deg"].values

    # create radii bins
    cats = np.linspace(obj_size.min(), obj_size.max(), nr_size_cats)
    # sort objects into their size category
    objects["size_cat"] = np.digitize(obj_size, cats, right=True)
    # count nr of objects in categories
    cat_idx, count = np.unique(objects["size_cat"].values, return_counts=True)
    # select bins with enough voids
    cat_idx = cat_idx[count >= min_obj_nr]
    # filter objects in valid categories
    objects = objects.loc[objects["size_cat"].isin(cat_idx)]
    return objects


def minimal_voids(voids, tracers, args):
    """
    Approximate prescription of minimal voids by weighting voids by their average
    tracer number density within voids compared to the mean tracer density.
    DOI: 10.1093/mnras/stv1994

    Args:
        tracers: dict
            E.g. peaks or HOD galaxies
    """
    from scipy import spatial

    # Background density
    density_tot = len(tracers) / args.field_width ** 2

    # Void density
    tree = spatial.cKDTree(tracers["POS"]["pix"])
    # Scipy v1.1.0
    neighbour_nr = np.asarray(
        [
            len(
                tree.query_ball_point(
                    np.array(voids["POS"]["pix"][oo]), voids["RAD"]["pix"][oo]
                )
            )
            for oo in range(len(np.array(voids["RAD"]["pix"])))
        ]
    )
    void_volums = np.pi * voids["RAD"]["pix"] ** 2
    density_voids = neighbour_nr / void_volums

    # density contrast
    delta = density_voids / density_tot
    select = delta < 1

    print("++++++++", density_tot, density_voids.min(), density_voids.max())
    print("++++++++", delta.min(), delta.max())
    print("++++++++", np.unique(select))

    voids["minimal"] = delta < 1
    return voids


def trim_edges(data, extend: float, npix: int) -> dict:
    """
    Remove edge effects.

    Either:
        - Trim away voids within 1*max-void-radius of the edge of the map
        - Trim away voids within kpe (kappa profile extent, args["extend"])
          times their own radii of the edge of the map

    Returns:
        object: dict
            radius, centers in x and y (degrees or pixel numbers available)
    """
    # remove voids outside of 1 * max void radius
    # max_void_radius = np.max(void_radius)
    # trim_length = 1 * max_void_radius
    # bool_1_x = void_x <= (10 - trim_length)
    # bool_2_x = void_x >= trim_length
    # bool_1_y = void_y <= (10 - trim_length)
    # bool_2_y = void_y >= trim_length
    # trim_a = np.logical_and(
    #    np.logical_and(bool_1_x,bool_2_x),
    #    np.logical_and(bool_1_y,bool_2_y),
    # )

    # remove voids within kpe of their own radii from the edge
    # bool_3_i_x = objects["POS"]["pix"][:, 0] + \
    #             args.extend * objects["RAD"]["pix"] < args.Npix
    # bool_3_ii_x = objects["POS"]["pix"][:, 0] - \
    #              args.extend * objects["RAD"]["pix"] > 0
    # bool_3_i_y = objects["POS"]["pix"][:, 1] + \
    #             args.extend * objects["RAD"]["pix"] < args.Npix
    # bool_3_ii_y = objects["POS"]["pix"][:, 1] - \
    #              args.extend * objects["RAD"]["pix"] > 0
    bool_3_i_x = data["x_pix"].values + extend * data["rad_pix"].values < npix
    bool_3_ii_x = data["x_pix"].values - extend * data["rad_pix"].values > 0
    bool_3_i_y = data["y_pix"].values + extend * data["rad_pix"].values < npix
    bool_3_ii_y = data["y_pix"].values - extend * data["rad_pix"].values > 0

    indx_b = np.logical_and(
        np.logical_and(bool_3_i_x, bool_3_ii_x), np.logical_and(bool_3_i_y, bool_3_ii_y)
    )
    # trim = np.logical_and(trim_a,trim_b)

    # iterate over nested dictionary and index for all keys
    # for major_key in objects.keys():
    #    for minor_key in objects[major_key].keys():
    #        objects[major_key][minor_key] = objects[major_key][minor_key][indx_b]

    return data[indx_b]
