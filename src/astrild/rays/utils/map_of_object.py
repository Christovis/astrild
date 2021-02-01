import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.interpolate import RectBivariateSpline
import time


def timing(f):
    def wrap(*args):
        time1 = time.time()
        ret = f(*args)
        time2 = time.time()
        print(
            "{:s} function took {:.3f} ms".format(
                f.__name__, (time2 - time1) * 1000.0
            )
        )
        return ret

    return wrap


@timing
def trim_object(objects, tile_conv, args):
    """
    - Load data from one of the tunnels text files,
    - Requires input of smoothed convergence map
    Either:
        - Trim away voids within 1*max-void-radius of the edge of the map
        - Trim away voids within kpe (kappa profile extent, args["extend"])
          times their own radii of the edge of the map

    Returns:
        void radius, and void centers in x and y (degrees or pixel numbers available)
    """
    # remove voids within kpc of their own radii from the edge
    bool_3_i_x = objects["pos_x"]["pix"] + args["extend"] * objects["radius"][
        "pix"
    ] < len(tile_conv)
    bool_3_ii_x = (
        objects["pos_x"]["pix"] - args["extend"] * objects["radius"]["pix"] > 0
    )
    bool_3_i_y = objects["pos_y"]["pix"] + args["extend"] * objects["radius"][
        "pix"
    ] < len(tile_conv)
    bool_3_ii_y = (
        objects["pos_y"]["pix"] - args["extend"] * objects["radius"]["pix"] > 0
    )
    indx_b = np.logical_and(
        np.logical_and(bool_3_i_x, bool_3_ii_x),
        np.logical_and(bool_3_i_y, bool_3_ii_y),
    )
    # trim = np.logical_and(trim_a,trim_b)

    # iterate over nested dictionary and index for all keys
    for major_key in objects.keys():
        for minor_key in objects[major_key].keys():
            objects[major_key][minor_key] = objects[major_key][minor_key][
                indx_b
            ]
    return objects


def objectmap_from_map(objects, mapp, args):
    """
    Generate maps for a list of objects [peaks, voids].

    Args:
        objects : dict
        mapp : np.ndarray
        args : dict

    Returns:
        profiles : np.array
    """
    # round up to make sure all values are included
    object_radius_upper = np.ceil(
        objects["radius"]["pix"] * args["extend"]
    ).astype(int)
    print("max object radius in this bin:", object_radius_upper.max())

    objectmaps = np.zeros(
        (
            2 * object_radius_upper.max(),
            2 * object_radius_upper.max(),
            len(objects["radius"]["pix"]),
        )
    )
    # Run through objects
    for i in range(len(objects["radius"]["pix"])):
        # create tile of mapp around object center within its radii
        objectmap = mapp[
            centre[0]
            - object_radius_upper[i] : centre[0]
            + object_radius_upper[i],
            centre[1]
            - object_radius_upper[i] : centre[1]
            + object_radius_upper[i],
        ]

        if len(mapp_tile) < len(objectmaps[:, 0, 0]):
            coord = np.arange(len(objectmap[:, 0]))
            objectmap_spline = RectBivariateSpline(coord, coord, objectmap)
            print("old shape", objectmap.shape())
            coord = np.arange(len(objectmaps[:, 0, 0]))
            objectmap = objectmap_spline(coord, coord)
            print("new shape", objectmap.shape())

        objectmaps[:, :, i] = mapp_tile
    return True
