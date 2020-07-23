import os, re, sys, glob
import argparse
import pandas as pd
import numpy as np
from scipy import ndimage

from astropy import units as un

from wys_ars.rays.utils.tunnels import TunnelsFinder


def get_args(argv=None):
    parser = argparse.ArgumentParser(description="Sky-map creation settings.")
    parser.add_argument(
        "--mapdir",
        dest="mapdir",
        action="store",
        type=str,
        default="/cosma7/data/dp004/dc-beck3/3_Proca/lightcone_cvg_b3_100/1_maps/",
        help="File in which maps is saved.",
    )
    parser.add_argument(
        "--outdir",
        dest="outdir",
        action="store",
        type=str,
        default="/cosma7/data/dp004/dc-beck3/3_Proca/lightcone_cvg_b3_100/2_voids/tunnels/",
        help="File in which tunnels is saved.",
    )
    parser.add_argument(
        "--Npix",
        dest="Npix",
        action="store",
        type=int,
        default="2048",
        help="Number of pixels per edge",
    )
    parser.add_argument(
        "--fov",
        dest="fov",
        action="store",
        type=float,
        default=10.0,
        help="Opening angle of light-cone. [deg]",
    )
    parser.add_argument(
        "--kernel_width",
        dest="k_width",
        action="store",
        type=float,
        default="2.5",
        help="Width of convolution kernel.",
    )
    parser.add_argument(
        "--verbose",
        dest="verbose",
        action="store_true",
        default=True,
        help="File in which map is saved.",
    )
    args = parser.parse_args()

    return args


def run(box_nrs, ray_nrs, sigmas, args):
    """
    """
    if box_nrs[0] is 0:
        print("# Find tunnels on observed path-of-sky")
        mapfile = args.mapdir + "kappa2_flat_lc.npy"

        first = True
        # run through dipol signal significances
        for sigma in sigmas:
            # find tunnels
            tunnels = TunnelsFinder(mapfile, args)
            tunnels.find_peaks(args)
            tunnels.find_voids([sigma], args)

            if first is True:
                voids_df_sum = tunnels.voids
                peaks_df_sum = tunnels.peaks
                first = False
            else:
                voids_df_sum = voids_df_sum.append(tunnels.voids, ignore_index=True)
                peaks_df_sum = peaks_df_sum.append(tunnels.peaks, ignore_index=True)

        voids_df_sum.to_hdf(args.outdir + "voids_tunnels_kappa2_lc.h5", key="s")
        peaks_df_sum.to_hdf(args.outdir + "peaks_tunnels_kappa2_lc.h5", key="s")

    else:

        for box_nr in box_nrs:

            if not ray_nrs:
                # read all ray-outputs for this box
                fname = args.mapdir + "slices/kappa2_flat_box%d_ray*.npy" % box_nr
                ray_map_files = glob.glob(fname)
                # sort them with decreasing redshift
                ray_nrs = [
                    int(re.findall(r"\d+", rfile.split("_")[-1].split(".")[0])[0])
                    for rfile in ray_map_files
                ]
                ray_map_files = [ray_map_files[nn] for nn in np.argsort(ray_nrs)]
                ray_nrs = np.sort(ray_nrs)

            else:
                # read indicated ray-outputs for this box
                ray_map_files = args.mapdir + "slices/kappa2_flat_box%d" % box_nr
                ray_map_files = [ray_map_files + "_ray%d.npy" % ii for ii in ray_nrs]
                # sort them with decreasing redshift
                ray_nrs = [
                    int(re.findall(r"\d+", rfile.split("_")[-1].split(".")[0])[0])
                    for rfile in ray_map_files
                ]
                ray_map_files = [ray_map_files[nn] for nn in np.argsort(ray_nrs)]
                ray_nrs = np.sort(ray_nrs)

            # run through snapshots of box
            for ii in range(len(ray_nrs)):
                print("Box-Nr %d Ray-Nr %d" % (box_nr, ray_nrs[ii]))

                if (box_nr is 4) and (ray_nrs[ii] is 8):
                    continue

                mapfile = args.mapdir + "slices/kappa2_flat_box%d_ray%d.npy" % (
                    box_nr,
                    ray_nrs[ii],
                )

                first = True
                # run through dipol signal significances
                for sigma in sigmas:
                    # find tunnels
                    tunnels = TunnelsFinder(mapfile, args)
                    tunnels.find_peaks(args)
                    tunnels.find_voids([sigma], args)

                    if first is True:
                        voids_df_sum = tunnels.voids
                        peaks_df_sum = tunnels.peaks
                        first = False
                    else:
                        voids_df_sum = voids_df_sum.append(
                            tunnels.voids, ignore_index=True,
                        )
                        peaks_df_sum = peaks_df_sum.append(
                            tunnels.peaks, ignore_index=True,
                        )

                file_out = (
                    args.outdir
                    + "slices/voids_tunnels_kappa2_box%d_ray%d.h5"
                    % (box_nr, ray_nrs[ii])
                )
                voids_df_sum.to_hdf(file_out, key="df")
                file_out = (
                    args.outdir
                    + "slices/peaks_tunnels_kappa2_box%d_ray%d.h5"
                    % (box_nr, ray_nrs[ii])
                )
                peaks_df_sum.to_hdf(file_out, key="df")


if __name__ == "__main__":
    args = get_args()
    box_nrs = [0]  # if =0 read full integral
    ray_nrs = []  # if =0 read full integral
    sigmas = [0.0, 1.0, 2.0, 3.0]  # peaks signal significances

    run(box_nrs, ray_nrs, sigmas, args)
