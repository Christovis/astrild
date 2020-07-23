import numpy as np
import matplotlib.pyplot as plt


def plot_halo_mass_function(mass, nbins=20, color="black", label="default"):

    if isinstance(mass, numpy.ndarray):
        mass_count, mass_bin = halo_mass_function(mass)
        plt.loglog(
            mass_bin, mass_count, marker="o", markersize=3.0, color=color, label=label
        )

    elif isinstance(mass, list):
        # Run through mass-functions
        for mm in mass:
            mass_func, edges = np.histogram(mm, bins=bins)
            plt.loglog(
                mass_bin,
                mass_count,
                marker="o",
                markersize=3.0,
                color=color,
                label=label,
            )

    plt.ylabel("Number of halos")
    plt.xlabel(r"$M_{200c}$")
