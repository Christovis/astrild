import numpy as np
from scipy.stats import binned_statistic
from scipy.optimize import curve_fit, newton

from halotools.mock_observables import tpcf
from halotools import mock_observables as mo

from wys_ars.utils.arepo_hdf5_library import read_hdf5


class SubFind:
    def halo_mass_fct(
        snapshot: read_hdf5.snapshot, limits: tuple = (11.78, 16), nbins: int = 20,
    ) -> tuple:
        """
        Compute the halo mass function
        
        Args:
            data:
                halo mass in [M_{\odot}/h]
        """
        if "Group_M_Crit200" not in snapshot.cat:
            raise ValueError(f"'Group_M_Crit200' data not loaded")
        bins = np.logspace(min(limits), max(limits), nbins + 1)
        mass = snapshot.cat["Group_M_Crit200"][:]  # [Msun/h]
        mass = mass[min(limits) < mass]
        mass_count, edges = np.histogram(mass, bins=bins)
        mass_bin = (edges[1:] + edges[:-1]) / 2.0
        return mass_bin, mass_count

    def two_point_corr_fct(
        snapshot: read_hdf5.snapshot,
        limits: tuple = None,
        nbins: int = None,
        boxsize: float = None,
    ):
        """
        Comput the real space two point correlation function using halotools

        Args:
            data:
                3D array with the cartesian coordiantes of the tracers.
            boxsize:
                box size of the simulation in the same units as positions.
        """
        if boxsize is None:
            boxsize = snapshot.header.boxsize / 1e3  #[Mpc/h]
        if limits is None:
            limits = (0.3, boxsize / 5)
        if nbins is None:
            nbins = int(2 / 3 * max(limits))

        r = np.geomspace(min(limits), max(limits), nbins)
        r_c = 0.5 * (r[1:] + r[:-1])
        real_tpcf = mo.tpcf(
            snapshot.cat["GroupPos"][:] * snapshot.header.hubble / 1e3,  #[Mpc/h]
            rbins=r,  #[Mpc/h]
            period=boxsize,
            estimator="Landy-Szalay",
        )
        return r_c, real_tpcf

    def mean_pairwise_velocity(
        snapshot: read_hdf5.snapshot,
        limits: tuple,
        nbins: int,
        boxsize: float,
        seperate: dict = None, #{"Group_M_Crit200": 14, "compare": [1, 2]},
    ) -> tuple:
        """
        Comput the real space two point correlation function using halotools

        Args:
            data:
                3D array with the cartesian coordiantes of the tracers.
            boxsize:
                box size of the simulation in the same units as positions.
        """
        if boxsize is None:
            boxsize = snapshot.header.boxsize / 1e3  # [Mpc/h]
        if limits is None:
            limits = (0.3, boxsize / 5)
        if nbins is None:
            nbins = int(2 / 3 * max(limits))

        r = np.geomspace(min(limits), max(limits), nbins)
        r_c = 0.5 * (r[1:] + r[:-1])

        if seperate is None:
            idx1 = np.ones(len(snapshot.cat["GroupVel"][:]), dtype=bool)
            idx2 = np.ones(len(snapshot.cat["GroupVel"][:]), dtype=bool)
        else:
            split_quantity = list(seperate.keys())[0]
            idx1 = snapshot.cat[split_quantity][:] < 10 ** seperate[split_quantity]
            idx2 = snapshot.cat[split_quantity][:] > 10 ** seperate[split_quantity]

            if seperate["compare"][0] == 1:
                idx1 = snapshot.cat[split_quantity][:] < 10 ** seperate[split_quantity]
            elif seperate["compare"][0] == 2:
                idx1 = snapshot.cat[split_quantity][:] > 10 ** seperate[split_quantity]
            
            if seperate["compare"][1] == 1:
                idx2 = snapshot.cat[split_quantity][:] < 10 ** seperate[split_quantity]
            elif seperate["compare"][1] == 2:
                idx2 = snapshot.cat[split_quantity][:] > 10 ** seperate[split_quantity]


        print(
            f"Group one has {len(np.arange(len(idx1))[idx1])} halos and "
            + f"Group two has {len(np.arange(len(idx2))[idx2])} halos"
        )


        pos1 = snapshot.cat["GroupPos"][idx1, :] * snapshot.header.hubble / 1e3  # [Mpc/h]
        vel1 = snapshot.cat["GroupVel"][idx1, :]  # [km/sec]

        pos2 = snapshot.cat["GroupPos"][idx2, :] * snapshot.header.hubble / 1e3  # [Mpc/h]
        vel2 = snapshot.cat["GroupVel"][idx2, :]  # [km/sec]

        pv12 = mo.mean_radial_velocity_vs_r(
            pos1,
            vel1,
            rbins_absolute=r_c,
            sample2=pos2,
            velocities2=vel2,
            period=boxsize,
            # do_auto=False,
            # do_cross=True,
        )
        return r_c[1:], pv12  # TODO this should be

    def concentration_mass_rel(
        snapshot: read_hdf5.snapshot,
        limits: tuple = None,
        nbins: int = 20,
        method: str = "prada",
    ) -> tuple:
        """
        Comput the concentration/mass relation.
        
        Args:
            method:
                prada: DOI: 10.1111/j.1365-2966.2012.21007.x
                faltenbacher:
                nfw:
        """
        _m200c = snapshot.cat["Group_M_Crit200"][:] * snapshot.header.hubble
        _scale_factor = 1.0 / (1.0 + snapshot.header.redshift)
        _r200c_physical = (
            snapshot.cat["Group_R_Crit200"][:] * _scale_factor / 1e3
        )  # [Mpc] ???
        _vmax = snapshot.cat["SubhaloVmax"][:]

        if limits is None:
            # TODO double check untis
            limits = (np.log10(_m200c.min()), np.log10(_m200c.max()))

        # filter mass range
        mass_idx = np.logical_and(
            10 ** min(limits) < _m200c, 10 ** max(limits) > _m200c
        )
        _m200c = _m200c[mass_idx]
        _r200c_physical = _r200c_physical[mass_idx]
        _vmax = _vmax[mass_idx]

        # circular velocity at virial radius
        _v200 = np.sqrt(snapshot.const.G * _m200c / _r200c_physical) * (
            snapshot.const.Mpc / 1000.0
        )  # [km/s]

        if method == "prada":
            _m200c, concentration = _concentration_prada(_m200c, _vmax, _v200)
        elif method == "nfw":
            raise ValueError("not implemented yet")
            # _m200c, concentration = _concentration_nfw(snapshot)

        if len(_m200c) != 0:
            c_mean, edges, _ = binned_statistic(
                _m200c,
                values=concentration,
                statistic="mean",
                bins=np.logspace(min(limits), max(limits), nbins + 1),
            )
            mass_bins = (edges[1:] + edges[:-1]) / 2.0
        else:
            mass_bins = None
            c_mean = None
        return mass_bins, c_mean

    def _concentration_prada(_m200c: np.array, _vmax: np.array, _v200: np.array):
        """
        Computer halo concentration according to Prada et al. (2012)
        (DOI: 10.1111/j.1365-2966.2012.21007.x ; arxiv: 1104.5130)
        """

        def y(x, _vmax, _v200):
            func = np.log(1 + x) - (x / (1 + x))
            return ((0.216 * x) / func) ** 0.5 - (_vmax / _v200)

        concentration = []
        mass_idx = np.zeros(len(_m200c), dtype=bool)
        for halo_idx in range(len(_m200c)):
            if _v200[halo_idx] < _vmax[halo_idx]:
                try:
                    concentration.append(
                        newton(y, x0=5.0, args=(_vmax[halo_idx], _v200[halo_idx]))
                    )
                    mass_idx[halo_idx] = True
                except:
                    continue
        return _m200c[mass_idx], concentration

    #    def _concentration_nfw():
    #        nfw_profile_fitting()
    #        return _m200c, concentration
    #
    #
    #    def nfw_profile_fitting(
    #        snapshot: read_hdf5.snapshot,
    #        nbins: int = 20,
    #        boxsize: float = 500.,
    #    ):
    #        rho_crit = snapshot.const.rho_crit  #[Msun/Mpc^3]
    #
    #        self.nfw_concentration = np.zeros((self.N_halos))
    #        self.rho_s = np.zeros((self.N_halos))
    #        self.chisq = np.zeros((self.N_halos))
    #        self.nfw_profiles_value = np.zeros((self.N_halos, nbins))
    #        self.nfw_profiles_radii = np.zeros((self.N_halos, nbins))
    #
    #        for halo_idx in range(self.N_halos):
    #            bin_radii, bin_densities = self.get_one_profile(halo_idx, 'density', nbins)
    #
    #            # use only radii-bins to fit nfw, where a measurement exists
    #            fit_densities = bin_densities[bin_densities > 0.0]
    #            fit_radii = bin_radii[bin_densities > 0.0]
    #
    #            if (len(fit_densities) > 2) & (self.N_particles[halo_idx] >= 5000):
    #                try:
    #                    popt, pcov = curve_fit(
    #                        nfw,
    #                        fit_radii,
    #                        np.log10(fit_densities),
    #                        p0=(8000 * rho_crit, 5.0),
    #                    )
    #                except:
    #                    popt = (-9999, -9999)
    #
    #                self.rho_s[halo_idx] = popt[0]
    #                self.nfw_concentration[halo_idx] = popt[1]
    #
    #                # use all radii-bins to create nfw-profile
    #                fit = nfw(bin_radii, *popt)
    #                self.nfw_profiles_radii[halo_idx, :] = bin_radii
    #                self.nfw_profiles_value[halo_idx, :] = fit
    #                self.chisq[halo_idx] = (
    #                    1 / len(bin_radii) * np.sum((np.log10(fit_densities) - fit) ** 2)
    #                )
    #
    #            else:
    #                self.rho_s[halo_idx] = -9999
    #                self.nfw_concentration[halo_idx] = -9999
    #                self.chisq[halo_idx] = -9999
    #                self.nfw_profiles_radii[halo_idx, :] = -9999
    #                self.nfw_profiles_value[halo_idx, :] = -9999

    # def density_profiles(
    #    snapshot: read_hdf5.snapshot,
    #    limits: tuple = (11.78, 16),
    #    nbins: int = 20,
    #    boxsize: float = 500.,
    # ):
