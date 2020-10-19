from halotools.mock_observables import tpcf, s_mu_tpcf 
import numpy as np


def compute_real_tpcf(r, pos, vel, boxsize, num_threads = 1):
    '''

    Computes the real space two point correlation function using halotools

    Args:
        r: np.array
             binning in pair distances.
        pos: np.ndarray
             3-D array with the position of the tracers.
        vel: np.ndarray
            3-D array with the velocities of the tracers.
        boxsize: float
            size of the simulation's box.
        num_threads: int
            number of threads to use.

    Returns:
        real_tpcf: np.array
            1-D array with the real space tpcf.

    '''


    real_tpcf = tpcf(pos, r, period = boxsize, num_threads = num_threads)

    return real_tpcf

def compute_tpcf_s_mu(s, mu, pos, vel, los_direction, 
        cosmology, boxsize, num_threads = 1): 
    '''

    Computes the redshift space two point correlation function

    Args:
        s: np.array
            binning in redshift space pair distances.
        mu: np.array
             binning in the cosine of the angle respect to the line of sight.
        pos: np.ndarray
            3-D array with the position of the tracers, in Mpc/h.
        vel: np.ndarray
             3-D array with the velocities of the tracers, in km/s.
        los_direction: int
            line of sight direction either 0(=x), 1(=y), 2(=z)
        cosmology: dict
            dictionary containing the simulatoin's cosmological parameters.
        boxsize:  float
            size of the simulation's box.
        num_threads: int 
            number of threads to use.

    Returns:

        tpcf_s_mu: np.ndarray
            2-D array with the redshift space tpcf.

    '''

    s_pos = pos.copy()

    # Move tracers to redshift space
    s_pos[:, los_direction] += vel[:, los_direction]/100. # to Mpc/h

    # Apply periodic boundary conditions
    s_pos[:, los_direction] = np.where(s_pos[:, los_direction] > boxsize, 
            s_pos[:, los_direction] - boxsize, s_pos[:, los_direction])
    s_pos[:, los_direction] = np.where(s_pos[:, los_direction] < 0., 
            s_pos[:, los_direction] + boxsize, s_pos[:, los_direction])

    assert np.prod( (s_pos > 0) & (s_pos < boxsize))

    # Halotools tpcf_s_mu assumes the line of sight is always the z direction

    if(los_direction != 2):
        s_pos_old = s_pos.copy()
        s_pos[:,2] = s_pos_old[:, los_direction]
        s_pos[:, los_direction] = s_pos_old[:,2]

    tpcf_s_mu = s_mu_tpcf(s_pos, s, mu, period = boxsize,
            estimator = u'Landy-Szalay', num_threads = num_threads)


    return tpcf_s_mu

def tpcf_wedges(s_mu_tcpf_result, mu_bins, n_wedges = 3): 
    '''

    Calculate the wedges of the two point correlation function
    after first computing `~halotools.mock_observables.s_mu_tpcf`.


    Args:
        s_mu_tcpf_result : np.ndarray
            2-D array with the two point correlation function calculated in bins
            of :math:`s` and :math:`\mu`.  See `~halotools.mock_observables.s_mu_tpcf`.
        n_wedges : int, optional 
            number of wedges to be returned

    Returns:
        xi_w : list 
             the indicated number of wedges of ``s_mu_tcpf_result``.
    '''

    # process inputs
    s_mu_tcpf_result = np.atleast_1d(s_mu_tcpf_result)

    mu_bins = np.atleast_1d(mu_bins)

    n_wedges = int(n_wedges)

    # calculate the center of each mu bin
    mu_bin_centers = (mu_bins[:-1]+mu_bins[1:])/(2.0)

    # split mu bins into wedges
    mu_wedges = np.linspace( 0., 1, n_wedges + 1)

    # Ensure mu binning has the appropiate edges (contains wedge edges)
    assert np.product( np.isin(mu_wedges, mu_bins)) == 1, 'Wedge boundaries not included in mu_bins values'

    wedge_sizes = mu_wedges[1:] - mu_wedges[:-1]

    wedge_index = np.digitize(mu_bin_centers, mu_wedges) - 1 


    # numerically integrate over mu
    result = [ 1 / wedge_sizes[wedge] * np.sum(np.diff(mu_bins)[wedge_index == wedge ] *\
            s_mu_tcpf_result[:, wedge_index == wedge], axis = 1) for wedge in range(n_wedges)]

    return np.asarray(result)


