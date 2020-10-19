import numpy as np
import h5py
import pickle
import tpcf_tools
from halotools.mock_observables import tpcf_multipole

def read_simulation(hdf5_filename, pos_field, vel_field, mass_field,
		mass_range = [None, None]):
	'''

	Reads positions and velocities from a given hdf5 file applying a mass threshold

	Args:
		hdf5_filename: string
			path to the hdf5 file.
		pos_field: string
			name of the hdf5 file field containing the tracer's positions
		vel_field: string
			name of the hdf5 file field containing the tracer's velocities 
		mass_field: string
			name of the hdf5 file field containing the tracer's masses 
		mass_range: list
			list containing (Minimum mass, Maximum mass)
	
	Returns:
		pos: np.ndarray
			 3-D array with the position of the tracers.
		vel: np.ndarray
			 3-D array with the velocity of the tracers.

	'''



	with h5py.File( hdf5_filename, 'r') as f:

		pos = f[pos_field][:]
		vel = f[vel_field][:]
		m200c = f[mass_field][:]

	if mass_range[0] is None:
		mass_range[0] = np.min(m200c)

	if mass_range[1] is None:
		mass_range[1] = np.max(m200c)

	mass_mask = (m200c > mass_range[0]) & (m200c < mass_range[1])

	return pos[mass_mask, :], vel[mass_mask, :]

def save_tpcfs(r, s, mu, pos, vel, los_direction, cosmology, boxsize, saveto, n_wedges = 3):


	print('Computing real space correlation function...')
	tpcf_real = tpcf_tools.compute_real_tpcf(r, pos, vel, boxsize)
	print('Done!')

	print('Computing redshift space correlation function...')
	tpcf_s_mu = tpcf_tools.compute_tpcf_s_mu(s, mu, pos, vel, los_direction,
			cosmology, boxsize)
	print('Done!')

	print('Computing multipoles...')
	monopole = tpcf_multipole(tpcf_s_mu, mu, order = 0)

	quadrupole = tpcf_multipole(tpcf_s_mu, mu, order = 2)

	hexadecapole= tpcf_multipole(tpcf_s_mu, mu, order = 4)
	print('Done!')

	# wedges for mu

	n_mu_bins_per_wedge = 50

	mu_wedges = np.concatenate(
			[np.linspace(i * (1./n_wedges), (i+1) * 1./n_wedges - 0.001, n_mu_bins_per_wedge) for i in range(n_wedges)],
			)


	mu_wedges[-1] = 1.

	print('Computing wedges...')
	tpcf_s_mu_wedges = tpcf_tools.compute_tpcf_s_mu(s, mu_wedges, pos, vel, los_direction,
			cosmology, boxsize)

	wedges = tpcf_tools.tpcf_wedges(tpcf_s_mu_wedges, mu_wedges, n_wedges = n_wedges)
	print('Done!')


	tpcf_dict = {
				'real':{
						'tpcf': tpcf_real,
						'r': r,
						},
				'redsfhit': {
						's': s,
						'mu': mu,
						'tpcf_s_mu': tpcf_s_mu,
						'monopole': monopole,
						'quadrupole': quadrupole,
						'hexadecapole': hexadecapole,
						'wedges': wedges
						}
				}


	with open(saveto + '.pickle', 'wb') as fp:
		pickle.dump(tpcf_dict, fp, protocol = pickle.HIGHEST_PROTOCOL)



if __name__=='__main__':


	data_dir =  '/cosma6/data/dp004/dc-cues1/simulations/RSD/'
	tracer = 'halos'
	boxsize = 1024.
	box = 1
	hdf5_filename = data_dir + f'{tracer}_{int(boxsize)}_b{box}.h5'

	n_r_bins = 300
	r = np.logspace(-0.4, np.log10(150.), n_r_bins)

	n_mu_bins = 60
	mu = np.sort( 1. - np.geomspace(0.0001, 1., n_mu_bins))

	s = np.arange(0., 50., 1.)
	s[0] = 0.0001


	pos, vel = read_simulation(hdf5_filename, 'GroupPos', 'GroupVel', 'GroupMass',
			mass_range = [5.e14, None])

	print(f'Found {pos.shape[0]} halos withing the given mass range')	
	cosmology = {'Omega_lambda':0.2}

	save_tpcfs(r, s, mu, pos, vel, 2, cosmology, boxsize, 'tpcf_dict')
		
