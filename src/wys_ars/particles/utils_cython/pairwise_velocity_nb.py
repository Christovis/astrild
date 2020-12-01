import copy
import numpy as np
import numba as nb


def mean_pv_from_tv(
    tree,
    ppos: np.ndarray, # declare as 2D C-contiguous array
    vvel: np.ndarray, # declare as 2D C-contiguous array
    ffirst: int,
    ssecond: int,
    dist_bins: int,
):
    """
    Function to compute the pairwise velocity PDF for 1D case along the z axis,
    but without the sign convention. Z-axis is taken as the line of sight axis.
    The pairwise velocity calculated in this function is as below:
    v_{12}(r) = 

    Args:
        tree: The ball tree data structure which was trained on the position data set.
        ppos: The array containing position of the particles used in the simulation.
        vvel: The array containing velocities of the particles.
        ffirst: Denotes the index from where the looping should start, developed in
        keeping mind of the parallelisation of the for-loop.
        ssecond: Denotes the end index for the for-loop.
        dist_bins: No of bins for the distance. For now the bin size is fixed to 1 Mpc/h
    Returns:
    """
    e_x = np.array([1., 0., 0.])
    e_y = np.array([0., 1., 0.])
    nom = np.zeros(len(dist_bins))
    denom = np.zeros(len(dist_bins))

    theta_max = 0
    theta_min = 999
    phi_max = 0
    phi_min = 999
    # loop over i-th particle
    for i in range(ffirst, ssecond):
        pos_i = ppos[i, :]  # position in cart. coord., r_i
        tvel_i = vvel[i, :]  # transverse velocity in spher. coord.
        # find j-th particle within max. distance
        pairs = tree.query_radius(pos_i[np.newaxis, :], np.max(dist_bins), return_distance=True)
        if len(pairs[1][0]) == 0:
            continue
        
        j_distances = pairs[1][0][pairs[0][0] > i]
        j_indices = pairs[0][0][pairs[0][0] > i]
        j_dist_bins = np.digitize(j_distances, dist_bins)
        j_indices = j_indices[j_dist_bins < len(dist_bins)]
        j_dist_bins = j_dist_bins[j_dist_bins < len(dist_bins)]
        
        # get angle between i-particle position vec and y,x-axis
        # (as light-cone is along z-axis)
        # get angle with y-axis
        pos_i /= np.linalg.norm(pos_i)
        theta = np.arccos(pos_i[1])
        # project pos_i on x,z-plane and get angle with x-axis
        pos_i_proj = pos_i - pos_i[1] * e_y
        pos_i_proj /= np.linalg.norm(pos_i_proj)
        phi = np.arccos(pos_i_proj[0])
        # translate transverse-to-los velocity vector from 
        # spherical coords. (e_theta, e_phi) to
        # cartesian coords. (e_x, e_y, e_z)
        tv_theta_i = tvel_i[1] * np.array([
            np.cos(theta)*np.cos(phi), -np.sin(theta), np.cos(theta)*np.sin(phi),
        ])
        tv_phi_i = tvel_i[0] * np.array([-np.sin(phi), 0, np.cos(phi)])
        tv_i = tv_theta_i + tv_phi_i

        #print("distances  ", np.min(j_distances), np.max(j_distances))
        nom_i, denom_i = loop_over_j(
            ppos, vvel, j_distances, j_indices, j_dist_bins, pos_i, tv_i,
        )
        nom += nom_i
        denom += denom_i

        if theta > theta_max:
            theta_max = copy.deepcopy(theta)
        if theta < theta_min:
            theta_min = copy.deepcopy(theta)
        if phi > phi_max:
            phi_max = copy.deepcopy(phi)
        if phi < phi_min:
            phi_min = copy.deepcopy(phi)

    vij = nom / denom
    print("theta", theta_min*180/np.pi, theta_max*180/np.pi)
    print("phi", phi_min*180/np.pi, phi_max*180/np.pi)
    return vij
        
@nb.jit(nopython=True)
def loop_over_j(
    ppos, vvel, j_distances, j_indices, j_dist_bins, pos_i, tv_i,
):
    e_x = np.array([1., 0., 0.])
    e_y = np.array([0., 1., 0.])
    nom = np.zeros(40)
    denom = np.zeros(40)
    # loop over j-th particle
    for idx, j in enumerate(j_indices):
        pos_j = ppos[j, :]  # position in cart. coord., r_j
        tvel_j = vvel[j, :]  # transverse velocity in spher. coord.

        # get angle between j-particle position vec and y-axis
        pos_j /= np.linalg.norm(pos_j)
        theta = np.arccos(pos_j[1])
        # project pos_j on x,z-plane and get angle with x-axis
        pos_j_proj = pos_j - pos_j[1] * e_y
        pos_j_proj /= np.linalg.norm(pos_j_proj)
        phi = np.arccos(pos_j_proj[0])
        # translate transverse-to-los velocity vector from 
        # spherical coords. (e_theta, e_phi) to
        # cartesian coords. (e_x, e_y, e_z)
        tv_theta_j = tvel_j[1] * np.array([
            np.cos(theta)*np.cos(phi), -np.sin(theta), np.cos(theta)*np.sin(phi),
        ])
        tv_phi_j = tvel_j[0] * np.array([-np.sin(phi), 0, np.cos(phi)])
        tv_j = tv_theta_j + tv_phi_j

        tv_ij = tv_i - tv_j
        pos_ij = pos_i - pos_j
        pos_ij /= np.linalg.norm(pos_ij)
        dotprod_i = np.dot(pos_ij, pos_i)
        dotprod_j = np.dot(pos_ij, pos_j)
        q_ij = 0.5 * (2.*pos_ij - pos_i*dotprod_i - pos_j*dotprod_j)
        nom[j_dist_bins[idx]] += np.dot(tv_ij, q_ij)
        denom[j_dist_bins[idx]] += q_ij[0]**2 + q_ij[1]**2 + q_ij[2]**2

    return nom, denom
