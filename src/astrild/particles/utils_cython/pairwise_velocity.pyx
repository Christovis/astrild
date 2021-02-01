from __future__ import division
from libc.math cimport sqrt, sin, cos, acos, floor, fabs
from libc.stdlib cimport malloc, free

import numpy as np
cimport numpy as np

cimport cython
from cython.parallel import prange

DTYPEf = np.float32
ctypedef np.float32_t DTYPEf_t

DTYPEf = np.float64
ctypedef np.float64_t DTYPEff_t

DTYPEi = np.int64
ctypedef np.int64_t DTYPEi_t

DTYPEu = np.uint8
ctypedef np.uint8_t DTYPEu_t

#CFLAGS="$CFLAGS -D CYTHON_CLINE_IN_TRACEBACK=0"

# --------------------
# Defining functions
# --------------------

@cython.boundscheck(True)  # Deactivate bounds checking
@cython.wraparound(False)  # Deactivate negative indexing
@cython.cdivision(True)
def mean_pv_from_tv(
    tree,
    double[:, ::1] ppos, # declare as 2D C-contiguous array
    double[:, ::1] vvel, # declare as 2D C-contiguous array
    int ffirst,
    int ssecond,
    double rmax,
    double[:] dist_bins,
):
    """
    Function to compute the pairwise velocity PDF for 1D case along the z axis,
    but without the sign convention. Z-axis is taken as the line of sight axis.
    The pairwise velocity calculated in this function is as below:
    v_{12}(r) = 

    Args:
        tree:     The ball tree data structure which was trained on the position data set.
        ppos:     The array containing position of the particles used in the simulation.
        vvel:     The array containing velocities of the particles.
        ffirst:   Denotes the index from where the looping should start, developed in
            keeping mind of the parallelisation of the for-loop.
        ssecond:  Denotes the end index for the for-loop.
        rmax: Max. distance out to which pairwise velocity is calculated.
        dist_bins: No of bins for the distance. For now the bin size is fixed to 1 Mpc/h
    Returns:
        vij:  A flattened array containing the counts of the pairwise velocity which fall
            into respective bins of distance r.
    """
    cdef Py_ssize_t nr_dist_bins = len(dist_bins)
    cdef int i, j, leng
    cdef double dist, theta, phi
    cdef int[:] e_x = np.array([1, 0, 0])
    cdef int[:] e_y = np.array([0, 1, 0])
    cdef int[:] e_z = np.array([0, 0, 1])
    cdef double[:] pos_i = np.empty(3, dtype=DTYPEf)
    cdef double[:] tvel_i = np.empty(2, dtype=DTYPEf)
    cdef double[:] tv_theta_i = np.empty(3, dtype=DTYPEf)
    cdef double[:] tv_phi_i = np.empty(3, dtype=DTYPEf)
    cdef double[:] tv_i = np.empty(3, dtype=DTYPEf)
    cdef double[:] pos_j = np.empty(3, dtype=DTYPEf)
    cdef double[:] tvel_j = np.empty(2, dtype=DTYPEf)
    cdef double[:] tv_theta_j = np.empty(3, dtype=DTYPEf)
    cdef double[:] tv_phi_j = np.empty(3, dtype=DTYPEf)
    cdef double[:] tv_j = np.empty(3, dtype=DTYPEf)
    cdef double[:] tv_ij = np.empty(3, dtype=DTYPEf)
    cdef double[:] pos_ij= np.empty(3, dtype=DTYPEf)
    cdef double[:] q_ij= np.empty(3, dtype=DTYPEf)
    cdef double[:] nom = np.zeros(nr_dist_bins, dtype=DTYPEf)
    cdef double[:] denom = np.zeros(nr_dist_bins, dtype=DTYPEf)
    cdef double[:] vij = np.empty(nr_dist_bins, dtype=DTYPEf)
    #cdef np.ndarray[DTYPEff_t, ndim=1] e_x = np.zeros(3)

    # loop over i-th particle
    for i in range(ffirst, ssecond):
        pos_i = ppos[i, :]  # position in cart. coord., r_i
        tvel_i = vvel[i, :]  # transverse velocity in spher. coord.
        # find j-th particle within max. distance
        pairs = tree.query_radius(pos_i, rmax, return_distance=True)
        leng = len(pairs[0][0]) # nr of pairs
                
        # get angle between i-particle position vec and y,x-axis
        # (as light-cone is along z-axis)
        theta = angle(pos_i, e_y)
        phi = angle(plane_projection(pos_i, e_y), e_x)
        # translate transverse-to-los velocity vector from 
        # spherical coords. (e_theta, e_phi) to
        # cartesian coords. (e_x, e_y, e_z)
        tv_theta_i[0] = tvel_i[0] * cos(theta)*cos(phi)
        tv_theta_i[1] = tvel_i[0] * -sin(theta)
        tv_theta_i[2] = tvel_i[0] * cos(theta)*sin(phi)
        tv_phi_i[0] = tvel_i[1] * -sin(phi)
        tv_phi_i[1] = 0
        tv_phi_i[2] = tvel_i[1] * cos(phi)
        tv_i[0] = tv_theta_i[0] + tv_phi_i[0]
        tv_i[1] = tv_theta_i[1] + tv_phi_i[1]
        tv_i[2] = tv_theta_i[2] + tv_phi_i[2]

        # loop over j-th particle
        for jj in range(leng):
            j = pairs[0][0][jj] # index of j-particle
            if (j > i): # to avoid double counting
                dist = pairs[1][0][jj]  # distance between pair, |r_ij|
                pos_j = ppos[j, :]  # position in cart. coord., r_j
                tvel_j = vvel[j, :]  # transverse velocity in spher. coord.

                # get angle between i-particle position vec and y,x-axis
                # (as light-cone is along z-axis)
                theta = angle(pos_j, e_y)
                phi = angle(plane_projection(pos_j, e_y), e_x)
                # translate transverse-to-los velocity vector from 
                # spherical coords. (e_theta, e_phi) to
                # cartesian coords. (e_x, e_y, e_z)
                tv_theta_j[0] = tvel_j[0] * cos(theta)*cos(phi)
                tv_theta_j[1] = tvel_j[0] * -sin(theta)
                tv_theta_j[2] = tvel_j[0] * cos(theta)*sin(phi)
                tv_phi_j[0] = tvel_j[1] * -sin(phi)
                tv_phi_j[1] = 0
                tv_phi_j[2] = tvel_j[1] * cos(phi)
                tv_j[0] = tv_theta_j[0] + tv_phi_j[0]
                tv_j[1] = tv_theta_j[1] + tv_phi_j[1]
                tv_j[2] = tv_theta_j[2] + tv_phi_j[2]

                tv_ij[0] = tv_i[0] - tv_j[0]
                tv_ij[1] = tv_i[1] - tv_j[1]
                tv_ij[2] = tv_i[2] - tv_j[2]

                pos_ij[0] = pos_i[0] - pos_j[0]
                pos_ij[1] = pos_i[1] - pos_j[1]
                pos_ij[2] = pos_i[2] - pos_j[2]
                
                dotprod_i = pos_ij[0]*pos_i[0] + pos_ij[1]*pos_i[1] + pos_ij[2]*pos_i[2]
                dotprod_j = pos_ij[0]*pos_j[0] + pos_ij[1]*pos_j[1] + pos_ij[2]*pos_j[2]

                q_ij[0] = 0.5 * (2.*pos_ij[0] - pos_i[0]*dotprod_i - pos_j[0]*dotprod_j)
                q_ij[1] = 0.5 * (2.*pos_ij[1] - pos_i[1]*dotprod_i - pos_j[1]*dotprod_j)
                q_ij[2] = 0.5 * (2.*pos_ij[2] - pos_i[2]*dotprod_i - pos_j[2]*dotprod_j)

                for bin_idx in range(nr_dist_bins-1):
                    if (dist_bins[bin_idx] < dist) and (dist < dist_bins[bin_idx+1]):
                        nom[(<int>bin_idx)] += (
                            tv_ij[0]*q_ij[0] + tv_ij[1]*q_ij[1] + tv_ij[2]*q_ij[2]
                        )
                        denom[(<int>bin_idx)] += (
                            q_ij[0]*q_ij[0] + q_ij[1]*q_ij[1] + q_ij[2]*q_ij[2]
                        )

    for bin_idx in range(nr_dist_bins):
        vij[(<int>bin_idx)] = nom[(<int>bin_idx)] / denom[(<int>bin_idx)]

    return vij


@cython.cdivision(True)
cdef double angle(
    double[:] vec1,
    int[:] vec2,
):
    cdef double vec1_norm, vec2_norm, dotprod = 0
    vec1_norm = sqrt(vec1[0]*vec1[0] + vec1[1]*vec1[1] + vec1[2]*vec1[2])
    vec2_norm = sqrt(vec2[0]*vec2[0] + vec2[1]*vec2[1] + vec2[2]*vec2[2])
    for i in range(3):
        dotprod += vec1[i] / vec1_norm * vec2[i] / vec2_norm
    return acos(dotprod)


@cython.cdivision(True)
cdef double[:] plane_projection(
    double[:] vec1,
    int[:] vec2,
):
    cdef double vec2_norm_squared, dotprod
    cdef double[:] vec1proj = np.empty(3, dtype=DTYPEf)
    vec2_norm_squared = vec2[0]*vec2[0] + vec2[1]*vec2[1] + vec2[2]*vec2[2]
    dotprod = vec1[0] * vec2[0] + vec1[1] * vec2[1] + vec1[2] * vec2[2]
    vec1proj[0] = vec1[0] - vec2[0] * dotprod / vec2_norm_squared
    vec1proj[1] = vec1[1] - vec2[1] * dotprod / vec2_norm_squared
    vec1proj[2] = vec1[2] - vec2[2] * dotprod / vec2_norm_squared
    return vec1proj

@cython.boundscheck(True)  # Deactivate bounds checking
@cython.wraparound(False)  # Deactivate negative indexing
@cython.cdivision(True)
def mean_pv_z_sign(
    tree,
    np.ndarray[DTYPEff_t, ndim=2] ppos,
    np.ndarray[DTYPEff_t, ndim=2] vvel,
    int ffirst,
    int ssecond,
    float r,
    int dist_bin,
    int vel_bin,
):
    """
    Function to compute the pairwise velocity PDF for 1D case along the z axis.
    Z-axis is taken as the line of sight axis.
    The pairwise velocity calculated in this function is as below:
    v_{12} = (u_{2z}-u_{1z})*sign(r_{2z}-r_{1z})

    Args:
        tree:     The ball tree data structure which was trained on the position data set.
        ppos:     The array containing position of the particles used in the simulation.
        vvel:     The array containing velocities of the particles.
        ffirst:   Denotes the index from where the looping should start, developed in
            keeping mind of the parallelisation of the for-loop.
        ssecond:  Denotes the end index for the for-loop.
        dist_bin: No of bins for the distance. For now the bin size is fixed to 1 h^{-1} Mpc
        vel_bin:  No of bins for the velocity. Binning goes from -(vel_bin/2) to +(vel_bin/2).
            Currently the bin size is 1.
    Returns:
        counter:  A flattened array containing the counts of the pairwise velocity which fall
            into respective bins of distance r.
    """
    cdef int i, j, leng, jj
    cdef float diff, dist, buff_vel, rubbish_counter
    cdef int offset=(vel_bin/2)
    cdef np.ndarray[DTYPEff_t, ndim=2] counter = np.zeros((dist_bin, vel_bin))

    for i in range(ffirst, ssecond):
        pairs = tree.query_radius(ppos[i][np.newaxis], r, return_distance=True)
        leng = len(pairs[0][0])
        buff_vel = vvel[i,2]
        buff_pos = ppos[i,2]
        
        for j in range(leng):
            jj = pairs[0][0][j]
            if (jj > i):
                dist = pairs[1][0][j]
                diff = (
                    (vvel[jj,2] - buff_vel) * (
                        bool((ppos[jj,2] - buff_pos) > 0) - \
                        bool((ppos[jj,2] - buff_pos) < 0)
                    )
                ) + offset
                #offset is added to take care of the negative velocities, should
                #look into a better technique for binning negative numbers. Might
                #need to use copysign for the sign than using bool due to presence of 0.
                if (diff > vel_bin or diff < 0 or dist > r):
                    rubbish_counter += 1
                else:
                    print("dist diff", dist, diff)
                    counter[(<int>dist),(<int>diff)] += 1
    return counter.flatten()


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def mean_pv_radial(
    tree,
    np.ndarray[DTYPEf_t, ndim=2] ppos,
    np.ndarray[DTYPEf_t, ndim=2] vvel,
    int ffirst,
    int ssecond,
    float r,
    int dist_bin,
    int vel_bin
):
    """
    Function to compute the pairwise velocity PDF along the separation vector for 1D case.
    The pairwise velocity calculated in this function is as below:
    v_{12} = ((u_{2x}-u_{1x})\cdot(r_{2x}-r_{1x})) + \
             ((u_{2y}-u_{1y})\cdot(r_{2y}-r_{1y})) + \
             ((u_{2z}-u_{1z})\cdot(r_{2z}-r_{1z})) / |r_{12}|
    
    Args:
        tree:     The ball tree data structure which was trained on the position data set.
        ppos:     The array containing position of the particles used in the simulation.
        vvel:     The array containing velocities of the particles.
        ffirst:   Denotes the index from where the looping should start, developed in
            keeping mind of the parallelisation of the for-loop.
        ssecond:  Denotes the end index for the for-loop.
        dist_bin: No of bins for the distance. For now the bin size is fixed to 1 h^{-1} Mpc
        vel_bin:  No of bins for the velocity. Binning goes from -(vel_bin/2) to +(vel_bin/2).
            Currently the bin size is 1.
    Returns:
        counter:  A flattened array containing the counts of the pairwise velocity which fall
            into respective bins of distance r.
    """
    cdef int i, j, leng
    cdef float diff, dist, buff_vel, rubbish_counter
    cdef int offset=(vel_bin/2)
    cdef np.ndarray[DTYPEff_t, ndim=2] counter = np.zeros((dist_bin, vel_bin))

    for i in range(ffirst, ssecond):
        pairs = tree.query_radius(ppos[i], r, return_distance=True)
        leng = len(pairs[0][0])
        for j in range(leng):
            jj = pairs[0][0][j]
            if (jj > i):
                dist = pairs[1][0][j]
                diff = (
                    (
                        ((vvel[jj,0]-vvel[i,0]) * (ppos[jj,0]-ppos[i,0])) + \
                        ((vvel[jj,1]-vvel[i,1]) * (ppos[jj,1]-ppos[i,1])) + \
                        ((vvel[jj,2]-vvel[i,2]) * (ppos[jj,2]-ppos[i,2]))
                    ) / dist
                ) + offset
                if (diff > vel_bin or diff < 0 or dist > r):
                    rubbish_counter += 1
                else:
                    counter[(<int>dist),(<int>diff)] += 1
    return counter.flatten()
