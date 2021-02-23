# File Description:
# Contains cosmological distance calculations implemented with astropy
# and cython bridge from python to c data-types for strong lensing
# analysis

apr = 206269.43  #radians to arcsec

def Dc(z, unit, cosmo):
    """
    Input:
        z: redshift
        unit: distance unit in kpc, Mpc, ...
        cosmo: dicitonary of cosmology parameters
    Output:
        res: comoving distance in unit as defined by variable 'unit'
    """
    res = cosmo.comoving_distance(z).to_value(unit)  #*cosmo.h
    return res

def Da(z, unit, cosmo):
    res = cosmo.angular_diameter_distance(z).to_value(unit)  #*cosmo.h
    return res

def Dc2(z1, z2, unit, cosmo):
    Dcz1 = cosmo.comoving_distance(z1).to_value(unit)  #*cosmo.h
    Dcz2 = cosmo.comoving_distance(z2).to_value(unit)  #*cosmo.h
    res = (Dcz2-Dcz1+1e-8)
    return res

def Da2(z1, z2, unit, cosmo):
    Dcz1 = cosmo.angular_diameter_distance(z1).to_value(unit)  #*cosmo.h
    Dcz2 = cosmo.angular_diameter_distance(z2).to_value(unit)  #*cosmo.h
    res = (Dcz2-Dcz1+1e-8)
    return res

#---------------------------------------------------------------------------------
import numpy as np
import ctypes as ct
import os

lib_path = "/cosma5/data/dp004/dc-beck3/StrongLensing/LensingAnalysis/lib/"
#---------------------------------------------------------------------------------
sps = ct.CDLL(lib_path+"lib_so_sph_w_omp/libsphsdens.so")

sps.cal_sph_sdens_weight.argtypes =[np.ctypeslib.ndpointer(dtype = ct.c_float), \
                                    np.ctypeslib.ndpointer(dtype = ct.c_float), \
                                    np.ctypeslib.ndpointer(dtype = ct.c_float), \
                                    np.ctypeslib.ndpointer(dtype = ct.c_float), \
                                    ct.c_float,ct.c_long,ct.c_float,ct.c_long,ct.c_long, \
                                    ct.c_float,ct.c_float,ct.c_float, \
                                    np.ctypeslib.ndpointer(dtype = ct.c_float), \
                                    np.ctypeslib.ndpointer(dtype = ct.c_float), \
                                    np.ctypeslib.ndpointer(dtype = ct.c_float)]

sps.cal_sph_sdens_weight.restype  = ct.c_int

def call_sph_sdens_weight(x1,x2,x3,mpp,Bsz,Ncc):

    x1 = np.array(x1, dtype=ct.c_float)
    x2 = np.array(x2, dtype=ct.c_float)
    x3 = np.array(x3, dtype=ct.c_float)
    mpp = np.array(mpp, dtype=ct.c_float)
    dcl = ct.c_float(Bsz/Ncc)
    Ngb = ct.c_long(32)
    xc1 = ct.c_float(0.0)
    xc2 = ct.c_float(0.0)
    xc3 = ct.c_float(0.0)
    Np  = len(mpp)
    posx1 = np.zeros((Ncc,Ncc),dtype=ct.c_float)
    posx2 = np.zeros((Ncc,Ncc),dtype=ct.c_float)
    sdens = np.zeros((Ncc,Ncc),dtype=ct.c_float)

    sps.cal_sph_sdens_weight(x1,x2,x3,mpp,ct.c_float(Bsz),ct.c_long(Ncc),dcl,Ngb,ct.c_long(Np),xc1,xc2,xc3,posx1,posx2,sdens);
    return sdens
#---------------------------------------------------------------------------------
sps.cal_sph_sdens_weight_omp.argtypes =[np.ctypeslib.ndpointer(dtype = ct.c_float), \
                                        np.ctypeslib.ndpointer(dtype = ct.c_float), \
                                        np.ctypeslib.ndpointer(dtype = ct.c_float), \
                                        np.ctypeslib.ndpointer(dtype = ct.c_float), \
                                        ct.c_float,ct.c_long,ct.c_float,ct.c_long,ct.c_long, \
                                        ct.c_float,ct.c_float,ct.c_float, \
                                        np.ctypeslib.ndpointer(dtype = ct.c_float), \
                                        np.ctypeslib.ndpointer(dtype = ct.c_float), \
                                        np.ctypeslib.ndpointer(dtype = ct.c_float)]

sps.cal_sph_sdens_weight_omp.restype  = ct.c_int

def call_sph_sdens_weight_omp(x1,x2,x3,mpp,Bsz,Nc):
    x1 = np.array(x1,dtype=ct.c_float)
    x2 = np.array(x1,dtype=ct.c_float)
    x3 = np.array(x1,dtype=ct.c_float)
    mpp = np.array(mpp,dtype=ct.c_float)
    dsx = ct.c_float(Bsz/Nc)
    Ngb = ct.c_long(32)
    xc1 = ct.c_float(0.0)
    xc2 = ct.c_float(0.0)
    xc3 = ct.c_float(0.0)
    Np  = len(mpp)
    posx1 = np.zeros((Nc,Nc),dtype=ct.c_float)
    posx2 = np.zeros((Nc,Nc),dtype=ct.c_float)
    sdens = np.zeros((Nc,Nc),dtype=ct.c_float)

    sps.cal_sph_sdens_weight_omp(x1,x2,x3,mpp,ct.c_float(Bsz),ct.c_long(Nc),dsx,Ngb,ct.c_long(Np),xc1,xc2,xc3,posx1,posx2,sdens);
    return sdens

#---------------------------------------------------------------------------------
gls = ct.CDLL(lib_path+"lib_so_cgls/libglsg.so")
gls.kappa0_to_alphas.argtypes = [np.ctypeslib.ndpointer(dtype = ct.c_double), \
                                 ct.c_int,ct.c_double,\
                                 np.ctypeslib.ndpointer(dtype = ct.c_double), \
                                 np.ctypeslib.ndpointer(dtype = ct.c_double)]
gls.kappa0_to_alphas.restype  = ct.c_void_p

def call_cal_alphas(Kappa, Bsz, Ncc):
    kappa0 = np.array(Kappa, dtype=ct.c_double)
    alpha1 = np.array(np.zeros((Ncc,Ncc)), dtype=ct.c_double)
    alpha2 = np.array(np.zeros((Ncc,Ncc)), dtype=ct.c_double)
    gls.kappa0_to_alphas(kappa0, Ncc, Bsz, alpha1, alpha2)
    return alpha1,alpha2

gls.kappa0_to_phi.argtypes = [np.ctypeslib.ndpointer(dtype = ct.c_double), \
                              ct.c_int,ct.c_double,\
                              np.ctypeslib.ndpointer(dtype = ct.c_double)]
gls.kappa0_to_phi.restype  = ct.c_void_p

def call_cal_phi(Kappa, Bsz, Ncc):
	kappa0 = np.array(Kappa, dtype=ct.c_double)
	phi = np.array(np.zeros((Ncc, Ncc)), dtype=ct.c_double)
	gls.kappa0_to_phi(kappa0, Ncc, Bsz, phi)

	return phi

#--------------------------------------------------------------------
lzos = ct.CDLL(lib_path+"lib_so_lzos/liblzos.so")
lzos.lanczos_diff_2_tag.argtypes = [np.ctypeslib.ndpointer(dtype = ct.c_double), \
                                    np.ctypeslib.ndpointer(dtype = ct.c_double), \
                                    np.ctypeslib.ndpointer(dtype = ct.c_double), \
                                    np.ctypeslib.ndpointer(dtype = ct.c_double), \
                                    np.ctypeslib.ndpointer(dtype = ct.c_double), \
                                    np.ctypeslib.ndpointer(dtype = ct.c_double), \
                                    ct.c_double,ct.c_int,ct.c_int]
lzos.lanczos_diff_2_tag.restype  = ct.c_void_p

def call_lanczos_derivative(alpha1,alpha2,Bsz,Ncc):
    dif_tag = ct.c_int(2)
    dcl = ct.c_double(Bsz/Ncc)
    m1 = np.array(alpha1,dtype=ct.c_double)
    m2 = np.array(alpha2,dtype=ct.c_double)
    m11 = np.zeros((Ncc, Ncc))
    m12 = np.zeros((Ncc, Ncc))
    m21 = np.zeros((Ncc, Ncc))
    m22 = np.zeros((Ncc, Ncc))

    lzos.lanczos_diff_2_tag(m1,m2,m11,m12,m21,m22,dcl,ct.c_int(Ncc),dif_tag)

    return m11,m12,m21,m22

#---------------------------------------------------------------------------------
# Cloud-in-Cell scheme
rtf = ct.CDLL(lib_path + "lib_so_icic/librtf.so")

rtf.inverse_cic.argtypes = [np.ctypeslib.ndpointer(dtype =  ct.c_double),\
                            np.ctypeslib.ndpointer(dtype =  ct.c_double), \
                            np.ctypeslib.ndpointer(dtype =  ct.c_double), \
                            ct.c_double,ct.c_double,ct.c_double, \
                            ct.c_int,ct.c_int,ct.c_int,ct.c_int,\
                            np.ctypeslib.ndpointer(dtype = ct.c_double)]
rtf.inverse_cic.restype  = ct.c_void_p

def call_inverse_cic(img_in, yc1, yc2, yi1, yi2, dsi):
    ny1,ny2 = np.shape(img_in)
    nx1,nx2 = np.shape(yi1)

    img_in = np.array(img_in,dtype=ct.c_double)

    yi1 = np.array(yi1,dtype=ct.c_double)
    yi2 = np.array(yi2,dtype=ct.c_double)

    img_out = np.zeros((nx1,nx2))

    rtf.inverse_cic(img_in,yi1,yi2,ct.c_double(yc1),ct.c_double(yc2),ct.c_double(dsi),ct.c_int(ny1),ct.c_int(ny2),ct.c_int(nx1),ct.c_int(nx2),img_out)
    return img_out.reshape((nx1,nx2))

#--------------------------------------------------------------------
rtf.inverse_cic_omp.argtypes = [np.ctypeslib.ndpointer(dtype =  ct.c_double),\
                                np.ctypeslib.ndpointer(dtype =  ct.c_double), \
                                np.ctypeslib.ndpointer(dtype =  ct.c_double), \
                                ct.c_double,ct.c_double,ct.c_double, \
                                ct.c_int,ct.c_int,ct.c_int,ct.c_int,\
                                np.ctypeslib.ndpointer(dtype = ct.c_double)]
rtf.inverse_cic_omp.restype  = ct.c_void_p

def call_inverse_cic_omp(img_in,yc1,yc2,yi1,yi2,dsi):
    ny1,ny2 = np.shape(img_in)
    nx1,nx2 = np.shape(yi1)
    img_in = np.array(img_in,dtype=ct.c_double)
    yi1 = np.array(yi1,dtype=ct.c_double)
    yi2 = np.array(yi2,dtype=ct.c_double)
    img_out = np.zeros((nx1,nx2))

    rtf.inverse_cic_omp(img_in,yi1,yi2,ct.c_double(yc1),ct.c_double(yc2),ct.c_double(dsi),ct.c_int(ny1),ct.c_int(ny2),ct.c_int(nx1),ct.c_int(nx2),img_out)
    return img_out.reshape((nx1,nx2))

#--------------------------------------------------------------------
rtf.inverse_cic_single.argtypes = [np.ctypeslib.ndpointer(dtype = ct.c_float), \
                                   np.ctypeslib.ndpointer(dtype = ct.c_float), \
                                   np.ctypeslib.ndpointer(dtype = ct.c_float), \
                                   ct.c_float,ct.c_float,ct.c_float,ct.c_int,ct.c_int,ct.c_int, \
                                   np.ctypeslib.ndpointer(dtype = ct.c_float)]
rtf.inverse_cic_single.restype  = ct.c_void_p

def call_inverse_cic_single(img_in,yc1,yc2,yi1,yi2,dsi):
    ny1,ny2 = np.shape(img_in)
    img_in = np.array(img_in,dtype=ct.c_float)
    yi1 = np.array(yi1,dtype=ct.c_float)
    yi2 = np.array(yi2,dtype=ct.c_float)
    nlimgs = len(yi1)
    img_out = np.zeros((nlimgs),dtype=ct.c_float)

    rtf.inverse_cic_single(img_in,yi1,yi2,ct.c_float(yc1),ct.c_float(yc2),ct.c_float(dsi),ct.c_int(ny1),ct.c_int(ny2),ct.c_int(nlimgs),img_out)
    return img_out

#--------------------------------------------------------------------
rtf.inverse_cic_omp_single.argtypes = [np.ctypeslib.ndpointer(dtype = ct.c_float), \
                                       np.ctypeslib.ndpointer(dtype = ct.c_float), \
                                       np.ctypeslib.ndpointer(dtype = ct.c_float), \
                                       ct.c_float,ct.c_float,ct.c_float,ct.c_int,ct.c_int,ct.c_int, \
                                       np.ctypeslib.ndpointer(dtype = ct.c_float)]
rtf.inverse_cic_omp_single.restype  = ct.c_void_p

def call_inverse_cic_single_omp(img_in,yc1,yc2,yi1,yi2,dsi):
    """
    Input:
        img_in: Magnification Map
        yc1, yc2: Lens position
        yi1, yi2: Source position
        dsi: pixel size on grid
    """
    ny1,ny2 = np.shape(img_in)
    img_in = np.array(img_in,dtype=ct.c_float)
    yi1 = np.array(yi1,dtype=ct.c_float)
    yi2 = np.array(yi2,dtype=ct.c_float)
    nlimgs = len(yi1)
    img_out = np.zeros((nlimgs),dtype=ct.c_float)

    rtf.inverse_cic_omp_single(img_in,yi1,yi2,ct.c_float(yc1),ct.c_float(yc2),ct.c_float(dsi),ct.c_int(ny1),ct.c_int(ny2),ct.c_int(nlimgs),img_out)
    return img_out

#--------------------------------------------------------------------
tri = ct.CDLL(lib_path+"lib_so_tri_roots/libtri.so")
tri.mapping_triangles.argtypes = [np.ctypeslib.ndpointer(dtype = ct.c_double), \
                                  np.ctypeslib.ndpointer(dtype = ct.c_double), \
                                  np.ctypeslib.ndpointer(dtype = ct.c_double), \
                                  np.ctypeslib.ndpointer(dtype = ct.c_double), \
                                  np.ctypeslib.ndpointer(dtype = ct.c_double), \
                                  ct.c_int, \
                                  np.ctypeslib.ndpointer(dtype = ct.c_double)]
tri.mapping_triangles.restype  = ct.c_void_p

def call_mapping_triangles(pys, xi1, xi2, yi1, yi2):
    pys_in = np.array(pys, dtype=ct.c_double)
    xi1_in = np.array(xi1, dtype=ct.c_double)
    xi2_in = np.array(xi2, dtype=ct.c_double)
    yi1_in = np.array(yi1, dtype=ct.c_double)
    yi2_in = np.array(yi2, dtype=ct.c_double)
    nc_in = ct.c_int(np.shape(xi1)[0])

    # assuming there will not be more than 40 lensed images
    xroots_out = np.array(np.ones(1000)*(-99999), dtype=ct.c_double)

    tri.mapping_triangles(pys_in,xi1_in,xi2_in,yi1_in,yi2_in,nc_in,xroots_out)

    aroots = len(xroots_out[xroots_out!=(-99999)])

    xroot1 = xroots_out[:aroots:2]
    xroot2 = xroots_out[1:aroots:2]
    return xroot1, xroot2
#--------------------------------------------------------------------

def make_r_coor(bs, nc):
    ds = bs/nc
    x1edge = np.linspace(0,bs-ds,nc)-bs/2.0+ds/2.0
    x2,x1 = np.meshgrid(x1edge,x1edge)
    return x1, x2, x1edge

def make_c_coor(bs, nc):
    ds = bs/nc
    x1 = np.linspace(0,bs-ds,nc)-bs/2.0+ds/2.0
    x2 = np.linspace(0,bs-ds,nc)-bs/2.0+ds/2.0
    x1,x2 = np.meshgrid(x1,x2)
    return x1,x2
