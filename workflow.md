# Creation of CMB powerspectrum using CAMB
1. pars.set_cosmology(H0=67.74,ombh2=0.022,mnu=0.0,omk=0,tau=0.06,)
2. pars.InitPower.set_params(As=2e-9, ns=0.965, r=0)
3. pars.set_for_lmax(9000, lens_potential_accuracy=0)

# Creation of CMB flat sky map using Healpy
1. cart_proj = hp.projector.CartesianProj(lonra=0,latra=20.,xsize=8192,ysize=8192,)
   map_array = cart_proj.projmap(
       self.data[of],rot=(0., 0.),vec2pix_func=partial(hp.vec2pix, self._nside),
   )

# Creation of ISW-RS map
1. on-the-fly ray tracing simulations in range z=[0., 6.]

# Creation of final sky map
1. unlensed CMB map + ISW-RS map

# Transverse velocity measurement
1. apply gaussian smoothing filter of fwhm = 1 arcmin on full DeltaT map
2. apply gaussian high pass filter of fwhm = 5 arcmin on full DeltaT and alpha map
3. [apply](https://github.com/Christovis/wys-ars/blob/9b4d32bfc292cbf5623fc93e00147cddb3200c1f/src/wys_ars/rays/dipole_finder.py#L440) [DGD3 filter](https://github.com/Christovis/wys-ars/blob/787ccf1ce590f31582b962190dacc330f20a7370/src/wys_ars/rays/utils/filters.py#L299) on a cropped DeltaT and alpha map extending 30xR200 from halo centre
4. apodize using [Hann window](https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.signal.hann.html)
