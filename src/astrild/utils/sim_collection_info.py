import numpy as np
import pandas as pd

import astropy
import astropy.units as u
from astropy.cosmology import cvG

# from astropy.cosmology import LambdaCDM

from nbodykit.lab import cosmology

theta = 10  # [deg] light-cone opening angle
# astropy background cosmology class
cosmo_params = {
    "H0": 67.74,
    "Om0": 0.3089,
    "Ob0": 0.0,
    "Ode0": 0.6911,
}
astropy_cosmo = cvG(**cosmo_params)

# nbodykit background cosmology class
# cosmo_params = cosmology.cosmology.astropy_to_dict(astropy_cosmo)
# cosmo_params["T0_cmb"] = 2.7255
# cosmo_params["Omega0_b"] = 0.0482754208891869
# nbodykit_cosmo = cosmology.Cosmology(**cosmo_params)

# dataframe index
indx = [
    np.array([1] * 2 + [2] * 2 + [3] * 2 + [4] * 2 + [5] * 2),
    np.array(
        list(np.arange(2) + 1)[::-1]
        + list(np.arange(2) + 1)[::-1]
        + list(np.arange(2) + 1)[::-1]
        + list(np.arange(2) + 1)[::-1]
        + list(np.arange(2) + 1)[::-1]
    ),
]

# redshifts of ray-snapshots
z_box1 = [0.08441153419063716, 0.0][::-1]
z_box2 = [0.26076155161814074, 0.1711614118069671][::-1]
z_box3 = [0.4510919664608182, 0.3538262356242269][::-1]
z_box4 = [0.6619103668943251, 0.5534385981919647][::-1]
z_box5 = [0.902348846364959, 0.77773617315665][::-1]

# hubble-parameter H(z)
Hz_box1 = [
    astropy_cosmo.efunc(z) * astropy_cosmo.H0.value for z in z_box1
]  # [km/sec/Mpc]
Hz_box2 = [
    astropy_cosmo.efunc(z) * astropy_cosmo.H0.value for z in z_box2
]  # [km/sec/Mpc]
Hz_box3 = [
    astropy_cosmo.efunc(z) * astropy_cosmo.H0.value for z in z_box3
]  # [km/sec/Mpc]
Hz_box4 = [
    astropy_cosmo.efunc(z) * astropy_cosmo.H0.value for z in z_box4
]  # [km/sec/Mpc]
Hz_box5 = [
    astropy_cosmo.efunc(z) * astropy_cosmo.H0.value for z in z_box5
]  # [km/sec/Mpc]

# lookback-times
tlb_box1 = [astropy_cosmo.lookback_time(z).value for z in z_box1]  # [Gyr]
tlb_box2 = [astropy_cosmo.lookback_time(z).value for z in z_box2]  # [Gyr]
tlb_box3 = [astropy_cosmo.lookback_time(z).value for z in z_box3]  # [Gyr]
tlb_box4 = [astropy_cosmo.lookback_time(z).value for z in z_box4]  # [Gyr]
tlb_box5 = [astropy_cosmo.lookback_time(z).value for z in z_box5]  # [Gyr]

# scale-factor
a_box1 = [1 / (z + 1) for z in z_box1]
a_box2 = [1 / (z + 1) for z in z_box2]
a_box3 = [1 / (z + 1) for z in z_box3]
a_box4 = [1 / (z + 1) for z in z_box4]
a_box5 = [1 / (z + 1) for z in z_box5]

# comoving-distance
chi_box1 = [astropy_cosmo.comoving_distance(z).value for z in z_box1]  # [Mpc]
chi_box2 = [astropy_cosmo.comoving_distance(z).value for z in z_box2]  # [Mpc]
chi_box3 = [astropy_cosmo.comoving_distance(z).value for z in z_box3]  # [Mpc]
chi_box4 = [astropy_cosmo.comoving_distance(z).value for z in z_box4]  # [Mpc]
chi_box5 = [astropy_cosmo.comoving_distance(z).value for z in z_box5]  # [Mpc]

# growth-factor (D1 = Linear order growth function)
# bg = cosmology.background.PerturbationGrowth(nbodykit_cosmo)
# D_box1 = [bg.D1(a) for a in a_box1]
# D_box2 = [bg.D1(a) for a in a_box2]
# D_box3 = [bg.D1(a) for a in a_box3]
# D_box4 = [bg.D1(a) for a in a_box4]
# D_box5 = [bg.D1(a) for a in a_box5]

# extrema of ISW-RS maps
dic = {}
dic["box"] = indx[0]
dic["ray"] = indx[1]
dic["lookback_time"] = np.array(tlb_box1 + tlb_box2 + tlb_box3 + tlb_box4 + tlb_box5)
dic["redshift"] = np.array(z_box1 + z_box2 + z_box3 + z_box4 + z_box5)
dic["scale_factor"] = np.array(a_box1 + a_box2 + a_box3 + a_box4 + a_box5)
dic["comoving_distance"] = np.array(
    chi_box1 + chi_box2 + chi_box3 + chi_box4 + chi_box5
)
# dic["growth_factor"] = np.array(D_box1 + D_box2 + D_box3 + D_box4 + D_box5)

df = pd.DataFrame(dic)
df.set_index(["box", "ray"], inplace=True)

df.to_hdf(
    "/cosma/home/dp004/dc-beck3/3_Proca/src/configs/ray_snapshot_info.h5", key="df",
)

# dataframe index
indx = [
    np.array([1] * 12 + [2] * 12 + [3] * 12 + [4] * 12 + [5] * 12),
    np.array(
        list(np.arange(12)[::-1] + 1)
        + list(np.arange(12)[::-1] + 1)
        + list(np.arange(12)[::-1] + 1)
        + list(np.arange(12)[::-1] + 1)
        + list(np.arange(12)[::-1] + 1)
    ),
]

# scale-factor of particle-snapshots
a_box1 = [
    0.02,
    0.3,
    0.4,
    0.5,
    0.57,
    0.6,
    0.66,
    0.7,
    0.8,
    0.8869459927696931,
    0.9597463497474074,
    1.0,
][::-1]
a_box2 = [
    0.02,
    0.3,
    0.4,
    0.5,
    0.57,
    0.6,
    0.66,
    0.7,
    0.7652175560504898,
    0.8,
    0.8226597157260597,
    1.0,
][::-1]
a_box3 = [
    0.02,
    0.3,
    0.4,
    0.5,
    0.57,
    0.6,
    0.66,
    0.6659700624272444,
    0.7,
    0.713326496838544,
    0.8,
    1.0,
][::-1]
a_box4 = [
    0.02,
    0.3,
    0.4,
    0.5,
    0.57,
    0.5817946665994816,
    0.6,
    0.6223408211627445,
    0.66,
    0.7,
    0.8,
    1.0,
][::-1]
a_box5 = [
    0.02,
    0.3,
    0.4,
    0.5,
    0.5126961892210254,
    0.5438193022917206,
    0.57,
    0.6,
    0.66,
    0.7,
    0.8,
    1.0,
][::-1]

# scale-factor
z_box1 = [1 / a - 1 for a in a_box1]
z_box2 = [1 / a - 1 for a in a_box2]
z_box3 = [1 / a - 1 for a in a_box3]
z_box4 = [1 / a - 1 for a in a_box4]
z_box5 = [1 / a - 1 for a in a_box5]

# hubble-parameter H(z)
Hz_box1 = [
    astropy_cosmo.efunc(z) * astropy_cosmo.H0.value for z in z_box1
]  # [km/sec/Mpc]
Hz_box2 = [
    astropy_cosmo.efunc(z) * astropy_cosmo.H0.value for z in z_box2
]  # [km/sec/Mpc]
Hz_box3 = [
    astropy_cosmo.efunc(z) * astropy_cosmo.H0.value for z in z_box3
]  # [km/sec/Mpc]
Hz_box4 = [
    astropy_cosmo.efunc(z) * astropy_cosmo.H0.value for z in z_box4
]  # [km/sec/Mpc]
Hz_box5 = [
    astropy_cosmo.efunc(z) * astropy_cosmo.H0.value for z in z_box5
]  # [km/sec/Mpc]

# lookback-times
tlb_box1 = [astropy_cosmo.lookback_time(z).value for z in z_box1]  # [Gyr]
tlb_box2 = [astropy_cosmo.lookback_time(z).value for z in z_box2]  # [Gyr]
tlb_box3 = [astropy_cosmo.lookback_time(z).value for z in z_box3]  # [Gyr]
tlb_box4 = [astropy_cosmo.lookback_time(z).value for z in z_box4]  # [Gyr]
tlb_box5 = [astropy_cosmo.lookback_time(z).value for z in z_box5]  # [Gyr]

# comoving-distance
chi_box1 = [astropy_cosmo.comoving_distance(z).value for z in z_box1]  # [Mpc]
chi_box2 = [astropy_cosmo.comoving_distance(z).value for z in z_box2]  # [Mpc]
chi_box3 = [astropy_cosmo.comoving_distance(z).value for z in z_box3]  # [Mpc]
chi_box4 = [astropy_cosmo.comoving_distance(z).value for z in z_box4]  # [Mpc]
chi_box5 = [astropy_cosmo.comoving_distance(z).value for z in z_box5]  # [Mpc]

# growth-factor (D1 = Linear order growth function)
# bg = cosmology.background.PerturbationGrowth(nbodykit_cosmo)
# D_box1 = [bg.D1(a) for a in a_box1]
# D_box2 = [bg.D1(a) for a in a_box2]
# D_box3 = [bg.D1(a) for a in a_box3]
# D_box4 = [bg.D1(a) for a in a_box4]
# D_box5 = [bg.D1(a) for a in a_box5]

dic = {}
dic["box"] = indx[0]
dic["snap"] = indx[1]
dic["lookback_time"] = np.array(tlb_box1 + tlb_box2 + tlb_box3 + tlb_box4 + tlb_box5)
dic["redshift"] = np.array(z_box1 + z_box2 + z_box3 + z_box4 + z_box5)
dic["scale_factor"] = np.array(a_box1 + a_box2 + a_box3 + a_box4 + a_box5)
dic["comoving_distance"] = np.array(
    chi_box1 + chi_box2 + chi_box3 + chi_box4 + chi_box5
)
# dic["growth_factor"] = np.array(D_box1 + D_box2 + D_box3 + D_box4 + D_box5)
df = pd.DataFrame(dic)
df.set_index(["box", "snap"], inplace=True)
df.to_hdf(
    "/cosma/home/dp004/dc-beck3/3_Proca/src/configs/particle_snapshot_info.h5",
    key="df",
)
