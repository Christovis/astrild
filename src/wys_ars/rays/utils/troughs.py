######################################
######################################

# Typical parameters I use for the below functions

mr = 4096  # number of pixels on side
mw = 20.0  # map width in degrees
nTroughs = 5000  # number of troughs to calculate
frac = 0.2
nbins = 50
rads = [
    10.0 / 60.0,
    20.0 / 60.0,
    30.0 / 60.0,
]  # size of trough radii in degrees

# The functions I use


def radMask(index, radius, res):
    """return a circular mask.
    index: the center coordinates of the mask
    radius: radius of the mask
    res: the resolution of the grid the mask is applied to
    """
    a, b = index
    y, x = np.ogrid[a - res : a + res, b - res : b + res]
    mask = x * x + y * y <= radius * radius
    return mask


def find_troughs(data, nTroughs, lowestFraction, rad, mw, res, conv):
    """Find troughs as per Gruen16 (I think) and return the fraction with the lowest mean convergence
    data: the square convergence map
    nTroughs: number of random troughs to place
    lowestFraction: fraction of nTroughs to return with the lowest mean enclosed convergence
    rad: trough radius in degrees
    mw: map width in degrees
    res: map resolution in pixel number
    conv: whether or not the input map is a convergence map. This decides if the mean or sum of values enclosed by the trough is calculated
    if conv is True, the mean is calculated. Otherwise the sum is calculated
    """

    trough_radius = rad * res / mw  # convert from degrees into number of pixels
    tr = int(trough_radius) + 1  # round up since pixels are discrete

    troughMean = np.zeros(
        (nTroughs)
    )  # create an empty array to store mean enclosed trough values

    lower = 1024  # upper and lower limits of the square map to place troughs in
    upper = 3072  # these should be moved to inputs for the function
    rand_points = np.random.randint(
        low=lower, high=upper + 1, size=[nTroughs, 2]
    )  # create a list of nTroughs trough center positions

    mask = radMask(
        [0, 0], tr, tr
    )  # create a circular mask that blocks out pixels outside the circle

    for i in range(nTroughs):  # for every randomly placed trough
        b, a = rand_points[i]  # get the centers
        if conv == True:
            troughMean[i] = data[a - tr : a + tr, b - tr : b + tr][mask].mean()
        if conv == False:
            troughMean[i] = data[a - tr : a + tr, b - tr : b + tr][mask].sum()

    N = np.rint(lowestFraction * nTroughs).astype(
        "int"
    )  # Take the lowest lowestFraction of troughs and discard the rest.
    ind = np.argpartition(troughMean, N)[:N]  # sorted by mean enclosed value
    pos = rand_points[ind]
    means = troughMean[ind]

    pos = pos.astype(np.float64) * mw / res

    return (
        pos,
        means,
    )  # return the positions of the accepted troughs and their mean enclosed values. (or sum if conv == False)


def radMasks(
    index, radius, nbins, mw, mr
):  # similar to radMask except this creates nbin annuli masks for radial profile calculations
    """
    index: coordinates for center of mask in form [x,y]
    radius: radius of mask in pixel number
    nbins: number of bins
    res: number of pixels on side of patch
       """
    a, b = index
    y, x = np.ogrid[a - radius : a + radius, b - radius : b + radius]
    grid = x * x + y * y

    delta_radius = radius / float(nbins)

    rbins = delta_radius * (np.arange(0, nbins, 1) + (1.0 / 2.0))

    eta = np.arange(0, nbins, 1)
    radius_upper = (eta + 1) * delta_radius
    radius_lower = eta * delta_radius

    mask_lower = [grid >= r * r for r in radius_lower]
    mask_upper = [grid <= r * r for r in radius_upper]

    mask = np.logical_and(mask_lower, mask_upper)

    return rbins, mask


def find_trough_profiles(
    data, pos, rad, nbins, mw, mr
):  # similar to find troughs except radial trough profiles are calculated here

    rad = int(rad * mr / mw) + 1
    rbins, mask = radMasks([0, 0], rad, nbins, mw, mr)
    profiles = np.zeros((len(pos), nbins))

    for i in range(len(pos)):
        b, a = np.int64(pos[i] * mr / mw)
        for j in range(nbins):
            profiles[i, j] = data[a - rad : a + rad, b - rad : b + rad][
                mask[j]
            ].mean()

    return rbins * mw / mr, profiles.mean(axis=0)


# Typical lines I use to call the above:

pos, means = find_troughs(mapi.data, nTroughs, frac, rad, mw, mr, conv=True)
pos_mean[k] = np.column_stack([pos, means])
r, profile_conv[k] = find_trough_profiles(
    mapi_rough.data, pos, 2 * rad, nbins, mw, mr
)

delta_radius = 2.0 / float(nbins)
r = delta_radius * (np.arange(0, nbins, 1) + (1.0 / 2.0))
