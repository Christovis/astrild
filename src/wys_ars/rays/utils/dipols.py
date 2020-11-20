from copy import copy
import numpy as np

from scipy import ndimage
from scipy import ndimage as ndi
from scipy.interpolate import RectBivariateSpline
from scipy.integrate import dblquad

from skimage.transform import resize
from skimage.morphology import white_tophat, black_tophat
from skimage.morphology import disk, watershed, local_minima
from skimage.feature import peak_local_max

from sklearn.cluster import DBSCAN


def dgd3_filter(img, sigma, direction):
    """
    Extract dipole signal.
    DOI: 10.3847/2041-8213/ab0bfe
    arXiv: 1812.04241 
    
    Args:
        sigma: ideally it should be the width of halo, R200. units are [pix]
    """
    gauss_1 = ndimage.gaussian_filter(
        img, sigma * 0.5, order=3 * direction, output=np.float64, mode="nearest"
    )
    gauss_2 = ndimage.gaussian_filter(
        img, sigma * 1.0, order=3 * direction, output=np.float64, mode="nearest"
    )
    gauss_3 = ndimage.gaussian_filter(
        img, sigma * 2.0, order=3 * direction, output=np.float64, mode="nearest"
    )
    return gauss_1 - gauss_2 + gauss_3


def hpf_gauss(img, kernel_width):
    """Gaussian high-pass filter"""
    lowpass = ndimage.gaussian_filter(img, kernel_width)
    img -= lowpass

    return img


def find_cluster(X, kernel_width, min_samples):
    """
    Find clusters of points that elevate from background and their centres.
    """
    db = DBSCAN(eps=kernel_width, min_samples=min_samples).fit(X)

    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # dipol_coord = np.zeros((len(np.unique(labels)), 2))
    cluster_centre = []
    cluster_radii = []
    for k in set(labels):
        if k == -1:
            # Black used for noise.
            continue

        class_member_mask = labels == k
        xy = X[class_member_mask & core_samples_mask]
        # get radii
        area_pix = len(xy)
        cluster_radii.append(np.sqrt(area_pix / np.pi))
        # get centre
        xy = np.array([np.median(xy[:, 0]), np.median(xy[:, 1])])
        cluster_centre.append(xy)

    cluster_centre = np.array(cluster_centre).astype(int)
    cluster_radii = np.array(cluster_radii).astype(int)

    return cluster_centre, cluster_radii


class DipoleFinder:
    def __init__(
        self,
        img,
        fov,
        Npix,
        significance,
        kernel_width=None,
        file_name=None,
        dataframe=None,
    ):
        """
        Args:
            img: 2D np.array
                Image on which to search for dipols.
            fov: float
                Field-of-view, sky coverage in [deg^2]
            significance: float
                Minim signal strenght to be concidered a dipole.
            kernel_width: float
                Width of convolution kernel. Recommended: 5 or 200
            file_name: str
                If dipols have already been identified but further properties
                need to be found.
        """
        self.img_orig = copy(img)
        self.fov = fov
        self.Npix = Npix
        self.significance = significance

        if kernel_width is not None:
            self.kernel_width = kernel_width
            # Gaussian high-pass filter
            img_hpf = hpf_gauss(img, kernel_width=kernel_width)
            self.img_hpf = img_hpf

        if file_name is not None:
            self.data = pd.read_hdf(file_name, key="df")
            self.pos = {
                "deg": self.data[["x_deg", "y_deg"]].values,
                "pix": self.data[["x_pix", "y_pix"]].values,
            }
            self.rad = {
                "deg": self.data["r200_deg"].values,
                "pix": self.data["r200_pix"].values,
            }

        if dataframe is not None:
            self.data = dataframe
            self.pos = {
                "deg": self.data[["x_deg", "y_deg"]].values,
                "pix": self.data[["x_pix", "y_pix"]].values,
            }
            self.rad = {
                "deg": self.data["r200_deg"].values,
                "pix": self.data["r200_pix"].values,
            }

    def position(self, *argv):
        """
        Find dipole signal in DeltaT sky maps.
        """
        # omni-directional DGD3 filter
        img_dgd3 = dgd3_filter(self.img_hpf, self.kernel_width, direction=1)

        # Smoothing filter
        self.img_dgd3_gauss = ndimage.gaussian_filter(
            np.abs(img_dgd3),
            sigma=self.kernel_width,
            order=0,
            output=np.float64,
            mode="nearest",
        )

        # find peaks above sigma-limit
        values_dgd3_gauss = self.img_dgd3_gauss.flatten()
        sigma = values_dgd3_gauss / np.std(values_dgd3_gauss)
        lower_limit = values_dgd3_gauss[
            sigma.max() * self.significance <= sigma
        ].min()

        # find dipols centres
        indx = np.where(lower_limit <= self.img_dgd3_gauss)
        X = np.vstack((indx[1], indx[0])).T
        pos, rad = find_cluster(X, self.kernel_width, 2)

        try:
            # if part of image is viewed
            if x_edge.__class__ == int:
                pos[:, 0] += x_edge
                pos[:, 1] += y_edge
            else:
                pos[:, 0] += x_edge[0]
                pos[:, 1] += y_edge[0]

        except:
            pass

        self.pos = {
            "pix": pos,
            "deg": pos
            * self.fov
            / self.img_orig.shape[0],  # convert pix to deg
        }
        self.rad = {
            "pix": rad,
            "deg": rad
            * self.fov
            / self.img_orig.shape[0],  # convert pix to deg
        }

    def velocity(self, deflt_x, deflt_y, area_of_integration, *argv):
        """
        Find dipole velocity vector in DeltaT sky maps.

        Args:
            deflt_x, deflt_y: 2D np.ndarray
                Deflection angle map.
            area_of_integration: float
                gold-section-search="gss"
                cluster of pixels over sigma threshold in dipole detection="sigma"
                associated group-halo R200="r200"
            dist_bound: 1D np.array
                Bounding radii within which to search for local maxima.
            tol: float
                Tolerance when to stop golden-section search.
        """
        # initialize
        self.vel = np.zeros((len(self.pos["pix"]), 2))
        thetas = np.array([[0, 1], [1, 0]])  # 1st theta1, 2nd theta2

        img_hpf = hpf_gauss(self.img_orig, kernel_width=200)
        defltx_hpf = hpf_gauss(deflt_x, kernel_width=200)
        deflty_hpf = hpf_gauss(deflt_y, kernel_width=200)

        if area_of_integration is "r200":
            self.kernel_width = self.rad["pix"]
        else:
            self.kernel_width = [self.kernel_width] * len(self.rad["pix"])

        # if area_of_integration is 'gss':
        #    # TODO maybe include in dipole for-loop
        #    # create mask with pixels distances to a pixel centre
        #    self.max_dist = dist_bound.max()
        #    # distance of pixels to centre
        #    pix_x = pix_y = np.arange(
        #        -int(dist_bound.max()), int(dist_bound.max())
        #    )
        #    self.pix = {}
        #    self.pix['xx'], self.pix['yy'] = np.meshgrid(pix_x, pix_y)
        #    self.pix['dist'] = np.sqrt(self.pix['xx']**2 + self.pix['yy']**2)

        # Run through theta1 & theta2 los-direction
        for ii in range(2):

            # Run through dipols
            for dd in range(len(self.pos["pix"])):
                centre = self.pos["pix"][dd]
                rad_pix = self.rad["pix"][dd]
                rad_deg = self.rad["deg"][dd]

                # crop images to around dipole out to twice the radius
                xaxis_min = int(centre[1] - int(rad_pix * 2))
                xaxis_max = int(centre[1] + int(rad_pix * 2)) + 1
                yaxis_min = int(centre[0] - int(rad_pix * 2))
                yaxis_max = int(centre[0] + int(rad_pix * 2)) + 1
                img_hpf_zoom = img_hpf[xaxis_min:xaxis_max, yaxis_min:yaxis_max]
                # uni-directional DGD3 filter, dipols in direction of theta
                iswrs_dgd3 = dgd3_filter(
                    img_hpf_zoom, self.kernel_width[dd], direction=thetas[ii]
                )
                if ii is 0:
                    deflt_x_zoom = defltx_hpf[
                        xaxis_min:xaxis_max, yaxis_min:yaxis_max
                    ]
                    deflt_dgd3 = dgd3_filter(
                        deflt_x_zoom,
                        self.kernel_width[dd],
                        direction=thetas[ii],
                    )
                else:
                    deflt_y_zoom = deflty_hpf[
                        xaxis_min:xaxis_max, yaxis_min:yaxis_max
                    ]
                    deflt_dgd3 = dgd3_filter(
                        deflt_y_zoom,
                        self.kernel_width[dd],
                        direction=thetas[ii],
                    )

                if area_of_integration is "gss":
                    # gold section search
                    inner, outer = dist_bound.min(), dist_bound.max()
                    # check if kernel is overshooting simulaiton boundary
                    outer = self._check_simulation_boundaries(
                        iswrs_dgd3.shape[0], centre, dist_bound.max()
                    )

                    if outer == -999999:
                        self.vel[dd, ii] = -999999
                        continue

                    else:
                        # Converge to maximum value of Eq. 9
                        self.vel[dd, ii] = self._gold_section_search(
                            iswrs_dgd3, deflt_dgd3, centre, inner, outer, tol
                        )
                elif area_of_integration is "sigma":
                    # distance of pixels to centre
                    pix_x = pix_y = np.arange(-int(rad_pix), int(rad_pix))
                    self.pix = {}
                    self.pix["xx"], self.pix["yy"] = np.meshgrid(pix_x, pix_y)
                    self.pix["dist"] = np.sqrt(
                        self.pix["xx"] ** 2 + self.pix["yy"] ** 2
                    )

                    self.vel[dd, ii] = self._magnitude(
                        iswrs_dgd3, deflt_dgd3, centre, rad_pix, rad_deg
                    )

                elif area_of_integration is "r200":
                    self.vel[dd, ii] = self._magnitude(
                        iswrs_dgd3, deflt_dgd3, centre, rad_pix, rad_deg
                    )

    def _check_simulation_boundaries(self, Npix, centre, outer):
        """
        Check whether the initial radii within which dipole velocity is evaluated
        shoots over the simulation box
        """
        extend = np.array(
            [
                centre[0] - outer,
                centre[0] + outer,
                centre[1] - outer,
                centre[1] + outer,
            ]
        )

        indx_min = np.where(extend < 0)[0]
        indx_max = np.where(extend > Npix)[0]

        if 0 < len(indx_min):
            print("Need to adjust searching distance")
            outer -= abs(extend[indx_min].min())

        if 0 < len(indx_max):
            print("Need to adjust searching distance")
            outer -= abs(extend[indx_max].max())

        if outer < 20:  # arbitrary
            return -999999
        else:
            # distance of pixels to centre
            pix_x = pix_y = np.arange(-int(outer), int(outer))
            self.pix_new = {}
            self.pix_new["xx"], self.pix_new["yy"] = np.meshgrid(pix_x, pix_y)
            self.pix_new["dist"] = np.sqrt(
                self.pix_new["xx"] ** 2 + self.pix_new["yy"] ** 2
            )

            return outer

    def _gold_section_search(
        self, iswrs_dgd3, deflt_dgd3, centre, inner, outer, tol
    ):
        """
        Perform search where magnitude of velocity vector is maximum.
        to find the minimum of f on [a,b]
        f: a strictly unimodal function on [a,b]
        Args:
            tol: float
                tolerance
        """
        gr = (np.sqrt(5) + 1) / 2  # golden ratio
        outer_orig = copy(outer)

        def cal_r1(inner, outer):
            return outer - round((outer - inner) / gr)

        def cal_r2(inner, outer):
            return inner + round((outer - inner) / gr)

        r1 = cal_r1(inner, outer)
        r2 = cal_r2(inner, outer)

        counter = 0
        while abs(r1 - r2) > tol:

            vel_r1 = self._magnitude(iswrs_dgd3, deflt_dgd3, centre, r1)
            vel_r2 = self._magnitude(iswrs_dgd3, deflt_dgd3, centre, r2)

            if abs(vel_r1) < abs(vel_r2):  # search for maximum
                inner = r1
                r1 = r2
                r2 = cal_r1(r1, outer)
            else:
                outer = r2
                r2 = r1
                r1 = cal_r2(inner, r2)

            print(abs(r1 - r2), (r1 + r2) / 2, abs(vel_r1))

            counter += 1
            if counter > 1000:
                print("Not converged ------------------------")
                break

        if counter > 1000:
            # If it didn't converge in time
            return -999999
        elif outer_orig == (r1 + r2) / 2:
            # If maximum is beyond maximum radii
            return -999999
        else:
            return self._magnitude(
                iswrs_dgd3, deflt_dgd3, centre, (r1 + r2) / 2
            )

    def _magnitude(self, iswrs_dgd3, deflt_dgd3, centre, rad_pix, rad_deg):
        """
        Velocity vector component from ISWRS and deflection angle map.
        Solve Eq. 9 in arxiv:1812.04241
        """
        c_light = 299792.458  # [km/s]
        iswrs_sum = self._solve_integral(iswrs_dgd3, centre, rad_pix, rad_deg)
        deflt_sum = self._solve_integral(deflt_dgd3, centre, rad_pix, rad_deg)

        print("iswrs_sum & deflt_sum", iswrs_sum, deflt_sum)
        velocity = -c_light * iswrs_sum / deflt_sum

        return velocity

    def _polar_integrand(self, _theta, _phi):
        x = _theta * np.cos(_phi)
        y = _theta * np.sin(_phi)
        return self.zoom_spline(x, y)

    def _solve_integral(self, img, centre, rad_pix, rad_deg):
        """
        Sum of image values around centre within radii.
        Args:
            rad_pix, rad_deg: np.int, float
                Radius in units of pixel/degrees.
        """
        if "pix_new" in dir(self):
            # select circular region
            indx = self.pix_new["dist"] <= radii

            # select square region with edge length equal to circular diameter
            zoom = img[
                int(centre[1] + self.pix_new["yy"].min()) : int(
                    centre[1] + self.pix_new["yy"].max()
                )
                + 1,
                int(centre[0] + self.pix_new["xx"].min()) : int(
                    centre[0] + self.pix_new["xx"].max()
                )
                + 1,
            ]
            del self.pix_new

        else:
            # select square region with edge length equal to circular diameter
            xaxis_min = int(len(img) / 2) - int(rad_pix)
            xaxis_max = int(len(img) / 2) + int(rad_pix) + 1
            yaxis_min = int(len(img) / 2) - int(rad_pix)
            yaxis_max = int(len(img) / 2) + int(rad_pix) + 1
            zoom = img[xaxis_min:xaxis_max, yaxis_min:yaxis_max]

            # increase resolution of square region
            _coord = np.linspace(
                -int(len(zoom) / 2) * self.fov / self.Npix,
                int(len(zoom) / 2) * self.fov / self.Npix,
                len(zoom),
            )
            self.zoom_spline = RectBivariateSpline(_coord, _coord, zoom)

            result = dblquad(
                self._polar_integrand,
                1e-10,
                rad_deg,  # [deg]
                lambda x: 0,
                lambda x: 2 * np.pi,  # [deg]
                # args=(_centre)
            )[0]
            # coord = np.linspace(0, len(zoom)-1, len(zoom)*2)
            # zoom = zoom_spline(coord, coord)
            #
            ## select circular region
            # pix_x = pix_y = np.arange(-int(len(zoom)/2), int(len(zoom)/2))
            # pix_xx, pix_yy  = np.meshgrid(pix_x, pix_y)
            # pix_dist = np.sqrt(pix_xx**2 + pix_yy**2)
            # indx = pix_dist <= radius
        return result
