from copy import copy
import numpy as np

from scipy import ndimage
from scipy import ndimage as ndi

from skimage.transform import resize
from skimage.morphology import white_tophat, black_tophat
from skimage.morphology import disk, watershed, local_minima
from skimage.feature import peak_local_max

from sklearn.cluster import DBSCAN


class VoidFinder:
    def __init__(self, img, fov, kernel_width, res_ratio):
        """
        Args:
            kernel_width: float
                [deg]
        """
        self.img_orig = img
        self.fov = fov
        self.kernel_width = kernel_width

        # lower resolution
        map_lowres = resize(img, np.array(img.shape) * res_ratio)

        deg_per_pix = 20 / map_lowres.shape[0]
        kernel_width_pix = int(kernel_width / deg_per_pix)  # convert deg to pix

        # low-pass filter (smoothing)
        map_lowres = ndimage.gaussian_filter(
            map_lowres,
            sigma=kernel_width_pix,
            order=0,
            output=np.float64,
            mode="nearest",
        )

        self.img_smooth = map_lowres

    def find_void_positions(self):
        # find local minima
        local_min = np.array(
            local_minima(self.img_smooth, indices=True)
        ).astype(float)
        self.pos = {}
        self.pos["pix"] = local_min[::-1].T.astype(int)
        self.pos["deg"] = (
            local_min[::-1].T.astype(int) * self.fov / self.img_smooth.shape[0]
        )

    def find_void_radii(self):

        # create compensated top-hat filter
        deg_per_pix = 20 / self.img_smooth.shape[0]
        kernel_width_pix = int(
            self.kernel_width / deg_per_pix
        )  # convert deg to pix
        self.cth_rad = np.array(
            [int(2 * kernel_width_pix), int(2 * kernel_width_pix * np.sqrt(2))]
        )

        selem = (
            2
            * np.pad(
                disk(self.cth_rad[0]),
                self.cth_rad[1] - self.cth_rad[0],
                mode="constant",
                constant_values=0,
            )
            + disk(self.cth_rad[1]) * -1
        )

        # apply compensated top-hat filter
        self.img_cth = black_tophat(self.img_smooth, selem)

        # create marker for watershedding
        marker = np.zeros(self.img_smooth.shape)
        for src in range(len(self.pos)):
            marker[self.pos[src, 1], self.pos[src, 0]] = src + 1
        self.marker = marker

        # create mask for watershedding
        mask = np.zeros(self.img_cth.shape)
        limit = np.percentile(self.img_cth.flatten(), 80)
        mask[limit <= self.img_cth] = 1
        self.mask = mask[::-1]

        # create image for watershedding
        distance = ndi.distance_transform_edt(mask)

        # watershedding
        labels = watershed(
            -distance,  # map_lowres_smooth,
            markers=marker,
            mask=mask,
            watershed_line=True,
        )

        self.labels = labels

        rad = np.zeros(len(self.pos))
        for src in range(len(self.pos)):
            lab = labels[self.pos[src, 1], self.pos[src, 0]]
            area_pix = len(labels[labels == lab])
            rad[src] = np.sqrt(area_pix / np.pi)

        self.radii = rad
