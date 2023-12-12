"""

"""
import numpy as np
import shapely as sl
from skimage.measure import find_contours

from .spideymaps_calculation import *

# default parameters for smoothening boundary
BNDPARS = dict(lam=0.39, mu=-0.4, N=500, sigma=1.5)

# default parameters for smoothening ridge
RDGPARS = dict(lam=0.39, mu=-0.4, N=500, sigma=2)


class Spideymap:


    def __init__(self, mask: np.ndarray = None):
        """
        constructor
        """ 
        if mask is not None:
            self.mask = mask
        else:
            # space for constructing default mask if none provided
            pass

    def make_grid(self, n_cols=16, n_rings=4, radius=10, bndpars=BNDPARS, rdgpars=RDGPARS):
        """
        """
        # define outermost boundary
        bnd = find_contours(self.mask, level=0.5)[0][:,::-1] # bnd skin (outer boundary)
        bnd = smooth_skin(bnd, **bndpars)
        self.bnd = sl.LinearRing(bnd)

        # define inner ridge
        rdg = get_spine(self.mask)[:,::-1]
        rdg = smooth_spine(rdg, **rdgpars) # smooth spine
        self.rdg = sl.LineString(rdg)
        self.rdg = extend_spine(self.rdg, self.bnd)

        # define foci where transition from cartesian to polar
        rdglen = sl.length(self.rdg)
        self.rdgpts = sl.line_interpolate_point(
            self.rdg, 
            np.linspace(radius, rdglen - radius, n_cols + 1)
            )
        # self.rdgpts = sl.LineString(rdgpts)


        

        










