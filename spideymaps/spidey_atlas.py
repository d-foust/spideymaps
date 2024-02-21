"""
Class perfomring calculations on a collection of Spideymaps
"""
import numpy as np
import pandas as pd
import shapely as sl

from .spideymap import extend_spine
from .spideymap import Spideymap
from .spideymap import OUTPARS, MIDPARS
from .spideymaps_rendering import calc_model_cell

DEFAULT_GRID_PARAMS = dict(n_cols = 8,
                           col_edges = None,
                           n_shells = 3,
                           shell_edges = None,
                           n_phi = None,
                           phi_edges  =None,
                           radius = 10,
                           outpars = OUTPARS,
                           midpars = MIDPARS)

DEFAULT_CC_PARAMS = dict(xl = -10, 
                         xr = 10, 
                         a0 = 0, 
                         a1 = 0, 
                         a2 = 0, 
                         r = 10)

class SpideyAtlas:

    def __init__(self, maps=None, map_names=None):
        """
        Parameters
        ----------
        maps : iterable containing Spideymaps
        """
        if maps is not None:
            
            # alternatively, keys could be file names, for example

            # organize maps into dictionary, collect coords into single DataFrame
            # if no keys (map_names) provided just use integers
            if map_names is None: map_names = np.arange(len(maps), dtype='int')
            self.maps = {}
            coords_list = []
            for k, m in zip(map_names, maps):
                if hasattr(m, 'coords'):
                    m.coords['map_name'] = k
                    coords_list.append(m.coords.copy())
                self.maps[k] = m
            if len(coords_list) > 0:
                self.coords = pd.concat(coords_list, ignore_index=True)

            self.pkeys = maps[0].polygons.keys() # polygon keys

        else:
            self.maps = {}

        self.data = {}

    def sum(self, data_key='counts'):
        """
        Sum, element-wise, the data[data_key]. Result stored in data[data_key+'_sum'].
        """
        # label for storage
        sum_key = data_key+'_sum'

        # initialize storage site
        self.data[sum_key] = {k: 0 for k in self.pkeys}

        # loop over everymap
        for map_key, map in self.maps.items():
            # loop over every polygon
            for pkey in self.pkeys:
                # add to previous total
                self.data[sum_key][pkey] += map.data[data_key][pkey]

        return self.data[sum_key]

    def density(self, num_key='counts_sum', den_key='areas_sum'):
        """
        Default behavior is to divide counts_sum and areas_sum
        """
        # check numerator values have been calculated
        if num_key=='counts_sum' and 'counts_sum' not in self.data:
            self.sum(data_key='counts')

        # check denominator already calculated
        if den_key=='areas_sum' and 'areas_sum' not in self.data:
            self.sum(data_key='areas')

        # divide key-wise
        rho_key = num_key+'_per_'+den_key
        self.data[rho_key] = {} 
        for pkey in self.pkeys: 
            self.data[rho_key][pkey] = self.data[num_key][pkey] / self.data[den_key][pkey]
            
        return self.data[rho_key]
            
    def find_symmetric_sets(self):
        """
        """
        pkey_array = np.array(list(self.pkeys))
        i_r_max = pkey_array[:,0].max()
        i_l_max = pkey_array[:,1].max()

        sym_key_sets = []

        for i_r in range(i_r_max + 1):
            pkey_ring = pkey_array[(pkey_array[:,0]==i_r)]
            i_p_max = pkey_ring[:,2].max()

            i_p = 0
            for i_p in range(1, i_p_max//2 + 1):
                sym_key_sets.append(((i_r, 0, i_p),
                    (i_r, 0, i_p_max - (i_p - 1)),
                    (i_r, i_l_max, i_p),
                    (i_r, i_l_max, i_p_max - (i_p - 1))))
                
            i_p += 1
            if i_p_max % 2 == 1:
                sym_key_sets.append(((i_r, 0, i_p),
                    (i_r, i_l_max, i_p_max - (i_p - 1))))
            
            i_l = 0
            for i_l in range(1, ((i_l_max - 1) // 2) + 1):
                sym_key_sets.append(((i_r, i_l, 0),
                                    (i_r, i_l, -1),
                                    (i_r, i_l_max-i_l, 0),
                                    (i_r, i_l_max-i_l, -1)))
            
            i_l += 1
            if i_l_max % 2 == 0:
                sym_key_sets.append(((i_r, i_l, 0),
                    (i_r, i_l, -1)))
    
        self.sym_key_sets = sym_key_sets

        return self.sym_key_sets
    
    def add_symmetric_elements(self, data_key='counts_sum', style='quad'):
        """
        """
        sym_key = data_key + '_sym'
        self.data[sym_key] = {}

        if hasattr(self, 'sym_key_sets') == False:
            self.find_symmetric_sets()

        for sym_set in self.sym_key_sets:
            set_sum = 0
            for pkey in sym_set:
                set_sum += self.data[data_key][pkey]
            for pkey in sym_set:
                self.data[sym_key][pkey] = set_sum

        return self.data[sym_key]
            
    def create_rep_grid(self, mode='binaries', grid_params=DEFAULT_GRID_PARAMS, cc_params=DEFAULT_CC_PARAMS):
        """
        Create a representative grid for the atlas.

        If mode is from binaries. Use binary images to generate a repensentative cell.
        """
        # use binary images of cells to find a representative grid
        if mode == 'binaries':
            cell_bimages = [sm.bimage for sm in self.maps.values()]
            rep_bimage = calc_model_cell(cell_bimages)

            map = Spideymap(bimage=rep_bimage)
            map.make_grid(**grid_params)

            self.rep_grid = map.polygons

        # use colicoords parameters to calculate a representative grid
        elif mode == 'colicoords':
            map = Spideymap()
            out = calc_outline(**cc_params)
            out = sl.LinearRing(out)
            mid = calc_midline(**cc_params)
            mid = sl.LineString(mid)
            mid = extend_spine(mid, out)
            map.make_grid(out=out, mid=mid, **grid_params)

            self.rep_grid = map.polygons


def calc_midline(x_arr, a0, a1, a2):
    """
    From colicoords.
    Calculate p(x).

    The function p(x) describes the midline of the cell.

    Parameters
    ----------
    x_arr : :class:`~numpy.ndarray`
        Input x values.
    a0, a1, a2
        Coefficients for 2nd order polynomial.

    Returns
    -------
    p : :class:`~numpy.ndarray`
        Evaluated polynomial p(x)
    """
    y = a0 + a1 * x_arr + a2 * x_arr ** 2
    mid = np.array([x_arr, y]).T
    
    return mid

def calc_outline(xl, xr, a0, a1, a2, r):
    """
    From colicoords.
    Plot the outline of the cell based on the current coordinate system.

    The outline consists of two semicircles and two offset lines to the central parabola.[1]_[2]_

    Parameters
    ----------
    ax : :class:`~matplotlib.axes.Axes`, optional
        Matplotlib axes to use for plotting.
    **kwargs
        Additional kwargs passed to ax.plot().

    Returns
    -------
    line : :class:`~matplotlib.lines.Line2D`
        Matplotlib line artist object.


    .. [1] T. W. Sederberg. "Computer Aided Geometric Design". Computer Aided Geometric Design Course Notes.
        January 10, 2012
    .. [2] Rida T.Faroukia, Thomas W. Sederberg, Analysis of the offset to a parabola, Computer Aided Geometric Design
        vol 12, issue 6, 1995

    """

    # Parametric plotting of offset line
    # http://cagd.cs.byu.edu/~557/text/ch8.pdf
    #
    # Analysis of the offset to a parabola
    # https://doi-org.proxy-ub.rug.nl/10.1016/0167-8396(94)00038-T

    numpoints = 500
    t = np.linspace(xl, xr, num=numpoints)
    # a0, a1, a2 = self.cell_obj.coords.coeff

    x_top = t + r * ((a1 + 2 * a2 * t) / np.sqrt(1 + (a1 + 2 * a2 * t) ** 2))
    y_top = a0 + a1*t + a2*(t**2) - r * (1 / np.sqrt(1 + (a1 + 2*a2*t)**2))

    x_bot = t + - r * ((a1 + 2 * a2 * t) / np.sqrt(1 + (a1 + 2 * a2 * t) ** 2))
    y_bot = a0 + a1*t + a2*(t**2) + r * (1 / np.sqrt(1 + (a1 + 2*a2*t)**2))

    #Left semicirlce
    psi = np.arctan(-p_dx(xl, a1, a2))

    th_l = np.linspace(-0.5*np.pi+psi, 0.5*np.pi + psi, num=200)
    cl_dx = r * np.cos(th_l)
    cl_dy = r * np.sin(th_l)

    cl_x = xl - cl_dx
    cl_y = calc_midline(xl, a0, a1, a2) + cl_dy

    #Right semicircle
    psi = np.arctan(-p_dx(xr, a1, a2))

    th_r = np.linspace(0.5*np.pi - psi, -0.5*np.pi - psi, num=200)
    cr_dx = r * np.cos(th_r)
    cr_dy = r * np.sin(th_r)

    cr_x = cr_dx + xr
    cr_y = cr_dy + calc_midline(xr, a0, a1, a2)

    x_all = np.concatenate((cl_x[::-1], x_top, cr_x[::-1], x_bot[::-1]))
    y_all = np.concatenate((cl_y[::-1], y_top, cr_y[::-1], y_bot[::-1]))

    out = np.array([x_all, y_all]).T

    return out

def p_dx(x_arr, a1, a2):
    """
    Calculate the derivative p'(x) evaluated at x.

    Parameters
    ----------
    x_arr :class:`~numpy.ndarray`:
        Input x values.

    Returns
    -------
    p_dx : :class:`~numpy.ndarray`
        Evaluated function p'(x).
    """
    return a1 + 2 * a2 * x_arr
