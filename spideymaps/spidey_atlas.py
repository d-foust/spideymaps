"""
Class performing calculations on a collection of Spideymaps
"""
import numpy as np
import pandas as pd
import shapely as sl
from shapely.ops import unary_union

from .spideymap import extend_spine
from .spideymap import get_colicoords_row
from .spideymap import OUTPARS, MIDPARS
from .spideymap import Spideymap
from .spideymaps_rendering import calc_model_cell

# default grid parameters
DEFAULT_GRID_PARAMS = dict(n_cols = 8,
                           col_edges = None,
                           n_shells = 3,
                           shell_edges = None,
                           n_phi = None,
                           phi_edges  =None,
                           radius = 10,
                           outpars = OUTPARS,
                           midpars = MIDPARS)

# default colicoords parameters
DEFAULT_CC_PARAMS = dict(xl = -10, 
                         xr = 10, 
                         a0 = 0, 
                         a1 = 0, 
                         a2 = 0, 
                         r = 10)


class SpideyAtlas:

    def __init__(self, maps, map_names=None):
        """
        Parameters
        ----------
        maps : iterable containing Spideymaps
        """
            
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
            # if hasattr(m.data, 'areas'):
            #     self.areas 
            # else: # find areas then append
        if len(coords_list) > 0:
            self.coords = pd.concat(coords_list, ignore_index=True)

        self.pkeys = maps[0].polygons.keys() # polygon keys

        self.areas = pd.DataFrame(index=map_names, columns=self.pkeys)
        for k, m in zip(map_names, maps):
            pass

        # self.data = {}
        self.data = pd.DataFrame(index=self.pkeys)
    
    def add_symmetric_elements(self, col_name='count', symcol_name=None, style='quad'):
        """
        """
        if symcol_name is None:
            symcol_name = col_name + '_symsum'

        if hasattr(self, 'sym_key_sets') == False:
            self.find_symmetric_sets()

        for sym_set in self.sym_key_sets:
            self.data.loc[[*sym_set], symcol_name] = self.data.loc[[*sym_set], col_name].sum()
    
    def align_hamburger_style(self):
        """
        Align all spideymaps in atlas hamburger style (relative to short axis)
        """
        i_l_max = 0
        for i_r, i_l, i_p in list(self.pkeys):
            if i_l > i_l_max:
                i_l_max = i_l

        maps = self.coords['map_name']

        for map in maps:
            map_filt = self.coords['map_name'] == map
            left = (self.coords[map_filt]['i_l'] <= i_l_max // 2 + i_l_max % 2 - 1).sum()
            right = (self.coords[map_filt]['i_l'] > i_l_max // 2).sum()


            if left > right:
                self.coords[map_filt]['i_l'] = i_l_max - self.coords[map_filt]['i_l']

    def calculate_histograms(self, 
                             pixel_size=0.049, 
                             l_rep=None, 
                             r_rep=None, 
                             r_buffer=0.1, 
                             delta=1e-6,
                             num_l_bins=None,
                             num_r_bins=None):
        """
        """
        ## check that colicoords have been calculate
        colicoords_columns = ['l_abs', 'r_abs', 'theta', 'l_rel', 'r_abs_signed']
        if set(colicoords_columns).issubset(self.coords) is not True:
            self.get_colicoords(delta=delta)

        ## use rep_grid to find representative absolute dimensions if needed
        if l_rep is None or r_rep is None:
            if hasattr(self, 'rep_grid') is False:
                self.create_rep_grid()

            rep_cell = unary_union(list(self.rep_grid.values()))
            cell_bounding_box = rep_cell.bounds

            xrange = (cell_bounding_box[2] - cell_bounding_box[0]) * pixel_size
            yrange = (cell_bounding_box[3] - cell_bounding_box[1]) * pixel_size

            l_rep = xrange
            r_rep = (yrange + 2*r_buffer) / 2

        ## generate cell length histogram
        l_hist, l_rel_edges = np.histogram(
            self.coords['l_rel'],
            bins = np.linspace(0, 1, num_l_bins+1)
        )

        l_abs_edges = l_rel_edges * l_rep

        l_hist_asym = l_hist / l_hist.sum()

        l_hist_sym = (l_hist + l_hist[::-1]) / 2
        l_hist_sym = l_hist_sym / l_hist_sym.sum()

        # l_hist_edges = (l_hist_edges * xrange) - (xrange / 2)

        l_hist_df = pd.DataFrame(
            data = {
                'l_rel_left': l_rel_edges[:-1],
                'l_rel_right': l_rel_edges[1:],
                'l_um_left': l_abs_edges[:-1],
                'l_um_right': l_abs_edges[1:],
                'prob_asym': l_hist_asym,
                'prob_sym': l_hist_sym,
            }
        )

        r_hist, r_abs_edges = np.histogram(
            self.coords['r_abs_signed'] * pixel_size,
            bins=np.linspace(-1, 1, num_r_bins+1) * r_rep
            )
        
        r_rel_edges = np.linspace(0, 1, num_r_bins+1)

        r_hist_asym = r_hist / r_hist.sum()
        
        r_hist_sym = r_hist + r_hist[::-1]
        r_hist_sym = r_hist_sym / r_hist_sym.sum()
        
        r_hist_df = pd.DataFrame(
            data = {
                'r_rel_left': r_rel_edges[:-1],
                'r_rel_right': r_rel_edges[1:],
                'r_um_left': r_abs_edges[:-1],
                'r_um_right': r_abs_edges[1:],
                'prob_asym': r_hist_asym,
                'prob_sym': r_hist_sym,
            }
        )

        self.l_hist_df = l_hist_df
        self.r_hist_df = r_hist_df

    def create_rep_grid(self, 
                        mode='binaries', 
                        grid_params=DEFAULT_GRID_PARAMS, 
                        cc_params=DEFAULT_CC_PARAMS,
                        min_length=0,
                        max_length=np.inf):
        """
        Create a representative grid for the atlas.

        If mode is from binaries. Use binary images to generate a repensentative cell.
        """
        # use binary images of cells to find a representative grid
        if mode == 'binaries':
            cell_bimages = [map.bimage for map in self.maps.values() if (map.mid.length >= min_length and map.mid.length < max_length)]
            # cell_lengths = [map.mid.length for map in maps.values()]
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
        This is only for four-fold symmetry for now.
        """
        # convert polygon keys to array to work with easier
        pkey_array = np.array(list(self.pkeys))

        # find max indices for i_r and i_l
        # i_p max index depends on i_r, do downstream
        i_r_max = pkey_array[:,0].max()
        i_l_max = pkey_array[:,1].max()

        # returned as list
        sym_key_sets = []

        # loop over rings
        for i_r in range(i_r_max + 1):
            # find i_p_max for this ring
            pkey_ring = pkey_array[(pkey_array[:,0] == i_r)]
            i_p_max = pkey_ring[:,2].max()

            # start at i_p = 0, top side of spideymap
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

    def get_cell_lengths(self):
        """
        Length of cells in pixels added in new column of SpideyAtlas.coords.
        """
        map_names = self.coords['map_name'].unique()

        for map_name in map_names:
            map = self.maps[map_name]
            self.coords.loc[self.coords['map_name'] == map_name, 'cell_length'] = map.mid.length

    def get_colicoords(self, delta=1e-6):
        """
        """
        map_names = self.coords['map_name'].unique()

        for map_name in map_names:
            map_coords = self.coords[self.coords['map_name'] == map_name]
            map = self.maps[map_name]

            map_coords[['l_abs', 'r_abs', 'theta']] = map_coords.apply(
                lambda row: pd.Series(
                    get_colicoords_row(
                        row,
                        map.mid,
                        xcol=map.xcol,
                        ycol=map.ycol,
                        delta=delta
                    )
                ),
                axis=1
            )

            self.coords.loc[self.coords['map_name'] == map_name, 'l_abs'] = map_coords['l_abs']
            self.coords.loc[self.coords['map_name'] == map_name, 'r_abs'] = map_coords['r_abs']
            self.coords.loc[self.coords['map_name'] == map_name, 'theta'] = map_coords['theta']

            self.coords.loc[self.coords['map_name'] == map_name, 'cell_length'] = map.mid.length

            self.coords.loc[self.coords['map_name'] == map_name, 'l_rel'] = map_coords['l_abs'] / map_coords['cell_length']
            self.coords.loc[self.coords['map_name'] == map_name, 'r_abs_signed'] = map_coords['r_abs'] * np.sign(np.cos(map_coords['theta'] * np.pi/180.))

    def sum_maps(self, map_data_col='count', atlas_data_col=None, min_length=0, max_length=np.inf):
        """
        Sum map level data, Spideymap.data
        """
        map_data_list = [m.data[map_data_col] for m in self.maps.values() if (m.mid.length >= min_length and m.mid.length < max_length)]
        if atlas_data_col is None: atlas_data_col = map_data_col
        self.data[atlas_data_col] = pd.concat(map_data_list, axis=1).sum(axis=1)

    def sum_coords(self, sumcol_name='count', filt_col=None, filt_val=True):
        """
        Sum coordinate level data using index frequency.
        """
        if filt_col is None: 
            coords = self.coords
        elif filt_col is not None: 
            coords = self.coords[self.coords[filt_col] == filt_val]

        self.data[sumcol_name] = coords.groupby(['i_r', 'i_l', 'i_p']).size()
        self.data[sumcol_name].fillna(0, inplace=True)


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
