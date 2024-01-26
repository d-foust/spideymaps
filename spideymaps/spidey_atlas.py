"""
Class perfomring calculations on a collection of Spideymaps
"""
import numpy as np
import pandas as pd

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
            
    def create_rep_grid(self, mode='binaries', grid_params=DEFAULT_GRID_PARAMS):
        """
        Create a representative grid for the atlas.

        If mode is from binaries. Use binary images to generate a repensentative cell.
        """
        cell_bimages = [sm.bimage for sm in self.maps.values()]
        rep_bimage = calc_model_cell(cell_bimages)

        map = Spideymap(bimage=rep_bimage)
        map.make_grid(**grid_params)

        self.rep_grid = map.polygons


