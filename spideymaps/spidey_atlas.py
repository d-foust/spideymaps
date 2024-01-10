"""
Class perfomring calculations on a collection of Spideymaps
"""
import numpy as np

from .spideymap import Spideymap

class SpideyAtlas:

    def __init__(self, maps=None, map_names=None):
        """
        Parameters
        ----------
        maps : iterable containing Spideymaps
        """
        if maps is not None:
            # if no keys provided just use integers
            # alternatively, keys could be file names, for example
            if map_names is None: map_names = np.arange(len(maps), dtype='int')

            # organize maps into dictionary
            self.maps = {k: m for k, m in zip(map_names, maps)}
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

    def density(self, num_key='counts', den_key='areas'):
        """
        """
        # check numerator values have been calculated
        if num_key+'_sum' not in self.data.keys():
            self.sum(data_key=num_key)

        # check denominator already calculated
        if den_key+'_sum' not in self.data.keys():
            self.sum(data_key=den_key)

        # divide key-wise
        self.data[num_key+'_per_'+den_key] = {} 
        for pkey in self.pkeys: 
            self.data[num_key+'_per_'+den_key][pkey] = self.data[num_key+'_sum'][pkey]\
                                                       / self.data[den_key+'_sum'][pkey]
            

        

