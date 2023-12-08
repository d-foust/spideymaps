#####################
### Load packages ###
#####################
# builtin Python packages
import os
from os.path import join
from pathlib import Path
import sys

# third party packages installed through Anaconda
import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import regionprops

# local packages
script_path = sys.argv[0]
spidey_path = Path(script_path).parents[1]
os.chdir(os.path.dirname(os.path.abspath(sys.argv[0])))
if join(spidey_path, "spideymaps") not in sys.path:
    sys.path.append(join(spidey_path, "spideymaps"))
import spideymaps as sm

########################
### Where's the data ###
########################
labels_folders = ["..\data\cwx2695_LAM"] # can specify multiple folders separated by comma
locs_folders = ["..\data\cwx2695_LAM"]

###################
### Define Grid ###
###################
pixel_size = 0.049 # in um
vertebrae_frac_pos = np.arange(1/24, 0.99, 1/24) # how to divide cell along long axis in non-polar region
rings_frac_pos = np.arange(1/6, 0.99, 1/6) # how to divide cell radially
angles = [np.array([np.pi/2]), # what angles to sample in polar region, innermost ring to outermost
        np.array([np.pi/4, np.pi/2]),
        np.array([np.pi/6, np.pi/3, np.pi/2]),
        np.array([np.pi/8, np.pi/4, 3*np.pi/8, np.pi/2]),
        np.array([np.pi/8, np.pi/4, 3*np.pi/8, np.pi/2]),
        np.array([np.pi/8, np.pi/4, 3*np.pi/8, np.pi/2]),]
radius = 0.5 / pixel_size # used to decide where to transition from polar to non-polar, um / (um/px)

grid_params = {
                'vertebrae_frac_pos': vertebrae_frac_pos,
                'rings_frac_pos': rings_frac_pos,
                'angles': angles,
                'radius': radius,
               }

#####################
### Length ranges ###
#####################
# in um
length_ranges = np.array([
    [2.5, 3.5],
    [3.5, 4.5],
    [4.5, 5.5]
])


#####################
### read map data ###
#####################
labels_list, locs_list = sm.read_map_data(
    labels_folders=labels_folders,
    locs_folders=locs_folders,
    labels_pattern="_PhaseMask.mat",
    locs_pattern="_AccBGSUB_fits.mat",
    labels_format="smalllabs", # alternative "cellpose"
    locs_format="smalllabs", # alternative "smalllabs"
    pixel_size=1, # for conversion if units are not pixels
    coord_cols=["row", "col"] # if using csv, specify which columns contain coordinates
)

####################################################################
### subtract 1 from Matlab coordinates to get Python coordinates ###
####################################################################
for locs in locs_list:
    locs.loc[:,'row-1'] = locs.loc[:,'row'] - 1
    locs.loc[:,'col-1'] = locs.loc[:,'col'] - 1


###########################################
### Calculate maps for individual cells ###
###########################################
cell_maps, cell_bools = sm.calc_cell_maps(
    labels_list,
    locs_list,
    grid_params,
    pixel_size=1,
    coord_cols=('row-1', 'col-1'),
    weights_col='goodfit',
    label_col='roinum'
    )

##############################
### Calculate cell lengths ###
##############################
cell_lengths = np.array([regionprops(cell_bool.astype('int'))[0]["feret_diameter_max"] for cell_bool in cell_bools])
cell_lengths = cell_lengths * pixel_size # convert to um

####################################
### Calculate map for each range ###
####################################
model_cells = []
normalized_maps = []

for length_range in length_ranges:
    min_length = length_range[0]
    max_length = length_range[1]
    length_filter = (cell_lengths >= min_length) & (cell_lengths < max_length)
    if length_filter.sum() == 0:
        continue
    cell_bools_subset = [cell_bool for cell_bool, lf in zip(cell_bools, length_filter) if lf]
    cell_maps_subset = {k: cell_map for (k, cell_map), lf in zip(cell_maps.items(), length_filter) if lf}

    model_cell_bool = sm.calc_model_cell(cell_bools_subset)
    model_cell = sm.get_cell_grid(model_cell_bool, 
                            grid_params['vertebrae_frac_pos'],
                            grid_params['rings_frac_pos'],
                            grid_params['angles'], 
                            grid_params['radius'],
                            sigma_spine=1.5,
                            sigma_skin=2)

    sym_elements = sm.symmetric_grid_elements(model_cell["polygons"])

    counts_total = sm.sum_cell_maps(cell_maps_subset, val_key='counts')
    areas_total  = sm.sum_cell_maps(cell_maps_subset, val_key='areas')
    counts_per_area = sm.divide_dicts(counts_total, areas_total)
    cpa_normed = sm.normalize_map(counts_per_area, weights=areas_total)
    cpa_norm_sym = sm.average_sym_elements(cpa_normed, sym_elements)

    model_cells.append(model_cell)
    normalized_maps.append(cpa_norm_sym)

####################
### Render maps ###
####################
for model_cell, normalized_map in zip(model_cells, normalized_maps):
    fig, ax = sm.render_map(model_cell["polygons"], normalized_map)

