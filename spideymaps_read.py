from glob import glob
from os.path import join, basename, exists

import h5py
import pandas as pd
import numpy as np
from scipy.io import loadmat

from spideymaps import count_cell, prepare_fits

pixel_size = 0.049

vertebrae_frac_pos = np.arange(0.1, 1, 0.1) # how to divide cell along long axis in non-polar region
rings_frac_pos = np.array([1./5, 2./5, 3./5, 4./5]) # how to divide cell radially
angles = [np.array([np.pi/2]), # what angles to sample in polar region, innermost ring to outermost
        np.array([np.pi/4, np.pi/2]),
        np.array([np.pi/6, np.pi/3, np.pi/2]),
        np.array([np.pi/8, np.pi/4, 3*np.pi/8, np.pi/2]),
        np.array([np.pi/10, np.pi/5, 3*np.pi/10, 2*np.pi/5, np.pi/2])]
radius = 0.5 / pixel_size # used to decide where to transition from polar to non-polar

default_grid_params = {
    'vertebrae_frac_pos': vertebrae_frac_pos,
    'rings_frac_pos': rings_frac_pos,
    'angles': angles,
    'radius': radius
}

def read_masks_file(file, format='smalllabs'):
    """
    """
    if format == 'smalllabs':
        masks = loadmat(file)['PhaseMask']
        
    elif format == 'cellpose':
        masks = np.load(file, allow_pickle=True).item()['masks']

    return masks

def read_locs_file(file, format='smalllabs', pixel_size=1, column_names=['x', 'y']):
    """
    return array 2 columns: rows, columns
    """
    if format == 'smalllabs':
        locs_obj = h5py.File(file)
        locs_data, weights = prepare_fits(locs_obj, gf_only=True)
    elif format == 'csv':
        locs_df = pd.read_csv(file)
        locs_data = locs_df[column_names].values / pixel_size

    return locs_data

def read_map_data(mask_folders, locs_folders,
                  masks_pattern='_seg.npy', locs_pattern='.locs',
                  masks_format='smalllabs', locs_format='smalllabs'):
    """
    """
    masks_list = []
    locs_list = []

    for mask_folder, locs_folder in zip(mask_folders, locs_folders):
        mask_files = glob(join(mask_folder, '*' + masks_pattern))
        base_names = [basename(file).split(masks_pattern)[0] for file in mask_files]
        for base_name in base_names:
            masks_file = join(mask_folder, base_name+masks_pattern)
            locs_file  = join(locs_folder, base_name+locs_pattern)
            if exists(masks_file) and exists(locs_file):
                masks = np.load(masks_file, allow_pickle=True).item()['masks']
                locs  = pd.read_csv(locs_file)

                masks_list.append(masks)
                locs_list.append(locs)

    return masks_list, locs_list


def calc_cell_maps(folders,
                    subfolder='',
                    locs_pattern='_fits.mat',
                    masks_pattern='_PhaseMask.mat',
                    masks_format='smalllabs',
                    locs_format='smalllabs',
                    grid_params=default_grid_params,
                    column_names=['x', 'y'], # 'row' column name, 'column' column name (palmari reversed)
                    pixel_size=0.049):
    """
    """
    j = 0; num_cells_total = 0
    data_dict = {}
    cell_bool_list = []

    for folder in folders:
        print(folder)
        files = glob(join(folder, '*' + masks_pattern)) # identify mask files
        base_names = [basename(file).split(masks_pattern)[0] for file in files]
        for base_name in base_names:
            masks_file = join(folder, base_name+masks_pattern)
            locs_file = join(folder, subfolder, base_name+locs_pattern)
            if exists(masks_file) and exists(locs_file):
                masks = read_masks_file(masks_file, format=masks_format)
                locs_data = read_locs_file(locs_file,
                                           format=locs_format,
                                           pixel_size=pixel_size,
                                           column_names=column_names)
                num_cells = masks.max()
                for i_cell in range(1, num_cells+1):
                    data_dict[j] = {}
                    cell_bool = masks == i_cell
                    data_cell = count_cell(cell_bool, grid_params, locs_data)
                    if data_cell:
                        data_dict[j] = data_cell
                        cell_bool_list.append(cell_bool)
                        num_cells_total += 1
                        j += 1
                    else:
                        print('Could not compute adaptive grid for: ',
                                join(folder, base_name), ' cell #', i_cell)

    return data_dict, cell_bool_list

def filter_by_nlocs(locs, minlocs=2, maxlocs=50):
    """
    filter locs data to exclude locs not in tracks with a minimum number of locs
    """
    locs_ngroups = locs.groupby('n')
    locs_linked = locs_ngroups.filter(lambda x: (x['n'].count() >= minlocs) & (x['n'].count() <= maxlocs))
    
    return locs_linked.reset_index()

def get_steps(df: pd.DataFrame):
    """
    """
    df_sorted = df.sort_values(['n','frame'])
    n_filt = df_sorted['n'].values[1:] == df_sorted['n'].values[:-1]
    f_filt = (df_sorted['frame'].values[1:] - df_sorted['frame'].values[:-1]) == 1
    t_filt = n_filt & f_filt

    steps = np.sqrt((df_sorted['y'].values[1:] - df_sorted['y'].values[:-1])**2 + (df_sorted['x'].values[1:] - df_sorted['x'].values[:-1])**2)
    y = (df_sorted['y'].values[1:] + df_sorted['y'].values[:-1]) / 2
    x = (df_sorted['x'].values[1:] + df_sorted['x'].values[:-1]) / 2
    rois = df_sorted['rois'].values[:-1]

    steps = steps[t_filt]
    y = y[t_filt]
    x = x[t_filt]
    rois = rois[t_filt]

    steps_df = pd.DataFrame(data={'step_size': steps,
                                    'y': y,
                                    'x': x,
                                    'rois': rois})
    return steps_df

def filter_by_stepsize(df: pd.DataFrame, minsize: float = 0, maxsize: float = 100):
    """
    """
    size_filter = (df['step_size'] >= minsize) & (df['step_size'] < maxsize)
    df_filt = df[size_filter]

    return df_filt