from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import numpy as np
import shapely as sl
from skimage.measure import regionprops
from skimage.transform import rotate
import seaborn as sns

def calc_model_cell(rois_list): 
    """
    rois_list : list of 2d arrays of type bool
    """
    
    total_cells = len(rois_list)

    rois_sum = np.zeros([301,301])
    
    for roi in rois_list:
        
        props = regionprops(roi.astype('int'))
        theta = props[0]['orientation']
        centroid = props[0]['centroid']
        roi_rot = rotate(roi, 
                         angle = 90 - theta * (180 / np.pi), 
                         center = centroid[::-1], 
                         resize = True)
        roi_bbox = regionprops(roi_rot.astype('int'))[0]['bbox']
        roi_rot = roi_rot[roi_bbox[0]:roi_bbox[2], roi_bbox[1]:roi_bbox[3]]
        n_rows, n_cols = roi_rot.shape
        rois_sum[150 - n_rows // 2 : 150 + n_rows // 2 + n_rows % 2, 
                 150 - n_cols // 2 : 150 + n_cols // 2 + n_cols % 2] += roi_rot
        
    rois_med = rois_sum >= total_cells / 2
    
    rois_med_sym = (rois_med.astype('int') + rois_med[:,::-1] + rois_med[::-1,:] + rois_med[::-1,::-1]) >= 2
        
    return rois_med_sym

def combine_symmetric_elements(counts, areas, sym_elements, polygons):
    """
    Combine counts and areas in symmetric elements.
    """
    counts_sym = {}
    areas_sym = {}
    
    for se_key in sym_elements:
        sels = sym_elements[se_key]
        counts_sym[sels[0]] = counts[sels[0]]
        areas_sym[sels[0]] = areas[sels[0]]
        for sel in sels[1:]:
            counts_sym[sels[0]] += counts[sel]
            areas_sym[sels[0]] += areas[sel]
        for sel in sels[1:]:
            counts_sym[sel] = counts_sym[sels[0]]
            areas_sym[sel] = areas_sym[sels[0]]
            
    counts_sym_sorted = {}
    areas_sym_sorted = {}
    
    for pkey in polygons.keys():
        counts_sym_sorted[pkey] = counts_sym[pkey]
        areas_sym_sorted[pkey] = areas_sym[pkey]
            
    return counts_sym_sorted, areas_sym_sorted

def density_symmetric(counts_sym, areas_sym, sym_elements):
    """
    Calculate densities for data that has been symmetrified.
    """
    density_sym = {}
    for ckey in counts_sym:
        density_sym[ckey] = counts_sym[ckey] / areas_sym[ckey]
        
    total_counts = 0
    total_area = 0
    
    for sels_key in sym_elements:
        total_counts += counts_sym[sym_elements[sels_key][0]]
        total_area += areas_sym[sym_elements[sels_key][0]]
        
    average_density = total_counts / total_area
    
    density_sym_norm = {}
    density_sym_norm_min = 1
    for dkey in density_sym:
        density_sym_norm[dkey] = density_sym[dkey] / average_density
        if density_sym_norm[dkey] < density_sym_norm_min and density_sym_norm[dkey] != 0:
            density_sym_norm_min = density_sym_norm[dkey]
    
    density_sym_log2_min = np.log2(density_sym_norm_min)
    density_sym_log2 = {}
    for dkey in density_sym_norm:
        if density_sym_norm[dkey] != 0:
            density_sym_log2[dkey] = np.log2(density_sym_norm[dkey])
        else:
            density_sym_log2[dkey] = density_sym_log2_min
        
    return density_sym, density_sym_norm, density_sym_log2

def get_color(val, vmin, vmax, cmap):
    if val < vmin: val = vmin
    if val > vmax: val = vmax
    return cmap.colors[np.round(255*((val - vmin) / (vmax - vmin))).astype('int')]

def get_densities(data_dict):
    """
    data: dict containing counts for all cells in all polygons
    
    return
    density_sym: counts / area, symmetrified
    density_sym_norm: density_sym divided by the average density
    density_sym_log2: log base-2 of density_sym_norm
    counts: total counts in each polygon, symmetrified
    areas: total area corresponding to each polygon symmetrified
    """
    n_cells = len(data_dict)
    
    areas_cell_total = 0 # sum of all areas in all cells
    counts_cell_total = 0 # sum of all counts in all cells
    
    # initialize counts and areas trackers
    counts_total = {pg: 0 for pg in data_dict[0]['polygons']} # sum counts in each polygon
    areas_total = {pg: 0 for pg in data_dict[0]['polygons']} # sum areas in each polygon

    for nc in range(n_cells):
        for gp_key in data_dict[nc]['polygons']:
            counts_total[gp_key] += data_dict[nc]['counts'][gp_key]
            areas_total[gp_key] += data_dict[nc]['areas'][gp_key]
            areas_cell_total += data_dict[nc]['areas'][gp_key]
            counts_cell_total += data_dict[nc]['counts'][gp_key]

    density, density_norm, density_log2 = polygon_density(counts_total, areas_total)

    sym_els = symmetric_grid_elements(data_dict[0]['polygons']) # groups polygons together with their symmetric counterparts
    counts_sym, areas_sym = combine_symmetric_elements(counts_total, areas_total, sym_els, density)
    density_sym, density_sym_norm, density_sym_log2 = density_symmetric(counts_sym, areas_sym, sym_els)
    
    return density_sym, density_sym_norm, density_sym_log2, counts_sym, areas_sym

def normalize_map(values_dict, weights=None):
    """
    Find weighted average, divide by it, return new dict
    """
    if weights is None:
        weights_array = np.ones(len(values_dict))
    else:
        weights_array = np.array(list(weights.values()))

    values_array = np.array(list(values_dict.values()))

    average = (weights_array * values_array).sum() / weights_array.sum()

    values_normed = {key: val / average for key, val in values_dict.items()}

    return values_normed

def polygon_density(counts, areas):
    """
    Calculate density in counts per pixel.
    """
    density = {}
    total_area = 0
    total_count = 0
    for ckey in counts:
        density[ckey] = counts[ckey] / areas[ckey]
        total_area += areas[ckey]
        total_count += counts[ckey]
        
    average_density = total_count / total_area
    norm_density = {}
    for dkey in density:
        norm_density[dkey] = density[dkey] / average_density
        
    min_norm_density = np.array(list(norm_density.values()))[np.array(list(norm_density.values()))>0].min()
    
    log2_density = {}
    for ndkey in norm_density:
        if norm_density[ndkey] > 0:
            log2_density[ndkey] = np.log2(norm_density[ndkey])
        else:
            log2_density[ndkey] = np.log2(min_norm_density)
        
    return density, norm_density, log2_density

def render_map(polygons_dict, values_dict, vmin=None, vmax=None, cmap=None, pixel_size=1):

    if cmap is None: cmap = sns.color_palette("vlag", as_cmap=True)

    values_array = np.array(list(values_dict.values()))
    if vmin is None:
        vmin = np.nanmin(values_array)
    if vmax is None:
        vmax = np.nanmax(values_array)

    sm = ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax, clip=True), cmap=cmap)

    x_center, y_center = sl.MultiPolygon(polygons_dict.values()).centroid.xy
    x_center = x_center[0] * pixel_size
    y_center = y_center[0] * pixel_size

    fig, ax = plt.subplots(figsize=(10,5))
    # x_min = 1000
    # x_max = -1000
    # y_min = 1000
    # y_max = -1000
    for key in polygons_dict:
        x = np.array(polygons_dict[key].boundary.xy[0], dtype='float') * pixel_size - x_center
        y = np.array(polygons_dict[key].boundary.xy[1], dtype='float') * pixel_size - y_center
        plt.fill(x, y,
                facecolor=get_color(values_dict[key], vmin=vmin, vmax=vmax, cmap=cmap),
                edgecolor='xkcd:white', linewidth=1.5)
        # if x.min() < x_min:
        #     x_min = x.min()
        # if x.max() > x_max:
        #     x_max = x.max()
        # if y.min() < y_min:
        #     y_min = y.min()
        # if y.max() > y_max:
        #     y_max = y.max()

    # x_center = (x_max - x_min) / 2
    # y_center = (y_max - y_min) / 2

    # range_x = x_max - x_min
    # range_y = y_max - y_min

    # ax.set_xlim(x_center - range_x / 2, x_center + range_x / 2)
    # ax.set_ylim(y_center - range_y / 2, y_center + range_y / 2)

    # ticks_x = plt.xticks()[0]
    # ticks_y = plt.yticks()[0]
    # ax.set_xticklabels([int(tick - x_center) for tick in ticks_x])
    # ax.set_yticklabels([int(tick - y_center) for tick in ticks_y])

    plt.gca().set_aspect('equal')
    cb = plt.colorbar(mappable=sm, ax=ax, shrink=0.30, aspect=10)
    cb.outline.set_color('xkcd:white')
    cb.ax.tick_params(labelsize=18)

    plt.gca().spines['bottom'].set_color(4*[0])
    plt.gca().spines['top'].set_color(4*[0])
    plt.gca().spines['left'].set_color(4*[0])
    plt.gca().spines['right'].set_color(4*[0])

    plt.gca().set_aspect('equal')

    return fig, ax

def symmetric_grid_elements(polygons):
    """
    Find grid elements that are symmetric assuming four-fold symmetry.
    """

    max_rad_idx = 0
    for element in polygons:
        (a, b), c = element
        if b > max_rad_idx:
            max_rad_idx = b

    sym_elements = {}; j = 0
    for element in polygons:
        (a, b), c = element
        if a < b and b <= max_rad_idx // 2:
            if a == 0:
                d = max_rad_idx
            else:
                d = -(max_rad_idx-a)
            sym_elements[j] = [((a , b), c),
                               ((-a , -b), c),
                               ((max_rad_idx-b, max_rad_idx-a), c),
                               ((-(max_rad_idx-b), d), c)]
            j += 1
        elif a < b and a < max_rad_idx/2 and b > max_rad_idx/2 and max_rad_idx%2 == 1:
            sym_elements[j] = [((a , b), c),
                               ((-a , -b), c)]
            j += 1

    return sym_elements