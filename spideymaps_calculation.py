"""
Functions for calculating adaptive grid given a binary mask for a rod-shaped bacterium
"""
import numpy as np
from numpy.linalg import matrix_power
from scipy.ndimage import convolve
from scipy.stats import linregress
import shapely
from shapely import distance, get_coordinates, get_parts, get_x, get_y, intersection, line_interpolate_point
from shapely import LinearRing, LineString, STRtree
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
from skimage.morphology import skeletonize, binary_dilation, binary_erosion, disk
from skimage.measure import find_contours

def add_radials(vb_xy, skin_isxn, rings_frac_pos, radials, rad_idx):
    """
    """
    last_pos = vb_xy; i=0
    for rfp in rings_frac_pos:
        pos = rfp*skin_isxn + (1-rfp)*vb_xy
        radials[rad_idx,i] = np.array([last_pos[::-1], pos[::-1]])
        last_pos = pos; i+=1
    radials[rad_idx,i] = np.array([last_pos[::-1], skin_isxn[::-1]])
    return radials

def average_sym_elements(values_dict, sym_elements):
    """
    """
    values_sym = {}
    for k, sels in sym_elements.items():
        n_sels = len(sels)
        sum_sels = 0
        for sel in sels:
            sum_sels += values_dict[sel]
        average_sels = sum_sels / n_sels
        for sel in sels:
            values_sym[sel] = average_sels

    values_sym = {key: values_sym[key] for key in values_dict} # make sure in same order

    return values_sym

def calc_cell_diff(mask, grid_params, diff_dict):
    """
    """
    try:
        skin, skin_isxns, spine, vertebrae_idx, radials, rings, polygons = get_cell_grid(mask, 
                                                                                     grid_params['vertebrae_frac_pos'],
                                                                                     grid_params['rings_frac_pos'],
                                                                                     grid_params['angles'])
    except:
        print('Warning: Could not compute adaptive grid for this shape.')
        
    diff_total = {}
    weights_total = {}
    shapely_polygons = {}
    
    for pkey in polygons:
        diff_total[pkey] = 0
        weights_total[pkey] = 0
        shapely_polygons[pkey] = Polygon(polygons[pkey])
        
    for loc1, loc2, weight, D_app in zip(diff_dict['loc1'],
                                         diff_dict['loc2'],
                                         diff_dict['weights'],
                                         diff_dict['D_app']):
        pt1 = Point(loc1)
        pt2 = Point(loc2)
        for pkey in shapely_polygons:
            if shapely_polygons[pkey].contains(pt1):
                diff_total[pkey] += (weight/2) * D_app
                weights_total[pkey] += (weight/2)
            if shapely_polygons[pkey].contains(pt2):
                diff_total[pkey] += (weight/2) * D_app
                weights_total[pkey] += (weight/2)
                
    return {'diff_total': diff_total, 'weights_total': weights_total}

def calc_cell_maps(labels_list, locs_list, grid_params, pixel_size=1, coord_cols=('row', 'col'), weights_col=None, label_col=None):
    """


    Parameters
    ----------
    labels_list : list[np.ndarray]
        Elements are 2d integer arrays (images)
    locs_list : list[pd.DataFrame]
        list of DataFrames containing localization data
    grid_params : dict
        Defines grid.
    pixel_size : float
        Locs in locs_list should have units of pixels. If not, use to convert.
    """
    data_dict = {}; j=0
    cell_bool_list = []
    for labels, locs in zip(labels_list, locs_list):
        num_cells = labels.max()

        coords = locs[list(coord_cols)].values / pixel_size
        if label_col is not None:
            label_vals = locs[label_col].values
        else:
            label_vals = None
        if weights_col is not None:
            weights = locs[weights_col].values
        else:
            weights = np.ones(len(locs), dtype='float')
        
        for i_cell in range(1, num_cells+1):
            data_dict[j] = {}
            cell_bool = labels == i_cell
            if label_vals is not None:
                labels_bool = label_vals == i_cell
                coords_cell = coords[labels_bool]
                weights_cell = weights[labels_bool]
            else:
                coords_cell = coords
                weights_cell = weights
            data_cell = count_cell(cell_bool, grid_params, coords_cell, weights=weights_cell)
            if data_cell:
                data_dict[j] = data_cell
                cell_bool_list.append(cell_bool)
                j += 1
            else:
                print('Could not compute adaptive grid for cell #', i_cell)

    return data_dict, cell_bool_list

def calculate_volumes(spine, skin, polygons, dx=0.1):
    """
    Inputs
    spine : shapely LineString
    skin : shapely LinearRing
    polygons : shapely polygons, array of Polygon

    Output :
    volumes : array of volume values corresponding to each polygon
    """
    spine_extended = extend_spine(spine, skin)

    spine_pts = line_interpolate_point(spine_extended, np.arange(-dx/2, spine_extended.length+dx/2, dx), normalized=False)
    spine_idx = np.arange(len(spine_pts))

    sample_pts_list = []
    heights_list = []

    xmin, ymin, xmax, ymax = skin.bounds

    for i, pt in zip(spine_idx[1:-1], spine_pts[1:-1]):
        prev_pt = spine_pts[i-1]
        next_pt = spine_pts[i+1]

        x0, y0 = get_x(prev_pt), get_y(prev_pt)
        x,  y  = get_x(pt),      get_y(pt)
        x1, y1 = get_x(next_pt), get_y(next_pt)

        if y0 != y1: # rib not perfectly vertical
            slope = -(x1 - x0) / (y1 - y0)
            rib_line = LineString(([xmin, slope * (xmin - x) + y],
                                   [xmax, slope * (xmax - x) + y]))
        else: # rib needs to be vertical
            rib_line = LineString(([x, ymin],
                                   [x, ymax]))
        # could be problem if rib_line perfectly vertical, slope = infinity
        rib_line = LineString(get_parts(intersection(skin, rib_line)))
        rib_pts = line_interpolate_point(rib_line, np.arange(dx/2, rib_line.length-dx/2, dx), normalized=False)

        bigR = rib_line.length / 2
        midpt = line_interpolate_point(rib_line, distance=0.5, normalized=True)
        littleRs = distance(midpt, rib_pts)
        hs = np.sqrt(bigR**2 - littleRs**2)

        sample_pts_list.append(rib_pts)
        heights_list.append(hs)

    sample_pts = np.concatenate(sample_pts_list)
    heights = np.concatenate(heights_list)

    polygon_rtree = STRtree(polygons)

    raw_counts = shapely_count(sample_pts, polygon_rtree, weights=None)
    heights_totaled = shapely_count(sample_pts, polygon_rtree, weights=heights)

    areas = np.array([polygon.area for polygon in polygons])
    volumes = (heights_totaled / raw_counts) * areas

    return volumes

def ccw(A,B,C):
    """
    Helper for intersect.
    """
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

def combine_polygons(polygons, angles, angles_dict, num_ribs, max_rad_idx):
    """
    Use angles list to combine polygons near poles to be more appropriately sized.
    polygons: dictionary containing polygons, dict keys follow weird coordinate system I made up
    angles: list of 1d numpy array containing angles for each ring, start with innermost, units are radians
    """
    new_polygons = {}
    
    for ir, angles_ring in enumerate(angles): # ir: ring index, angles_ring: numpy array of angles in radians
        angles_ring = [0]+list(angles_ring) # include 0 angle by default
        radial_idx = []
        
        for ar in angles_ring:
            radial_idx.append(angles_dict[ar])

        for ipoly in range(len(radial_idx)-1):
            start_idx = radial_idx[ipoly]
            end_idx = radial_idx[ipoly+1]
            wedge_list_q1 = []
            wedge_list_q2 = []
            wedge_list_q3 = []
            wedge_list_q4 = []

            for iwedge in range(start_idx, end_idx):
                wedge_list_q1.append(Polygon(polygons[(iwedge, iwedge+1), ir]))
                wedge_list_q2.append(Polygon(polygons[(convert_rad_idx(iwedge, max_rad_idx, 2),
                                                       convert_rad_idx(iwedge+1, max_rad_idx, 2)), ir]))
                wedge_list_q3.append(Polygon(polygons[(convert_rad_idx(iwedge+1, max_rad_idx, 3),
                                                       convert_rad_idx(iwedge, max_rad_idx, 3)), ir]))
                wedge_list_q4.append(Polygon(polygons[(convert_rad_idx(iwedge+1, max_rad_idx, 4),
                                                       convert_rad_idx(iwedge, max_rad_idx, 4)), ir]))

            new_polygons[(radial_idx[ipoly], radial_idx[ipoly+1]), ir] = unary_union(wedge_list_q1)

            new_polygons[(convert_rad_idx(radial_idx[ipoly], max_rad_idx, 2),
                          convert_rad_idx(radial_idx[ipoly+1], max_rad_idx, 2)), ir] = unary_union(wedge_list_q2)

            new_polygons[(convert_rad_idx(radial_idx[ipoly+1], max_rad_idx, 3),
                          convert_rad_idx(radial_idx[ipoly], max_rad_idx, 3)), ir] = unary_union(wedge_list_q3)

            new_polygons[(convert_rad_idx(radial_idx[ipoly+1], max_rad_idx, 4),
                          convert_rad_idx(radial_idx[ipoly], max_rad_idx, 4)), ir] = unary_union(wedge_list_q4)
           
    # copy non-polar polygons
    num_rings = len(angles)
    num_angles = len(angles_dict) - 1
    for irib in range(num_angles, num_angles+num_ribs-1):
        for iring in range(num_rings):
            new_polygons[(irib, irib+1), iring] = Polygon(polygons[(irib, irib+1), iring])
            new_polygons[(-irib, -irib-1), iring] = Polygon(polygons[(-irib, -irib-1), iring])
            
    return new_polygons

def convert_rad_idx(idx_q1, max_rad_idx, quadrant):
    """
    Convert radial line index in quadrant 1 to equivalent radial line index in quadrant 2, 3, or 4.

    Parameters
    ----------
    idx_q1 : int
        Quadrant 1 index to be converted.
    max_rad_idx : int
        Highest radial idx in grid.
    quadrant : int
        2, 3, or 4.

    Returns
    -------
    idx : int
    """
    
    if quadrant == 2:
        idx = -idx_q1
    elif quadrant == 3:
        idx = max_rad_idx - idx_q1
    elif quadrant == 4:
        if idx_q1 == 0:
            idx = max_rad_idx
        else:
            idx = -max_rad_idx + idx_q1
        
    return idx

def count_cell(mask, grid_params, coords, weights=None):
    """
    Calculate adaptive grid. Calculate areas and localizations for each polygon in grid.
    -
    Input parameters
    mask : boolean 2d array
    grid_params : dict containing vertebrae_frac_pos, rings_frac_pos, angles
    coords : Nx2 array, first column has rows, second column has columns
    weights : Nx1 array
    """
    data = {}
    
    try:
        cg_dict = get_cell_grid(mask,
                                grid_params['vertebrae_frac_pos'],
                                grid_params['rings_frac_pos'],
                                grid_params['angles'],
                                grid_params['radius'],
                                sigma_spine=2,
                                sigma_skin=1.5)
        
        spine = LineString(cg_dict['spine'])
        skin  = LinearRing(cg_dict['skin'])
        polygons = shapely.polygons(np.array([LinearRing(rc) for rc in cg_dict['polygons'].values()]))

        data['polygons']    = cg_dict['polygons']
        data['counts']      = polygon_counts(coords=coords, polygons=cg_dict['polygons'], weights=weights)
        data['areas']       = polygon_areas(polygons=cg_dict['polygons'])
        volumes = calculate_volumes(spine, skin, polygons, dx=0.2)
        data['volumes']     = {k: v for k, v in zip(cg_dict['polygons'].keys(), volumes)}
    except:
        print('Warning: Could not compute adaptive grid for this shape.')
    
    return data

def divide_dicts(numerator: dict, denominator: dict):
    """
    """
    return {key: numerator[key]/denominator[key] if denominator[key] != 0 else np.nan for key in numerator}

def extend_spine(spine, skin):
    """
    Add points to both ends of spine so it intersects with skin.
    Extension lines have same slopes as end segments of input spine.
    ---
    Inputs
    spine : shapely LineString
    skin : shapely LinearRing
    Outputs
    spine_extended : shapely LineString
    """
    xmin, ymin, xmax, ymax = skin.bounds
    diag = np.sqrt((np.array([xmin-xmax, ymin-ymax])**2).sum())

    # first get new segment for one end
    x0 = spine.coords.xy[0][0]
    y0 = spine.coords.xy[1][0]
    x1 = spine.coords.xy[0][1]
    y1 = spine.coords.xy[1][1]
    v = np.array([x0 - x1, y0 - y1])
    mag = np.sqrt((v**2).sum())
    probe = LineString(np.array([[x0,y0], (diag/mag)*v + np.array([x0,y0])]))
    new_end0 = intersection(probe, skin)

    # second segment for other end
    x0 = spine.coords.xy[0][-1]
    y0 = spine.coords.xy[1][-1]
    x1 = spine.coords.xy[0][-2]
    y1 = spine.coords.xy[1][-2]
    v = np.array([x0 - x1, y0 - y1])
    mag = np.sqrt((v**2).sum())
    probe = LineString(np.array([[x0,y0], (diag/mag)*v + np.array([x0,y0])]))
    new_end1 = intersection(probe, skin)

    spine_extended = LineString(
        np.concatenate([
            get_coordinates(new_end0),
            get_coordinates(spine),
            get_coordinates(new_end1)
            ])
            )

    return spine_extended

def find_angle(vector1, vector2):
    """
    """
    arg = (np.dot(vector1, vector2) / ((vector1**2).sum()**0.5 * (vector2**2).sum()**0.5)).round(decimals=3)
    return np.arccos(arg)

def find_skin_intercepts(segment, skin):
    """
    """
    intersections = []
    isxn_idx_skin = []
    for i in range(len(skin)-1):
        skin_seg = [skin[i][::-1], skin[i+1][::-1]]
        if intersect(segment[0], segment[1], skin_seg[0], skin_seg[1]):
            intersections.append(findIntersection(segment[0][0], segment[0][1],
                                   segment[1][0], segment[1][1],
                                   skin_seg[0][0], skin_seg[0][1],
                                   skin_seg[1][0], skin_seg[1][1]))
            # isxn_idx_skin.append(i+1)
            isxn_idx_skin.append((i,i+1))
    return np.asarray(intersections), isxn_idx_skin

def find_skin_intersection(vb_xy, slope, ref_segment, skin):
    """
    """
    if ~np.isinf(slope):
        intercept = vb_xy[1] - slope*vb_xy[0]
        test_start = [vb_xy[0]-10000, slope * (vb_xy[0]-10000) + intercept] # need check for verticality
        test_end   = [vb_xy[0]+10000, slope * (vb_xy[0]+10000) + intercept]
        test_segment = np.array([test_start, test_end])
    elif np.isinf(slope):
        test_start = [vb_xy[0], -1]
        test_end   = [vb_xy[0], 10000]
        test_segment = np.array([test_start, test_end])
    
    intersections, isxn_idx_skin = find_skin_intercepts(test_segment, skin)
    isxn_vectors = np.array([isxn - vb_xy for isxn in intersections])
    angles = np.array([find_angle(ref_segment, iv) for iv in isxn_vectors])
    which_intersection = angles.argmin()
    correct_intersection = np.array(intersections[which_intersection])
    correct_idx = isxn_idx_skin[which_intersection]
    
    return correct_idx, correct_intersection

def findIntersection(x1, y1, x2, y2, x3, y3, x4, y4):
    """
    Find where two line segments intersect
    """
    px = ( (x1*y2-y1*x2)*(x3-x4) - (x1-x2)*(x3*y4-y3*x4) ) / ( (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4) ) 
    py = ( (x1*y2-y1*x2)*(y3-y4) - (y1-y2)*(x3*y4-y3*x4) ) / ( (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4) )
    return [px, py]

def gauss_kernel(x, sigma):
    gk = np.exp(-x**2 / (2 * sigma**2))
    gk[x==0] = 0
    gk /= gk.sum()
    return gk

def get_cell_grid(mask, vertebrae_frac_pos, rings_frac_pos, angles, radius,
                 sigma_spine=2, sigma_skin=1.5):
    """
    Main function for finding grid from binary mask and grid specifications.
    
    Parameters
    ----------
    mask : np.ndarray
        Boolean 2d array describing cell shape
    vertebrae_frac_pos : 
        Fractional positions to have grid lines. 0 and 1 included by default.
    rings_frac_pos : 
        Fractional ring positions. 0 and 1 included by default.
    angles : 
        Angles at which to draw radial gridlines
    radius : float
        Distance from cell ends to transition from rounded to rectangular grid elements.
    sigma_spine : float, default is 2


    Returns
    -------
    grid_data : dict
        Keys : 
            'skin'
            'skin_isxns'
            'spine'
            'vertebrae_idx'
            'radials'
            'rings'
            'polygons'
    """
    skin = find_contours(mask, level=0.5)[0] # initial skin (outer boundary)
    skin = smooth_skin(skin, lam=0.39, mu=-0.4, N=500, sigma=sigma_skin) # smooth skin
    spine = get_spine(mask) # find initial spine (center axis)
    spine = smooth_spine(spine, lam=0.39, mu=-0.4, N=500, sigma=sigma_spine) # smooth spine
    spine = set_radius(spine, skin, radius)
    spine, vertebrae_idx = insert_vertebrae(spine, vertebrae_frac_pos) # insert positions on spine that correspond to fractional positions
    
    spine_segs = {}
    for i, (vi1, vi2) in enumerate(zip(vertebrae_idx[:-1], vertebrae_idx[1:])):
        spine_segs[i] = spine[vi1:vi2+1] # organize spine into dictionary holding individual segments
    
    radials = {} # to hold segments going from spine to skin
    skin_isxns = {} # to hold [indices or positions?] where radials intersect skin
    
    unique_angles = np.sort( np.unique( np.concatenate(angles) ) )
    
    angles_dict = {0: 0}
    for i, angle in enumerate(unique_angles):
        angles_dict[angle] = i+1
    
    num_angles = len(unique_angles)
    num_ribs = len(vertebrae_idx)
    num_rings = len(rings_frac_pos)
    ext2_idx = 2*(num_angles-1) + num_ribs + 1 # max radial index
    
    # first extension
    vb_xy1 = np.array([spine[0,1], spine[0,0]]) # position of spine end, (row, col) -> (x, y) (flipped)
    x1 = spine[:7,1] # x positions, end of spine
    y1 = spine[:7,0] # y positions, end of spine
    line1 = linregress(x1, y1) # find line that fits end of spine
    slope1 = line1.slope # slope of line 
    spine_seg1 = np.array([spine[0,1] - spine[7,1] , slope1*(spine[0,1] - spine[7,1])]) # reference vector for finding nearest angle intersection vector
    skin_isxns[0] = find_skin_intersection(vb_xy1, slope1, ref_segment=spine_seg1, skin=skin) # store where first radial intersects skin
    radials = add_radials(vb_xy1, skin_isxns[0][1], rings_frac_pos, radials, rad_idx=0) # determine radial lines that go from spine end to skin at specified angles
    
    # second extension 
    vb_xy2 = np.array([spine[-1,1], spine[-1,0]]) # position of other spine end
    x2 = spine[-7:,1]
    y2 = spine[-7:,0]
    line2= linregress(x2, y2)
    slope2 = line2.slope
    spine_seg2 = np.array([spine[-1,1] - spine[-7,1] , slope2*(spine[-1,1] - spine[-7,1])]) # reference vector for finding nearest angle intersection vector
    skin_isxns[ext2_idx] = find_skin_intersection(vb_xy2, slope2, ref_segment=spine_seg2, skin=skin)
    radials = add_radials(vb_xy2, skin_isxns[ext2_idx][1], rings_frac_pos, radials, rad_idx=ext2_idx)
    
    ref_seg1p = radials[0,0][1][::-1] - radials[0,0][0][::-1]
    ref_seg1m = radials[0,0][1][::-1] - radials[0,0][0][::-1]
    ref_seg2p = radials[ext2_idx,0][1][::-1] - radials[ext2_idx,0][0][::-1]
    ref_seg2m = radials[ext2_idx,0][1][::-1] - radials[ext2_idx,0][0][::-1]
    for ia, angle in enumerate(unique_angles):
        theta1 = np.arctan(slope1)
        theta1_pangle = theta1 + angle
        slope1_pangle = np.tan(theta1_pangle)
        skin_isxns[ia+1] = find_skin_intersection(vb_xy1, slope1_pangle, ref_seg1p, skin)
        radials = add_radials(vb_xy1, skin_isxns[ia+1][1], rings_frac_pos, radials, rad_idx=ia+1)
        ref_seg1p = radials[ia+1,0][1][::-1] - radials[ia+1,0][0][::-1]
        
        theta1_mangle = theta1 - angle
        slope1_mangle = np.tan(theta1_mangle)
        skin_isxns[-ia-1] = find_skin_intersection(vb_xy1, slope1_mangle, ref_seg1m, skin)
        radials = add_radials(vb_xy1, skin_isxns[-ia-1][1], rings_frac_pos, radials, rad_idx=-ia-1)
        ref_seg1m = radials[-ia-1,0][1][::-1] - radials[-ia-1,0][0][::-1]
        
        theta2 = np.arctan(slope2)
        theta2_pangle = theta2 + angle
        slope2_pangle = np.tan(theta2_pangle)
        skin_isxns[-ext2_idx+ia+1] = find_skin_intersection(vb_xy2, slope2_pangle, ref_seg2p, skin)
        radials = add_radials(vb_xy2, skin_isxns[-ext2_idx+ia+1][1], rings_frac_pos, radials, rad_idx=-ext2_idx+ia+1)
        ref_seg2p = radials[-ext2_idx+ia+1,0][1][::-1] - radials[-ext2_idx+ia+1,0][0][::-1]
        
        theta2_mangle = theta2 - angle
        slope2_mangle = np.tan(theta2_mangle)
        skin_isxns[ext2_idx-ia-1] = find_skin_intersection(vb_xy2, slope2_mangle, ref_seg2m, skin)
        radials = add_radials(vb_xy2, skin_isxns[ext2_idx-ia-1][1], rings_frac_pos, radials, rad_idx=ext2_idx-ia-1)
        ref_seg2m = radials[ext2_idx-ia-1,0][1][::-1] - radials[ext2_idx-ia-1,0][0][::-1]
        
        
    ref_segp = radials[ia+1,0][1][::-1] - radials[ia+1,0][0][::-1]
    ref_segm = radials[-ia-1,0][1][::-1] - radials[-ia-1,0][0][::-1]

    for iv, vbi in enumerate(vertebrae_idx[1:-1]):
        vb_xy = np.array([spine[vbi,1], spine[vbi,0]])
        lb = np.max([0, vbi-3])
        ub = np.min([vbi+4, len(spine)])
        x = spine[lb:ub, 1]
        y = spine[lb:ub, 0]
        line = linregress(x, y) ############## problem line
        if line.slope==0: slope = np.inf
        else: slope = -line.slope**-1
        skin_isxns[ia+1+iv+1] = find_skin_intersection(vb_xy, slope, ref_segp, skin)
        radials = add_radials(vb_xy, skin_isxns[ia+1+iv+1][1], rings_frac_pos, radials, rad_idx=ia+1+iv+1)
        
        skin_isxns[-ia-1-iv-1] = find_skin_intersection(vb_xy, slope, ref_segm, skin)
        radials = add_radials(vb_xy, skin_isxns[-ia-1-iv-1][1], rings_frac_pos, radials, rad_idx=-ia-1-iv-1)
        
    rings = {}
    
    # add rings on (+)-side
    for rad_idx in range(ext2_idx):
        for ring_idx in range(num_rings):
            rings[(rad_idx,rad_idx+1), ring_idx] = np.array([radials[rad_idx, ring_idx][1],
                                                             radials[rad_idx+1, ring_idx][1]])
    # add rings on (-)-side        
    for rad_idx in range(0,-ext2_idx+1,-1):
        for ring_idx in range(num_rings):
            rings[(rad_idx,rad_idx-1), ring_idx] = np.array([radials[rad_idx, ring_idx][1],
                                                             radials[rad_idx-1, ring_idx][1]])
    # seal (+)-side and (-)-side
    for ring_idx in range(num_rings):
        rings[(-ext2_idx+1,ext2_idx), ring_idx] = np.array([radials[-ext2_idx+1, ring_idx][1],
                                                             radials[ext2_idx, ring_idx][1]])
    
    # populate polygons
    polygons = {}
    for rad_idx in range(num_angles):
        for ring_idx in range(num_rings):
            polygons[(rad_idx,rad_idx+1), ring_idx] = np.concatenate((radials[rad_idx, ring_idx],
                                                                      radials[rad_idx+1, ring_idx][::-1]), axis=0)
            polygons[(-rad_idx,-rad_idx-1), ring_idx] = np.concatenate((radials[-rad_idx, ring_idx],
                                                                        radials[-rad_idx-1, ring_idx][::-1]), axis=0)
            polygons[(ext2_idx-rad_idx-1,ext2_idx-rad_idx), ring_idx] = np.concatenate((radials[ext2_idx-rad_idx-1, ring_idx],
                                                                                        radials[ext2_idx-rad_idx, ring_idx][::-1]), axis=0)
            
    for rad_idx in range(num_angles-1):
        for ring_idx in range(num_rings):
            polygons[(-ext2_idx+2+rad_idx, -ext2_idx+1+rad_idx), ring_idx] = np.concatenate((radials[-ext2_idx+2+rad_idx, ring_idx],
                                                                                        radials[-ext2_idx+1+rad_idx, ring_idx][::-1]), axis=0)
    for ring_idx in range(num_rings):
        polygons[(-ext2_idx+1, ext2_idx), ring_idx] = np.concatenate((radials[-ext2_idx+1, ring_idx],
                                                                     radials[ext2_idx, ring_idx][::-1]), axis=0)
        
    for rad_idx in range(num_angles, ext2_idx-num_angles):
        polygons[(rad_idx,rad_idx+1),0] = np.concatenate((radials[rad_idx,0],
                                                         radials[rad_idx+1,0][::-1],
                                                         spine_segs[rad_idx-num_angles][::-1]), axis=0)
        polygons[(-rad_idx,-rad_idx-1),0] = np.concatenate((radials[-rad_idx,0],
                                                         radials[-rad_idx-1,0][::-1],
                                                         spine_segs[rad_idx-num_angles][::-1]), axis=0)
        for ring_idx in range(1, num_rings):
            polygons[(rad_idx,rad_idx+1),ring_idx] = np.concatenate((radials[rad_idx,ring_idx],
                                                                     radials[rad_idx+1,ring_idx][::-1]), axis=0)
            polygons[(-rad_idx,-rad_idx-1),ring_idx] = np.concatenate((radials[-rad_idx,ring_idx],
                                                                       radials[-rad_idx-1,ring_idx][::-1]), axis=0)
    
    # link outermost ring to skin
    ####
    for rad_idx in range(ext2_idx):
        skin_seg = get_skin_segment(skin, skin_isxns[rad_idx][0], skin_isxns[rad_idx+1][0])
        if isinstance(skin_seg, str):
            polygons[(rad_idx, rad_idx+1), num_rings] = np.concatenate((radials[rad_idx, num_rings],
                                                                        radials[rad_idx+1, num_rings][::-1]), axis=0)
        else:
            polygons[(rad_idx, rad_idx+1), num_rings] = np.concatenate((radials[rad_idx, num_rings],
                                                                        skin_seg,
                                                                        radials[rad_idx+1, num_rings][::-1]), axis=0)
    ####
    
    for rad_idx in range(0, -ext2_idx+1, -1):
        skin_seg = get_skin_segment(skin, skin_isxns[rad_idx][0], skin_isxns[rad_idx-1][0])
        if isinstance(skin_seg, str):
            polygons[(rad_idx, rad_idx-1), num_rings] = np.concatenate((radials[rad_idx, num_rings], 
                                                                        radials[rad_idx-1, num_rings][::-1]), axis=0)
        else:
            polygons[(rad_idx, rad_idx-1), num_rings] = np.concatenate((radials[rad_idx, num_rings], 
                                                                        skin_seg,
                                                                        radials[rad_idx-1, num_rings][::-1]), axis=0)
            
    skin_seg = get_skin_segment(skin, skin_isxns[-ext2_idx+1][0], skin_isxns[ext2_idx][0])
    if isinstance(skin_seg, str):
        polygons[(-ext2_idx+1,ext2_idx), num_rings] = np.concatenate((radials[-ext2_idx+1, num_rings],
                                                                      radials[ext2_idx, num_rings][::-1]), axis=0)
    else:
        polygons[(-ext2_idx+1,ext2_idx), num_rings] = np.concatenate((radials[-ext2_idx+1, num_rings],
                                                                      skin_seg,
                                                                      radials[ext2_idx, num_rings][::-1]), axis=0)
        
    polygons = combine_polygons(polygons, angles, angles_dict, num_ribs, ext2_idx)
    
    for pg_key in polygons:
        polygons[pg_key] = polygon2numpy(polygons[pg_key])
    
    return {'skin': skin, 'skin_isxns': skin_isxns, 'spine': spine,'vertebrae_idx': vertebrae_idx,
            'radials': radials, 'rings': rings, 'polygons': polygons}

def get_long_axis_vals(values_dict, polygons_dict):
    """
    """
    max_ring_idx = 0
    max_rad_idx = 0
    for vk in values_dict.keys():
        (rad_idx1, rad_idx2), ring_idx = vk
        if ring_idx > max_ring_idx:
            max_ring_idx = ring_idx
        if rad_idx2 > max_rad_idx:
            max_rad_idx = rad_idx2

    axis_vals_left = []; axis_pos_left = []
    axis_vals_right = []; axis_pos_right = []

    for ring_idx in range(max_ring_idx, -1, -1):
        for vk in values_dict.keys():
            (rad_idx1, rad_idx2), ring_idx_curr = vk
            if rad_idx1 == 0 and rad_idx2 > 0 and ring_idx_curr == ring_idx:
                axis_vals_left.append(values_dict[vk])
                axis_pos_left.append(Polygon(polygons_dict[vk]).centroid.y)
                if ring_idx == 0:
                    left_idx = rad_idx2
            if rad_idx2 == max_rad_idx and rad_idx1 > 0 and ring_idx_curr == ring_idx:
                axis_vals_right.append(values_dict[vk])
                axis_pos_right.append(Polygon(polygons_dict[vk]).centroid.y)
                if ring_idx == 0:
                    right_idx = rad_idx1

    axis_vals_mid = []; axis_pos_mid = []

    while left_idx != right_idx:
        for vk in values_dict.keys():
            (rad_idx1, rad_idx2), ring_idx = vk
            if ring_idx == 0 and rad_idx1 == left_idx:
                axis_vals_mid.append(values_dict[vk])
                axis_pos_mid.append(Polygon(polygons_dict[vk]).centroid.y)
                left_idx = rad_idx2

    axis_vals = np.array(axis_vals_left + axis_vals_mid + axis_vals_right[::-1])
    axis_pos = np.array(axis_pos_left + axis_pos_mid + axis_pos_right[::-1])

    return axis_vals, axis_pos

def get_skel_coords(skel):
    """
    Get ordered coordinates of skeleton and cumulative lengths of skeleton
    """
    X,Y = np.meshgrid(range(skel.shape[1]), range(skel.shape[0]))
    weights = np.ones([3,3]); weights[1,1]=0
    skel_nconn = convolve(skel.astype('int'), weights=weights)
    skel_nconn[~skel] = 0
    x_ends = X[skel_nconn==1]
    y_ends = Y[skel_nconn==1]
    
    start_pt = [y_ends[0], x_ends[0]]
    end_pt = [y_ends[1], x_ends[1]]
    neighbors = [[1,0], [1,1], [0,1], [-1,1],
                 [-1, 0], [-1,-1], [0,-1], [1,-1]]
    
    curr_pt = start_pt
    last_pt = [-1, -1]

    skel_rc = np.zeros([skel.sum(), 2])
    skel_rc[0] = start_pt    
    
    i = 1
    while ~(curr_pt[0] == end_pt[0] and curr_pt[1] == end_pt[1]):
        for neighbor in neighbors:
            check_pt = [curr_pt[0] + neighbor[0], curr_pt[1] + neighbor[1]]
            if skel[check_pt[0], check_pt[1]] == True and ~(check_pt[0]==last_pt[0] and check_pt[1]==last_pt[1]):
                last_pt = curr_pt
                curr_pt = check_pt
                skel_rc[i] = curr_pt
                i+=1
                
    return skel_rc

def get_skin_segment(skin, idx0, idx1):
    """
    """
    max_idx = len(skin) - 1
    if idx1[0] > idx0[1]:
        len_simple = idx1[0] - idx0[1]
        len_0cross_rev = max_idx - idx1[1] + 1 + idx0[0]
        if len_simple <= len_0cross_rev:
            segment = skin[idx0[1]:idx1[0]+1]
        else:
            segment = np.concatenate((skin[idx1[1]:], skin[1:idx0[0]+1]), axis=0)[::-1]
    elif idx1[1] < idx0[0]:
        len_0cross = max_idx - idx0[1] + idx1[0]
        len_rev = idx0[0] - idx1[1]
        if len_rev <= len_0cross:
            segment = skin[idx1[1]:idx0[0]+1][::-1]
        else:
            segment = np.concatenate((skin[idx0[1]:], skin[1:idx1[0]+1]), axis=0)
    elif idx0[1] == idx1[0]:
        segment = np.array([skin[idx0[1]]])
    elif idx0[0] == idx1[1]:
        segment = np.array([skin[idx0[0]]])
    elif idx0[0] == idx1[0]:
        segment = 'radials intercept same segment' # this might not work
    
    return segment

def get_spine(mask):
    """
    """
    # mask_smooth = binary_erosion(mask, footprint=disk(3))
    # mask_smooth = binary_dilation(mask_smooth, footprint=disk(2))
    mask_smooth = binary_dilation(mask, footprint=disk(2))
    spine_image = skeletonize(mask_smooth)
    spine = get_skel_coords(spine_image)

    return spine

def insert_vertebrae(spine, vertebrae_frac_pos):
    """
    """
    cum_length = np.zeros(len(spine))
    for iv in range(1, len(spine)):
        length = ((spine[iv] - spine[iv-1])**2).sum()**0.5
        cum_length[iv] = cum_length[iv-1] + length
    cum_length_norm = cum_length / cum_length[-1]
    
    spine_seeds = np.zeros([len(vertebrae_frac_pos), 2])
    seed_idx = np.zeros(len(vertebrae_frac_pos))
    for ivfp, vfp in enumerate(vertebrae_frac_pos):
        idx = np.arange(len(cum_length_norm))[cum_length_norm > vfp][0].astype('int')
        S = (vfp - cum_length_norm[idx-1]) / (cum_length_norm[idx] - cum_length_norm[idx-1])
        spine_seeds[ivfp] = spine[idx-1] + S*(spine[idx] - spine[idx-1])
        # spine_seeds[ivfp] = (spine[idx] + spine[idx-1]) / 2
        seed_idx[ivfp] = idx
    new_spine = np.insert(spine, seed_idx.astype('int'), spine_seeds, axis=0)
    vertebrae_idx = seed_idx + np.arange(len(seed_idx))
    vertebrae_idx = np.concatenate(([0], vertebrae_idx, [len(new_spine)-1]))
    
    return new_spine, vertebrae_idx.astype('int')

def intersect(A,B,C,D):
    """
    Return true if line segments AB and CD intersect
    """
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def polygon_areas(polygons):
    """
    polygons: dict holding polygons
    """
    return {pkey: Polygon(polygon).area for pkey, polygon in polygons.items()}

def polygon_counts(coords, polygons, weights=None):
    # data['counts']      = polygon_counts(locs=coords, polygons=cg_dict['polygons'], weights=weights)
    """
    coords: row, col (unit is pixels) from csv or mat - not shapely points
    polygons: iterable, arrays of polygon coords row, col (in pixels) - not shapely polygons
    weights: default each loc counts for one 
    """
    # convert to shapely
    shapely_points = shapely.points(coords)
    shapely_polygons = shapely.polygons([shapely.linearrings(polygon) for polygon in polygons.values()])

    polygon_rtree = STRtree(shapely_polygons)
    counts = shapely_count(shapely_points, polygon_rtree, weights=weights)
       
    return {pkey: counts[i] for (i, pkey) in enumerate(polygons)}

def polygon2numpy(pg):
    """
    pg : shapely Polygon
    """
    return np.array([pg.boundary.coords.xy[0], pg.boundary.coords.xy[1]]).T

def prepare_diff_data(fits_obj, dT=0.04, px_sz=0.049):
    """
    Get D_app and appropriate weights from fits object
    """
    diff_dict = {}
    tracks = fits_obj['tracks'][:].T
    
    if tracks.ndim == 2:
        traj_filter = tracks[:-1:,3] == tracks[1:,3]

        diff_dict['loc1'] = tracks[:-1,1:3][traj_filter]
        diff_dict['loc2'] = tracks[1:,1:3][traj_filter]
        diff_dict['n_frames'] = tracks[1:,0][traj_filter] - tracks[:-1,0][traj_filter]

        diff_dict['D_app'] = ((diff_dict['loc2'] - diff_dict['loc1'])**2).sum(axis=1) / (4 * dT * diff_dict['n_frames']) * px_sz**2

        traj_ids = np.unique(tracks[:,3])

        diff_dict['weights'] = np.zeros(traj_filter.sum())
        for ti in traj_ids:
            ti_filt = tracks[:-1,3][traj_filter] == ti
            n_segs = ti_filt.sum()
            weight = n_segs**-1.
            diff_dict['weights'][ti_filt] = weight
    else:
        diff_dict = None
    
    return diff_dict

def prepare_fits(fits_obj, use_tracks=False, weight_tracks=True, gf_only=True, segment_range=None):
    """
    Prepare fits to go into heat mapping.
    -
    fits_obj : object produced by h5py reading of _fits.mat file
    use_tracks : whether to use tracked localizations, default use all localizations
    weight_tracks : whether to weigh localizations in tracks by the inverse of the trajectories length
    gf_only : use goodfits only, setting to False will include all guesses as well, ignored if using tracks
    segment_range : length 2 tuple, first entry min seg length, second entry max seg length
    """
    if use_tracks and segment_range is None:
        tracks = fits_obj['tracks'][:].T
        if tracks.ndim == 2:
            track_idx = np.unique(tracks[:,3])
            weights = np.ones(tracks.shape[0])
            if weight_tracks:
                for ti in track_idx:
                    weights[tracks[:,3]==ti] = (tracks[:,3]==ti).sum()**-1.
            locs = tracks[:,1:3] - 1 # subtract 1 to convert to Python coordinates
        else:
            locs = None; weights = None
    elif use_tracks and segment_range is not None:
        tracks = fits_obj['tracks'][:].T
        if tracks.ndim == 2:
            track_idx = np.unique(tracks[:,3])
            weights = np.ones(tracks.shape[0])
            for ti in track_idx:
                track = tracks[tracks[:,3]==ti]
                seg_lengths = (((track[1:,1:3] - track[:-1,1:3])**2).sum(axis=1))**0.5
                fr_filt = (track[1:,0] - track[:-1,0]) == 1
                len_filt = (seg_lengths >= segment_range[0]) & (seg_lengths < segment_range[1])
                tot_filt = fr_filt & len_filt
                w_trk = (track.shape[0] - 1)**-1 / 2
                weights_trk = np.zeros(track.shape[0])
                weights_trk[:-1] += tot_filt * w_trk
                weights_trk[1:] += tot_filt * w_trk
                weights[tracks[:,3]==ti] = weights_trk
            locs = tracks[:,1:3] - 1
        else:
            locs = None; weights = None
    else:
        locs = np.array([fits_obj['fits']['row'][0,:],
                         fits_obj['fits']['col'][0,:]]).T - 1 # subtract 1 to convert to Python coordinates
        gfs = fits_obj['fits']['goodfit'][0,:]
        if gf_only:
            locs = locs[gfs==1]
        weights = np.ones(locs.shape[0])

    return locs, weights

def set_radius(spine, skin, radius):
    """
    spine : intial spine, coordinates
    skin : outer boundary, coordinates
    radius : target distance from spine endpoints to skin (in pixels)
    ---
    returns new spine with endpoints radius distance from skin
    """
    vb_xy = np.array([spine[0,1], spine[0,0]])
    slope = (spine[0,0] - spine[1,0]) / (spine[0,1] - spine[1,1])

    ref_segment = np.array([spine[0,1] - spine[1,1], 
                            spine[0,0] - spine[1,0]])

    correct_idx, correct_intersection_0 = find_skin_intersection(vb_xy, slope, ref_segment, skin)

    vb_xy = np.array([spine[-1,1], spine[-1,0]])
    slope = (spine[-1,0] - spine[-2,0]) / (spine[-1,1] - spine[-2,1])

    ref_segment = np.array([spine[-1,1] - spine[-2,1],
                            spine[-1,0] - spine[-2,0]])

    correct_idx, correct_intersection_1 = find_skin_intersection(vb_xy, slope, ref_segment, skin)
    
    spine_long = np.concatenate(([correct_intersection_0[::-1]], spine, [correct_intersection_1[::-1]]))
    
    trimmed_len = 0
    curr_pt = spine_long[1]
    old_pt = spine_long[0]
    i = 2
    while trimmed_len < radius:
        trimmed_len += (((curr_pt - old_pt)**2).sum())**0.5
        if trimmed_len >= radius:
            dist_pts = (((curr_pt - old_pt)**2).sum())**0.5
            prev_dist = trimmed_len - dist_pts # distance to old_pt
            f_old = (trimmed_len - radius) / (trimmed_len - prev_dist)
            f_new = (radius - prev_dist) / (trimmed_len - prev_dist)
            new_end0 = f_old*old_pt + f_new*curr_pt
            new_spine = np.concatenate([[new_end0], spine_long[i-1:]])
        old_pt = curr_pt
        curr_pt = spine_long[i]
        i += 1
        
    trimmed_len = 0
    curr_pt = new_spine[-2]
    old_pt = new_spine[-1]
    i = -3
    while trimmed_len < radius:
        trimmed_len += (((curr_pt - old_pt)**2).sum())**0.5
        if trimmed_len >= radius:
            dist_pts = (((curr_pt - old_pt)**2).sum())**0.5
            prev_dist = trimmed_len - dist_pts # distance to old_pt
            f_old = (trimmed_len - radius) / (trimmed_len - prev_dist)
            f_new = (radius - prev_dist) / (trimmed_len - prev_dist)
            new_end1 = f_old*old_pt + f_new*curr_pt
            newer_spine = np.concatenate([new_spine[:i+1], [new_end1]])
        old_pt = curr_pt
        curr_pt = new_spine[i]
        i -= 1
        
    return newer_spine

def shapely_count(shapely_points, polygon_rtree, weights=None):
    """
    """
    query_results = polygon_rtree.query(shapely_points, predicate='intersects')

    n_polygons = len(polygon_rtree.geometries)

    if weights is None:
        counts = np.bincount(query_results[1,:], minlength=n_polygons)
    else:
        counts = np.bincount(query_results[1,:], minlength=n_polygons, weights=weights[query_results[0,:]])
             
    return counts

def smooth_skin(skin, lam=0.39, mu=-0.4, N=500, sigma=1.5):
    """
    """
    nv = len(skin) - 1
    I = np.identity(nv)
    W = np.zeros(I.shape)
    
    x = np.arange(-nv//2, nv//2)
    gk = gauss_kernel(x, sigma)
    gk = np.roll(gk, -nv//2)
    
    for v in range(nv):
        W[v,:] = gk
        gk = np.roll(gk, 1)

    K = I - W
    smooth_operator = matrix_power(np.matmul((I - mu*K), (I - lam*K)), N)
    skin_smooth = np.matmul(smooth_operator, skin[:-1]) # ignore second end
    skin_smooth = np.concatenate([skin_smooth,
                                  [skin_smooth[0]]]) # make ends match
    
    return skin_smooth

def smooth_spine(spine, lam=0.39, mu=-0.4, N=500, sigma=2):
    """
    ---
    Reference
    Taubin, G. 1995. Curve and surface smoothing without shrinkage. In: Proceedings of IEEE International Conference on Computer Vision. . pp. 852–857.
    DOI 10.1109/ICCV.1995.466848
    """
    nv = len(spine)
    I = np.identity(nv)
    W = np.zeros(I.shape)
    
    for v in range(nv):
        W[v,:] = gauss_kernel(np.arange(0-v, nv-v), sigma)

    K = I - W
    smooth_operator = matrix_power(np.matmul((I - mu*K), (I - lam*K)), N)
    spine_smooth = np.matmul(smooth_operator, spine)
    
    return spine_smooth

def sum_cell_maps(cell_data, val_key='counts'):

    init_cell_key = list(cell_data.keys())[0]

    data_sum = {poly_key: 0 for poly_key in cell_data[init_cell_key][val_key]} # initial dict to hold sums
    for cell_key in cell_data:
        for poly_key in data_sum:
            data_sum[poly_key] += cell_data[cell_key][val_key][poly_key] 

    return data_sum