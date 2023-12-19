"""

Nomenclature
bimage : binary image
mid : midline
out : outline
"""
import numpy as np
import shapely as sl
from skimage.measure import find_contours

from .spideymaps_calculation import *

# default parameters for smoothening boundary
OUTPARS = dict(lam=0.39, mu=-0.4, N=500, sigma=1.5)

# default parameters for smoothening ridge
MIDPARS = dict(lam=0.39, mu=-0.4, N=500, sigma=2)


class Spideymap:


    def __init__(self, bimage: np.ndarray = None):
        """
        constructor

        Parameters
        ----------
        bimage 
            Binary image.
        """ 
        if bimage is not None:
            self.bimage = bimage
        else:
            # space for constructing default mask if none provided
            pass

    def make_grid(self,
                  n_cols=8,
                  col_edges=None,
                  n_shells=3,
                  shell_edges=None,
                  n_phi=None,
                  phi_edges=None,
                  radius=10,
                  outpars=OUTPARS, 
                  midpars=MIDPARS):
        """
        """
        self.radius = radius

        # shell edges overrides n_shells
        if shell_edges is not None:
            n_shells = len(shell_edges) + 1

        if phi_edges is not None:
            n_phi = (len(pe)+1 for pe in phi_edges)

        if n_phi is None: 
            n_phi = np.arange(1, n_shells+1)
        elif shell_edges is not None:
            n_phi = np.arange(1, len(shell_edges) + 2)
        

        # define outline
        out = find_contours(self.bimage, level=0.5)[0][:,::-1] # outline
        out = smooth_skin(out, **outpars)
        self.out = sl.LinearRing(out)

        # define midline
        mid = get_spine(self.bimage)[:,::-1] # midline, make sure in x,y-format
        mid = smooth_spine(mid, **midpars) # smooth spine
        self.mid = sl.LineString(mid)
        self.mid = extend_spine(self.mid, self.out)

        # build high resolution rings
        if shell_edges is None:
            self.ring_pos = np.linspace(0, 1, n_shells+1)[1:-1]
        else: 
            self.ring_pos = shell_edges 
        self.build_rings(n_cols=30, n_theta=12, midpt_offset=0.5)

        # initialize dict to store grid elements
        self.polygons = {}

        # build polygons in nonpolar region
        if col_edges is None:
            self.col_edges = np.linspace(self.radius, self.mid.length - self.radius, n_cols + 1)
        else:
            self.col_edges = np.array([self.radius, 
                                      *(np.array(col_edges) * (self.mid.length - 2*self.radius) + self.radius), 
                                      self.mid.length - self.radius])
        self.build_nonpolar_polygons()

        # build polygons in polar regions
        self.phi_list = [np.linspace(0, np.pi, n+1) for n in n_phi]
        self.build_polar_polygons()

    def build_rings(self, n_cols=30, n_theta=12, midpt_offset=0.5):
        """

        """
        dists = np.linspace(self.radius, self.mid.length - self.radius, n_cols + 1)

        # sources of radial vectors on midline
        midpts = sl.line_interpolate_point(
            self.mid, 
            dists
            )

        # points to 
        midpts_l = sl.line_interpolate_point(
                    self.mid, 
                    (dists - midpt_offset).clip(0)
                    )

        midpts_r = sl.line_interpolate_point(
                    self.mid, 
                    (dists + midpt_offset).clip(self.mid.length)
                    )
        
        rads_north = []  # polar region north end of ridgeline
        rads_south = []  # polar region south end of ridgeline
        rads_top = []    # above ridgeline
        rads_bottom = [] # below ridgeline

        # construct polar radials
        for theta in np.linspace(-np.pi/2, np.pi/2, n_theta):
            rads_north.append(build_rad(midpts_l[0], midpts_r[0], self.out, origin=midpts[0], theta=theta+np.pi))
            rads_south.append(build_rad(midpts_l[-1], midpts_r[-1], self.out, origin=midpts[-1], theta=theta))

        # construct radials extending from midline
        for i in range(1, len(midpts)-1):
            # radial above midline
            rads_top.append(
                build_rad(midpts_l[i], midpts_r[i], self.out, origin=midpts[i], theta=np.pi/2)
            )
            # radials below midline
            rads_bottom.append(
                build_rad(midpts_l[i], midpts_r[i], self.out, origin=midpts[i], theta=-np.pi/2)
            )

        # put all radials together in the right order
        rads_sorted = np.array([*rads_north[::-1],
                        *rads_top,
                        *rads_south[::-1],
                        *rads_bottom[::-1]])

        # link radials to build rings
        rings = []
        for rp in self.ring_pos:
            ring = sl.LinearRing(
                (rad.line_interpolate_point(rp, normalized=True) for rad in rads_sorted)
            )
            rings.append(ring)

        self.rings = rings

    def build_nonpolar_polygons(self, midpt_offset=0.5):
        """
        Build polygons for nonpolar region.
        """
        dists = self.col_edges

        midpts = sl.line_interpolate_point(
                    self.mid, 
                    dists
                    )

        midsegs = {i+1:
                   sl.ops.substring(self.mid, start_dist=d0, end_dist=d1)
                   for i, (d0, d1)
                   in enumerate(zip(dists[:-1], dists[1:]))}

        midpts_l = sl.line_interpolate_point(self.mid, (dists - midpt_offset).clip(0))

        midpts_r = sl.line_interpolate_point(
                    self.mid, 
                    (dists + midpt_offset).clip(self.mid.length)
                    )
        
        self._midpts = midpts
        self._midpts_l = midpts_l
        self._midpts_r = midpts_r

        # build polygons above midline
        rad0 = build_rad(midpts_l[0], midpts_r[0], bnd=self.out, origin=midpts[0])
        for i_l, midseg in midsegs.items():
            rad1 = build_rad(midpts_l[i_l], midpts_r[i_l], bnd=self.out, origin=midpts[i_l])
            arc0 = midseg

            for i_r, ring in enumerate(self.rings):
                pt0 = sl.intersection(rad0, ring)
                pt1 = sl.intersection(rad1, ring)
                arc1 = get_arc(pt0, pt1, ring)

                self.polygons[i_r,i_l,0] = sl.Polygon(
                    (*arc0.coords, 
                    *arc1.coords[::-1],)
                    )
                arc0 = arc1

            i_r += 1
            arc1 = get_arc(sl.Point(rad0.coords[-1]), sl.Point(rad1.coords[-1]), self.out)
            self.polygons[i_r,i_l,0] = sl.Polygon(
                    (*arc0.coords, 
                    *arc1.coords[::-1],)
                    )
            
            rad0 = rad1

        # build polygons below midline
        rad0 = build_rad(midpts_l[0], midpts_r[0], bnd=self.out, origin=midpts[0], theta=-np.pi/2)
        for i_l, midseg in midsegs.items():
            rad1 = build_rad(midpts_l[i_l], midpts_r[i_l], bnd=self.out, origin=midpts[i_l], theta=-np.pi/2)

            arc0 = midseg

            for i_r, ring in enumerate(self.rings):
                pt0 = sl.intersection(rad0, ring)
                pt1 = sl.intersection(rad1, ring)
                arc1 = get_arc(pt0, pt1, ring)

                self.polygons[i_r,i_l,-1] = sl.Polygon(
                    (*arc0.coords, 
                    *arc1.coords[::-1],)
                    )
                arc0 = arc1

            i_r += 1
            arc1 = get_arc(sl.Point(rad0.coords[-1]), sl.Point(rad1.coords[-1]), self.out)
            self.polygons[i_r,i_l,-1] = sl.Polygon(
                    (*arc0.coords, 
                    *arc1.coords[::-1],)
                    )
            
            rad0 = rad1

    def build_polar_polygons(self,):
        """
        """
        midpts = self._midpts
        midpts_l = self._midpts_l
        midpts_r = self._midpts_r
        all_rings = [*self.rings, self.out]

        ## first pole
        # innermost ring, anchored to end of midline
        phi = self.phi_list[0]
        rad0 = build_rad(midpts_l[0], midpts_r[0], bnd=all_rings[0], origin=midpts[0], theta=np.pi/2+phi[0])
        for i_p, p in enumerate(phi[1:]):
            rad1 = build_rad(midpts_l[0], midpts_r[0], bnd=all_rings[0], origin=midpts[0], theta=np.pi/2 + p)
            arc = get_arc(sl.Point(rad0.coords[-1]), sl.Point(rad1.coords[-1]), bnd=all_rings[0])
            self.polygons[0,0,i_p+1] = sl.Polygon((*midpts[0].coords, *arc.coords))
            rad0 = rad1

        # remaining rings
        for i_r, phi in enumerate(self.phi_list[1:]):
            rad0 = build_rad(midpts_l[0], midpts_r[0], bnd=all_rings[i_r+1], origin=midpts[0], theta=np.pi/2 + phi[0])
            for i_p, p in enumerate(phi[1:]):
                rad1 = build_rad(midpts_l[0], midpts_r[0], bnd=all_rings[i_r+1], origin=midpts[0], theta=np.pi/2 + p)
                pt0 = sl.intersection(rad0, all_rings[i_r])
                pt1 = sl.intersection(rad1, all_rings[i_r])
                arc0 = get_arc(pt0, pt1, bnd=all_rings[i_r])
                arc1 = get_arc(sl.Point(rad0.coords[-1]), sl.Point(rad1.coords[-1]), bnd=all_rings[i_r+1])
                self.polygons[i_r+1,0,i_p+1] = sl.Polygon((*arc0.coords, *arc1.coords[::-1]))
                rad0 = rad1

        ## second pole
        i_l_max = len(self._midpts) + 1
        phi = self.phi_list[0]
        rad0 = build_rad(midpts_l[-1], midpts_r[-1], bnd=all_rings[0], origin=midpts[-1], theta=np.pi/2 - phi[0])
        for i_p, p in enumerate(phi[1:]):
            rad1 = build_rad(midpts_l[-1], midpts_r[-1], bnd=all_rings[0], origin=midpts[-1], theta=np.pi/2 - p)
            arc = get_arc(sl.Point(rad0.coords[-1]), sl.Point(rad1.coords[-1]), bnd=all_rings[0])
            self.polygons[0,i_l_max,i_p+1] = sl.Polygon((*midpts[-1].coords, *arc.coords))
            rad0 = rad1

        # remaining rings
        for i_r, phi in enumerate(self.phi_list[1:]):
            rad0 = build_rad(midpts_l[-1], midpts_r[-1], bnd=all_rings[i_r+1], origin=midpts[-1], theta=np.pi/2 - phi[0])
            for i_p, p in enumerate(phi[1:]):
                rad1 = build_rad(midpts_l[-1], midpts_r[-1], bnd=all_rings[i_r+1], origin=midpts[-1], theta=np.pi/2 - p)
                pt0 = sl.intersection(rad0, all_rings[i_r])
                pt1 = sl.intersection(rad1, all_rings[i_r])
                arc0 = get_arc(pt0, pt1, bnd=all_rings[i_r])
                arc1 = get_arc(sl.Point(rad0.coords[-1]), sl.Point(rad1.coords[-1]), bnd=all_rings[i_r+1])
                self.polygons[i_r+1,i_l_max,i_p+1] = sl.Polygon((*arc0.coords, *arc1.coords[::-1]))
                rad0 = rad1


def rotate_and_scale(linestring, theta, origin=None, neworigin=None, scale=1):
    """
    Parameters
    ----------
    linestring : shapely.LineString
    origin : shapely.point
    newstart : shapely.point

    Returns
    -------
    sl.LineString
    """
    # if no origin provided, use first coordinate in linestring as origin
    if origin is None:
        origin = sl.Point(linestring.coords[0])

    # bring back to original position if new origin not provided
    if neworigin is None:
        neworigin = origin

    # define rotation matrix
    rotmat = np.array([[np.cos(theta), -np.sin(theta)],
                       [np.sin(theta),  np.cos(theta)]])
    
    # shift
    v = np.asarray(linestring.coords) - np.asarray(origin.coords) 

    # rotate
    v = np.matmul(rotmat, v.T).T 

    # scale
    v = scale * v 

    # shift to new origin
    v = v + np.asarray(neworigin.coords) 

    return sl.LineString(v)


def build_rad(pt0, pt1, bnd, origin=None, theta=np.pi/2):
    """
    """
    if origin is None:
        origin = pt0
    rdgseg = sl.LineString((pt0, pt1))

    rad= rotate_and_scale(rdgseg,
                        theta=theta, 
                        origin=pt0,
                        neworigin=origin,
                        scale=5)
    
    isxn = sl.intersection(rad, bnd)

    rad = sl.LineString((rad.coords[0], isxn.coords[0]))

    return rad


def get_arc(pt0, pt1, bnd):
    """
    """
    d0 = sl.line_locate_point(bnd, pt0, normalized=True)
    d1 = sl.line_locate_point(bnd, pt1, normalized=True)

    delta = d1 - d0
    ordered = delta > 0
    cross_nick = np.abs(delta) > 0.5

    if ordered == True and cross_nick == False:
        arc = sl.ops.substring(bnd, d0, d1, normalized=True)
    elif ordered == False and cross_nick == False:
        arc = sl.ops.substring(bnd, d0, d1, normalized=True)
    elif ordered == True and cross_nick == True:
        arc1 = sl.ops.substring(bnd, d0, 0, normalized=True)
        arc2 = sl.ops.substring(bnd, 1, d1, normalized=True)
        arc = sl.LineString((*arc1.coords, *arc2.coords))
    elif ordered == False and cross_nick == True:
        arc1 = sl.ops.substring(bnd, d0, 1, normalized=True)
        arc2 = sl.ops.substring(bnd, 0, d1, normalized=True)
        arc = sl.LineString((*arc1.coords, *arc2.coords))

    return arc


        

        










