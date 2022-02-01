import pandas as pd
import random
from sklearn.neighbors import KDTree
import numpy as np
from scipy import ndimage

def pred_to_xy(fg, bo, pred_thr=0.5, n_elem_thr=10):
    """Uses prediction images for foreground and boundary to determine cell center co-ordinates.

    Args:
        fg: foreground prediction image
        bo: boundary prediction image
        pred_thr: binarization threshold
        n_elem_thr: connected component size threshold

    Returns:
        com: center of mass co-ordinates
    """

    strel = [[1, 1, 1],
             [1, 1, 1],
             [1, 1, 1]]

    thr_im = fg - bo >= pred_thr

    #center of mass for connected components:
    lbl, num_features = ndimage.label(thr_im, structure=strel)
    com = ndimage.measurements.center_of_mass(
        thr_im, labels=lbl, index=np.arange(1, num_features+1))
    com = np.array(com)

    #number of elements in each connected component
    n_elem = np.zeros((num_features,))
    for i, l in enumerate(range(1, num_features+1)):
        n_elem[i] = np.sum(lbl == l)

    if com.size>0:
        com = com[n_elem > n_elem_thr, :]
        n_elem = n_elem[n_elem > n_elem_thr]
    else:
        com = np.empty((0, 2))
        n_elem = np.empty((0, ))
    return com, n_elem


def remove_duplicate_points_postprocessing(coord, r=10, n=50):
    """
    Removes points that redundantly mark the same cell as another point. Takes in the dataframe
    containing x-coordinates, y-coordinates, and the sizes of corresponding cells for each point
    and (optionally) the radius used to identify points that are close together. Returns arrays
    of points that were not removed and points that were removed.
    """

    df_size_thresholded = coord['n'] >= n
    coord_subset = coord[df_size_thresholded].to_numpy()
    kdt = KDTree(coord_subset[:,:2], leaf_size=30, metric='euclidean')
    close_points = []
    close_set =[]
    indices_to_drop = []
    close = kdt.query_radius(coord_subset[:,:2], r)
    for i in close:
        if i.size > 1:
            close_points.append(list(i))
    for j in close_points:
        if not j in close_set:
            close_set.append(j)
    #removing points that are too close together
    for k in close_set:
        while len(k)>1:
            index = random.randrange(len(k))
            indices_to_drop.append(k[index])
            del k[index]
    header = ['x','y','n']
    coord_clean = np.array([i for j, i in enumerate(coord_subset) if j not in indices_to_drop])
    coord_removed = np.array([i for j, i in enumerate(coord_subset) if j in indices_to_drop])
    coord_clean = np.vstack([header, coord_clean])
    if coord_removed.size==0:
        pass
    else:
        coord_removed = np.vstack([header, coord_removed])

    return coord_clean, coord_removed

def identify_yellow_points(coord_r, coord_g, dist=8):
    red_zeroes = np.zeros((coord_r.shape[0],1))
    green_ones = np.ones((coord_g.shape[0],1))

    #0 for red, 1 for green
    coord_r = np.append(coord_r, red_zeroes, axis=1)
    coord_g = np.append(coord_g, green_ones, axis=1)
    coord_r = np.delete(coord_r, (0), axis=0)
    coord_g = np.delete(coord_g, (0), axis=0)
    coord_full = np.append(coord_r, coord_g, axis=0)

    kdt = KDTree(coord_full[:,:2], leaf_size=20, metric='euclidean')
    close_red_dist, close_red_ind = kdt.query(coord_full[:len(coord_r),:2], k=2)
    close_red_ind = np.sort(close_red_ind)

    close_indices = []
    indices_to_drop =[]
    for i, j in zip(close_red_ind, close_red_dist):
        if j[1]<=dist:
            if not list(i) in close_indices:
                coord_full[i[0]][3]=2
                indices_to_drop.append(i[1])
                close_indices.append(list(i))

    coord_clean = np.array([i for j, i in enumerate(coord_full) if j not in indices_to_drop])

    #header = ['x','y','n']
    red_cells = coord_clean[coord_clean[:, 3] == 0][:,:3]
    green_cells = coord_clean[coord_clean[:, 3] == 1][:,:3]
    yellow_cells = coord_clean[coord_clean[:, 3] == 2][:,:3]
    #red_cells = np.vstack([header, red_cells])
    #green_cells = np.vstack([header, green_cells])
    #yellow_cells = np.vstack([header, yellow_cells])

    return red_cells, green_cells, yellow_cells
