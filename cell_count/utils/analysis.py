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


def remove_duplicate_points(coord, r=10, cell_size=50):
    """
    Removes points that redundantly mark the same cell as another point. Takes in the file path
    to a .csv containing x-coordinates, y-coordinates, and the sizes of corresponding cells for
    each point and (optionally) the radius used to identify
    points that are close together. Returns a list containing a numpy array of points that were
    not removed from the input .csv and a numpy array of points that were removed from the input
    .csv file. Author: Cameron Trader
    """

    #removing points identifying cells smaller than 50 pixels
    coord_df = pd.DataFrame(coord)
    coord_df.columns = ['y', 'x', 'size']
    size_over_50 = coord_df['size'] >= cell_size
    coord_df = coord_df[size_over_50]
    coord_s50 = coord_df.to_numpy()

    #finding points that are within a certain distance of each other
    kdt = KDTree(coord_s50[:, :2], leaf_size=30, metric='euclidean')
    close_points = []
    close_set = []
    indices_to_drop = []
    close = kdt.query_radius(coord_s50[:, :2], r)
    for i in close:
        if i.size > 1:
            close_points.append(list(i))
    for i in close_points:
        if not i in close_set:
            close_set.append(i)
    #removing points that are too close together
    for i in close_set:
        while len(i) > 1:
            index = random.randrange(len(i))
            indices_to_drop.append(i[index])
            del i[index]

    coord_s50_clean = np.array(
        [i for j, i in enumerate(coord_s50) if j not in indices_to_drop])
    coord_s50_removed = np.array(
        [i for j, i in enumerate(coord_s50) if j in indices_to_drop])
    return coord_s50_clean, coord_s50_removed
