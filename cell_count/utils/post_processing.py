from sklearn.neighbors import KDTree
import numpy as np
import random
import pandas as pd

def remove_duplicate_points(coord, r=10, n=50):
    
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
    coord_clean = np.array([i for j, i in enumerate(coord_subset) if j not in indices_to_drop])
    coord_removed = np.array([i for j, i in enumerate(coord_subset) if j in indices_to_drop])

    return coord_clean, coord_removed

