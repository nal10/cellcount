from sklearn.neighbors import KDTree
import numpy as np
import random
import pandas as pd

def remove_duplicate_points(coord, r=10, cell_size=50):
    """
    Removes points that redundantly mark the same cell as another point. Takes in the file path
    to a .csv containing x-coordinates, y-coordinates, and the sizes of corresponding cells for
    each point and (optionally) the radius used to identify
    points that are close together. Returns a list containing a numpy array of points that were
    not removed from the input .csv and a numpy array of points that were removed from the input
    .csv file.
    """
        
    #removing points identifying cells smaller than 50 pixels
    coord_df = pd.DataFrame(coord)
    coord_df.columns = ['y', 'x', 'size']
    size_over_50 = coord_df['size'] >= cell_size
    coord_df = coord_df[size_over_50]
    coord_s50 = coord_df.to_numpy()
    
    #finding points that are within a certain distance of each other
    kdt = KDTree(coord_s50[:,:2], leaf_size=30, metric='euclidean')
    close_points = []
    close_set =[]
    indices_to_drop = []
    close = kdt.query_radius(coord_s50[:,:2], r)
    for i in close:
        if i.size > 1:
            close_points.append(list(i))
    for i in close_points:
        if not i in close_set:
            close_set.append(i)
    #removing points that are too close together
    for i in close_set:
        while len(i)>1:
            index = random.randrange(len(i))
            indices_to_drop.append(i[index])
            del i[index]
    
    coord_s50_clean = np.array([i for j, i in enumerate(coord_s50) if j not in indices_to_drop])
    coord_s50_removed = np.array([i for j, i in enumerate(coord_s50) if j in indices_to_drop])
    return coord_s50_clean, coord_s50_removed;

