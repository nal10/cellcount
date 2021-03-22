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
