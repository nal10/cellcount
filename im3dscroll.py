"""im3dscroll version that that uses left and right keys move through the 1st index in the 3d volume."""
'''
# Example:
% matplotlib qt
import numpy as np
import im3dscroll as I
vol=np.random.rand(10,100,100)
I.im3dscroll(vol)
'''

def previous_slice(ax):
    """Go to the previous slice."""
    volume = ax.volume
    ax.index = (ax.index - 1) % volume.shape[0]  # wrap around using %
    ax.images[0].set_array(volume[ax.index])
    ax.set_title('current plane' + str(ax.index))


def next_slice(ax):
    """Go to the next slice."""
    volume = ax.volume
    ax.index = (ax.index + 1) % volume.shape[0]
    ax.images[0].set_array(volume[ax.index])
    ax.set_title('current plane: ' + str(ax.index))


def process_key(event):
    fig = event.canvas.figure
    ax = fig.axes[0]
    if event.key == 'left':
        previous_slice(ax)
    elif event.key == 'right':
        next_slice(ax)
    fig.canvas.draw()


def im3dscroll(volume):
    #import PyQt5 as qt
    #import matplotlib as mpl
    import matplotlib.pyplot as plt
    #mpl.rcParams['keymap.back'].remove('left')
    #mpl.rcParams['keymap.forward'].remove('right')
    fig, ax = plt.subplots()
    ax.volume = volume
    ax.index = 0
    ax.imshow(volume[ax.index],cmap = 'gray',vmin = 0,vmax = 1)
    ax.set_title('current plane' + str(ax.index))
    fig.canvas.mpl_connect('key_press_event', process_key)
    plt.show(block = False)
