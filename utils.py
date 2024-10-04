import os
import glob
import numpy as np
import matplotlib.pyplot as plt


def create_if_not_exists(dirname):
    """
    Create a directory if it doesn't exist.
    
    Parameters:
        dirname (str): Directory name to check and create.
    """
    if not os.path.exists(dirname):
        os.mkdir(dirname)


def cleardir(tempfolder):
    """
    Clear all files from the directory.
    
    Parameters:
        tempfolder (str): Directory to clear.
    """
    filepaths = glob.glob(tempfolder + "/*")
    for filepath in filepaths:
        os.unlink(filepath)


def show_mask(mask, ax, obj_id=None, random_color=False):
    """
    Visualize the segmentation mask on the provided axis.
    
    Parameters:
        mask (numpy.array): Segmentation mask.
        ax (matplotlib.axes.Axes): Axis to plot the mask on.
        obj_id (int): Object ID for color differentiation.
        random_color (bool): Whether to assign random colors to masks.
    """
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])

    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
