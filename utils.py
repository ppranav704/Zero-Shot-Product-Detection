import os
import glob
import numpy as np
import matplotlib.pyplot as plt


def create_if_not_exists(dirname):
    """
    Create a directory if it doesn't exist.
    
    Parameters:
       dirname (str): Directory name to check and create.
    
    Raises:
       OSError: If there is an error creating the directory.
    """
    try:
       if not os.path.exists(dirname):
           os.mkdir(dirname)
           print(f"Directory created: {dirname}")
       else:
           print(f"Directory already exists: {dirname}")
           
    except OSError as e:
       print(f"Error creating directory {dirname}: {e}")


def cleardir(tempfolder):
    """
    Clear all files from a directory.
    
    Parameters:
       tempfolder (str): Directory to clear.

   Raises:
       OSError: If there is an error deleting files.
   """
   try:
       filepaths = glob.glob(os.path.join(tempfolder + "/*"))
       for filepath in filepaths:
           os.unlink(filepath)
       print(f"Cleared directory: {tempfolder}")

   except OSError as e:
       print(f"Error clearing directory {tempfolder}: {e}")


def show_mask(mask, ax, obj_id=None):
   """
   Visualize a segmentation mask on a provided axis.

   Parameters:
       mask (numpy.array): Segmentation mask.
       ax (matplotlib.axes.Axes): Axis to plot the mask on.
       obj_id (int): Object ID for color differentiation.

   Raises:
       ValueError: If provided mask dimensions are invalid.
   """
   try:
       cmap = plt.get_cmap("tab10")
       cmap_idx = 0 if obj_id is None else obj_id % cmap.N  # Ensure valid index for colormap

       color = np.array([*cmap(cmap_idx)[:3], 0.6])
       
       h,w = mask.shape[-2:]
       
       if h <= 0 or w <= 0:
           raise ValueError("Invalid mask dimensions.")

       mask_image = mask.reshape(h,w,1) * color.reshape(1,1,-1)
       
       ax.imshow(mask_image)

   except Exception as e:
       print(f"Error showing mask: {e}")
