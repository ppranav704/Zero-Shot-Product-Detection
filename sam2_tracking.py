import os
import shutil
import glob
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sam2.build_sam import build_sam2, build_sam2_video_predictor
from utils import create_if_not_exists, cleardir, show_mask


def extract_bounding_box_from_mask(mask_image_path):
    """
    Extract the bounding box coordinates from a binary mask image.

    Parameters:
        mask_image_path (str): Path to the mask image file.

    Returns:
        tuple: Bounding box coordinates (xmin, ymin, xmax, ymax).
    
    Raises:
        ValueError: If no object is found in the mask.
    """
    try:
        # Open the mask image
        mask = Image.open(mask_image_path).convert('L')  # Convert to grayscale

        # Convert the image to a numpy array
        mask_np = np.array(mask)

        # Find the non-zero regions (i.e., the object in the mask)
        non_zero_indices = np.nonzero(mask_np)

        if len(non_zero_indices[0]) == 0 or len(non_zero_indices[1]) == 0:
            raise ValueError(f"No object found in the mask: {mask_image_path}")

        # Calculate the bounding box from the non-zero indices
        ymin, ymax = np.min(non_zero_indices[0]), np.max(non_zero_indices[0])
        xmin, xmax = np.min(non_zero_indices[1]), np.max(non_zero_indices[1])

        return xmin, ymin, xmax, ymax

    except Exception as e:
        print(f"Error extracting bounding box: {e}")
        return None


def track_item_boxes(imgpath1, imgpath2, img1boxclasslist, model_cfg, checkpoint, tempfolder='./tempdir', visualize=True):
    """
    Track objects from imgpath1 to imgpath2 using bounding boxes and visualize results.
    
    Parameters:
        imgpath1 (str): Path to the first image where the object is known.
        imgpath2 (str): Path to the second image where the object is to be tracked.
        img1boxclasslist (list): List of bounding boxes in format [([xmin, xmax, ymin, ymax], object_id)].
        model_cfg (str): Configuration file for the SAM2 model.
        checkpoint (str): Path to the SAM2 model checkpoint.
        tempfolder (str): Temporary folder for intermediate files.
        visualize (bool): Whether to visualize the results.

    Returns:
        dict: Video segments with object masks.
    
    Raises:
        FileNotFoundError: If input images or model files are not found.
    """
    if not os.path.exists(imgpath1) or not os.path.exists(imgpath2):
        raise FileNotFoundError("One or both image paths do not exist.")

    try:
        # Load the SAM2 model
        sam2 = build_sam2(model_cfg, checkpoint, device='cuda')
        predictor_vid = build_sam2_video_predictor(model_cfg, checkpoint, device='cuda')

        # Prepare temp directory
        create_if_not_exists(tempfolder)
        cleardir(tempfolder)
        
        shutil.copy(imgpath1, os.path.join(tempfolder, "00000.jpg"))
        shutil.copy(imgpath2, os.path.join(tempfolder, "00001.jpg"))

        # Initialize video inference state
        inference_state = predictor_vid.init_state(video_path=tempfolder)
        predictor_vid.reset_state(inference_state)

        video_segments = {}

        # Loop through each object and propagate its mask in the video (between two images)
        for img1boxclass in img1boxclasslist:
            (xmin, xmax, ymin, ymax), objectnumint = img1boxclass
            box = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)

            # Add new points or bounding box to track the object
            _, out_obj_ids, out_mask_logits = predictor_vid.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=0,
                obj_id=objectnumint,
                box=box
            )

            # Propagate segmentation in the video
            for out_frame_idx, out_obj_ids, out_mask_logits in predictor_vid.propagate_in_video():
                video_segments[out_frame_idx] = {
                    out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                    for i, out_obj_id in enumerate(out_obj_ids)
                }

        # Visualize the result if needed
        if visualize:
            visualize_tracking_results(tempfolder, video_segments, xmin, ymin, xmax, ymax)

        return video_segments

    except Exception as e:
        print(f"Error during object tracking: {e}")
        return {}


def visualize_tracking_results(tempfolder, video_segments, xmin, ymin, xmax, ymax):
    """
    Visualizes tracking results by overlaying bounding boxes and masks on images.
    
    Parameters:
        tempfolder (str): Temporary folder storing images.
        video_segments (dict): Video segments containing object masks.
        xmin (int), ymin (int), xmax (int), ymax (int): Bounding box coordinates.
    
    Raises:
        IndexError: If there are no segments available for visualization.
    """
    try:
        # Show the first image with the bounding box
        fig, ax = plt.subplots()
        plt.title("Original Image Object:")
        
        ax.imshow(Image.open(os.path.join(tempfolder + "/00000.jpg")))
        
        rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                 linewidth=1,
                                 edgecolor='green',
                                 facecolor='none')
        
        ax.add_patch(rect)
        
        plt.show()

        # Show the second image with detected objects and segmentation masks
        plt.figure(figsize=(6, 4))
        
        plt.title("Detected Object in Test Image:")
        
        ax = plt.gca()
        
        ax.imshow(Image.open(os.path.join(tempfolder + "/00001.jpg")))
        
        for out_obj_id in video_segments.get(1).keys():
            show_mask(video_segments[1][out_obj_id], ax=ax)

    except IndexError as e:
         print(f"No segments available for visualization: {e}")
         return


if __name__ == "__main__":
    # Example paths for input images and mask
    firstimgpath = './data_2D/can_chowder_000001.jpg'
    secondimgpath = './data_2D/can_chowder_000002.jpg'
    mask_img_path = './data_2D/can_chowder_000001_1_gt.png'
    
    model_cfg = "./configs/sam2_hiera_t.yaml"  # Configuration file for SAM2
    checkpoint = "./checkpoints/sam2_hiera_tiny.pt"  # Model checkpoint

    # Extract the bounding box from the mask image
    bbox = extract_bounding_box_from_mask(mask_img_path)
    
    if bbox is not None:
      print(f"Bounding Box Coordinates: {bbox}")

      # Actual values from the mask
      img1boxclasslist = [(bbox, 1)]  # Use bounding box extracted from mask

      # Run object tracking between two images
      track_item_boxes(firstimgpath, secondimgpath, img1boxclasslist, model_cfg, checkpoint)
