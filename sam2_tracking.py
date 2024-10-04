import os
import shutil
import glob
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from utils import create_if_not_exists, cleardir, show_mask


# Function to track objects between two images using bounding boxes
def track_item_boxes(imgpath1, imgpath2, img1boxclasslist, model_cfg, checkpoint, tempfolder='./tempdir', visualize=True):
    """
    Track objects from imgpath1 to imgpath2 using bounding boxes and visualize results.
    
    Parameters:
        imgpath1 (str): Path to the first image where the object is known.
        imgpath2 (str): Path to the second image where the object is to be tracked.
        img1boxclasslist (list): List of bounding boxes in the format [([xmin, xmax, ymin, ymax], object_id)].
        model_cfg (str): Configuration file for the SAM2 model.
        checkpoint (str): Path to the SAM2 model checkpoint.
        tempfolder (str): Temporary folder for intermediate files.
        visualize (bool): Whether to visualize the results.

    Returns:
        dict: Video segments with object masks.
    """
    try:
        # Load the SAM2 model
        sam2 = build_sam2(model_cfg, checkpoint, device='cuda')
        predictor_vid = build_sam2_video_predictor(model_cfg, checkpoint, device='cuda')

        # Prepare temp directory
        create_if_not_exists(tempfolder)
        cleardir(tempfolder)
        shutil.copy(imgpath1, tempfolder + "/00000.jpg")
        shutil.copy(imgpath2, tempfolder + "/00001.jpg")

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
    Visualizes the tracking results by overlaying bounding boxes and masks on images.
    
    Parameters:
        tempfolder (str): Temporary folder storing images.
        video_segments (dict): Video segments containing object masks.
        xmin (int), ymin (int), xmax (int), ymax (int): Bounding box coordinates.
    """
    try:
        # Show the first image with the bounding box
        fig, ax = plt.subplots()
        plt.title("Original Image Object:")
        ax.imshow(Image.open(tempfolder + "/00000.jpg"))
        rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=1, edgecolor='green', facecolor='none')
        ax.add_patch(rect)
        plt.show()

        # Show the second image with the detected object and segmentation mask
        plt.figure(figsize=(6, 4))
        plt.title("Detected Object in Test Image:")
        plt.imshow(Image.open(tempfolder + "/00001.jpg"))
        for out_obj_id, out_mask in video_segments[1].items():
            show_mask(out_mask, plt.gca(), obj_id=out_obj_id)

    except Exception as e:
        print(f"Error in visualization: {e}")


if __name__ == "__main__":
    # Example paths for the input images and bounding box coordinates
    firstimgpath = '/content/data_2D/can_chowder_000001.jpg'
    secondimgpath = '/content/data_2D/can_chowder_000002.jpg'
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"  # Configuration file for SAM2
    checkpoint = "/content/sam2/checkpoints/sam2.1_hiera_large.pt"  # Model checkpoint

    # Example bounding box coordinates (xmin, xmax, ymin, ymax)
    img1boxclasslist = [([100, 200, 100, 200], 1)]  # Replace with actual values

    # Run the object tracking between two images
    track_item_boxes(firstimgpath, secondimgpath, img1boxclasslist, model_cfg, checkpoint)
