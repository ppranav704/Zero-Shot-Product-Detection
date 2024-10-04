SAM 2: Segment Anything in Images and Videos

# SAM2 Object Tracking

This project implements object tracking between two images using SAM2 (Segment Anything Model 2). It uses pre-trained models to segment and track objects and provides visualization of the tracking results.

## Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/facebookresearch/sam2.git

   cd sam2 & pip install -e .

2. Install the required dependencies:

   pip install -r requirements.txt


## Getting Started

### Download Checkpoints

First, we need to download a model checkpoint. All the model checkpoints can be downloaded by running:

```bash
cd checkpoints && \
./download_ckpts.sh && \
cd ..
```

or individually from:

- [sam2.1_hiera_tiny.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt)
- [sam2.1_hiera_small.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt)
- [sam2.1_hiera_base_plus.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt)
- [sam2.1_hiera_large.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt)

(note that these are the improved checkpoints denoted as SAM 2.1; see [Model Description](#model-description) for details.)

Then SAM 2 can be used in a few lines as follows for image and video prediction.

### Usage

Run the sam2_tracking.py script to track objects between two images:

```bash
python sam2_tracking.py
```

