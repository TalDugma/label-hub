# Label Hub - SAM2 Interactive Video Labeler

A lightweight, interactive annotation tool for video object segmentation powered by [Meta's Segment Anything Model 2 (SAM 2)](https://github.com/facebookresearch/sam2). This tool provides a Matplotlib-based GUI designed to run within Jupyter Notebooks, allowing users to label objects in video frames, propagate masks across the video timeline, and save the results for machine learning workflows.

## Features

- **Interactive Labeling**: Click to add positive (target) or negative (exclude) points to objects.
- **Multi-Object Support**: accurate mask management for multiple unique object IDs within the same video.
- **SAM 2 Integration**: Leverages SAM 2 for state-of-the-art zero-shot object selection and video tracking/propagation.
- **Notebook Native**: Built with `ipympl` (matplotlib widget) for seamless integration into Jupyter Lake/Lab/Notebook environments.
- **Automated Frame Extraction**: Automatically extracts frames from input video files using FFmpeg (via `imageio`/`ffmpeg`).
- **Mask Persistence**: Saves binary masks as `.npy` files for efficient storage and loading.

## Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd label-hub
```

### 2. Install Dependencies
Install the required Python packages. Note that you must have `ffmpeg` installed on your system for video processing.

```bash
pip install -r requirements.txt
```

### 3. Install SAM 2
This project requires the `sam2` library. As it is an evolving research project, please install it directly from the source:

```bash
git clone https://github.com/facebookresearch/sam2.git
cd sam2
pip install -e .
```

*Note: You will also need to download a SAM 2 checkpoint (e.g., `sam2.1_hiera_tiny.pt`, `sam2.1_hiera_large.pt`) and the corresponding model configuration file.*

## Usage

This tool is designed to be run inside a Jupyter Notebook.

### Basic Setup

```python
%matplotlib widget 
from main import run_app

# Configuration
checkpoint = "/path/to/sam2_checkpoint.pt"
cfg = "configs/sam2.1/sam2.1_hiera_t.yaml" # Relative to sam2 library or absolute path
video_path = "/path/to/video.mp4"

# Launch the App
app = run_app(video_path, checkpoint, cfg, log_file="labeling.log")
```

### GUI Controls

- **Frame Slider**: Navigate through the video timeline.
- **Label Type (Radio)**: Switch between **Positive** (add object) and **Negative** (remove area) clicks.
- **ID Controls**:
    - **< Prev ID / Next ID >**: Switch between different object masks. ID `-1` allows viewing all masks simultaneously.
- **Clear Points**: Remove all click prompts for the current frame and ID.
- **Submit/Track**: Propagate the masks from the current frame to the rest of the video using SAM 2's video tracker.
- **Save Masks**: Save the generated masks to the output directory (default: `<video_name>_masks`).
- **Reset All**: Clears all states, masks, and points to start over.

## Outputs

Masks are saved in the defined output directory with the following naming convention:
```
<frame_index>_<mask_id>.npy
```
Each `.npy` file contains a binary mask (uint8) for a specific object on a specific frame.

## Project Structure

- `main.py`: Main entry point and GUI implementation (`MatplotlibGUI`).
- `core.py`: Core logic wrapping SAM 2 interactions (`PromptGUI`).
- `utils.py`: Helper functions for image processing and frame extraction.
- `gradio_gui.py`: Legacy Gradio implementation (alternative interface).

## Requirements

- Python 3.10+
- PyTorch (CUDA supported for GPU acceleration)
- See `requirements.txt` for full list.
