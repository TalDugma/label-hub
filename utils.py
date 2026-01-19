import os
import datetime
import subprocess
import colorsys
import numpy as np
import cv2
from loguru import logger as guru

def isimage(p):
    ext = os.path.splitext(p.lower())[-1]
    return ext in [".png", ".jpg", ".jpeg"]

def get_hls_palette(n_colors: int, lightness: float = 0.5, saturation: float = 0.7) -> np.ndarray:
    """
    returns (n_colors, 3) tensor of colors,
        first is black and the rest are evenly spaced in HLS space
    """
    hues = np.linspace(0, 1, int(n_colors) + 1)[1:-1]  # (n_colors - 1)
    palette = [(0.0, 0.0, 0.0)] + [
        colorsys.hls_to_rgb(h_i, lightness, saturation) for h_i in hues
    ]
    return (255 * np.asarray(palette)).astype("uint8")

def compose_img_mask(img, color_mask, fac: float = 0.5):
    out_f = fac * img / 255 + (1 - fac) * color_mask / 255
    out_u = (255 * out_f).astype("uint8")
    return out_u

def colorize_masks(images, index_masks, fac: float = 0.5):
    if not index_masks:
        return [], []
    max_idx = max([m.max() for m in index_masks])
    guru.debug(f"{max_idx=}")
    palette = get_hls_palette(max_idx + 1)
    color_masks = []
    out_frames = []
    for img, mask in zip(images, index_masks):
        clr_mask = palette[mask.astype("int")]
        color_masks.append(clr_mask)
        out_u = compose_img_mask(img, clr_mask, fac)
        out_frames.append(out_u)
    return out_frames, color_masks

def draw_points(img, points, labels):
    """
    Draws points on the image.
    points: List or array of (x, y) coordinates.
    labels: List or array of labels (1.0 for positive, 0.0 for negative).
    """
    out = img.copy()
    for p, label in zip(points, labels):
        x, y = int(p[0]), int(p[1])
        color = (0, 255, 0) if label == 1.0 else (255, 0, 0)
        out = cv2.circle(out, (x, y), 10, color, -1)
    return out

def extract_frames(root_dir, vid_name, img_name, vid_file, start, end, fps, height, ext="png"):
    """
    Extracts frames from a video file using ffmpeg.
    """
    seq_name = os.path.splitext(vid_file)[0]
    vid_path = f"{root_dir}/{vid_name}/{vid_file}"
    out_dir = f"{root_dir}/{img_name}/{seq_name}"
    guru.debug(f"Extracting frames to {out_dir}")
    os.makedirs(out_dir, exist_ok=True)

    def make_time(seconds):
        return datetime.time(
            seconds // 3600, (seconds % 3600) // 60, seconds % 60
        )

    start_time = make_time(start).strftime("%H:%M:%S")
    end_time = make_time(end).strftime("%H:%M:%S")
    cmd = (
        f"ffmpeg -ss {start_time} -to {end_time} -i {vid_path} "
        f"-vf 'scale=-1:{height},fps={fps}' {out_dir}/%05d.{ext}"
    )
    print(cmd)
    subprocess.call(cmd, shell=True)
    return out_dir

def extract_video_frames(video_path, output_dir, fps=5, height=480, ext="jpg"):
    """
    Extracts frames from a video file to a specific directory.
    """
    if not os.path.exists(video_path):
        guru.error(f"Video file not found: {video_path}")
        return False
        
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if frames already exist to avoid re-extraction (simple check)
    if listdir(output_dir):
        guru.info(f"Frames already exist in {output_dir}. Skipping extraction. Clear directory to re-extract.")
        return True

    guru.info(f"Extracting frames from {video_path} to {output_dir}...")
    
    # Simple ffmpeg command
    cmd = (
        f"ffmpeg -i \"{video_path}\" "
        f"-vf 'scale=-1:{height},fps={fps}' \"{output_dir}/%05d.{ext}\" -hide_banner -loglevel error"
    )
    
    try:
        subprocess.call(cmd, shell=True)
        return True
    except Exception as e:
        guru.error(f"Error extracting frames: {e}")
        return False

def listdir(vid_dir):
    if vid_dir is not None and os.path.isdir(vid_dir):
        return sorted(os.listdir(vid_dir))
    return []
