import os
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider, RadioButtons
import numpy as np
import cv2
from loguru import logger as guru
from core import PromptGUI
from utils import compose_img_mask, draw_points, get_hls_palette, extract_frames, listdir

class MatplotlibGUI:
    def __init__(self, checkpoint_dir, model_cfg, device=None):
        self.prompts = PromptGUI(checkpoint_dir, model_cfg, device=device)
        
        self.frames_dir = None
        self.output_mask_dir = None
        
        self.fig = None
        self.ax = None
        self.img_plot = None
        
        # UI Elements
        self.slider_frame = None
        self.btn_next = None
        self.btn_prev = None
        self.radio_label = None
        self.btn_add_mask = None
        self.btn_clear = None
        self.btn_submit = None
        self.btn_save = None
        self.btn_reset = None
        self.btn_prev_id = None
        self.btn_next_id = None
        
        self.current_overlay_mask = None
        
        self.setup_gui()
        
    def setup_gui(self):
        # Create figure and axes
        # We need a layout that allows for the image and standard controls
        self.fig = plt.figure(figsize=(12, 8))
        self.ax = self.fig.add_axes([0.05, 0.2, 0.7, 0.75]) # Left, Bottom, Width, Height
        self.ax.set_title("SAM2 Interactive Labeler")
        self.ax.axis('off')

        # Controls area (Right side and Bottom)
        
        # Frame Slider (Bottom)
        ax_slider = self.fig.add_axes([0.1, 0.05, 0.6, 0.03])
        self.slider_frame = Slider(ax_slider, 'Frame', 0, 1, valinit=0, valstep=1)
        self.slider_frame.on_changed(self.on_frame_change)
        
        # Buttons (Right Side)
        btn_width = 0.15
        btn_height = 0.05
        start_x = 0.8
        start_y = 0.8
        gap = 0.06
        
        # Label Type (Radio)
        ax_radio = self.fig.add_axes([start_x, start_y, btn_width, 0.1])
        self.radio_label = RadioButtons(ax_radio, ('Positive', 'Negative'))
        self.radio_label.on_clicked(self.on_label_type_change)
        
        # Add Mask
        # Mask ID Controls
        # Previous ID
        ax_prev = self.fig.add_axes([start_x, start_y - 2*gap, btn_width / 2 - 0.01, btn_height])
        self.btn_prev_id = Button(ax_prev, '< Prev ID')
        self.btn_prev_id.on_clicked(self.on_prev_id)
        
        # Next ID
        ax_next = self.fig.add_axes([start_x + btn_width / 2 + 0.01, start_y - 2*gap, btn_width / 2 - 0.01, btn_height])
        self.btn_next_id = Button(ax_next, 'Next ID >')
        self.btn_next_id.on_clicked(self.on_next_id)
        
        # Clear Points
        ax_clear = self.fig.add_axes([start_x, start_y - 3*gap, btn_width, btn_height])
        self.btn_clear = Button(ax_clear, 'Clear Points')
        self.btn_clear.on_clicked(self.on_clear_points)
        
        # Submit / Track
        ax_submit = self.fig.add_axes([start_x, start_y - 4*gap, btn_width, btn_height])
        self.btn_submit = Button(ax_submit, 'Submit/Track')
        self.btn_submit.on_clicked(self.on_submit)
        
        # Save
        ax_save = self.fig.add_axes([start_x, start_y - 5*gap, btn_width, btn_height])
        self.btn_save = Button(ax_save, 'Save Masks')
        self.btn_save.on_clicked(self.on_save)
        
        # Reset
        ax_reset = self.fig.add_axes([start_x, start_y - 6*gap, btn_width, btn_height])
        self.btn_reset = Button(ax_reset, 'Reset All')
        self.btn_reset.on_clicked(self.on_reset)
        
        # Connect Click Event
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        
        self.update_display()

    def load_sequence(self, frames_dir, output_mask_dir=None, start_frame=0, step=1):
        """
        Load a sequence of frames from a directory.
        frames_dir: Path to the directory containing extracted frames.
        output_mask_dir: Path where masks will be saved.
        """
        self.frames_dir = frames_dir
        self.output_mask_dir = output_mask_dir
        
        # Configure slice info in prompts
        self.prompts.set_slice_config(start_frame=start_frame, step=step)
        
        if not os.path.exists(self.frames_dir):
            guru.error(f"Image directory not found: {self.frames_dir}")
            return
            
        num_imgs = self.prompts.set_img_dir(self.frames_dir)
        guru.info(f"Loaded {num_imgs} images from {frames_dir}")
        
        # Update slider
        self.slider_frame.valmax = max(0, num_imgs - 1)
        self.slider_frame.val = 0
        self.slider_frame.ax.set_xlim(0, max(0, num_imgs - 1))
        
        # Initialize SAM state for this sequence
        self.prompts.get_sam_features()
        
        self.update_display()

    def update_display(self):
        if self.prompts.image is None:
            self.ax.imshow(np.zeros((500, 500, 3)))
            self.fig.canvas.draw_idle()
            return
            
        # Base image
        display_img = self.prompts.image.copy()
        
        # Overlay Mask (if any)
        if hasattr(self, 'current_overlay_mask') and self.current_overlay_mask is not None:
             palette = get_hls_palette(self.current_overlay_mask.max() + 1)
             color_mask = palette[self.current_overlay_mask]
             display_img = compose_img_mask(display_img, color_mask)
        
        # Draw points
        if self.prompts.selected_points:
             display_img = draw_points(display_img, self.prompts.selected_points, self.prompts.selected_labels)
             
        self.ax.clear()
        self.ax.clear()
        self.ax.imshow(display_img)
        if self.prompts.cur_mask_idx == -1:
             title_str = f"Frame: {self.prompts.frame_index} | View: All Masks"
        else:
             title_str = f"Frame: {self.prompts.frame_index} | Current Mask ID: {self.prompts.cur_mask_idx}"
             
        self.ax.set_title(title_str)
        self.ax.axis('off')
        self.fig.canvas.draw_idle()

    def on_click(self, event):
        if event.inaxes != self.ax:
            return
        if self.prompts.image is None:
            return
            
        # Matplotlib coordinates
        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return

        if self.prompts.cur_mask_idx == -1:
            guru.warning("Cannot add points in 'All Masks' view. Switch to a specific ID to edit.")
            return
            
        # Add point
        # y is row, x is col
        mask = self.prompts.add_point(self.prompts.frame_index, int(y), int(x)) 
        guru.info(f"Added point at {int(x)}, {int(y)} frame {self.prompts.frame_index}")
        self.current_overlay_mask = mask
        self.update_display()

    def on_frame_change(self, val):
        idx = int(val)
        if idx != self.prompts.frame_index:
            self.prompts.set_input_image(idx)
            # Restore overlay mask if available (loaded by set_input_image's side effects or getter)
            self.current_overlay_mask = self.prompts.get_current_mask()
            self.update_display()

    def on_prev_id(self, event):
        # Allow going to -1 (All View)
        new_id = self.prompts.cur_mask_idx - 1
        if new_id < -1: 
             new_id = -1
             
        if new_id != self.prompts.cur_mask_idx:
            self.prompts.set_mask_id(new_id)
            self.current_overlay_mask = self.prompts.get_current_mask()
            self.update_display()

    def on_next_id(self, event):
        new_id = self.prompts.cur_mask_idx + 1
        self.prompts.set_mask_id(new_id)
        self.current_overlay_mask = self.prompts.get_current_mask()
        self.update_display()

    def on_label_type_change(self, label):
        if label == 'Positive':
            self.prompts.set_positive()
            guru.info("Switched to Positive Labels")
        else:
            self.prompts.set_negative()
            guru.info("Switched to Negative Labels")



    def on_clear_points(self, event):
        self.prompts.clear_points()
        guru.info("Cleared points for current mask.")
        self.current_overlay_mask = None
        self.update_display()

    def on_reset(self, event):
        self.prompts.reset()
        guru.info("Reset all states.")
        self.current_overlay_mask = None
        # Reload current image
        self.prompts.set_input_image(self.prompts.frame_index)
        self.update_display()

    def on_submit(self, event):
        guru.info("Running tracker... this might take a moment.")
        out_frames, color_masks = self.prompts.run_tracker()
        guru.info("Tracking complete. Results loaded into view.")
        
        # Refresh current view
        self.current_overlay_mask = self.prompts.get_current_mask()
        self.update_display()
        
    def on_save(self, event):
        if not self.output_mask_dir:
            guru.warning("No output directory set for masks.")
            return
        self.prompts.save_masks_to_dir(self.output_mask_dir)

    def block_until_closed(self):
        """
        Blocks the execution until the figure is closed.
        Uses standard plt.show(block=True) for best stability.
        """
        if not self.fig:
             return
             
        guru.info(f"Launching GUI with backend: {matplotlib.get_backend()}...")
        
        try:
            # block=True is the robust standard way to run the loop
            plt.show(block=True)
            guru.info("GUI closed.")
        except KeyboardInterrupt:
            guru.info("Execution stopped by user.")
            plt.close(self.fig)

def run_app(
    video_path, 
    checkpoint_dir, 
    model_cfg, 
    device=None, 
    block=False, 
    log_file=None, 
    output_mask_dir=None,
    start_frame=0,
    end_frame=None,
    step=None
):
    """
    Main function to run the app in a notebook with a video path.
    Supports slicing the video: start_frame, end_frame, step.
    """
    from utils import extract_video_frames
    import os
    
    # Configure logging if requested
    if log_file:
        guru.remove()
        guru.add(log_file, level="INFO", enqueue=True) # enqueue=True for thread safety/async
    
    if not os.path.exists(video_path):
        guru.error(f"Error: Video path {video_path} does not exist.")
        return None

    # Derive directory names
    video_msg = os.path.basename(video_path)
    video_name = os.path.splitext(video_msg)[0]
    base_dir = os.path.dirname(os.path.abspath(video_path))
    
    # Frames directory
    # Append slice info to dir name to avoid conflicts if user extracts different slices
    slice_suffix = ""
    if start_frame > 0 or end_frame is not None or step is not None:
         # Create a unique-ish suffix for the slice config
         s = start_frame
         e = "end" if end_frame is None else end_frame
         st = "1" if step is None else step
         slice_suffix = f"_s{s}_e{e}_st{st}"
         
    frames_dir = os.path.join(base_dir, f"{video_name}_frames{slice_suffix}")
    
    # Masks directory
    if output_mask_dir is None:
        output_mask_dir = os.path.join(base_dir, f"{video_name}_masks{slice_suffix}")
    
    # Extract frames
    guru.info(f"Preparing frames in {frames_dir}...")
    success = extract_video_frames(
        video_path, 
        frames_dir, 
        start_frame=start_frame, 
        end_frame=end_frame, 
        step=step
    )
    
    if not success:
        guru.error("Failed to extract frames.")
        return None
        
    # Init App
    app = MatplotlibGUI(checkpoint_dir, model_cfg, device=device)
    app.load_sequence(
        frames_dir, 
        output_mask_dir=output_mask_dir, 
        start_frame=start_frame,
        step=step if step is not None else 1
    )
    
    if block:
        app.block_until_closed()
    
    return app
