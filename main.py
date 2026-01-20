
import os
import io
import numpy as np
import ipywidgets as widgets
from ipyevents import Event
from PIL import Image as PILImage
from loguru import logger as guru
from loguru import logger as guru
from core import PromptGUI, init_model
from utils import compose_img_mask, draw_points, get_hls_palette, extract_video_frames

class IPyWidgetsGUI:
    def __init__(self, sam_model, device=None):
        self.prompts = PromptGUI(sam_model, device=device)
        
        self.frames_dir = None
        self.output_mask_dir = None
        
        # UI Elements
        # We will set layout dimensions dynamically in update_display
        self.image_widget = widgets.Image(format='jpeg')
        self.image_widget.layout.object_fit = 'contain'  # Ensure it fits if constraints apply, though we will set explicit size
        self.image_event = Event(source=self.image_widget, watched_events=['click'])
        self.image_event.on_dom_event(self.on_click)
        
        self.slider_frame = widgets.IntSlider(description='Frame:', min=0, max=0, value=0)
        self.slider_frame.observe(self.on_frame_change, names='value')
        
        self.btn_prev_frame = widgets.Button(description='< Prev', layout=widgets.Layout(width='80px'))
        self.btn_next_frame = widgets.Button(description='Next >', layout=widgets.Layout(width='80px'))
        self.btn_prev_frame.on_click(self.on_prev_frame)
        self.btn_next_frame.on_click(self.on_next_frame)
        
        self.radio_label = widgets.RadioButtons(
            options=['Positive', 'Negative'],
            value='Positive',
            description='Label:',
            disabled=False
        )
        self.radio_label.observe(self.on_label_type_change, names='value')
        
        self.btn_prev_id = widgets.Button(description='< Prev ID')
        self.btn_next_id = widgets.Button(description='Next ID >')
        self.btn_prev_id.on_click(self.on_prev_id)
        self.btn_next_id.on_click(self.on_next_id)
        
        self.mask_id_label = widgets.Label(value="Current Mask ID: 0")
        
        self.btn_clear = widgets.Button(description='Clear Points')
        self.btn_clear.on_click(self.on_clear_points)
        
        self.btn_submit = widgets.Button(description='Submit/Track', button_style='success')
        self.btn_submit.on_click(self.on_submit)
        
        self.btn_save = widgets.Button(description='Save Masks', button_style='info')
        self.btn_save.on_click(self.on_save)
        
        self.btn_reset = widgets.Button(description='Reset All', button_style='danger')
        self.btn_reset.on_click(self.on_reset)
        
        # Output widget for logs (optional, but good for feedback inside widget area)
        self.out = widgets.Output(layout={'border': '1px solid black', 'height': '150px', 'overflow_y': 'scroll'})
        
        self.current_overlay_mask = None
        
        # Build Layout
        self.container = self.build_layout()

    def build_layout(self):
        # Frame Controls
        frame_controls = widgets.HBox([self.btn_prev_frame, self.slider_frame, self.btn_next_frame])
        
        # Mask ID Controls
        id_controls = widgets.HBox([self.btn_prev_id, self.mask_id_label, self.btn_next_id])
        
        # Action Buttons
        actions1 = widgets.HBox([self.radio_label, self.btn_clear])
        actions2 = widgets.HBox([self.btn_submit, self.btn_save, self.btn_reset])
        
        # Sidebar
        sidebar = widgets.VBox([
            widgets.HTML("<h3>Controls</h3>"),
            frame_controls,
            widgets.HTML("<hr>"),
            id_controls,
            actions1,
            widgets.HTML("<hr>"),
            actions2,
            widgets.HTML("<hr>"),
            self.out
        ])
        
        # Main Layout
        main = widgets.HBox([self.image_widget, sidebar])
        return main

    def log(self, message, level="INFO"):
        with self.out:
            print(f"[{level}] {message}")
        guru.log(level, message)

    def load_sequence(self, frames_dir, output_mask_dir=None, start_frame=0, step=1):
        self.frames_dir = frames_dir
        self.output_mask_dir = output_mask_dir
        
        self.prompts.set_slice_config(start_frame=start_frame, step=step)
        
        if not os.path.exists(self.frames_dir):
            self.log(f"Image directory not found: {self.frames_dir}", "ERROR")
            return
            
        num_imgs = self.prompts.set_img_dir(self.frames_dir)
        self.log(f"Loaded {num_imgs} images from {frames_dir}")
        
        self.slider_frame.max = max(0, num_imgs - 1)
        self.slider_frame.value = 0
        
        self.prompts.get_sam_features()
        self.update_display()

    def update_display(self):
        if self.prompts.image is None:
            return

        display_img = self.prompts.image.copy()
        
        # Overlay Mask
        if hasattr(self, 'current_overlay_mask') and self.current_overlay_mask is not None:
             palette = get_hls_palette(self.current_overlay_mask.max() + 1)
             color_mask = palette[self.current_overlay_mask]
             display_img = compose_img_mask(display_img, color_mask)
        
        # Draw points
        if self.prompts.selected_points:
             display_img = draw_points(display_img, self.prompts.selected_points, self.prompts.selected_labels)
             
        # Convert to JPEG for widget
        pil_img = PILImage.fromarray(display_img)
        with io.BytesIO() as b:
            pil_img.save(b, format='JPEG')
        with io.BytesIO() as b:
            pil_img.save(b, format='JPEG')
            self.image_widget.value = b.getvalue()
            
        # Update Widget Dimensions to match aspect ratio
        # Start with a base width
        base_w = 600
        h, w = display_img.shape[:2]
        base_h = int(base_w * (h / w))
        
        self.image_widget.layout.width = f"{base_w}px"
        self.image_widget.layout.height = f"{base_h}px"
        
        # Store for click handler
        self.display_dims = (base_w, base_h)
            
        # Update labels
        if self.prompts.cur_mask_idx == -1:
             self.mask_id_label.value = "View: All Masks"
        else:
             self.mask_id_label.value = f"Current Mask ID: {self.prompts.cur_mask_idx}"

    def on_click(self, event):
        # event contains 'relativeX', 'relativeY' which are relative to the widget
        # We need to map them to image coordinates.
        # Assuming widget width/height matches image aspect ratio or handled by object-fit.
        # widgets.Image displays the image.
        
        if self.prompts.image is None:
            return
            
        # Image real dims
        h, w = self.prompts.image.shape[:2]
        
        # Widget display dims (approximate if fixed, but better if we can get actual client rect,
        # but ipyevents gives relativeX/Y to the element).
        # WARNING: If the image is scaled by CSS, relativeX might need scaling.
        # Simple approach: assume natural size or fixed size.
        # widgets.Image(width=600) might scale the image.
        # We can try to use raw data coordinate if possible, but IPyEvents gives pixel coords on element.
        
        # Let's assume the user sets width=600, height=400 in __init__.
        # We need to calculate scale.
        # Or better, let's not force width/height in widget to avoid aspect ratio issues, 
        # but huge images are bad.
        # For now, let's just use the coordinates and update logic if scaling is off.
        # Actually, standard ipywidgets Image auto-scales to width.
        
        # To make this robust:
        # We can rely on the fact that we sent a specific image size? No.
        
        # Let's try to map naively first.
        # If visual glitches, we might need a fixed aspect ratio container.
        
        # Actually, ipyevents usually gives coordinates relative to the DOM element.
        # If the image is 600px wide in DOM, x is 0-600.
        # If original image is 1920px, we need scale = 1920/600.
        
        # Ideally we know the displayed width/height.
        # self.image_widget.width is '600'.
        
        if not hasattr(self, 'display_dims'):
            # Fallback if update_display wasn't called or didn't set it (shouldn't happen)
            disp_w = 600
        else:
             disp_w, disp_h = self.display_dims

        scale_x = w / disp_w
        # We used explicit height calculation: h / base_h should equal w / base_w approx
        if hasattr(self, 'display_dims'):
             scale_y = h / self.display_dims[1]
        else:
             scale_y = scale_x

        x = event['relativeX'] * scale_x
        y = event['relativeY'] * scale_y
        
        if self.prompts.cur_mask_idx == -1:
            self.log("Cannot add points in 'All Masks' view.", "WARNING")
            return
            
        mask = self.prompts.add_point(self.prompts.frame_index, int(y), int(x))
        self.log(f"Added point at {int(x)}, {int(y)}")
        self.current_overlay_mask = mask
        self.update_display()

    def on_frame_change(self, change):
        idx = change['new']
        if idx != self.prompts.frame_index:
            self.prompts.set_input_image(idx)
            self.current_overlay_mask = self.prompts.get_current_mask()
            self.update_display()

    def on_prev_frame(self, b):
        if self.slider_frame.value > self.slider_frame.min:
            self.slider_frame.value -= 1
            
    def on_next_frame(self, b):
        if self.slider_frame.value < self.slider_frame.max:
            self.slider_frame.value += 1

    def on_prev_id(self, b):
        new_id = self.prompts.cur_mask_idx - 1
        if new_id < -1: new_id = -1
        if new_id != self.prompts.cur_mask_idx:
            self.prompts.set_mask_id(new_id)
            self.current_overlay_mask = self.prompts.get_current_mask()
            self.update_display()

    def on_next_id(self, b):
        new_id = self.prompts.cur_mask_idx + 1
        self.prompts.set_mask_id(new_id)
        self.current_overlay_mask = self.prompts.get_current_mask()
        self.update_display()

    def on_label_type_change(self, change):
        if change['new'] == 'Positive':
            self.prompts.set_positive()
        else:
            self.prompts.set_negative()

    def on_clear_points(self, b):
        self.prompts.clear_points()
        self.current_overlay_mask = None
        self.update_display()
        self.log("Points cleared.")

    def on_reset(self, b):
        self.prompts.reset()
        self.current_overlay_mask = None
        self.prompts.set_input_image(self.prompts.frame_index)
        self.update_display()
        self.log("Reset all.")

    def on_submit(self, b):
        self.log("Running tracker...")
        try:
            out_frames, color_masks = self.prompts.run_tracker()
            self.current_overlay_mask = self.prompts.get_current_mask()
            self.update_display()
            self.log("Tracking complete.")
        except Exception as e:
            self.log(f"Tracking failed: {e}", "ERROR")

    def on_save(self, b):
        if not self.output_mask_dir:
            self.log("No output directory set.", "WARNING")
            return
        self.prompts.save_masks_to_dir(self.output_mask_dir)
        self.log(f"Masks saved to {self.output_mask_dir}")

def run_app(
    video_path, 
    model,
    device=None, 
    log_file=None, 
    output_mask_dir=None,
    start_frame=0,
    end_frame=None,
    step=None
):
    """
    Main function to run the app in a notebook with a video path.
    """
    
    # Configure logging
    if log_file:
        guru.remove()
        guru.add(log_file, level="INFO", enqueue=True)
    
    if not os.path.exists(video_path):
        guru.error(f"Error: Video path {video_path} does not exist.")
        return None

    # Derive directory names
    video_msg = os.path.basename(video_path)
    video_name = os.path.splitext(video_msg)[0]
    base_dir = os.path.dirname(os.path.abspath(video_path))
    
    # Frames/Masks directories
    slice_suffix = ""
    if start_frame > 0 or end_frame is not None or step is not None:
         s = start_frame
         e = "end" if end_frame is None else end_frame
         st = "1" if step is None else step
         slice_suffix = f"_s{s}_e{e}_st{st}"
         
    frames_dir = os.path.join(base_dir, f"{video_name}_frames{slice_suffix}")
    
    if output_mask_dir is None:
        output_mask_dir = os.path.join(base_dir, f"{video_name}_masks{slice_suffix}")
    
    # Extract frames
    success = extract_video_frames(
        video_path, 
        frames_dir, 
        start_frame=start_frame, 
        end_frame=end_frame, 
        step=step
    )
    
    if not success:
        return None
        
    # Init App
    app = IPyWidgetsGUI(model, device=device)
    app.load_sequence(
        frames_dir, 
        output_mask_dir=output_mask_dir, 
        start_frame=start_frame,
        step=step if step is not None else 1
    )
    
    return app
