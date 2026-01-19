import os
import torch
import numpy as np
import imageio.v2 as iio
from loguru import logger as guru
from sam2.build_sam import build_sam2_video_predictor
from utils import isimage#, make_index_mask  # We need to implement make_index_mask in utils or keep it here.

# Ideally make_index_mask should be in utils or a static method. 
# Let's put it as a static method or helper here since it deals with mask logic.

def make_index_mask(masks):
    assert len(masks) > 0
    idcs = list(masks.keys())
    idx_mask = masks[idcs[0]].astype("uint8")
    for i in idcs:
        mask = masks[i]
        idx_mask[mask] = i + 1
    return idx_mask

class PromptGUI(object):
    def __init__(self, checkpoint_dir, model_cfg, device=None):
        self.checkpoint_dir = checkpoint_dir
        self.model_cfg = model_cfg
        
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            else:
                torch.autocast("cpu", dtype=torch.bfloat16).__enter__()
                self.device = "cpu"
        else:
            self.device = device
            
        self.sam_model = None
        self.inference_state = None

        self.selected_points = []
        self.selected_labels = []
        self.cur_label_val = 1.0

        self.frame_index = 0
        self.image = None
        self.cur_mask_idx = 0
        # can store multiple object masks
        # saves the masks and logits for each mask index
        self.cur_masks = {}
        self.cur_logits = {}
        self.index_masks_all = []
        self.color_masks_all = []

        self.img_dir = ""
        self.img_paths = []
        self.init_sam_model()

    def init_sam_model(self):
        if self.sam_model is None:
            # Check for float32 settings
            if "cuda" in self.device and torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            
            # Use bfloat16 for the entire notebook/session context generally, 
            # but we will just ensure the model is built correctly.
            self.sam_model = build_sam2_video_predictor(self.model_cfg, self.checkpoint_dir, device=self.device)
            guru.info(f"loaded model checkpoint {self.checkpoint_dir} on {self.device}")

    def clear_points(self):
        self.selected_points.clear()
        self.selected_labels.clear()

    def add_new_mask(self):
        self.cur_mask_idx += 1
        self.clear_points()
        guru.info(f"Creating new mask with index {self.cur_mask_idx}")

    def _clear_image(self):
        """
        clears image and all masks/logits for that image
        """
        self.image = None
        self.cur_mask_idx = 0
        self.frame_index = 0
        self.cur_masks = {}
        self.cur_logits = {}
        self.index_masks_all = []
        self.color_masks_all = []

    def reset(self):
        self._clear_image()
        if self.inference_state:
             self.sam_model.reset_state(self.inference_state)

    def set_img_dir(self, img_dir: str) -> int:
        self._clear_image()
        self.img_dir = img_dir
        self.img_paths = [
            f"{img_dir}/{p}" for p in sorted(os.listdir(img_dir)) if isimage(p)
        ]
        return len(self.img_paths)

    def set_input_image(self, i: int = 0) -> np.ndarray | None:
        guru.debug(f"Setting frame {i} / {len(self.img_paths)}")
        if i < 0 or i >= len(self.img_paths):
            return self.image
        self.clear_points()
        self.frame_index = i
        image = iio.imread(self.img_paths[i])
        self.image = image
        return image

    def get_sam_features(self):
        self.inference_state = self.sam_model.init_state(video_path=self.img_dir)
        self.sam_model.reset_state(self.inference_state)
        guru.info("SAM features extracted.")
        return self.image

    def set_positive(self):
        self.cur_label_val = 1.0

    def set_negative(self):
        self.cur_label_val = 0.0

    def add_point(self, frame_idx, i, j):
        """
        Add a point and return the current mask for the objects.
        i: row (y)
        j: col (x)
        """
        self.selected_points.append([j, i])
        self.selected_labels.append(self.cur_label_val)
        
        masks = self.get_sam_mask(
            frame_idx, np.array(self.selected_points, dtype=np.float32), np.array(self.selected_labels, dtype=np.int32)
        )
        if not masks:
            return None
            
        mask = make_index_mask(masks)
        return mask
    
    def get_sam_mask(self, frame_idx, input_points, input_labels):
        assert self.sam_model is not None
        
        # Ensure we are in the correct autocast text
        if "cuda" in self.device:
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
            ctx = torch.autocast(device_type="cuda", dtype=dtype)
        else:
            ctx = torch.no_grad()

        with ctx:
             _, out_obj_ids, out_mask_logits = self.sam_model.add_new_points_or_box(
                inference_state=self.inference_state,
                frame_idx=frame_idx,
                obj_id=self.cur_mask_idx,
                points=input_points,
                labels=input_labels,
            )

        return  {
                out_obj_id: (out_mask_logits[i] > 0.0).squeeze().cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }

    def run_tracker(self):
        # read images and drop the alpha channel
        images = [iio.imread(p)[:, :, :3] for p in self.img_paths]
        
        video_segments = {}  # video_segments contains the per-frame segmentation results
        
        if "cuda" in self.device:
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
            ctx = torch.autocast(device_type="cuda", dtype=dtype)
        else:
            ctx = torch.no_grad()

        with ctx:
            for out_frame_idx, out_obj_ids, out_mask_logits in self.sam_model.propagate_in_video(self.inference_state, start_frame_idx=0, ):
                masks = {
                    out_obj_id: (out_mask_logits[i] > 0.0).squeeze().cpu().numpy()
                    for i, out_obj_id in enumerate(out_obj_ids)
                }
                video_segments[out_frame_idx] = masks

        self.index_masks_all = []
        # Ensure frames are ordered
        for i in sorted(video_segments.keys()):
             masks = video_segments[i]
             if masks:
                 self.index_masks_all.append(make_index_mask(masks))
             else:
                 # If no mask for a frame (shouldn't happen in propagation usually unless lost), empty mask?
                 # For now assuming propagation returns masks.
                 # If it skips frames, we need to handle that. 
                 self.index_masks_all.append(np.zeros((images[0].shape[0], images[0].shape[1]), dtype=np.uint8))

        from utils import colorize_masks # Lazy import to avoid circular dependency if any
        # Although utils doesn't import core, so it's fine.
        out_frames, self.color_masks_all = colorize_masks(images, self.index_masks_all)
        return out_frames, self.color_masks_all

    def save_masks_to_dir(self, output_dir: str):
        if not self.color_masks_all:
            return "No masks to save."
        
        os.makedirs(output_dir, exist_ok=True)
        for img_path, clr_mask, id_mask in zip(self.img_paths, self.color_masks_all, self.index_masks_all):
            name = os.path.basename(img_path)
            out_path = f"{output_dir}/{name}"
            iio.imwrite(out_path, clr_mask)
            np_out_path = f"{output_dir}/{name[:-4]}.npy"
            np.save(np_out_path, id_mask)
        
        message = f"Saved masks to {output_dir}!"
        guru.info(message)
        return message
