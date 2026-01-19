import os
import torch
import numpy as np
import numpy as np
import imageio.v2 as iio
from tqdm import tqdm
from loguru import logger as guru
from sam2.build_sam import build_sam2_video_predictor
from utils import isimage#, make_index_mask  # We need to implement make_index_mask in utils or keep it here.

# Ideally make_index_mask should be in utils or a static method. 
# Let's put it as a static method or helper here since it deals with mask logic.

def make_index_mask(masks):
    if not masks:
        return None
    
    # Get shape from first mask
    first_mask = next(iter(masks.values()))
    idx_mask = np.zeros(first_mask.shape, dtype=np.uint8)
    
    for i, mask in masks.items():
        idx_mask[mask > 0] = i + 1
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
        
        # Persistence storage
        self.per_frame_points = {}
        self.per_frame_labels = {}
        self.per_frame_masks = {}

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
        
        # Clear specific mask from persistence
        if self.frame_index in self.per_frame_points:
            if self.cur_mask_idx in self.per_frame_points[self.frame_index]:
                del self.per_frame_points[self.frame_index][self.cur_mask_idx]
            if self.cur_mask_idx in self.per_frame_labels[self.frame_index]:
                del self.per_frame_labels[self.frame_index][self.cur_mask_idx]
            if self.cur_mask_idx in self.per_frame_masks[self.frame_index]:
                del self.per_frame_masks[self.frame_index][self.cur_mask_idx]

    def set_mask_id(self, mask_id):
        self.cur_mask_idx = mask_id
        
        # Restore or Clear
        if self.frame_index in self.per_frame_points and mask_id in self.per_frame_points[self.frame_index]:
             self.selected_points = list(self.per_frame_points[self.frame_index][mask_id])
             self.selected_labels = list(self.per_frame_labels[self.frame_index][mask_id])
        else:
             self.selected_points = []
             self.selected_labels = []
        
        guru.info(f"Switched to Mask ID {mask_id}. Loaded {len(self.selected_points)} points.")

    def add_new_mask(self):
        # We might deprecate this simple incr in favor of explicit ID set
        self.set_mask_id(self.cur_mask_idx + 1)

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
        self.per_frame_points = {}
        self.per_frame_labels = {}
        self.per_frame_masks = {}

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
        self.index_masks_all = [None] * len(self.img_paths)
        return len(self.img_paths)

    def set_input_image(self, i: int = 0) -> np.ndarray | None:
        guru.info(f"Setting frame {i} / {len(self.img_paths)}")
        if i < 0 or i >= len(self.img_paths):
            return self.image
        self.frame_index = i
        image = iio.imread(self.img_paths[i])
        self.image = image
        
        # Restore state if exists
        self.set_mask_id(self.cur_mask_idx)
            
        return image

    def get_current_masks(self):
        """
        Returns all mask dict for the current frame if it exists in persistence.
        """
        if self.frame_index in self.per_frame_masks:
            return self.per_frame_masks[self.frame_index]
        return {}

    def get_current_mask(self):
        """
        Returns the specific mask for the current ID.
        If current ID is -1, returns composite of all masks.
        """
        if self.cur_mask_idx == -1:
             # Composite view
             all_frame_masks = self.get_current_masks()
             if not all_frame_masks:
                 return None
             return make_index_mask(all_frame_masks)

        if self.frame_index in self.per_frame_masks and self.cur_mask_idx in self.per_frame_masks[self.frame_index]:
             return self.per_frame_masks[self.frame_index][self.cur_mask_idx]
        return None

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
            
        # FILTERING: Only keep the mask for the current ID
        # The model might return predictions for other propagated objects, but we only want to see/edit current.
        filtered_masks = {k: v for k, v in masks.items() if k == self.cur_mask_idx}
        
        if not filtered_masks:
             guru.warning(f"No mask found for ID {self.cur_mask_idx} in SAM output.")
             return None
             
        mask = make_index_mask(filtered_masks)
        
        # Save state nested
        if frame_idx not in self.per_frame_points:
            self.per_frame_points[frame_idx] = {}
            self.per_frame_labels[frame_idx] = {}
            self.per_frame_masks[frame_idx] = {}
            
        self.per_frame_points[frame_idx][self.cur_mask_idx] = list(self.selected_points)
        self.per_frame_labels[frame_idx][self.cur_mask_idx] = list(self.selected_labels)
        self.per_frame_masks[frame_idx][self.cur_mask_idx] = mask
        
        # Sync to index_masks_all to ensure it's saved even without running tracker
        self.index_masks_all[frame_idx] = make_index_mask(self.per_frame_masks[frame_idx])
        
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
        guru.info("Loading images for visualization...")
        images = [iio.imread(p)[:, :, :3] for p in self.img_paths]
        
        video_segments = {}  # video_segments contains the per-frame segmentation results
        
        if "cuda" in self.device:
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
            ctx = torch.autocast(device_type="cuda", dtype=dtype)
        else:
            ctx = torch.no_grad()

        guru.info("Propagating masks in video...")
        with ctx:
            guru.info("started propagation")
            for out_frame_idx, out_obj_ids, out_mask_logits in self.sam_model.propagate_in_video(self.inference_state, start_frame_idx=0):
                guru.info(f"propagated frame {out_frame_idx}")
                masks = {
                    out_obj_id: (out_mask_logits[i] > 0.0).squeeze().cpu().numpy()
                    for i, out_obj_id in enumerate(out_obj_ids)
                }
                video_segments[out_frame_idx] = masks

                # Update per_frame_masks for visualization
                if out_frame_idx not in self.per_frame_masks:
                    self.per_frame_masks[out_frame_idx] = {}
                
                # 'masks' is {obj_id: logical_mask}
                # We need to convert logical_mask to standard mask format used in this app (usually uint8 with values)
                # But actually, per_frame_masks[frame][id] expects a mask where that specific object is labeled. 
                # make_index_mask converts {id: mask} -> single index mask.
                # Here we want to store the mask for *each* object ID individually for our isolation logic.
                
                for obj_id, logical_mask in masks.items():
                    # Create a mask where this object is labeled with its ID (or just 1? isolation logic uses initialized zeros and sets obj_id)
                    # isolation logic in make_index_mask expects a dict of masks.
                    # let's look at how add_point saves it: it saves the result of make_index_mask(filtered).
                    # So we should save an index mask where pixels == obj_id (or 1 if we normalize).
                    # Our make_index_mask sets pixels to `i+1` where `i` is the key. 
                    # But the key coming from SAM is the arbitrary obj_id.
                    # If we follow add_point, it saves `make_index_mask({obj_id: logical})`.
                    
                    single_obj_dict = {obj_id: logical_mask}
                    # We reuse make_index_mask to format it correctly as a uint8 array
                    formatted_mask = make_index_mask(single_obj_dict)
                    if formatted_mask is not None:
                        self.per_frame_masks[out_frame_idx][obj_id] = formatted_mask

        self.index_masks_all = []
        # Ensure frames are ordered
        guru.info("Processing results...")
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
        if not self.index_masks_all or all(m is None for m in self.index_masks_all):
            guru.info("No masks to save.")
            return "No masks to save."
        guru.info(f"Saving masks to directory {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
        # Using enumerate to ensure frame_idx is consistent with the list order
        for frame_idx, (img_path, idx_mask) in tqdm(enumerate(zip(self.img_paths, self.index_masks_all)), total=len(self.img_paths), desc="Saving masks"):
            if idx_mask is None:
                continue
            
            unique_ids = np.unique(idx_mask)
            for uid in unique_ids:
                if uid == 0:
                    continue
                
                # Recover original mask_id (since make_index_mask uses i+1)
                mask_id = uid - 1
                
                # Create binary mask for this ID
                binary_mask = (idx_mask == uid).astype(np.uint8)
                
                # Save as {frame_idx}_{mask_id}.npy
                out_name = f"{frame_idx}_{mask_id}.npy"
                out_path = os.path.join(output_dir, out_name)
                np.save(out_path, binary_mask)
                guru.info(f"Saved mask {out_name}")
        
        message = f"Saved masks to {output_dir}!"
        guru.info(message)
        return message
