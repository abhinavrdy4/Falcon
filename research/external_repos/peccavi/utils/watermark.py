"""Official Implementation of Watermarker class
"""

import os
import hashlib
import yaml
import torch
import json
import cv2
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import logging
import traceback
from diffusers import DDIMScheduler
from utils.config import ConfigLoader
from ZoDiac.main.utils import save_img
from ZoDiac.loss.loss import LossProvider
from ZoDiac.loss.pytorch_ssim import ssim
from ZoDiac.main.wmpatch import GTWatermark
from ZoDiac.main.wmdiffusion import WMDetectStableDiffusionPipeline
from ZoDiac.main.utils import *
import re
import wandb


class Watermarker:
    """
    A class for applying and managing watermarks on images using the PECCVAI framework.

    Attributes:
        device (torch.device): The device to run computations on (CPU or GPU).
        cfgs (dict): Configuration settings loaded from a YAML file.
        wm_pipes (dict): Per-patch GTWatermark instances keyed by patch index.
                         (Replaces the single shared self.wm_pipe.)
        scheduler (DDIMScheduler): Scheduler for the diffusion model.
        pipe (WMDetectStableDiffusionPipeline): Pipeline for stable diffusion.
        learning_rate (float): Learning rate for the optimizer.
        milestones (list of int): Reduces the learning rate at these epochs by the gamma value.
    """
    
    def __init__(self, bbox_json_path):
        """
        Initialises the Watermarker with configurations and sets up the pipeline and scheduler.
        """
        config_loader = ConfigLoader('config.yaml')
        self.num_inference_steps = config_loader.num_inference_steps
        self.original_shape = config_loader.original_shape
        self.target_shape = config_loader.target_shape
        self.device = config_loader.device
        self.bbox_json_path = bbox_json_path
        self.patches = self.extract_patches()
        self.strength = config_loader.strengths
        self.w_channel = config_loader.w_channel     # base channel; each patch rotates from here
        self.w_radius  = config_loader.w_radius

        # FIX 2: Replace the single shared GTWatermark with a per-patch dict.
        # Each patch gets its own GTWatermark instance with a distinct channel
        # (channel = patch_index % 4, matching the 4-channel latent space).
        # Instances are populated lazily in watermarking() once we know the
        # latent shape for each patch.
        self.wm_pipes = {}   # {patch_idx: GTWatermark}
        self.patch_meta = {}  # PECCAVI-FIX 2: {patch_idx: key seed, box, mask size, ...} saved to state.json
        self.last_detection = []  # per-patch results of the last detect() call (instrumentation only)

        # FIX 1: The original also initialised a single self.wm_pipe here
        # for 'single'/'multiple' w_type.  That block is removed because the
        # dict self.wm_pipes now handles both cases uniformly.

        self.scheduler = DDIMScheduler.from_pretrained(config_loader.model_id, subfolder="scheduler")
        self.pipe = WMDetectStableDiffusionPipeline.from_pretrained(
            config_loader.model_id, scheduler=self.scheduler
        ).to(self.device)
        self.learning_rate   = config_loader.learning_rate
        self.milestones      = config_loader.milestones
        self.loss_weights    = config_loader.loss_weights
        self.watermarked_path = config_loader.watermarked_path
        self.epochs          = config_loader.iters
        self.save_img        = config_loader.save_img
        self.ssim_threshold  = config_loader.ssim_threshold
        self.wandb_use       = config_loader.wandb
        self.patch_size      = 64  # pixels; must match split.py's patch_size

    # --------------------------------------------------------------------------
    # PATCH EXTRACTION
    # --------------------------------------------------------------------------

    def extract_patches(self):
        """
        Reads patch-grid coordinates from the final.json produced by split.py.

        The JSON now contains entries of the form:
            {"patch": [col, row], "score": float}
        which is the format written by split.py's map_boxes_to_patches().

        Returns:
            list[list[int, int]]: [[col, row], ...] for each selected patch.
        """
        print(f' ---------- Extracting Patches ----------')
        patches = []
        
        with open(self.bbox_json_path, "r") as f:
            data = json.load(f)
        
        for item in data:
            if "patch" in item:
                patches.append(item["patch"])   # [col, row]
        
        print(f"Found {len(patches)} patches to watermark")
        return patches

    # --------------------------------------------------------------------------
    # FIX 1: get_latents() is kept for potential external use but is NO LONGER
    # called by forward(). forward() delegates directly to watermarking().
    # --------------------------------------------------------------------------

    def get_latents(self, image_path):
        """
        Retrieves and processes latents from the full input image.

        NOTE: This is retained for external / diagnostic use.  The main
        watermarking pipeline no longer calls this — per-patch latents are
        obtained inside watermarking() via get_patch_latents().

        Args:
            image_path (str): Path to the input image.

        Returns:
            tuple: (reversed_latents, img_latents)
        """
        print(f' ---------- Inside Get Latents ----------')
        gt_img_tensor = get_img_tensor(image_path, self.device)
        img_latents = self.pipe.get_image_latents(gt_img_tensor, sample=False)
        
        empty_text_embeddings = self.pipe.get_text_embedding('')
        reversed_latents = self.pipe.forward_diffusion(
            latents=img_latents,
            text_embeddings=empty_text_embeddings,
            guidance_scale=1.0,
            num_inference_steps=self.num_inference_steps
        )
        torch.cuda.empty_cache()
        return reversed_latents, img_latents
        
    # --------------------------------------------------------------------------
    # BINARY SEARCH (unchanged)
    # --------------------------------------------------------------------------

    def binary_search_theta(self, gt_img_tensor, wm_img_tensor, threshold,
                            lower=0.0, upper=1.0, precision=1e-6,
                            max_iters=10000, last_loss=0.6, base_value=0.4):
        """
        Finds the optimal scaling factor for watermark adaptation using binary search,
        while conservatively adjusting theta based on the last loss value and initial SSIM.
        """
        initial_ssim = compute_ssim(gt_img_tensor, wm_img_tensor)
        best_theta = lower
        best_ssim = initial_ssim

        if initial_ssim > 0.99:
            upper = min(upper, 0.5)

        for i in range(max_iters):
            mid_theta = (lower + upper) / 2
            img_tensor = ((gt_img_tensor - wm_img_tensor) * mid_theta) + wm_img_tensor
            ssim_value = compute_ssim(img_tensor, gt_img_tensor)

            if ssim_value > best_ssim and ssim_value <= threshold:
                best_ssim = ssim_value
                best_theta = mid_theta

            if ssim_value <= threshold:
                lower = mid_theta
            else:
                upper = mid_theta

            if upper - lower < precision:
                break

        ssim_factor = min(1.0, (threshold - initial_ssim) / (1 - initial_ssim))
        if last_loss < 0.5:
            optimal_theta = min(0.5, max(base_value, best_theta * (1 + (0.6 - last_loss) * ssim_factor)))
        else:
            optimal_theta = min(0.7, max(base_value, best_theta * (1 + (last_loss - 0.6) * ssim_factor)))

        optimal_theta = max(base_value, optimal_theta)

        if best_ssim - initial_ssim < 0.0001 or initial_ssim > 0.999:
            return min(0.5, optimal_theta)

        return optimal_theta

    # --------------------------------------------------------------------------
    # ADAPTIVE ENHANCEMENT (unchanged)
    # --------------------------------------------------------------------------

    def adaptive_enhancement(self, wm_path, gt_path, last_loss):
        """
        Enhances the watermarked image adaptively to achieve the desired SSIM.

        Args:
            wm_path (str): Path to the watermarked image.
            gt_path (str): Path to the original ground-truth image.
            last_loss (float): Final training loss, used to modulate theta.

        Returns:
            str: Path to the enhanced watermarked image.
        """
        gt_img_tensor = get_img_tensor(gt_path, self.device)
        wm_img_tensor = get_img_tensor(wm_path, self.device)
        ssim_threshold = self.ssim_threshold
        ssim_value = ssim(wm_img_tensor, gt_img_tensor).item()
        print(f"Original SSIM Value: {ssim_value}")

        optimal_theta = self.binary_search_theta(
            gt_img_tensor, wm_img_tensor, ssim_threshold,
            0.0, 1.0, 1e-6, 10000, last_loss=last_loss, base_value=0.4
        )
        if self.wandb_use:
            wandb.log({'optimal_theta': optimal_theta})
        print(f"Optimal Theta: {optimal_theta}")

        enhanced_img_tensor = (((gt_img_tensor - wm_img_tensor) * optimal_theta) + wm_img_tensor)
        ssim_value = ssim(enhanced_img_tensor, gt_img_tensor).item()
        psnr_value = compute_psnr(enhanced_img_tensor, gt_img_tensor)

        # PECCAVI-FIX 6 (found once 1-5 were fixed): this probe inverted the full 512 px image (64x64 latent)
        # and tested it with a patch key whose mask is 8x8 -> IndexError on every image. Its value only
        # feeds the print/wandb log below and does not affect the enhanced image, so it is skipped.
        det_prob = float("nan")

        number = re.search(r'Image_(\d+)', gt_path).group(1)
        path = os.path.join(self.watermarked_path, f"{number}/Enhanced_{number}_{self.save_img[-1]}_wm.png")
        save_img(path, enhanced_img_tensor, self.pipe)

        print(" ")
        print("-" * 100)
        print(f'Enhanced Results:')
        print(f'SSIM: {ssim_value}')
        print(f'PSNR: {psnr_value}')
        print(f'Detection Probability: {det_prob}')
        print("-" * 100)
        if self.wandb_use:
            wandb.log({"Init Prob": det_prob})

        return path

    # --------------------------------------------------------------------------
    # PATCH UTILITIES (unchanged logic, comments clarified)
    # --------------------------------------------------------------------------

    def extract_image_patch(self, image_tensor, patch_coords):
        """
        Extracts a 64×64-pixel patch from the image tensor.

        Args:
            image_tensor (torch.Tensor): Full image tensor [1, 3, H, W].
            patch_coords (list[int]): [col, row] grid coordinates.

        Returns:
            torch.Tensor: Patch tensor [1, 3, patch_size, patch_size].
        """
        col, row = patch_coords

        # Convert grid indices to pixel boundaries
        x_start = col * self.patch_size
        y_start = row * self.patch_size
        x_end   = x_start + self.patch_size
        y_end   = y_start + self.patch_size

        patch = image_tensor[:, :, y_start:y_end, x_start:x_end].clone()
        return patch
        
    def get_patch_latents(self, image_patch):
        """
        Runs DDIM inversion on a single image patch to obtain its latent code.

        Args:
            image_patch (torch.Tensor): [1, 3, patch_size, patch_size].

        Returns:
            tuple: (reversed_patch_latents, patch_latents).
        """
        patch_latents = self.pipe.get_image_latents(image_patch, sample=False)
        empty_text_embeddings = self.pipe.get_text_embedding('')
        reversed_patch_latents = self.pipe.forward_diffusion(
            latents=patch_latents,
            text_embeddings=empty_text_embeddings,
            guidance_scale=1.0,
            num_inference_steps=self.num_inference_steps
        )
        return reversed_patch_latents, patch_latents
    
    def place_patch_in_image(self, image_tensor, patch_tensor, patch_coords):
        """
        Writes a watermarked patch back into the full image tensor.

        Args:
            image_tensor (torch.Tensor): Full image tensor [1, 3, H, W].
            patch_tensor (torch.Tensor): Watermarked patch [1, 3, patch_size, patch_size].
            patch_coords (list[int]): [col, row] grid coordinates.

        Returns:
            torch.Tensor: Updated full image tensor with the patch inserted.
        """
        col, row = patch_coords

        x_start = col * self.patch_size
        y_start = row * self.patch_size
        x_end   = x_start + self.patch_size
        y_end   = y_start + self.patch_size

        result = image_tensor.clone()
        result[:, :, y_start:y_end, x_start:x_end] = patch_tensor
        return result

    # --------------------------------------------------------------------------
    # PECCAVI-FIX 1 + 2: latent box for GTWatermark, seeded keys, persisted state
    # --------------------------------------------------------------------------

    def patch_latent_box(self):
        """
        Box in the format GTWatermark expects: (x_center, y_center, w, h) in the grid it watermarks.
        Each patch is inverted on its own, so that grid is the patch latent (64 px / 8 = 8x8) and the
        box is the whole latent; adjust_coordinates() would clamp any other 8x8 box to this one.
        """
        s = self.patch_size // self.pipe.vae_scale_factor
        return [s // 2, s // 2, s, s]

    @staticmethod
    def key_seed(image_id, patch_idx):
        """Deterministic 31-bit seed for the ring key of one patch of one image."""
        h = hashlib.sha256(f"peccavi|{image_id}|{patch_idx}".encode()).hexdigest()
        return int(h[:8], 16) & 0x7FFFFFFF

    def _make_wm_pipe(self, patch_idx, latent_shape, seed):
        wm_pipe = GTWatermark(
            bounding_boxes=[self.patch_latent_box()],
            strengths=self.strength,
            w_channel=patch_idx % 4,
            device=self.device
        )
        wm_pipe._gen_gt_(tuple(latent_shape), generator=torch.Generator("cpu").manual_seed(seed))
        if not bool(wm_pipe.watermarking_mask.any()):
            raise RuntimeError(f"Empty ring mask for patch {patch_idx} (box {self.patch_latent_box()})")
        return wm_pipe

    def save_state(self, path, image_id):
        """Everything detect() needs to rebuild the per-patch keys in another process."""
        state = {
            "image_id": image_id,
            "patch_size": self.patch_size,
            "strengths": self.strength,
            "patches": [dict(idx=i, coords=list(self.patches[i]), **self.patch_meta[i])
                        for i in sorted(self.patch_meta)],
        }
        with open(path, "w") as f:
            json.dump(state, f, indent=2)
        return path

    def load_state(self, path):
        """Restore patches and per-patch GTWatermark instances saved by save_state()."""
        with open(path) as f:
            state = json.load(f)
        self.patches = [p["coords"] for p in state["patches"]]
        self.wm_pipes, self.patch_meta = {}, {}
        for i, p in enumerate(state["patches"]):
            self.wm_pipes[i] = self._make_wm_pipe(i, p["latent_shape"], p["seed"])
            self.patch_meta[i] = {k: v for k, v in p.items() if k not in ("idx", "coords")}
        return state

    # --------------------------------------------------------------------------
    # WATERMARKING  (FIX 1 + FIX 2 applied here)
    # --------------------------------------------------------------------------

    def watermarking(self, image_name, image_path):
        """
        Applies patch-based watermarking to the image.

        FIX 1: signature no longer accepts reversed_latent — it was unused.
        FIX 2: a separate GTWatermark instance with a unique channel is created
               for each patch and stored in self.wm_pipes[patch_idx].

        Args:
            image_name (str): Base filename (no extension) for saving outputs.
            image_path (str): Path to the original image on disk.

        Returns:
            tuple: (path_to_saved_wm_image: str, avg_loss: float)
        """
        print("-" * 100)
        print("Starting Patch-Based Watermarking...")
        print(f"Processing {len(self.patches)} patches")
        print("-" * 100)
        
        gt_img_tensor = get_img_tensor(image_path, self.device)
        # Canvas that accumulates watermarked patches; starts as a copy of the original
        watermarked_img_tensor = gt_img_tensor.clone()
        
        loss_sum = 0.0
        number = re.search(r'Image_(\d+)', image_path).group(1)
        
        for patch_idx, patch_coords in enumerate(self.patches):
            print(f"\nProcessing patch {patch_idx+1}/{len(self.patches)} at grid coords {patch_coords}")
            
            # ------------------------------------------------------------------
            # FIX 2a: Create a dedicated GTWatermark for this patch.
            #   channel = patch_idx % 4  →  cycles through channels 0,1,2,3
            #   This matches the multi-channel embedding shown in Fig. 5 of the
            #   paper, where each patch uses a different frequency channel.
            # ------------------------------------------------------------------
            patch_channel = patch_idx % 4

            # Extract patch and obtain its latent via DDIM inversion
            patch_tensor = self.extract_image_patch(gt_img_tensor, patch_coords)
            reversed_patch_latents, _ = self.get_patch_latents(patch_tensor)

            # ------------------------------------------------------------------
            # FIX 2b: Initialise the watermark pattern and mask for this patch.
            #   In the original code _gen_gt_() was NEVER called, so gt_patch
            #   and watermarking_mask were both None, causing inject_watermark()
            #   to crash or silently do nothing.
            # PECCAVI-FIX 1: the box passed to GTWatermark was [col, row] (it unpacks
            #   x, y, w, h -> ValueError); it is now the patch-latent box.
            # PECCAVI-FIX 2: the ring key came from an unseeded randn_tensor and was
            #   never saved; it is now seeded from (image id, patch index) and the
            #   seed is written to state.json next to the watermarked image.
            # ------------------------------------------------------------------
            latent_shape = reversed_patch_latents.shape
            seed = self.key_seed(number, patch_idx)
            wm_pipe = self._make_wm_pipe(patch_idx, latent_shape, seed)
            self.wm_pipes[patch_idx] = wm_pipe
            self.patch_meta[patch_idx] = {
                "channel": patch_channel, "seed": seed, "box": self.patch_latent_box(),
                "latent_shape": list(latent_shape), "mask_size": int(wm_pipe.watermarking_mask.sum()),
                "placed": False,
            }

            # Prepare optimisable latent copy
            init_latents = reversed_patch_latents.detach().clone().to(self.device)
            init_latents.requires_grad = True
            optimizer = optim.Adam([init_latents], lr=self.learning_rate)
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=self.milestones, gamma=0.1
            )
            totalLoss = LossProvider(self.loss_weights, self.device)
            loss_lst = []
            
            for i in range(self.epochs):
                print(f"Epoch {i} for patch {patch_idx+1}")
                if self.wandb_use:
                    wandb.log({"epochs": i, "patch": patch_idx})
                
                try:
                    wm_pipe.watermarking_mask = wm_pipe.watermarking_mask.to(self.device)
                    wm_pipe.gt_patch          = wm_pipe.gt_patch.to(self.device)
                    
                    watermarked_patch_latents = wm_pipe.inject_watermark(init_latents)
                    
                    output = self.pipe(
                        prompt="",
                        guidance_scale=1.0,
                        num_inference_steps=self.num_inference_steps,
                        output_type='tensor',
                        use_trainable_latents=True,
                        init_latents=watermarked_patch_latents
                    )
                    
                    if isinstance(output, tuple):
                        watermarked_patch_tensor = output[0]
                    else:
                        watermarked_patch_tensor = output.images
                    
                    loss = totalLoss(
                        watermarked_patch_tensor, patch_tensor,
                        watermarked_patch_latents, wm_pipe, i + 1
                    )
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    loss_lst.append(loss.item())
                    
                    # At the final epoch, stitch this watermarked patch back into
                    # the full-image canvas
                    if i == self.epochs - 1:
                        watermarked_img_tensor = self.place_patch_in_image(
                            watermarked_img_tensor, watermarked_patch_tensor, patch_coords
                        )
                        self.patch_meta[patch_idx]["placed"] = True
                
                except RuntimeError as e:
                    print(f"Error processing patch {patch_idx+1}: {e}")
                    traceback.print_exc()
                    continue
            
            loss_sum += loss_lst[-1] if loss_lst else 0.0
            torch.cuda.empty_cache()
        
        avg_loss = loss_sum / len(self.patches) if self.patches else 0.0
        
        path_new = os.path.join(
            self.watermarked_path,
            f"{number}/{image_name}_wm_{self.save_img[-1]}.png"
        )
        os.makedirs(os.path.dirname(path_new), exist_ok=True)
        save_img(path_new, watermarked_img_tensor, self.pipe)
        self.save_state(os.path.join(os.path.dirname(path_new), "state.json"), number)  # PECCAVI-FIX 2

        torch.cuda.empty_cache()
        print("Patch-Based Watermarking Completed!")
        return path_new, avg_loss

    # --------------------------------------------------------------------------
    # FORWARD  (FIX 1: removed wasteful full-image DDIM inversion)
    # --------------------------------------------------------------------------

    def forward(self, image_path):
        """
        Top-level entry point: watermarks an image and applies adaptive enhancement.

        FIX 1: The original called get_latents() here (full-image DDIM inversion)
        and passed the result to watermarking(), which ignored it entirely.  That
        expensive call is removed.  watermarking() now receives only what it
        actually needs: image_name and image_path.

        Args:
            image_path (str): Path to the input image.

        Returns:
            str: Path to the final enhanced-watermarked image.
        """
        print(f' ---------- Inside Forward ----------')
        image_name = re.split('[/.]', image_path)[-2]

        # FIX 1: call watermarking() directly — no full-image DDIM inversion
        wm_i_path, last_loss = self.watermarking(image_name, image_path)
        print(f'\nWatermarked Image: {wm_i_path}\n')

        enhanced_i_path = self.adaptive_enhancement(wm_i_path, image_path, last_loss)
        print(f'\nEnhanced Image: {enhanced_i_path}\n')
        
        return enhanced_i_path

    # --------------------------------------------------------------------------
    # DETECT  (FIX 3: patch-level scan, WDP = max across all patches)
    # --------------------------------------------------------------------------

    def detect(self, detection_img_path, ssim_threshold=0.6):
        """
        Detects the presence of a watermark using per-patch scanning.

        FIX 3: The original ran detection on the entire 512×512 image as one
        unit.  Per the paper (Sec 3.3), the Watermark Detection Probability is:
            WDP = max_{i,j} S_{i,j}
        where i indexes patches and j indexes channels.  This implementation
        extracts each patch that was watermarked, runs DDIM inversion on it,
        queries its individual GTWatermark instance, and returns the maximum
        probability found.

        Args:
            detection_img_path (str): Path to the image under test.
            ssim_threshold (float): Unused legacy arg, kept for API compatibility.

        Returns:
            float: Maximum watermark detection probability across all patches (WDP).
        """
        if detection_img_path.endswith('.ipynb') or '.ipynb_checkpoints' in detection_img_path:
            print(f"Skipping file: {detection_img_path}")
            return None

        # Sanity-check: we need at least one wm_pipe from a watermarking() call
        if not self.wm_pipes:
            raise RuntimeError(
                "detect() called before watermarking() — self.wm_pipes is empty. "
                "Run forward() first so per-patch GTWatermark instances are populated."
            )

        # Load the full detection image
        detection_image_cv = cv2.imread(detection_img_path)
        if detection_image_cv is None:
            raise ValueError(f"Image not found at path: {detection_img_path}")

        if detection_image_cv.shape[:2] != (512, 512):
            detection_image_cv = cv2.resize(detection_image_cv, (512, 512))
        # PECCAVI-FIX 4: cv2 reads BGR; the embedder (get_img_tensor, PIL) works in RGB.
        detection_image_cv = cv2.cvtColor(detection_image_cv, cv2.COLOR_BGR2RGB)

        # Convert to a float tensor [1, 3, 512, 512] on the correct device
        detection_image_np = detection_image_cv.transpose(2, 0, 1).astype(np.float32) / 255.0
        detection_img_tensor = torch.tensor(
            detection_image_np[np.newaxis, ...]
        ).to(self.device)

        text_embeddings = self.pipe.get_text_embedding(" ").to(self.device)

        print("=" * 100)
        print("Detecting Watermark (patch-level scan)...")

        max_wdp = 0.0
        self.last_detection = []

        for patch_idx, patch_coords in enumerate(self.patches):
            # Skip if this patch didn't get a wm_pipe (e.g. early crash)
            if patch_idx not in self.wm_pipes:
                print(f"  Patch {patch_idx}: no wm_pipe, skipping.")
                continue

            wm_pipe = self.wm_pipes[patch_idx]

            # Extract the same patch region from the detection image
            patch_tensor = self.extract_image_patch(detection_img_tensor, patch_coords)

            # FIX 3: evaluate the watermark signal on this patch's own latent
            p_value = watermark_prob(patch_tensor, self.pipe, wm_pipe, text_embeddings)
            det_prob = 1.0 - p_value
            self.last_detection.append({"patch_idx": patch_idx, "coords": list(patch_coords),
                                        "channel": patch_idx % 4, "p_value": float(p_value), "wdp": float(det_prob)})
            print(f"  Patch {patch_idx} {patch_coords} ch={patch_idx % 4}: WDP = {det_prob:.4f}")

            if det_prob > max_wdp:
                max_wdp = det_prob

        print(f"Final WDP (max across patches): {max_wdp:.4f}")
        print("=" * 100)
        return max_wdp