"""
Official implementation of Split Saliency Detector for PECCAVI.
"""

# ==============================================================================
# IMPORTS
# ==============================================================================

import os
import saliency.core as saliency
import numpy as np
import PIL
from matplotlib import pyplot as P
import matplotlib.patches as patches
import torch
from torchvision import models, transforms
import tensorflow as tf
from huggingface_hub import snapshot_download
import json
from .boundingbox import Boundingbox
from shapely.geometry import box
from shapely.ops import unary_union
from glob import glob

# ==============================================================================
# CORE SALIENCY DETECTION
# ==============================================================================

class SaliencyDetector:
    """
    A class for detecting salient regions in images using InceptionV3 and XRAI.

    Provides a complete pipeline for preprocessing images, executing model
    inference with hooks to capture gradients/activations, and post-processing
    saliency maps into actionable bounding boxes.

    Attributes:
        model: Pre-trained InceptionV3 model.
        conv_layer_outputs: Cache for outputs/gradients from the target layer.
        conv_layer: The specific layer used for saliency (Mixed_7c).
        transformer: ImageNet-standard normalization transform.
    """

    def __init__(self, model_name='inception_v3'):
        """
        Initialize the detector and register model hooks.

        Args:
            model_name: Identifier for the model architecture. Defaults to 'inception_v3'.
        """
        # Load model and set to evaluation mode
        self.model = models.inception_v3(weights=True, init_weights=False)
        self.model.eval()

        # State management for Grad-CAM / XRAI hooks
        self.conv_layer_outputs = {}
        self.conv_layer = self.model.Mixed_7c
        self.conv_layer.register_forward_hook(self._conv_layer_forward)
        self.conv_layer.register_full_backward_hook(self._conv_layer_backward)

        # Standard ImageNet normalization: Mean and Std Dev
        self.transformer = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    # --------------------------------------------------------------------------
    # MODEL HOOKS
    # --------------------------------------------------------------------------

    def _conv_layer_forward(self, m, i, o):
        """
        Forward hook to capture layer activations.

        Notes:
            Moves dimensions from (B, C, H, W) to (B, H, W, C) for saliency library compatibility.
        """
        self.conv_layer_outputs[saliency.base.CONVOLUTION_LAYER_VALUES] = torch.movedim(o, 1, 3).detach().numpy()

    def _conv_layer_backward(self, m, i, o):
        """
        Backward hook to capture layer gradients.

        Notes:
            Moves dimensions from (B, C, H, W) to (B, H, W, C).
        """
        self.conv_layer_outputs[saliency.base.CONVOLUTION_OUTPUT_GRADIENTS] = torch.movedim(o[0], 1, 3).detach().numpy()

    # --------------------------------------------------------------------------
    # IMAGE PROCESSING
    # --------------------------------------------------------------------------

    def preprocess_images(self, images):
        """
        Normalize and convert image arrays to Torch tensors.

        Args:
            images: List of RGB images as numpy arrays (0-255 range).

        Returns:
            torch.Tensor: Normalized tensor with gradient tracking enabled.
        """
        images = np.array(images) / 255
        images = np.transpose(images, (0, 3, 1, 2))
        images = torch.tensor(images, dtype=torch.float32)
        images = self.transformer(images)
        return images.requires_grad_(True)

    def call_model_function(self, images, call_model_args=None, expected_keys=None):
        """
        Interface for the saliency library to query model outputs or gradients.

        Args:
            images: Preprocessed image list.
            call_model_args: Dict containing 'class_idx_str' (target class index).
            expected_keys: List of requested data types (e.g., INPUT_OUTPUT_GRADIENTS).

        Returns:
            dict: Dictionary containing captured gradients or convolution outputs.
        """
        images = self.preprocess_images(images)
        target_class_idx = call_model_args.get('class_idx_str', None)
        output = self.model(images)
        
        # Softmax to obtain class probabilities
        m = torch.nn.Softmax(dim=1)
        output = m(output)
        
        if saliency.base.INPUT_OUTPUT_GRADIENTS in expected_keys:
            # Path for Vanilla Gradients or Integrated Gradients
            outputs = output[:, target_class_idx]
            grads = torch.autograd.grad(outputs, images, grad_outputs=torch.ones_like(outputs))
            grads = torch.movedim(grads[0], 1, 3)
            gradients = grads.detach().numpy()
            return {saliency.base.INPUT_OUTPUT_GRADIENTS: gradients}
        else:
            # Path for XRAI/Grad-CAM (triggers registered hooks)
            one_hot = torch.zeros_like(output)
            one_hot[:, target_class_idx] = 1
            self.model.zero_grad()
            output.backward(gradient=one_hot, retain_graph=True)
            return self.conv_layer_outputs

    # --------------------------------------------------------------------------
    # FILE I/O & VISUALIZATION
    # --------------------------------------------------------------------------

    def load_image(self, file_path):
        """Loads and resizes an image to 512x512 for model input."""
        im = PIL.Image.open(file_path)
        im = im.resize((512, 512))
        return np.asarray(im)

    def save_image(self, im, folder, title):
        """Saves a standard image to the specified folder."""
        output_path = os.path.join(folder, f"{title}.jpg")
        P.imsave(output_path, im)
        print(f"Image saved: {output_path}")

    def save_grayscale_image(self, im, folder, title):
        """Saves a single-channel image using a gray colormap."""
        output_path = os.path.join(folder, f"{title}.jpg")
        P.imsave(output_path, im, cmap=P.cm.gray, vmin=0, vmax=1)
        print(f"Grayscale image saved: {output_path}")

    def save_heatmap(self, im, folder, title):
        """Saves a saliency map using the 'inferno' colormap."""
        output_path = os.path.join(folder, f"{title}.png")
        P.imsave(output_path, im, cmap='inferno')
        print(f"Heatmap saved: {output_path}")

    # --------------------------------------------------------------------------
    # HEATMAP ANALYSIS
    # --------------------------------------------------------------------------

    def analyze_heatmap(self, heatmap, folder, title):
        """
        Analyze saliency peaks and generate bounding box data.

        Side Effects:
            Saves a JSON file containing grid centers and merged boxes.
        """
        analyzer = Boundingbox(heatmap, grid_fraction=1/8)
        grid_centers = analyzer.find_highest_points_with_grids()
        merged_boxes_with_nms = analyzer.non_max_suppression(threshold=0.25)

        result = {
            'grid_centers': grid_centers,
            'merged_boxes': merged_boxes_with_nms
        }

        # Convert result to JSON, handling numpy integer types
        result_json = json.dumps(result, indent=4, default=lambda x: int(x) if isinstance(x, np.int64) else x)
        output_path = os.path.join(folder, f"{title}_data.json")
        with open(output_path, 'w') as f:
            f.write(result_json)
        print(f"Analysis results saved: {output_path}")

    def calculate_iou(self, box1, box2):
        """
        Calculate Intersection over Union (IoU) for two boxes.

        Args:
            box1: (minx, miny, maxx, maxy)
            box2: (minx, miny, maxx, maxy)

        Returns:
            float: IoU value between 0.0 and 1.0.
        """
        b1 = box(*box1)
        b2 = box(*box2)

        intersection_area = b1.intersection(b2).area
        union_area = b1.area + b2.area - intersection_area

        if union_area == 0:
            return 0.0
        
        return intersection_area / union_area

    # --------------------------------------------------------------------------
    # XRAI DETECTION
    # --------------------------------------------------------------------------
    
    def XRAI(self, paraphrase_folder):
        """
        Run XRAI saliency detection on all images in a folder.

        Processes images, identifies the predicted class, generates attributions,
        highlights top/bottom regions, and saves visual results.

        Args:
            paraphrase_folder: Path to directory containing image files.
        """
        for img_filename in os.listdir(paraphrase_folder):
            if img_filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                # Filter out previous analysis outputs
                if "Heatmap" in img_filename or "Top" in img_filename: 
                    print(f'Leaving out {img_filename}')
                    continue 

                img_path = os.path.join(paraphrase_folder, img_filename)
                print(f"Processing image: {img_path}")

                img_result_dir = os.path.join(paraphrase_folder, os.path.splitext(img_filename)[0])
                os.makedirs(img_result_dir, exist_ok=True)

                im_orig = self.load_image(img_path)
                im_tensor = self.preprocess_images([im_orig])
                predictions = self.model(im_tensor)
                predictions = predictions.detach().numpy()
                prediction_class = np.argmax(predictions[0])
                call_model_args = {'class_idx_str': prediction_class}

                print("Prediction class: " + str(prediction_class))

                # Initialize and run XRAI
                xrai_object = saliency.XRAI()
                xrai_params = saliency.XRAIParameters()
                xrai_params.algorithm = 'fast'
                xrai_attributions_fast = xrai_object.GetMask(
                    im_orig, self.call_model_function, call_model_args, 
                    extra_parameters=xrai_params, batch_size=20
                )

                self.save_image(im_orig, img_result_dir, title='Original Image')
                self.save_heatmap(xrai_attributions_fast, img_result_dir, title='XRAI Heatmap')

                # Thresholding: 75th percentile for Top (Red), 25th percentile for Bottom (Blue)
                top_15_mask = xrai_attributions_fast >= np.percentile(xrai_attributions_fast, 75)
                bottom_15_mask = xrai_attributions_fast <= np.percentile(xrai_attributions_fast, 25)

                im_combined = np.array(im_orig).copy()
                im_combined[top_15_mask] = [255, 0, 0] # Red
                im_combined[bottom_15_mask & ~top_15_mask] = [0, 0, 255] # Blue

                self.save_image(im_combined, img_result_dir, title='Top15_Bottom15_Combined')
                self.analyze_heatmap(xrai_attributions_fast, img_result_dir, title='XRAI_Analysis')
            else:
                print(f"Skipping file: {img_filename}'")
        
        self.process_and_score_boxes(base_path=paraphrase_folder)

    # --------------------------------------------------------------------------
    # SCORING & AGGREGATION
    # --------------------------------------------------------------------------

    def calculate_score(self, overlap_count, total_images):
        """
        Assign a confidence score based on box detection frequency.

        Args:
            overlap_count: How many images shared a similar box.
            total_images: Normalization base (usually the number of paraphrases).

        Returns:
            float: Score value from the predefined list.
        """
        score_list = [1, 0.75, 0.5, 0.25, 0.1]
        return score_list[min(overlap_count, total_images) - 1]

    def boxes_overlap(self, box1, box2):
        """Boolean check for spatial intersection using Shapely."""
        b1 = box(*box1)
        b2 = box(*box2)
        return b1.intersects(b2)

    def merge_boxes(self, boxes):
        """Merge multiple boxes into a single bounding envelope."""
        merged_box = unary_union([box(*b) for b in boxes])
        return merged_box.bounds

    def process_and_score_boxes(self, base_path, merge_iou_threshold=0.01, nms_iou_threshold=0.01, score_threshold=0.25, versions=5):
        """
        Aggregate saliency results across multiple image versions and save a final JSON.

        1. Scans version directories for JSON analysis files.
        2. Aggregates and counts occurrences of unique boxes.
        3. Merges overlapping boxes and calculates confidence scores.
        4. Filters by score and applies a secondary padding check.
        5. [NEW] Maps surviving NMP boxes onto the patch grid and writes
           patch-indexed entries so watermark.py can consume the output.

        Returns:
            str: Path to the generated final.json file.
        """
        box_map = {}

        # Identify all sub-directories containing version data
        directories = [name for name in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, name))]

        for directory in directories:
            json_files = glob(os.path.join(base_path, directory, "*.json"))
            for file in json_files:
                with open(file, 'r') as f:
                    data = json.load(f)
                    for b in data['merged_boxes']:
                        box_tuple = (b["x1"], b["y1"], b["x2"], b["y2"])
                        if box_tuple in box_map:
                            box_map[box_tuple] += 1
                        else:
                            box_map[box_tuple] = 1

        merged_boxes = []
        visited_boxes = set()
        for box_coords, count in box_map.items():
            if box_coords in visited_boxes:
                continue
            overlapping_boxes = [box_coords]
            for other_box in box_map:
                if other_box != box_coords and other_box not in visited_boxes:
                    iou = self.calculate_iou(box_coords, other_box)
                    if iou > merge_iou_threshold:
                        overlapping_boxes.append(other_box)
                        visited_boxes.add(other_box)

            # Assign score based on frequency or group size
            if len(overlapping_boxes) > 1:
                merged_box = self.merge_boxes(overlapping_boxes)
                score = self.calculate_score(len(overlapping_boxes), 5)
            else:
                merged_box = box_coords
                score = self.calculate_score(count, 5)

            merged_boxes.append({"box": list(merged_box), "score": score})
            visited_boxes.add(box_coords)

        # Non-Maximum Suppression (NMS) style filtering
        merged_boxes.sort(key=lambda x: x["score"], reverse=True)
        final_boxes = []

        for box_info in merged_boxes:
            if box_info["score"] < score_threshold:
                continue

            should_keep = True
            for kept_box in final_boxes:
                # Discard if box is nested within existing padded box
                if self.is_within_padding(box_info["box"], kept_box["box"]):
                    should_keep = False
                    break

                iou = self.calculate_iou(box_info["box"], kept_box["box"])
                if iou > nms_iou_threshold:
                    should_keep = False
                    break

            if should_keep:
                final_boxes.append(box_info)

        # Fallback: Default box if no regions pass the score threshold
        if not final_boxes:
            default_box = {"box": [256, 256, 512, 512], "score": 1.0}
            final_boxes.append(default_box)

        # ------------------------------------------------------------------
        # NEW: Map NMP bounding boxes → patch grid coordinates.
        #
        # Why this is needed:
        #   process_and_score_boxes() produces pixel-space boxes [x1,y1,x2,y2].
        #   watermark.py's extract_patches() reads {"patch": [col, row]} entries.
        #   Nothing previously bridged these two formats, so extract_patches()
        #   always returned an empty list and no patch was ever watermarked.
        #
        # What this does:
        #   Divides the 512×512 image into an 8×8 grid of 64×64 px patches.
        #   For every (col, row) patch, accumulates the weighted NMP area that
        #   falls inside it (weight = NMP score).  Selects the top-K unique
        #   patches by weighted coverage and appends {"patch": [col, row],
        #   "score": normalised_coverage} to final_boxes before writing the JSON.
        # ------------------------------------------------------------------
        patch_entries = self.map_boxes_to_patches(
            final_boxes,
            image_size=512,
            patch_size=64,
            top_k=4           # keep at most 4 patches (tune via config if needed)
        )
        final_boxes = patch_entries          # replace raw-box list with patch list

        output_path = os.path.join(base_path, "final.json")
        with open(output_path, 'w') as outfile:
            json.dump(final_boxes, outfile, indent=4)

        print(f"Final JSON created at {output_path} with {len(final_boxes)} patch entries.")
        return output_path

    # --------------------------------------------------------------------------
    # NEW METHOD: NMP boxes → patch grid indices
    # --------------------------------------------------------------------------

    def map_boxes_to_patches(self, nmp_boxes, image_size=512, patch_size=64, top_k=4):
        """
        Convert pixel-space NMP bounding boxes into patch-grid coordinates.

        The image is divided into an (image_size // patch_size) × (image_size // patch_size)
        uniform grid.  Each patch is scored by the total NMP-weighted area that
        overlaps with it.  The top-K highest-scoring patches are returned.

        Args:
            nmp_boxes (list[dict]): Output of process_and_score_boxes before this
                call — each entry is {"box": [x1, y1, x2, y2], "score": float}.
            image_size (int): Side length of the (square) image in pixels. Default 512.
            patch_size (int): Side length of each patch in pixels. Default 64,
                giving an 8×8 grid for a 512×512 image.
            top_k (int): Maximum number of patch entries to return.

        Returns:
            list[dict]: Each entry is {"patch": [col, row], "score": float},
                where col/row are zero-based grid indices and score is the
                normalised weighted coverage (0–1).  If top_k is larger than
                the number of patches that received any coverage, all covered
                patches are returned; a fallback centre patch is added when
                every patch scores zero.
        """
        n_patches = image_size // patch_size          # e.g. 8 for 512/64
        # Accumulator: patch_scores[row][col] = total weighted overlap area
        patch_scores = np.zeros((n_patches, n_patches), dtype=np.float64)

        for nmp in nmp_boxes:
            x1, y1, x2, y2 = nmp["box"]
            nmp_score = nmp["score"]

            # Clamp NMP box to image boundaries
            x1 = max(0.0, float(x1))
            y1 = max(0.0, float(y1))
            x2 = min(float(image_size), float(x2))
            y2 = min(float(image_size), float(y2))

            if x2 <= x1 or y2 <= y1:
                # Degenerate box — skip
                continue

            # Iterate only over patches that could possibly intersect this NMP box.
            # col_start/end and row_start/end are the range of patch indices that
            # the NMP box spans, so we avoid an O(n_patches^2) double loop.
            col_start = int(x1 // patch_size)
            col_end   = min(n_patches - 1, int((x2 - 1) // patch_size))
            row_start = int(y1 // patch_size)
            row_end   = min(n_patches - 1, int((y2 - 1) // patch_size))

            for row in range(row_start, row_end + 1):
                for col in range(col_start, col_end + 1):
                    # Pixel boundaries of this patch cell
                    px1 = col * patch_size
                    py1 = row * patch_size
                    px2 = px1 + patch_size
                    py2 = py1 + patch_size

                    # Intersection area between NMP box and patch cell
                    inter_x1 = max(x1, px1)
                    inter_y1 = max(y1, py1)
                    inter_x2 = min(x2, px2)
                    inter_y2 = min(y2, py2)

                    inter_w = max(0.0, inter_x2 - inter_x1)
                    inter_h = max(0.0, inter_y2 - inter_y1)
                    intersection = inter_w * inter_h

                    if intersection > 0:
                        # Weight the overlap by the NMP's confidence score so that
                        # highly-stable NMPs (score=1.0) contribute more than
                        # weakly-stable ones (score=0.25).
                        patch_scores[row, col] += intersection * nmp_score

        # Normalise so scores are in [0, 1] relative to the best patch
        max_score = patch_scores.max()
        if max_score == 0:
            # No NMP overlapped any patch at all — use the centre patch as fallback
            print("Warning: no NMP-patch overlap found; using centre patch as fallback.")
            centre = n_patches // 2
            return [{"patch": [centre, centre], "score": 1.0}]

        patch_scores_norm = patch_scores / max_score

        # Flatten, sort by score descending, keep top_k
        flat_indices = np.argsort(patch_scores_norm.ravel())[::-1]
        patch_entries = []
        for flat_idx in flat_indices[:top_k]:
            row = int(flat_idx // n_patches)
            col = int(flat_idx  % n_patches)
            score = float(patch_scores_norm[row, col])
            if score == 0:
                # Once we hit zero-score patches we can stop
                break
            patch_entries.append({
                "patch": [col, row],    # [col, row] matches watermark.py's (x, y) convention
                "score": round(score, 4)
            })

        print(f"Mapped {len(nmp_boxes)} NMP boxes → {len(patch_entries)} patch(es): "
              f"{[e['patch'] for e in patch_entries]}")
        return patch_entries

    # --------------------------------------------------------------------------
    # CONTAINMENT HELPER
    # --------------------------------------------------------------------------

    def is_within_padding(self, box1, box2):
        """
        Check if one box is contained within the expanded bounds of another.

        Args:
            box1: list [x1, y1, x2, y2]
            box2: list [x1, y1, x2, y2]

        Returns:
            bool: True if containment is detected under padding expansion.
        """
        def is_contained(inner, outer):
            return (inner[0] >= outer[0] and inner[1] >= outer[1] and 
                    inner[2] <= outer[2] and inner[3] <= outer[3])

        # Expand box2 by its dimensions to create a "safe" padding zone
        padded_box2 = [box2[0], box2[1], box2[2] + (box2[2] - box2[0]), box2[3] + (box2[3] - box2[1])]
        if is_contained(box1, padded_box2):
            return True

        # Expand box1 similarly
        padded_box1 = [box1[0], box1[1], box1[2] + (box1[2] - box1[0]), box1[3] + (box1[3] - box1[1])]
        if is_contained(box2, padded_box1):
            return True

        return False

# ==============================================================================
# END OF MODULE
# ==============================================================================