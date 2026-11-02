"""
Official implemnetation of Saliency Detector for PECCAVI.

This module provides a suite of tools for visual saliency detection, leveraging 
InceptionV3 as a backbone and integrating multiple saliency algorithms including 
XRAI, Grad-CAM, SmoothGrad, and MSI-Net. It also includes logic for bounding 
box aggregation and scoring based on cross-version consistency.
"""

# ==============================================================================
# IMPORTS
# ==============================================================================

import os
import saliency.core as saliency
import numpy as np
import PIL
import keras
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
import keras

# ==============================================================================
# SALIENCY DETECTION ENGINE
# ==============================================================================

class SaliencyDetector:
    """
    Orchestrates saliency map generation and analysis using deep learning models.

    This class integrates various interpretability techniques to identify 
    important regions in an image. It handles model hooks for gradient 
    extraction, image preprocessing, and post-inference spatial analysis.

    Attributes:
        model: The InceptionV3 model used for feature extraction and gradients.
        conv_layer_outputs: Storage for layer activations and gradients via hooks.
        conv_layer: The target layer (Mixed_7c) used for Grad-CAM style methods.
        transformer: Normalization transformation for InceptionV3 input.
        hf_dir: Local path to downloaded MSI-Net assets from Hugging Face.
    """

    def __init__(self, model_name='inception_v3', repo_id="alexanderkroner/MSI-Net"):
        """
        Initializes the detector, loads weights, and sets up instrumentation hooks.

        Args:
            model_name: Identifier for the backbone (defaults to 'inception_v3').
            repo_id: Hugging Face repository for MSI-Net saliency models.
        """
        # Load InceptionV3 with ImageNet weights
        self.model = models.inception_v3(weights=True, init_weights=False)
        self.model.eval()

        # Internal state for hooks
        self.conv_layer_outputs = {}
        # 'Mixed_7c' is the final convolutional layer in InceptionV3
        self.conv_layer = self.model.Mixed_7c
        
        # Register hooks to capture internal tensors for saliency calculation
        self.conv_layer.register_forward_hook(self._conv_layer_forward)
        self.conv_layer.register_full_backward_hook(self._conv_layer_backward)

        # Download external MSI-net model assets
        self.hf_dir = snapshot_download(repo_id)
        # Note: self.MSImodel initialization is currently commented out in source
        # self.MSImodel = keras.layers.TFSMLayer("/root/.cache/huggingface/hub/models--alexanderkroner--MSI-Net/snapshots/d950b35945db961ae63f84bc2b23f6bd578d0b8f",call_endpoint='serving_default')

        # Standard ImageNet normalization parameters
        self.transformer = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    # --------------------------------------------------------------------------
    # MODEL HOOKS
    # --------------------------------------------------------------------------

    def _conv_layer_forward(self, m, i, o):
        """
        Captures forward pass activations of the convolutional layer.

        Notes:
            - Moves dimensions from NCHW (Torch) to NHWC (expected by saliency.core).
        """
        self.conv_layer_outputs[saliency.base.CONVOLUTION_LAYER_VALUES] = torch.movedim(o, 1, 3).detach().numpy()

    def _conv_layer_backward(self, m, i, o):
        """
        Captures backward pass gradients of the convolutional layer.

        Notes:
            - Moves dimensions from NCHW to NHWC.
        """
        self.conv_layer_outputs[saliency.base.CONVOLUTION_OUTPUT_GRADIENTS] = torch.movedim(o[0], 1, 3).detach().numpy()

    # --------------------------------------------------------------------------
    # DATA PREPROCESSING
    # --------------------------------------------------------------------------

    def preprocess_images(self, images):
        """
        Prepares raw image arrays for InceptionV3 inference.

        Args:
            images: List of images as numpy arrays (0-255).

        Returns:
            torch.Tensor: Normalized tensor with gradients enabled.
        """
        images = np.array(images) / 255
        images = np.transpose(images, (0, 3, 1, 2))  # HWC to CHW
        images = torch.tensor(images, dtype=torch.float32)
        images = self.transformer(images)
        return images.requires_grad_(True)

    def call_model_function(self, images, call_model_args=None, expected_keys=None):
        """
        Standard interface for the saliency library to interact with the model.

        Args:
            images: List of numpy arrays to be processed.
            call_model_args: Dict containing 'class_idx_str' (the target class).
            expected_keys: List of saliency features requested (e.g., gradients).

        Returns:
            dict: Maps requested keys (Gradients or Layer Values) to numpy arrays.
        """
        images = self.preprocess_images(images)
        if call_model_args != None:
            target_class_idx = call_model_args.get('class_idx_str', None)
        
        output = self.model(images)
        m = torch.nn.Softmax(dim=1)
        output = m(output)
        
        if saliency.base.INPUT_OUTPUT_GRADIENTS in expected_keys:
            # Direct gradient computation (Saliency/Integrated Gradients)
            outputs = output[:, target_class_idx]
            grads = torch.autograd.grad(outputs, images, grad_outputs=torch.ones_like(outputs))
            grads = torch.movedim(grads[0], 1, 3)
            gradients = grads.detach().numpy()
            return {saliency.base.INPUT_OUTPUT_GRADIENTS: gradients}
        else:
            # Backpropagation to trigger hooks (Grad-CAM/XRAI)
            one_hot = torch.zeros_like(output)
            one_hot[:, target_class_idx] = 1
            self.model.zero_grad()
            output.backward(gradient=one_hot, retain_graph=True)
            return self.conv_layer_outputs

    # --------------------------------------------------------------------------
    # FILE I/O UTILITIES
    # --------------------------------------------------------------------------

    def load_image(self, file_path):
        """Loads and resizes an image to a fixed 512x512 resolution."""
        im = PIL.Image.open(file_path)
        im = im.resize((512, 512))
        return np.asarray(im)

    def save_image(self, im, folder, title):
        """Saves a standard RGB image using matplotlib."""
        output_path = os.path.join(folder, f"{title}.jpg")
        P.imsave(output_path, im)
        print(f"Image saved: {output_path}")

    def save_grayscale_image(self, im, folder, title):
        """Saves an image using a grayscale colormap."""
        output_path = os.path.join(folder, f"{title}.jpg")
        P.imsave(output_path, im, cmap=P.cm.gray, vmin=0, vmax=1)
        print(f"Grayscale image saved: {output_path}")

    def save_heatmap(self, im, folder, title):
        """Saves a saliency attribution map using the 'inferno' colormap."""
        output_path = os.path.join(folder, f"{title}.png")
        P.imsave(output_path, im, cmap='inferno')
        print(f"Heatmap saved: {output_path}")

    # --------------------------------------------------------------------------
    # SPATIAL ANALYSIS
    # --------------------------------------------------------------------------

    def analyze_heatmap(self, heatmap, folder, title):
        """
        Extracts bounding box data from a heatmap and saves as JSON.

        Side Effects:
            - Writes a JSON file containing grid centers and NMS-merged boxes.
        """
        analyzer = Boundingbox(heatmap, grid_fraction=1/8)
        grid_centers = analyzer.find_highest_points_with_grids()
        merged_boxes_with_nms = analyzer.non_max_suppression(threshold=0.25)

        result = {
            'grid_centers': grid_centers,
            'merged_boxes': merged_boxes_with_nms
        }

        # Handle numpy int64 serialization issues
        result_json = json.dumps(result, indent=4, default=lambda x: int(x) if isinstance(x, np.int64) else x)
        output_path = os.path.join(folder, f"{title}_data.json")
        with open(output_path, 'w') as f:
            f.write(result_json)
        print(f"Analysis results saved: {output_path}")

    def calculate_iou(self, box1_coords, box2_coords):
        """
        Calculates Intersection over Union using Shapely.

        Args:
            box1_coords: (minx, miny, maxx, maxy)
            box2_coords: (minx, miny, maxx, maxy)
        """
        b1 = box(*box1_coords)
        b2 = box(*box2_coords)

        intersection_area = b1.intersection(b2).area
        union_area = b1.area + b2.area - intersection_area

        if union_area == 0:
            return 0.0
        
        return intersection_area / union_area

    # --------------------------------------------------------------------------
    # SALIENCY ALGORITHMS
    # --------------------------------------------------------------------------

    # 
    def XRAI(self, paraphrase_folder):
        """
        Executes the XRAI saliency algorithm on images within a directory.

        Processes files, generates 'fast' XRAI attributions, and creates a combined 
        mask highlighting the top and bottom 25% of salient regions.

        Notes:
            - Although documentation in method comments mentions 'top 15%', 
              the logic uses `percentile(75)` which isolates the top 25%.
        """
        for img_filename in os.listdir(paraphrase_folder):
            if img_filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                if "Heatmap" in img_filename or "Top" in img_filename : 
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

                xrai_object = saliency.XRAI()
                xrai_params = saliency.XRAIParameters()
                xrai_params.algorithm = 'fast'
                xrai_attributions_fast = xrai_object.GetMask(
                    im_orig, self.call_model_function, call_model_args, 
                    extra_parameters=xrai_params, batch_size=20
                )

                self.save_image(im_orig, img_result_dir, title='Original Image')
                self.save_heatmap(xrai_attributions_fast, img_result_dir, title='XRAI Heatmap')

                # Percentile logic isolates top and bottom quartiles
                top_15_mask = xrai_attributions_fast >= np.percentile(xrai_attributions_fast, 75)
                bottom_15_mask = xrai_attributions_fast <= np.percentile(xrai_attributions_fast, 25)

                im_combined = np.array(im_orig).copy()
                # Red for High Saliency
                im_combined[top_15_mask] = [255, 0, 0]
                # Blue for Low Saliency
                im_combined[bottom_15_mask & ~top_15_mask] = [0, 0, 255]

                self.save_image(im_combined, img_result_dir, title='Top15_Bottom15_Combined')
                self.analyze_heatmap(xrai_attributions_fast, img_result_dir, title='XRAI_Analysis')
            else:
                print(f"Skipping file: {img_filename}'")
        
        self.process_and_score_boxes(base_path=paraphrase_folder)

    # --------------------------------------------------------------------------
    # MSI-NET METHODS
    # --------------------------------------------------------------------------

    def get_target_shape_MSI(self, original_shape):
        """Determines optimal aspect ratio padding for MSI-Net input (320px)."""
        original_aspect_ratio = original_shape[0] / original_shape[1]

        square_mode = abs(original_aspect_ratio - 1.0)
        landscape_mode = abs(original_aspect_ratio - 240 / 320)
        portrait_mode = abs(original_aspect_ratio - 320 / 240)

        best_mode = min(square_mode, landscape_mode, portrait_mode)

        if best_mode == square_mode:
            target_shape = (320, 320)
        elif best_mode == landscape_mode:
            target_shape = (240, 320)
        else:
            target_shape = (320, 240)

        return target_shape

    def preprocess_input_MSI(self, input_image, target_shape):
        """Resizes and pads image for MSI-Net compatibility."""
        input_tensor = tf.expand_dims(input_image, axis=0)

        input_tensor = tf.image.resize(
            input_tensor, target_shape, preserve_aspect_ratio=True
        )

        vertical_padding = target_shape[0] - input_tensor.shape[1]
        horizontal_padding = target_shape[1] - input_tensor.shape[2]

        vertical_padding_1 = vertical_padding // 2
        vertical_padding_2 = vertical_padding - vertical_padding_1
        horizontal_padding_1 = horizontal_padding // 2
        horizontal_padding_2 = horizontal_padding - horizontal_padding_1

        input_tensor = tf.pad(
            input_tensor,
            [
                [0, 0],
                [vertical_padding_1, vertical_padding_2],
                [horizontal_padding_1, horizontal_padding_2],
                [0, 0],
            ],
        )

        return (
            input_tensor,
            [vertical_padding_1, vertical_padding_2],
            [horizontal_padding_1, horizontal_padding_2],
        )

    def postprocess_output_MSI(self, output_tensor, vertical_padding, horizontal_padding, original_shape):
        """Removes padding and rescales saliency map to original dimensions."""
        output_tensor = output_tensor[
            :,
            vertical_padding[0] : output_tensor.shape[1] - vertical_padding[1],
            horizontal_padding[0] : output_tensor.shape[2] - horizontal_padding[1],
            :,
        ]

        output_tensor = tf.image.resize(output_tensor, original_shape)
        output_array = output_tensor.numpy().squeeze()
        output_array = P.cm.inferno(output_array)[..., :3]

        return output_array

    def MSI(self, paraphrase_folder):
        """Runs the MSI-Net saliency detection pipeline."""
        for img_filename in os.listdir(paraphrase_folder):
            if img_filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                if "Heatmap" in img_filename or "Top" in img_filename : 
                    print(f'Leaving out {img_filename}')
                    continue 

                img_path = os.path.join(paraphrase_folder, img_filename)
                print(f"Processing image: {img_path}")

                img_result_dir = os.path.join(paraphrase_folder, os.path.splitext(img_filename)[0])
                os.makedirs(img_result_dir, exist_ok=True)

                im_orig = self.load_image(img_path)

                input_image = tf.keras.utils.load_img(img_path)
                input_image = np.array(input_image, dtype=np.float32)
                original_shape = input_image.shape[:2]
                target_shape = (320, 320)

                input_tensor, vertical_padding, horizontal_padding = self.preprocess_input_MSI(
                    input_image, target_shape
                )
                # Note: self.MSImodel must be initialized elsewhere as it is commented in __init__
                output_tensor = self.MSImodel(input_tensor)['layer_from_saved_model']
                
                saliency_map = self.postprocess_output_MSI(
                    output_tensor, vertical_padding, horizontal_padding, original_shape
                )

                self.save_image(im_orig, img_result_dir, title='Original Image')
                saliency_map_2d = np.max(saliency_map, axis=2) 

                self.save_heatmap(saliency_map_2d, img_result_dir, title='MSI Heatmap')

                top_15_mask = saliency_map_2d >= np.percentile(saliency_map_2d, 75)
                bottom_15_mask = saliency_map_2d <= np.percentile(saliency_map_2d, 25)

                im_combined = np.array(im_orig).copy()
                im_combined[top_15_mask] = [255, 0, 0]
                im_combined[bottom_15_mask & ~top_15_mask] = [0, 0, 255]

                self.save_image(im_combined, img_result_dir, title='Top15_Bottom15_Combined')
                self.analyze_heatmap(saliency_map_2d, img_result_dir, title='MSInet_Analysis')
            else:
                print(f"Skipping file: {img_filename}'")
            
            self.process_and_score_boxes(base_path=paraphrase_folder)

    # --------------------------------------------------------------------------
    # ALTERNATIVE GRADIENT ALGORITHMS
    # --------------------------------------------------------------------------

    def Vanilla_Grad(self, paraphrase_folder):
        """Standard Gradient Saliency detection."""
        for img_filename in os.listdir(paraphrase_folder):
            if img_filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                if "Heatmap" in img_filename or "Top" in img_filename : 
                    continue 

                img_path = os.path.join(paraphrase_folder, img_filename)
                img_result_dir = os.path.join(paraphrase_folder, os.path.splitext(img_filename)[0])
                os.makedirs(img_result_dir, exist_ok=True)

                im_orig = self.load_image(img_path)
                im_tensor = self.preprocess_images([im_orig])
                predictions = self.model(im_tensor).detach().numpy()
                prediction_class = np.argmax(predictions[0])
                
                saliency_map = saliency.GradientSaliency().GetMask(im_orig, self.call_model_function, {'class_idx_str': prediction_class})
                saliency_map = saliency.VisualizeImageGrayscale(saliency_map)

                self.save_heatmap(saliency_map, img_result_dir, title='Vanilla_Grad Heatmap')
                self.analyze_heatmap(saliency_map, img_result_dir, title='Vanilla_Grad_Analysis')
            
            self.process_and_score_boxes(base_path=paraphrase_folder)

    def Smooth_grad(self, paraphrase_folder):
        """SmoothGrad (Gradient + Gaussian Noise) detection."""
        for img_filename in os.listdir(paraphrase_folder):
            if img_filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                if "Heatmap" in img_filename or "Top" in img_filename : 
                    continue 

                img_path = os.path.join(paraphrase_folder, img_filename)
                img_result_dir = os.path.join(paraphrase_folder, os.path.splitext(img_filename)[0])
                os.makedirs(img_result_dir, exist_ok=True)

                im_orig = self.load_image(img_path)
                im_tensor = self.preprocess_images([im_orig])
                prediction_class = np.argmax(self.model(im_tensor).detach().numpy()[0])
                
                saliency_map = saliency.GradientSaliency().GetSmoothedMask(im_orig, self.call_model_function, {'class_idx_str': prediction_class})
                saliency_map = saliency.VisualizeImageGrayscale(saliency_map)

                self.save_heatmap(saliency_map, img_result_dir, title='Smooth_Grad Heatmap')
                self.analyze_heatmap(saliency_map, img_result_dir, title='Smooth_Grad_Analysis')
            
            self.process_and_score_boxes(base_path=paraphrase_folder)

    def Vanilla_integrated(self, paraphrase_folder):
        """Integrated Gradients saliency detection."""
        for img_filename in os.listdir(paraphrase_folder):
            if img_filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                if "Heatmap" in img_filename or "Top" in img_filename : 
                    continue 

                img_path = os.path.join(paraphrase_folder, img_filename)
                img_result_dir = os.path.join(paraphrase_folder, os.path.splitext(img_filename)[0])
                os.makedirs(img_result_dir, exist_ok=True)

                im_orig = self.load_image(img_path)
                im_tensor = self.preprocess_images([im_orig])
                prediction_class = np.argmax(self.model(im_tensor).detach().numpy()[0])
                
                saliency_map = saliency.IntegratedGradients().GetMask(
                    im_orig, self.call_model_function, {'class_idx_str': prediction_class}, 
                    x_steps=25, x_baseline=np.zeros(im_orig.shape), batch_size=20
                )
                saliency_map = saliency.VisualizeImageGrayscale(saliency_map)

                self.save_heatmap(saliency_map, img_result_dir, title='Vanilla_integrated Heatmap')
                self.analyze_heatmap(saliency_map, img_result_dir, title='Vanilla_integrated_Analysis')
            
            self.process_and_score_boxes(base_path=paraphrase_folder)

    def Smooth_grad_integrated(self, paraphrase_folder):
        """Smoothed Integrated Gradients saliency detection."""
        for img_filename in os.listdir(paraphrase_folder):
            if img_filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                if "Heatmap" in img_filename or "Top" in img_filename : 
                    continue 

                img_path = os.path.join(paraphrase_folder, img_filename)
                img_result_dir = os.path.join(paraphrase_folder, os.path.splitext(img_filename)[0])
                os.makedirs(img_result_dir, exist_ok=True)

                im_orig = self.load_image(img_path)
                im_tensor = self.preprocess_images([im_orig])
                prediction_class = np.argmax(self.model(im_tensor).detach().numpy()[0])
                
                saliency_map = saliency.IntegratedGradients().GetSmoothedMask(
                    im_orig, self.call_model_function, {'class_idx_str': prediction_class}, 
                    x_steps=25, x_baseline=np.zeros(im_orig.shape), batch_size=20
                )
                saliency_map = saliency.VisualizeImageGrayscale(saliency_map)
                
                self.save_heatmap(saliency_map, img_result_dir, title='Smooth_grad_integrated Heatmap')
                self.analyze_heatmap(saliency_map, img_result_dir, title='Smooth_grad_integrated_Analysis')
            
            self.process_and_score_boxes(base_path=paraphrase_folder)

    # --------------------------------------------------------------------------
    # BOX AGGREGATION & SCORING
    # --------------------------------------------------------------------------

    def calculate_score(self, overlap_count, total_images):
        """
        Maps the frequency of a box detection across versions to a quality score.

        Args:
            overlap_count: Number of times a similar box appeared.
            total_images: Normalization factor for versions.

        Returns:
            float: Score from 0.1 to 1.0 based on consistency.
        """
        score_list = [1, 0.75, 0.5, 0.25, 0.1]
        return score_list[min(overlap_count, total_images) - 1]

    def boxes_overlap(self, box1_coords, box2_coords):
        """Boolean check for any intersection between two boxes."""
        b1 = box(*box1_coords)
        b2 = box(*box2_coords)
        return b1.intersects(b2)

    def merge_boxes(self, boxes_coords):
        """Creates the smallest encompassing bounds for multiple boxes."""
        merged_box = unary_union([box(*b) for b in boxes_coords])
        return merged_box.bounds

    def process_and_score_boxes(self, base_path, merge_iou_threshold=0.01, nms_iou_threshold=0.01, score_threshold=0.25, versions=5):
        """
        Aggregates detected regions across multiple folders and versions.

        1. Tallies occurrences of specific bounding boxes.
        2. Merges boxes with high IoU.
        3. Assigns scores based on cross-version consistency.
        4. Performs NMS and containment checks.

        Returns:
            str: Path to the generated 'final.json' file.
        """
        box_map = {}
        directories = [name for name in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, name))]

        # Step 1: Collect all boxes from all JSON files in the base directory
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

        # Step 2: Merge overlapping boxes based on IoU threshold
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

            if len(overlapping_boxes) > 1:
                merged_box = self.merge_boxes(overlapping_boxes)
                score = self.calculate_score(len(overlapping_boxes), 5)
            else:
                merged_box = box_coords
                score = self.calculate_score(count, 5)

            merged_boxes.append({"box": list(merged_box), "score": score})
            visited_boxes.add(box_coords)

        # Step 3: Sort by score and apply NMS/Padding logic
        merged_boxes.sort(key=lambda x: x["score"], reverse=True)
        final_boxes = []

        for box_info in merged_boxes:
            if box_info["score"] < score_threshold:
                continue

            should_keep = True
            for kept_box in final_boxes:
                # Discard if box is spatially redundant (nested with padding)
                if self.is_within_padding(box_info["box"], kept_box["box"]):
                    should_keep = False
                    break

                iou = self.calculate_iou(box_info["box"], kept_box["box"])
                if iou > nms_iou_threshold:
                    should_keep = False
                    break

            if should_keep:
                # Associate the box with the most overlapping patch grid cell
                patch = self.find_max_overlap_patches(box_info["box"])
                box_info["patch"] = patch
                final_boxes.append(box_info)

        # Ensure result is never empty by providing a default center box
        if not final_boxes:
            default_box = {"box": [256, 256, 512, 512], "score": 1.0, "patch": self.find_max_overlap_patches([256, 256, 512, 512])}
            final_boxes.append(default_box)

        output_path = os.path.join(base_path, "final.json")
        with open(output_path, 'w') as outfile:
            json.dump(final_boxes, outfile, indent=4)

        print(f"Final JSON created at {output_path}")
        return output_path

    def is_within_padding(self, box1, box2):
        """
        Determines if one box is contained within the 'double-sized' bounds of another.

        Args:
            box1: [x1, y1, x2, y2]
            box2: [x1, y1, x2, y2]

        Returns:
            bool: True if containment occurs under expanded padding.
        """
        def is_contained(inner, outer):
            return (inner[0] >= outer[0] and inner[1] >= outer[1] and 
                    inner[2] <= outer[2] and inner[3] <= outer[3])

        # Expand box2 bounds by its own width/height
        padded_box2 = [box2[0], box2[1], box2[2] + (box2[2] - box2[0]), box2[3] + (box2[3] - box2[1])]
        if is_contained(box1, padded_box2):
            return True

        # Expand box1 bounds by its own width/height
        padded_box1 = [box1[0], box1[1], box1[2] + (box1[2] - box1[0]), box1[3] + (box1[3] - box1[1])]
        if is_contained(box2, padded_box1):
            return True

        return False

    def find_max_overlap_patches(self, box_coords, patch_size=64, image_size=(512, 512)):
        """
        Identifies the patch grid index (i, j) that best represents the bounding box.

        Args:
            box_coords: The [x1, y1, x2, y2] of the salient region.
            patch_size: Resolution of the grid cells.
            image_size: Overall resolution.

        Returns:
            list: [i, j] index of the grid cell with the highest intersection ratio.
        """
        box_polygon = box(*box_coords)
        box_area = box_polygon.area
        
        n_patches_x = image_size[0] // patch_size
        n_patches_y = image_size[1] // patch_size
        
        max_overlap = 0
        max_overlap_patch = [0, 0]
        
        # Iterate through grid cells to calculate overlap with the box
        for i in range(n_patches_x):
            for j in range(n_patches_y):
                x1 = i * patch_size
                y1 = j * patch_size
                x2 = x1 + patch_size
                y2 = y1 + patch_size
                
                patch_polygon = box(x1, y1, x2, y2)
                intersection = box_polygon.intersection(patch_polygon)
                intersection_area = intersection.area
                
                if box_area > 0:
                    overlap_ratio = intersection_area / box_area
                    if overlap_ratio > max_overlap:
                        max_overlap = overlap_ratio
                        max_overlap_patch = [i, j]
        
        return max_overlap_patch

# ==============================================================================
# END OF FILE
# ==============================================================================