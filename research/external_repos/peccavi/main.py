"""
Main Execution Module for the PECCAVI Watermarking Pipeline.

This script orchestrates the end-to-end workflow of the PECCAVI system, 
including image paraphrasing, parallelized saliency detection, watermark 
injection, and detection verification with Weights & Biases (W&B) integration.
"""

# ==============================================================================
# IMPORTS
# ==============================================================================

import gc
import json
import math
import os
import re
import threading
import time
import warnings
import logging
import multiprocessing
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch
import wandb
import yaml
from tqdm import tqdm

# Disable matplotlib font debug messages to keep the CLI clean
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)

from utils.animation import loading_animation
from utils.config import ConfigLoader
from utils.paraphrase import Paraphraser
from utils.salient import SaliencyDetector
#from utils.split import SaliencyDetector
from utils.watermark import Watermarker

# Ignore non-critical runtime warnings
warnings.filterwarnings("ignore")

# Global marker placeholder
marker = None

# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def generate_run_name(config, image_name):
    """
    Generates a descriptive string for W&B logging based on hyperparameters.

    Args:
        config: Dict containing 'loss_weights', 'learning_rate', etc.
        image_name: Base filename of the image being processed.

    Returns:
        str: A formatted string for identifying the experiment run.
    """
    ssim_loss = f"LW-{config['loss_weights'][2]}"
    learning_rate = f"LR-{config['learning_rate']}"
    inference_steps = f"Steps-{config['num_inference_steps']}"
    iterations = f"Iters-{config['iters']}"
    return f"{image_name}-{ssim_loss}-{learning_rate}-{inference_steps}-{iterations}"


def clear_cuda_memory():
    """
    Forces garbage collection and empties the PyTorch CUDA cache.
    
    Used to prevent Out-of-Memory (OOM) errors during long sequential runs.
    """
    torch.cuda.empty_cache()
    gc.collect()


def process_single_saliency(image_path):
    """
    Standalone function for parallel saliency processing.

    Args:
        image_path: Path to the image requiring a saliency map.

    Returns:
        tuple: (bool success_status, str image_path)
    """
    try:
        # Instantiate a local detector for thread/process safety
        detector = SaliencyDetector()
        detector.XRAI(image_path)
        return True, image_path
    except Exception as e:
        print(f"Error processing saliency for {image_path}: {str(e)}")
        return False, image_path


# ==============================================================================
# MAIN PIPELINE LOGIC
# ==============================================================================

def main(config_path):
    """
    Orchestrates the sequential processing of a directory of images.

    The workflow per image is:
    1. Paraphrase original image (Diffusion).
    2. Parallel Saliency Detection (XRAI) on paraphrases.
    3. Calculate cross-version salient bounding boxes.
    4. Inject watermark into the original image based on salient regions.
    5. Paraphrase the watermarked image to test robustness.
    6. Run detection logic and log results.

    Args:
        config_path: Path to the 'config.yaml' file.
    """
    # Initialize configuration and resources
    config_loader = ConfigLoader(config_path)
    wandb_config = config_loader.wandb_config()
    device = config_loader.device
    wandb_use = config_loader.wandb
    image_folder = config_loader.base_images_dir
    json_data_path = config_loader.json_data_path
    paraphrasing_model = config_loader.paraphrasing_model
    paraphrased_images_dir = config_loader.paraphrased_images_dir
    wm_paraphrased_dir = config_loader.WM_Paraphrased_dir
    detection = config_loader.detection

    print("\n" * 2)
    print("-" * 100)
    print("PECCVAI Watermarking")
    print("-" * 100)
    print(f"Device: {device}")

    # Load image metadata/annotations
    with open(json_data_path) as file:
        json_data = json.load(file)

    result_dict = {}
    img_list = os.listdir(image_folder)

    # --------------------------------------------------------------------------
    # IMAGE ITERATION LOOP
    # --------------------------------------------------------------------------
    for img in img_list:
        image_path = os.path.join(image_folder, img)
        # Skip directories, process files only
        if os.path.isdir(image_path):
            continue

        start_time = time.time()
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        # Expecting format: 'prefix_ID.ext'
        image_id = image_name.split('_')[1] if '_' in image_name else None

        # Optional Weights & Biases initialization
        if wandb_use:
            run_name = generate_run_name(wandb_config, image_name)
            wandb.init(project=config_loader.project_name, name=run_name, config=wandb_config)

        print(f"\nWorking on {image_name}")
        print(f"ID: {image_id}")
        print(f"Image path: {image_path}\n")

        try:
            # 1. PARAPHRASING PHASE
            # ------------------------------------------------------------------
            image_description = json_data['annotations'][image_id][:5]
            paraphraser = Paraphraser(paraphrasing_model, device)
            img_paraphrase_saved_dir = paraphraser.paraphrase_images(
                image_path, paraphrased_images_dir, num_paraphrase=5, caption=image_description
            )
            
            if img_paraphrase_saved_dir:
                print(f'\nFor image ID {image_id}, paraphrased images saved at: {img_paraphrase_saved_dir}')
            else:
                print(f'No paraphrased images were generated for image ID {image_id}')

            # 2. PARALLEL SALIENCY DETECTION PHASE
            # ------------------------------------------------------------------
            # Collect paths of generated paraphrases
            paraphrased_images = [os.path.join(img_paraphrase_saved_dir, f) 
                                for f in os.listdir(img_paraphrase_saved_dir) 
                                if f.endswith(('.png', '.jpg', '.jpeg'))]
            
            print(f"\nProcessing saliency for {len(paraphrased_images)} images in parallel...")
            
            # Utilize multiprocessing for CPU-bound XRAI operations
            num_processes = max(1, multiprocessing.cpu_count() - 1)
            
            with ProcessPoolExecutor(max_workers=num_processes) as executor:
                futures = [executor.submit(process_single_saliency, image) 
                          for image in paraphrased_images]
                
                # Monitor task completion
                for future in as_completed(futures):
                    success, image = future.result()
                    if not success:
                        print(f"Failed to process saliency for {image}")

            # Aggregate cross-version heatmaps into scored bounding boxes
            detector = SaliencyDetector()
            # Note: Result of detector call is specifically reassigned to a path string below
            processes_box_json = detector.process_and_score_boxes(img_paraphrase_saved_dir)
            processes_box_json = f"data/Paraphrased/{image_id}/final.json"
            
            if processes_box_json:
                print("\nSaliency generation completed.")
            else:
                print("Error in generating saliency")

            # 3. WATERMARK INJECTION PHASE
            # ------------------------------------------------------------------
            print(f' ========== Starting Watermarking ==========')
            watermarker = Watermarker(processes_box_json)
            # Apply watermark to the ORIGINAL image
            watermarked_image_path = watermarker.forward(image_path)
            clear_cuda_memory()

            # 4. ROBUSTNESS TESTING PHASE
            # ------------------------------------------------------------------
            # Paraphrase the newly watermarked image to see if the mark survives
            paraphraser.paraphrase_images(
                watermarked_image_path, wm_paraphrased_dir, num_paraphrase=5, caption=image_description
            )

            # 5. DETECTION & LOGGING PHASE
            # ------------------------------------------------------------------ 
            if detection:
                result_dict[image_name] = []
                wm_dir = os.path.join(wm_paraphrased_dir, image_id)
                
                # Check every version of the watermarked paraphrase for the mark
                for wm_img in os.listdir(wm_dir):
                    wm_path = os.path.join(wm_dir, wm_img)
                    prob = watermarker.detect(wm_path)
                    
                    # Sanitize probability value for JSON/W&B compatibility
                    prob = 0.0 if prob is None or math.isnan(prob) else prob
                    
                    if wandb_use:
                        wandb.log({"image_id": image_id, "watermark_prob": prob})

                    result_dict[image_name].append(prob)
                    
                    # Continuous update of the results file
                    output_dir = 'output'
                    os.makedirs(output_dir, exist_ok=True)
                    with open(os.path.join(output_dir, f'{run_name}_results.json'), 'w') as file:
                        json.dump(result_dict, file, indent=4)
                
                print(f"Total Time for Image {image_id}: {time.time() - start_time}")
            else:
                print("Paraphrased Watermarked Images Generated.")

        except Exception as e:
            print(f"Error processing image {image_name}: {str(e)}")

        finally:
            # 6. CLEANUP PHASE
            # ------------------------------------------------------------------
            if wandb_use:
                wandb.finish()

            # Clean up heavy objects to free memory for next iteration
            if 'paraphraser' in locals():
                del paraphraser
            if 'watermarker' in locals():
                del watermarker
                
            clear_cuda_memory()
            print(f"CUDA memory cleared after processing {image_name}")
            print(f"Current CUDA memory allocated: {torch.cuda.memory_allocated(device)}")
            print(f"Current CUDA memory reserved: {torch.cuda.memory_reserved(device)}")

    print("Processing completed for all images.")

# ------------------------------------------------------------------------------
# ENTRY POINT
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    CONFIG_PATH = "config.yaml"
    main(CONFIG_PATH)