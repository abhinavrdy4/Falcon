"""
This file generates image paraphrases using Stable Diffusion models.

The module provides the Paraphraser class, which utilizes Stable Diffusion's 
Image-to-Image capabilities to create variations of existing images based on 
textual guidance and visual starting points.
"""

# ==============================================================================
# IMPORTS
# ==============================================================================

import os
from PIL import Image
import torch
from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image
from diffusers.utils import load_image
from typing import Optional, List
from utils.config import ConfigLoader

# ==============================================================================
# IMAGE PARAPHRASING LOGIC
# ==============================================================================

class Paraphraser:
    """
    A class to handle image paraphrasing using Stable Diffusion models.

    This class manages the transition from a base image to a "paraphrased" version 
    by leveraging the diffusers library. It initializes a Text-to-Image pipeline 
    and repurposes it for Image-to-Image tasks.

    Example:
        if __name__ == "__main__":
            model_id = 'your_model_id'
            paraphraser = Paraphraser(model_id)
            base_images_path = 'path/to/your/base/images/or/folder'
            paraphrased_images_dir = 'path/to/save/paraphrased/images'
            num_paraphrase = 5
            paraphraser.paraphrase_images(base_images_path, paraphrased_images_dir, num_paraphrase)
    """

    def __init__(self, model_id: str, device: torch.device):
        """
        Initialize the Paraphraser with model configurations and hardware settings.

        Args:
            model_id: The identifier for the pre-trained Stable Diffusion model.
            device: The torch.device (CPU/CUDA) to run the model on.

        Side Effects:
            - Loads a ConfigLoader from 'config.yaml'.
            - Downloads/Instantiates heavy model weights into memory/VRAM.
            - Transfers the pipeline to the specified device.

        Notes:
            - Explicitly uses `use_safetensors=True` for secure weight loading.
            - The `model_id` passed in the constructor is stored but 
              `config_loader.paraphrasing_model` is used for the actual load.
        """
        config_loader = ConfigLoader('config.yaml')
        self.model_id = model_id
        self.device = config_loader.device
        
        # Load the base Text2Image pipeline
        self.pipeline_text2image = AutoPipelineForText2Image.from_pretrained(
            config_loader.paraphrasing_model,
            use_safetensors=True,
        ).to(self.device)
        
        # Convert or wrap the pipe into an Image2Image pipeline
        self.pipeline = AutoPipelineForImage2Image.from_pipe(self.pipeline_text2image).to(self.device)

    def paraphrase_images(self, path: str, paraphrase_path: str, num_paraphrase: int, caption: Optional[str] = "A One way sign laying on the sidewalk.", neg_prompt: Optional[str] = 'deformity, bad anatomy, cloned face, amputee, people in background, asymmetric, disfigured, extra limbs, text, missing legs, missing arms, Out of frame, low quality, Poorly drawn feet'):
        """
        Generate paraphrased images from base images in the given directory or a single image file.

        Args:
            path: Path to a directory of images or a single image file.
            paraphrase_path: Directory where the generated results will be saved.
            num_paraphrase: The maximum number of variations to save per source image.
            caption: Prompt to guide the appearance of the output image.
            neg_prompt: Text describing elements to exclude from the generation.

        Returns:
            Optional[str]: The path to the last created unique directory, or None if failed.

        Side Effects:
            - Creates subdirectories for each image processed.
            - Performs GPU-intensive inference.
            - Saves multiple PNG files to the local file system.

        Notes:
            - Image identification (img_id) depends on filenames containing an underscore (e.g., "prefix_123.jpg").
            - The `strength` is hardcoded to 0.2 (low variance from original).
            - The `guidance_scale` is hardcoded to 7.5.
            - If `caption` is a list, `neg_prompt` is expanded to a list of 5 elements.
        """
        unique_save_dir = None  # Initialize unique_save_dir to None

        # Determine if target is a directory or a single file
        if os.path.isdir(path):
            image_paths = [os.path.join(path, img_filename) for img_filename in os.listdir(path) if img_filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
        elif os.path.isfile(path) and path.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
            image_paths = [path]
            # Internal variable set but not explicitly utilized in this branch
            negative_prompt = neg_prompt
        else:
            print(f"Invalid path or unsupported file type: {path}")
            return None 
        
        # Synchronize list length for batch-like prompt processing
        if isinstance(caption, list):
            neg_prompt = [neg_prompt] * 5

        for img_path in image_paths:
            img_filename = os.path.basename(img_path)
            # Logic: Expects filename format to be 'anything_ID.ext'
            img_name_parts = os.path.splitext(img_filename)[0].split("_")
            
            if len(img_name_parts) > 1:
                img_id = img_name_parts[1]
            else:
                print(f"Skipping file due to unexpected format: {img_filename}")
                continue

            print(f"Processing image: {img_path}")

            try:
                # Load and prepare image for the pipeline
                img = Image.open(img_path)
                init_image = load_image(img)
                
                # Execute Image-to-Image generation
                gen_images = self.pipeline(
                    caption, 
                    image=init_image, 
                    negative_prompt=neg_prompt, 
                    strength=0.2, 
                    guidance_scale=7.5
                ).images

                # Organization: Each source image ID gets its own subdirectory
                unique_save_dir = os.path.join(paraphrase_path, f'{img_id}')
                os.makedirs(unique_save_dir, exist_ok=True)

                # Save generated variations up to the requested count
                for idx, gen_img in enumerate(gen_images[:num_paraphrase]):
                    gen_save_path = os.path.join(unique_save_dir, f'{img_id}_gen_{idx}.png')
                    gen_img.save(gen_save_path)
                    print(f"Generated Image saved to {gen_save_path} for id: {img_id}")
                    
            except Exception as e:
                print(f"Error processing image {img_path}: {e}")

        # Returns the directory path for the last successful batch
        return unique_save_dir 

# ==============================================================================
# END OF FILE
# ==============================================================================