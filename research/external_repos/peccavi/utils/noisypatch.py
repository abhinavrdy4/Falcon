"""
Official Implementation of NoisyPatch.

This module provides a workflow to segment an image into patches, identify the 
patch most semantically related to a text description using CLIP, and inject 
random noise into that specific region.
"""

# ==============================================================================
# IMPORTS
# ==============================================================================

from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel
import numpy as np

# ==============================================================================
# NOISY PATCH CLASS
# ==============================================================================

class NoisyPatch:
    """
    Identifies the most relevant image patch via CLIP and applies noise to it.

    This class provides tools to pad images, slice them into uniform patches, 
    rank them against a text description, and modify the original image data.

    Example Usage:
        if __name__ == "__main__":
            image_path = "your/image/path"
            patch_size = 64
            folder = "name of the folder"
            description = ["description of your image"]
            noise_level = 50
            add = True
            ...
            main(image_path, patch_size, description, folder, noise_level, add)
    """

    def __init__(self):
        """
        Initializes the NoisyPatch instance. (No internal state maintained).
        """
        pass

    def pad_image(self, image, patch_size):
        """
        Pads the input image with black pixels to ensure divisibility by patch_size.

        Args:
            image: A PIL Image object.
            patch_size: int, the side length of the square patches.

        Returns:
            Image: A new PIL Image padded to the next multiple of patch_size.

        Notes:
            - Uses RGB mode with black padding (0, 0, 0).
            - The original image is pasted at the top-left (0, 0).
        """
        width, height = image.size
        # Calculate dimensions required to fit an exact number of patches
        new_width = (width + patch_size - 1) // patch_size * patch_size
        new_height = (height + patch_size - 1) // patch_size * patch_size

        padded_image = Image.new("RGB", (new_width, new_height), (0, 0, 0))
        padded_image.paste(image, (0, 0))
        return padded_image

    def extract_patches(self, image, patch_size):
        """
        Slices the image into a grid of non-overlapping square patches.

        Args:
            image: A PIL Image (typically padded).
            patch_size: int, the size of the square crop.

        Returns:
            list: A list of PIL Image patches.

        Notes:
            - Patches are extracted Column-wise first (nested height loop).
        """
        patches = []
        width, height = image.size

        for i in range(0, width, patch_size):
            for j in range(0, height, patch_size):
                box = (i, j, i + patch_size, j + patch_size)
                patch = image.crop(box)
                patches.append(patch)

        return patches

    def save_patches(self, patches, folder):
        """
        Writes all patches to the specified directory.

        Args:
            patches: List of PIL Image objects.
            folder: str, path to the destination directory.

        Side Effects:
            - Saves files named 'patch_idx.png' to the local disk.
        """
        for idx, patch in enumerate(patches):
            # Note: patch 1 is below patch 0 (column-wise extraction logic)
            patch.save(f"{folder}/patch_{idx}.png")

    def clip_embedding(self, image, description):
        """
        Calculates the similarity score between an image patch and text descriptions.

        Args:
            image: A PIL Image patch.
            description: list of str, text labels/descriptions for comparison.

        Returns:
            float or list: The image-text similarity score (logits).

        Side Effects:
            - Downloads/Loads CLIP model/processor from HuggingFace on every call.
        """
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        inputs = processor(text=description, images=image, return_tensors="pt", padding=True)

        outputs = model(**inputs)
        # Returns raw logits indicating similarity strength
        logits_per_image = (outputs.logits_per_image).squeeze().tolist()
        return logits_per_image

    def add_noise(self, img, image, patch_number, patch_size, folder, noise_level=255, add=False, output_path="/home/shreyas/Desktop/Shreyas/Projects/Panda/data/noisy"):
        """
        Applies random noise to a specific patch and reconstructs the image.

        Args:
            img: Original unpadded PIL Image (used for final cropping).
            image: Padded PIL Image used as the base for modification.
            patch_number: int, index of the patch to modify.
            patch_size: int, size of the square patch.
            folder: str, directory where extracted patches are stored.
            noise_level: int, max value for random noise (0-255).
            add: bool, if True, the noise modification is performed.
            output_path: str, destination directory for the final image.

        Side Effects:
            - Reads a patch file from the disk.
            - Saves the final 'final_image.png' to the output_path.
        """
        if add:
            # Reload the specific patch identified as important
            patch = Image.open(f"{folder}/patch_{patch_number}.png")
            patch_array = np.array(patch)
            
            # Generate random noise and add to the patch array
            noise_array = np.random.randint(0, noise_level + 1, (patch_size, patch_size, 3), dtype=np.uint8)
            noisy_patch_array = np.clip(patch_array + noise_array, 0, 255)
            noisy_patch = Image.fromarray(noisy_patch_array)
            
            # Reconstruct spatial coordinates based on patch index
            n = image.size[1] // patch.size[1] # Number of patches in height
            x = patch_number // n * patch_size
            y = patch_number % n * patch_size
            
            # Paste the noisy version back into the padded image
            image.paste(noisy_patch, (x, y))
            
        # Crop back to original dimensions to remove padding
        og_x, og_y = img.size
        final_image = image.crop((0, 0, og_x, og_y))
        final_image.save(f'{output_path}/final_image.png')

    def main(self, image_path, patch_size, description, folder, noise_level, add, output_path):
        """
        Orchestrates the full patch extraction, analysis, and noise injection process.

        Args:
            image_path: Path to the source image.
            patch_size: Side length of the square segments.
            description: Text used for CLIP semantic matching.
            folder: Directory for temporary patch storage.
            noise_level: Intensity of added noise.
            add: Whether to actually apply the noise.
            output_path: Path for the resulting image.

        Notes:
            - Currently uses clip_embeddings.index(min(...)), which targets the 
              LEAST similar patch based on the code logic.
        """
        image = Image.open(image_path)

        # 1. Prepare image geometry
        padded_image = self.pad_image(image, patch_size)

        # 2. Slicing
        patches = self.extract_patches(padded_image, patch_size)
        self.save_patches(patches, folder)

        # 3. Semantic Analysis
        clip_embeddings = []
        for patch in patches:
            clip_embeddings.append(self.clip_embedding(patch, description))

        print("-" * 100)
        print("Clip Matrix")
        print(clip_embeddings)
        print("-" * 100)
        
        # 4. Identification (Note: min() selects the patch with the lowest similarity)
        patch_number = clip_embeddings.index(min(clip_embeddings))

        # 5. Modification and Save
        self.add_noise(image, padded_image, patch_number, patch_size, folder, noise_level, add, output_path)

# ==============================================================================
# EXECUTION BLOCK
# ==============================================================================

if __name__ == "__main__":
    # Configuration parameters
    image_path = "/home/shreyas/Desktop/Shreyas/Projects/Panda/data/Image_279108.jpg"
    patch_size = 128
    folder = "/home/shreyas/Desktop/Shreyas/Projects/Panda/image_patches"
    description = ["A old women feeding her husband using a spoon."]
    noise_level = 25
    add = True
    output_path = "/home/shreyas/Desktop/Shreyas/Projects/Panda/data/noisy"

    import os
    # Directory maintenance
    if not os.path.exists(folder):
        os.makedirs(folder)
    else:
        # Warning: Shell command depends on Unix-like environment
        os.system('rm -rf image_patches/*')
        
    noisy = NoisyPatch()
    noisy.main(image_path, patch_size, description, folder, noise_level, add, output_path)