"""
Utility module for image processing, metrics, and watermark analysis.

This module provides a comprehensive set of tools for visualizing tensors, 
saving images from diffusion pipelines, and calculating similarity metrics 
such as FID, PSNR, SSIM, and MS-SSIM. It also includes specialized logic 
for Fourier-domain watermark detection and latent visualization.
"""

# ==============================================================================
# IMPORTS
# ==============================================================================

from PIL import Image
import math
import os
from pytorch_msssim import ssim, ms_ssim
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms.functional import pil_to_tensor
from torchvision import transforms
import torch
from utils.config import ConfigLoader
import torch.nn.functional as F
from torchvision.models import inception_v3
from scipy.linalg import sqrtm

# ==============================================================================
# GLOBAL CONFIGURATION
# ==============================================================================

config_loader = ConfigLoader("config.yaml")
device = config_loader.device

# ==============================================================================
# VISUALIZATION UTILITIES
# ==============================================================================

def show_images_side_by_side(images, titles=None, figsize=(8,4)):
    """
    Display a list of images side by side using matplotlib.
    
    Args:
        images: List of numpy arrays representing images.
        titles: Optional list of strings for subplot titles.
        figsize: Tuple defining the figure dimensions.
    """
    num_images = len(images)
    
    if titles is not None:
        if len(titles) != num_images:
            raise ValueError("Number of titles must match the number of images.")
    
    fig, axes = plt.subplots(1, num_images, figsize=figsize)
    
    # Handle single-image case where axes is not an array
    if num_images == 1:
        axes = [axes]

    for i in range(num_images):
        ax = axes[i]
        ax.imshow(images[i]) 
        ax.axis('off')
        
        if titles is not None:
            ax.set_title(titles[i])
    
    plt.tight_layout()
    plt.show()
    return

def show_latent_and_final_img(latent: torch.Tensor, img: torch.Tensor, pipe):
    """
    Decodes a latent tensor and shows it alongside the final generated image.

    Args:
        latent: The latent space representation (tensor).
        img: The final image tensor.
        pipe: The diffusion pipeline object providing decoding utilities.
    """
    with torch.no_grad():
        # Decode the latent into image space and convert to PIL
        latents_pil_img = pipe.numpy_to_pil(pipe.decode_latents(latent.detach()))[0]
        # Convert the generated image tensor to PIL
        pil_img = pipe.numpy_to_pil(pipe.img_tensor_to_numpy(img))[0]
    
    show_images_side_by_side([latents_pil_img, pil_img], ['Latent','Generated Image'])
    return

def visualize_tensor(tensor, name='./tmp.png'):
    """
    Visualizes the first 4 channels of a 4D tensor as grayscale images.

    Args:
        tensor: torch.Tensor, typically [1, C, H, W].
        name: Filename/path where the visualization will be saved.
    """
    # Remove batch dimension and convert to numpy
    channels = tensor.squeeze(0).cpu().numpy()
    fig, axs = plt.subplots(1, 4, figsize=(12, 3))

    for i, ax in enumerate(axs):
        if i < len(channels):
            ax.imshow(channels[i], cmap='gray')
            ax.set_title(f'Channel {i+1}')
        ax.axis('off')

    plt.savefig(name)
    # plt.show() # Commented out in source

# ==============================================================================
# FILE & TENSOR I/O
# ==============================================================================

def save_img(path, img: torch.Tensor, pipe):
    """Saves a diffusion image tensor to the specified file path."""
    pil_img = pipe.numpy_to_pil(pipe.img_tensor_to_numpy(img))[0]
    pil_img.save(path)
    return

def get_img_tensor(img_path, device):
    """Loads an image and returns a normalized 4D tensor [1, C, H, W]."""
    img_tensor = pil_to_tensor(Image.open(img_path).convert("RGB"))/255
    return img_tensor.unsqueeze(0).to(device)

def load_img(img_path, device):
    """Loads an image into a normalized tensor using standard torchvision transforms."""
    img = Image.open(img_path).convert('RGB')
    x = (transforms.ToTensor()(img)).unsqueeze(0).to(device)
    return x

def create_output_folder(cfgs):
    """
    Creates directory structures for saving original and processed images.

    Args:
        cfgs: Dict containing 'save_img', 'dataset', 'method', 'case', and 'model_id'.

    Returns:
        tuple: (watermarked_image_path, original_image_path)
    """
    parent = os.path.join(cfgs['save_img'], cfgs['dataset'])
    wm_path = os.path.join(parent, cfgs['method'], cfgs['case'])
    
    # Handle specific naming conventions for CompVis models
    special_model = ['CompVis']
    for key in special_model:
        if key in cfgs['model_id']:
            wm_path = os.path.join(parent, cfgs['method'], '_'.join([cfgs['case'][:-1], key+'/']))
            break
        
    os.makedirs(wm_path, exist_ok=True)
    ori_path = os.path.join(parent, 'OriImgs/')
    os.makedirs(ori_path, exist_ok=True)
    return wm_path, ori_path

# ==============================================================================
# SIMILARITY METRICS
# ==============================================================================


def compute_fid(a, b):
    """
    Compute the Fréchet Inception Distance (FID) between two image tensors.

    Args:
        a: First image tensor.
        b: Second image tensor.

    Notes:
        - Uses InceptionV3 feature extraction.
        - Calculated as: ||mu1 - mu2||^2 + Tr(sigma1 + sigma2 - 2*sqrt(sigma1*sigma2)).
    """
    model = inception_v3(pretrained=True, transform_input=False).eval()
    for param in model.parameters():
        param.requires_grad = False
        
    with torch.no_grad():
        # Get feature activations from InceptionV3
        a1 = model(a)
        b1 = model(b)
        
    a1 = a1.numpy().flatten()
    b1 = b1.numpy().flatten()
    
    # Calculate Mean and Covariance
    mu1, sigma1 = a1.mean(), np.cov(a1)
    mu2, sigma2 = b1.mean(), np.cov(b1)

    diff = mu1 - mu2
    # Matrix square root of the product of covariances
    covmean, _ = sqrtm(sigma1 * sigma2, disp=False)

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff**2 + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid

def compute_psnr(a, b):
    """Calculates Peak Signal-to-Noise Ratio (PSNR) between two tensors."""
    mse = torch.mean((a - b) ** 2).item()
    if mse == 0:
        return 100 # Perfect reconstruction
    return 20 * math.log10(1.) - 10 * math.log10(mse)

def compute_msssim(a, b):
    """Calculates Multi-Scale Structural Similarity Index (MS-SSIM)."""
    return ms_ssim(a, b, data_range=1.).item()

def compute_ssim(a, b):
    """Calculates Structural Similarity Index (SSIM)."""
    return ssim(a, b, data_range=1.).item()

# --------------------------------------------------------------------------
# AGGREGATED EVALUATION WRAPPERS
# --------------------------------------------------------------------------

def eval_psnr_ssim_msssim(ori_img_path, new_img_path, device):
    """Loads images and computes PSNR, SSIM, and MS-SSIM scores."""
    ori_x, new_x = load_img(ori_img_path, device), load_img(new_img_path, device)
    return compute_psnr(ori_x, new_x), compute_ssim(ori_x, new_x), compute_msssim(ori_x, new_x)

def eval_lpips(ori_img_path, new_img_path, metric, device):
    """Computes Learned Perceptual Image Patch Similarity (LPIPS) using a provided metric object."""
    ori_x, new_x = load_img(ori_img_path, device), load_img(new_img_path, device)
    return metric(ori_x, new_x).item()

# ==============================================================================
# WATERMARK DETECTION & ANALYSIS
# ==============================================================================


def watermark_prob(img, dect_pipe, wm_pipe, text_embeddings, tree_ring=True, device=torch.device('cpu')):
    """
    Calculates the probability that an image contains a specific watermark.

    The process involves reversing the diffusion process to find the initial 
    noise/latent and analyzing it for specific patterns (like Tree-Ring).

    Args:
        img: Path string or torch.Tensor.
        dect_pipe: Pipeline for inversion/detection.
        wm_pipe: Object providing statistical p-value calculations for watermarks.
        text_embeddings: Embeddings used for the conditional inversion.
        tree_ring: Boolean, if True, uses Tree-Ring p-value logic.

    Returns:
        float: Probability score of watermark presence.
    """
    if isinstance(img, str):
        img_tensor = pil_to_tensor(Image.open(img).convert("RGB"))/255
        img_tensor = img_tensor.unsqueeze(0).to(device)
    elif isinstance(img, torch.Tensor):
        img_tensor = img
    
    print(img_tensor.shape)
    # 1. Obtain image latents (encoding)
    img_latents = dect_pipe.get_image_latents(img_tensor, sample=False)
    
    # 2. Reverse diffusion to recover the noise/starting latent
    reversed_latents = dect_pipe.forward_diffusion(
        latents=img_latents,
        text_embeddings=text_embeddings,
        guidance_scale=1.0,
        num_inference_steps=50,
    )
    
    # 3. Frequency domain analysis for visualization
    see = torch.Tensor(reversed_latents.detach().cpu().numpy())
    reversed_latents_w_fft = torch.fft.fftshift(torch.fft.fft2(see), dim=(-1, -2)).real
    visualize_tensor(reversed_latents_w_fft, name='output/reversed_real.png')
    
    # 4. Statistical detection
    det_prob = wm_pipe.one_minus_p_value(reversed_latents) if not tree_ring else wm_pipe.tree_ring_p_value(reversed_latents)
    return det_prob

def visualize_latents_in_fourier(init_latents, init_latents_wm, i, visualize_tensor, radius=10):
    """
    Generates Fourier magnitude spectrum visualizations to compare original and watermarked latents.

    High-frequency components are often used to hide watermarks; this function 
    applies a High-Pass filter to highlight those differences.

    Args:
        init_latents: Original noise/latent tensor.
        init_latents_wm: Watermarked noise/latent tensor.
        i: Current index for filename ordering.
        visualize_tensor: Callback function to save/show results.
        radius: Pixel radius for the High-Pass Filter mask center.
    """
    see_original = torch.Tensor(init_latents.detach().cpu().numpy())
    see_watermarked = torch.Tensor(init_latents_wm.detach().cpu().numpy())

    # Step 1: FFT + Shift zero-frequency to center
    fft_original = torch.fft.fftshift(torch.fft.fft2(see_original), dim=(-1, -2))
    fft_watermarked = torch.fft.fftshift(torch.fft.fft2(see_watermarked), dim=(-1, -2))

    # Step 2: Compute Magnitude Spectrum
    magnitude_original = torch.abs(fft_original)
    magnitude_watermarked = torch.abs(fft_watermarked)

    # Step 3: Logarithmic scaling for visual range
    magnitude_original_log = torch.log(magnitude_original + 1e-8)
    magnitude_watermarked_log = torch.log(magnitude_watermarked + 1e-8)

    # Step 4: Visualize Full Spectrum
    visualize_tensor(magnitude_original_log, name=f'output/{i+1}_original_latents_magnitude.png')
    visualize_tensor(magnitude_watermarked_log, name=f'output/{i+1}_watermarked_latents_magnitude.png')

    # Step 5: High-pass filter (highlights subtle high-freq watermarking patterns)
    rows, cols = see_watermarked.shape[-2], see_watermarked.shape[-1]
    crow, ccol = rows // 2, cols // 2

    # Create a circular/square mask to zero out low frequencies
    mask = torch.ones((rows, cols))
    mask[crow - radius:crow + radius, ccol - radius:ccol + radius] = 0

    # Apply mask to FFT components
    fft_original_high_pass = fft_original * mask
    fft_watermarked_high_pass = fft_watermarked * mask

    # Extract magnitude of high-frequency components
    magnitude_original_high_pass = torch.abs(fft_original_high_pass)
    magnitude_watermarked_high_pass = torch.abs(fft_watermarked_high_pass)

    magnitude_original_high_pass_log = torch.log(magnitude_original_high_pass + 1e-8)
    magnitude_watermarked_high_pass_log = torch.log(magnitude_watermarked_high_pass + 1e-8)

    # High-pass visualization (Currently disabled in source)
    # visualize_tensor(magnitude_original_high_pass_log, name=f'output/{i+1}_original_high_pass.png')
    # visualize_tensor(magnitude_watermarked_high_pass_log, name=f'output/{i+1}_watermarked_high_pass.png')

    print(f"Visualization for latent {i+1} complete.")

# ==============================================================================
# END OF FILE
# ==============================================================================