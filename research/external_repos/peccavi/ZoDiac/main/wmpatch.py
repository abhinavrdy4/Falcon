"""Official Implementation of WM Patch for PECCVAI"""

import numpy as np
from scipy.stats import norm
import scipy
import torch
from diffusers.utils.torch_utils import randn_tensor
import torch
import numpy as np
import logging
from scipy import stats
from scipy.stats import norm
import matplotlib.pyplot as plt

class GTWatermark():
    """
    GTWatermark class is responsible for generating and injecting watermarks into image data, 
    as well as evaluating and performing statistical analysis on the watermarks.

    Attributes:
        bounding_boxes (list): List of bounding boxes defining the regions for watermarking.
        strengths (list): List of strengths associated with each bounding box.
        w_channel (int): The channel to apply the watermark to (default: 0).
        device (str): Device to use for computations ('cpu' or 'cuda').
        shape (tuple): The shape of the latent tensor.
        gt_patch (torch.Tensor): Watermarked tensor.
        watermarking_mask (torch.Tensor): Mask applied to the tensor to represent the watermark region.
        mu (float): Mean of the watermark evaluation results over random samples.
        sigma (float): Standard deviation of watermark evaluation results over random samples.
    """
    def __init__(self, bounding_boxes, strengths, w_channel=0, device='cpu'):
        """
        Initializes the GTWatermark object with bounding boxes, strengths, watermark channel, and device.

        Args:
            bounding_boxes (list): Bounding boxes for watermarking regions.
            strengths (list or float): Strength values for watermarking.
            w_channel (int): Channel for watermark application.
            device (str): Computation device ('cpu' or 'cuda').
        """
        self.bounding_boxes = bounding_boxes
        self.strengths = strengths if isinstance(strengths, list) else [strengths]
        self.w_channel = w_channel
        self.device = device
        self.shape = None
        self.gt_patch = None
        self.watermarking_mask = None
        self.mu = None
        self.sigma = None

    def _circle_mask(self, size, x_offset, y_offset, r=10):
        """
        Generates a circular mask centered at (x_offset, y_offset) with a radius of r.

        Args:
            size (int): Size of the image.
            x_offset (int): X-coordinate of the center.
            y_offset (int): Y-coordinate of the center.
            r (int): Radius of the circle.
        
        Returns:
            np.ndarray: A binary mask with a circle.
        """
        center = (x_offset, y_offset)  # Use the x_offset and y_offset as the exact center
        Y, X = np.ogrid[:size, :size]
        distance = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
        mask = distance <= r
        return mask

    def adjust_coordinates(self, x, y, size,w,h):
        """
        Adjusts coordinates to be within the valid image boundaries.

        Args:
            x (int): X-coordinate.
            y (int): Y-coordinate.
            size (int): Size of the image.

        Returns:
            tuple: Adjusted (x, y) coordinates.
        """
        x = max(w // 2, min(x, size - w // 2))
        y = max(h // 2, min(y, size - h // 2))
        return x, y
    
    def _get_watermarking_pattern(self, gt_init):
        """
        Generates the watermarking pattern by modifying the FFT of the initialized tensor.

        Args:
            gt_init (torch.Tensor): Initialized tensor.
        
        Returns:
            torch.Tensor: Modified tensor with watermarking pattern.
        """
        gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2))
        for (bbox, strength) in zip(self.bounding_boxes, self.strengths):
            x, y, w, h = bbox
            r = min(w, h) // 2
            while r > 0:
                tmp_mask = torch.tensor(self._circle_mask(gt_init.shape[-1], r=int(r), x_offset=x, y_offset=y)).to(self.device)
                if strength > 0.0:
                    gt_patch[:, self.w_channel, tmp_mask] += gt_patch[0, self.w_channel, 0, int(r)].item() * strength
                r -= strength
        return gt_patch
    
    def visualize_tensor(self,tensor, name='./tmp.png'):
            channels = tensor.squeeze(0).cpu().numpy()
            fig, axs = plt.subplots(1, 4, figsize=(12, 3))

            for i, ax in enumerate(axs):
                ax.imshow(channels[i], cmap='gray')
                ax.set_title(f'Channel {i+1}')
                ax.axis('off')

            plt.savefig(name)
            # plt.show()
    
    
    def _get_watermark_mask(self, gt_patch):
        """
        Creates the watermarking mask based on bounding boxes and watermarking strength.

        Args:
            gt_patch (torch.Tensor): Generated watermarking pattern.
        
        Returns:
            torch.Tensor: Mask tensor representing the watermark regions.
        """
        watermarking_mask = torch.zeros(gt_patch.shape, dtype=torch.bool).to(self.device)
        
        size = gt_patch.shape[-1]
        for (bbox, strength) in zip(self.bounding_boxes, self.strengths):
            x, y, w, h = bbox
            x, y = self.adjust_coordinates(x, y, size, w, h)
            r = min(w, h) // 2
            while r > 0: 
                mask = torch.tensor(self._circle_mask(size=size,x_offset=x,y_offset=y,r=int(r))).to(self.device)
                if mask.sum() > 0.0 and strength > 0.0:
                    print(f"Mask sum : {mask.sum().item()} with strength : {strength}")
                    watermarking_mask[:, self.w_channel] |= mask
                    logging.debug(f'Adding mask for bounding box: x={x}, y={y}, w={w}, h={h}, strength={strength}, mask sum: {mask.sum().item()}')
                    break
            r -= strength
        print("="*100)
        
        watermark_mask = watermarking_mask
        self.visualize_tensor(watermark_mask, name='watermark_mask.png')
        return watermarking_mask

    def _gen_gt_(self, shape, generator=None):
        """
        Generates the ground truth (GT) watermark pattern and mask.

        Args:
            shape (tuple): Shape of the latent tensor.
            generator (torch.Generator, optional): Optional generator for randomness.
        
        Returns:
            tuple: (Generated GT patch, Watermarking mask).
        """
        self.shape = shape
        gt_init = randn_tensor(self.shape, generator=generator, device=self.device, dtype=torch.float32) 
        gt_patch = self._get_watermarking_pattern(gt_init)
        watermarking_mask = self._get_watermark_mask(gt_patch)
        self.gt_patch = gt_patch
        self.watermarking_mask = watermarking_mask
        return gt_patch, watermarking_mask
    
    def inject_watermark(self, latents):
        """
        Injects the watermark into the given latent tensor.

        Args:
            latents (torch.Tensor): Latent tensor to apply the watermark to.
        
        Returns:
            torch.Tensor: Latent tensor with watermark injected.
        """
        latents_fft = torch.fft.fftshift(torch.fft.fft2(latents), dim=(-1, -2))
        latents_fft = latents_fft * ~self.watermarking_mask + self.gt_patch * self.watermarking_mask
        latents_w = torch.fft.ifft2(torch.fft.ifftshift(latents_fft, dim=(-1, -2))).real
        return latents_w
    
    def eval_watermark(self, latent_w):
        """
        Evaluates the quality of the watermark by comparing the injected watermark to the original GT patch.

        Args:
            latent_w (torch.Tensor): Latent tensor with watermark.
        
        Returns:
            float: L1 metric measuring the difference between the GT patch and the watermark in the latent.
        """
        latents_w_fft = torch.fft.fftshift(torch.fft.fft2(latent_w), dim=(-1, -2))
        l1_metric = torch.abs(latents_w_fft[self.watermarking_mask] - self.gt_patch[self.watermarking_mask]).mean().item()
        return l1_metric
    
    def watermark_stat(self):
        """
        Computes statistics (mean and standard deviation) over multiple watermark evaluations.

        Returns:
            tuple: Mean (mu) and standard deviation (sigma) of the watermark evaluations.
        """
        dis_all = []
        for i in range(1000):
            rand_latents = randn_tensor(self.shape, device=self.device, dtype=torch.float32)
            dis = self.eval_watermark(rand_latents)
            dis_all.append(dis)
        dis_all = np.array(dis_all)
        self.mu = dis_all.mean()
        self.sigma = dis_all.std()
        return self.mu, self.sigma
    
    def one_minus_p_value(self, latents):
        """
        Computes the 1 minus p-value for the watermark evaluation using a normal distribution.

        Args:
            latents (torch.Tensor): Latent tensor with watermark.
        
        Returns:
            float: The 1 - p-value result.
        """
        if self.mu is None or self.sigma is None:
            raise ValueError("Must compute watermark statistics before calculating p-value.")
        l1_metric = self.eval_watermark(latents)
        return abs(0.5 - norm.cdf(l1_metric, self.mu, self.sigma)) * 2
    
    def tree_ring_p_value(self, latents):
        """
        Computes the p-value using a tree-ring structure comparison between the GT patch and latent tensor.

        Args:
            latents (torch.Tensor): Latent tensor with watermark.
        
        Returns:
            float: p-value for the tree-ring test.
        """
        target_patch = self.gt_patch[self.watermarking_mask].flatten()
        target_patch = torch.concatenate([target_patch.real, target_patch.imag])

        reversed_latents_w_fft = torch.fft.fftshift(torch.fft.fft2(latents), dim=(-1, -2))[self.watermarking_mask].flatten()
        reversed_latents_w_fft = torch.concatenate([reversed_latents_w_fft.real, reversed_latents_w_fft.imag])

        sigma_w = reversed_latents_w_fft.std()
        lambda_w = (target_patch ** 2 / sigma_w ** 2).sum().item()
        x_w = (((reversed_latents_w_fft - target_patch) / sigma_w) ** 2).sum().item()
        p_w = stats.ncx2.cdf(x=x_w, df=len(target_patch), nc=lambda_w)
        return p_w
