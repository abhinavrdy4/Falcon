import logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

import torch
import torch.nn as nn
from .watson_vgg import WatsonDistanceVgg
from .pytorch_ssim import SSIM
import wandb

wandb_use = True

class LossProvider(nn.Module):
    def __init__(self, loss_weights: list, device):
        super(LossProvider, self).__init__()
        self.loss_weights = loss_weights

        self.loss_img, self.loss_w = nn.MSELoss(), nn.L1Loss()
        self.loss_ssim = SSIM()

        # Add perceptual loss
        loss_percep = WatsonDistanceVgg(reduction='sum')
        loss_percep.load_state_dict(torch.load('ZoDiac/loss/rgb_watson_vgg_trial0.pth', map_location='cpu'))
        loss_percep = loss_percep.to(device)
        self.loss_per = lambda pred_img, gt_img: loss_percep((1 + pred_img) / 2.0, (1 + gt_img) / 2.0) / pred_img.shape[0]

    def __call__(self, pred_img_tensor, gt_img_tensor, init_latents, wm_pipe, epoch):
        """
        Compute the loss and log it to wandb with respect to the current epoch.

        Args:
            pred_img_tensor (torch.Tensor): Predicted image tensor.
            gt_img_tensor (torch.Tensor): Ground truth image tensor.
            init_latents (torch.Tensor): Initial latent tensors.
            wm_pipe: Watermarking pipeline containing masks and patches.
            epoch (int): Current epoch number.
        
        Returns:
            torch.Tensor: Total computed loss.
        """
        # Compute FFT of initial latents
        init_latents_fft = torch.fft.fftshift(torch.fft.fft2(init_latents), dim=(-1, -2))
        
        # Compute individual losses
        lossW = self.loss_w(init_latents_fft[wm_pipe.watermarking_mask], wm_pipe.gt_patch[wm_pipe.watermarking_mask]) * self.loss_weights[3]
        lossI = self.loss_img(pred_img_tensor, gt_img_tensor) * self.loss_weights[0]
        lossP = self.loss_per(pred_img_tensor, gt_img_tensor) * self.loss_weights[1]
        lossS = (1 - self.loss_ssim(pred_img_tensor, gt_img_tensor)) * self.loss_weights[2]
        
        # Total loss
        loss = lossW + lossI + lossP + lossS
        
        # Logging
        logger.debug(f'Epoch {epoch}: Watermark {lossW.item():.4f}, Image {lossI.item():.4f}, '
                     f'Perp {lossP.item():.4f}, SSIM {lossS.item():.4f} Total Loss {loss.item():.4f}')
        
        # Log losses to wandb with epoch as the step
        if wandb_use:
            wandb.log({
                'Perceptual Loss': lossP.item(),
                'SSIM Loss': lossS.item(),
                'Image Loss': lossI.item(),
                'Watermark Loss': lossW.item(),
                'Total Loss': loss.item()
            }, step=epoch)
        
        return loss
