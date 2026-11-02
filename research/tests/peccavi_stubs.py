"""CPU stand-ins for the diffusion models PECCAVI loads, so its control flow runs without weights.

FakeSDPipe replaces WMDetectStableDiffusionPipeline: an exactly invertible linear "VAE" (8x8 pixel block
<-> 4 latent values) and an identity "DDIM inversion". A latent watermark therefore survives an
encode/decode round trip up to uint8 rounding, which is enough to exercise embed -> save -> detect.
FakeImg2Img replaces the SDXL img2img paraphraser: small seeded jitter of the input, one output per caption.
"""
from __future__ import annotations

import types

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class FakeSDPipe:
    vae_scale_factor = 8

    def __init__(self, device="cpu", **_):
        g = torch.Generator().manual_seed(0)
        self.D = torch.randn(3 * 64, 4, generator=g) * 0.05  # one latent pixel -> one 3x8x8 pixel block
        self.E = torch.linalg.pinv(self.D)
        self.device = torch.device(device)

    @classmethod
    def from_pretrained(cls, *a, **kw):
        return cls()

    def to(self, device):
        self.device = torch.device(device)
        return self

    def decode(self, z):
        b, c, h, w = z.shape
        blocks = torch.einsum("pc,bchw->bphw", self.D.to(z), z).reshape(b, 3 * 64, h * w)
        return (0.5 + F.fold(blocks, (h * 8, w * 8), 8, stride=8)).clamp(0, 1)

    def get_image_latents(self, image, sample=True, rng_generator=None):
        b, _, H, W = image.shape
        blocks = F.unfold(image - 0.5, 8, stride=8)  # (b, 192, n)
        z = torch.einsum("cp,bpn->bcn", self.E.to(image), blocks)
        return z.reshape(b, 4, H // 8, W // 8)

    def get_text_embedding(self, prompt):
        return torch.zeros(1, 77, 8)

    def forward_diffusion(self, latents=None, **_):
        return latents.clone()

    def __call__(self, prompt=None, init_latents=None, output_type="tensor", **_):
        return types.SimpleNamespace(images=self.decode(init_latents))

    @staticmethod
    def img_tensor_to_numpy(tensor):
        return tensor.detach().cpu().permute(0, 2, 3, 1).float().numpy()

    @staticmethod
    def numpy_to_pil(images):
        return [Image.fromarray((im * 255).round().astype("uint8")) for im in images]


class FakeScheduler:
    @classmethod
    def from_pretrained(cls, *a, **kw):
        return cls()


class FakeImg2Img:
    """Same call signature as diffusers' img2img pipelines; returns len(prompt) jittered copies."""

    def __call__(self, prompt, image=None, negative_prompt=None, strength=0.2, guidance_scale=7.5, **_):
        prompts = prompt if isinstance(prompt, list) else [prompt]
        base = np.asarray(image.convert("RGB"), dtype=np.float32)
        out = []
        for k in range(len(prompts)):
            rng = np.random.default_rng(k)
            out.append(Image.fromarray(np.clip(base + rng.normal(0, 3, base.shape), 0, 255).astype(np.uint8)))
        return types.SimpleNamespace(images=out)


def textured_image(seed: int = 0, size: int = 512) -> Image.Image:
    """Smooth random texture (natural-image-like spectrum) so SSIM/PSNR and saliency are non-degenerate."""
    rng = np.random.default_rng(seed)
    small = rng.uniform(0, 255, (size // 32, size // 32, 3)).astype(np.uint8)
    return Image.fromarray(small).resize((size, size), Image.BICUBIC)
