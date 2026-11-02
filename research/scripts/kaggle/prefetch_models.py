#!/usr/bin/env python
"""Download every model/dataset the PECCAVI smoke run needs, once, before the two GPU processes start
(so they don't race on the same files and per-image timings exclude downloads). Prints seconds per item."""
import sys
import time
from pathlib import Path

RESEARCH = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RESEARCH / "src"))


def timed(name, fn):
    t = time.time()
    fn()
    print(f"{name:40s} {time.time() - t:7.1f}s", flush=True)


def main():
    from huggingface_hub import snapshot_download

    fp32 = ["*.json", "*.txt", "unet/diffusion_pytorch_model.safetensors", "vae/diffusion_pytorch_model.safetensors",
            "text_encoder/model.safetensors"]
    fp16 = ["*.json", "*.txt", "*.fp16.safetensors"]
    timed("SD-2-base mirror (PECCAVI embed/detect)",
          lambda: snapshot_download("sd2-community/stable-diffusion-2-base", allow_patterns=fp32))
    timed("SDXL base fp16 (PECCAVI paraphrase)",
          lambda: snapshot_download("stabilityai/stable-diffusion-xl-base-1.0", allow_patterns=fp16))
    timed("InstructPix2Pix fp16", lambda: snapshot_download("timbrooks/instruct-pix2pix", allow_patterns=fp16))
    timed("DINOv2-small", lambda: snapshot_download("facebook/dinov2-small"))

    def clip():
        import open_clip

        open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")

    def tv():
        import lpips
        import torchvision.models as m

        m.inception_v3(weights="IMAGENET1K_V1")  # XRAI (split.py)
        m.vgg16(weights="IMAGENET1K_V1")  # Watson-VGG loss
        lpips.LPIPS(net="alex", verbose=False)

    def pie():
        from wmedit.data import load_pie_bench

        load_pie_bench(RESEARCH / "datasets" / "hf_cache", None, 10, 5, 4, 42)

    timed("CLIP ViT-L-14 (edit metrics)", clip)
    timed("torchvision InceptionV3, VGG16, LPIPS", tv)
    timed("PIE-Bench++ parquet (categories 1-9)", pie)


if __name__ == "__main__":
    main()
