#!/usr/bin/env bash
# One-time setup on a rented Vast.ai GPU instance (PyTorch template, CUDA >= 12.1), over SSH:
#   git clone -b peccavi-baseline https://github.com/abhinavrdy4/Falcon.git /workspace/Falcon
#   bash /workspace/Falcon/research/scripts/vast/setup.sh
# Same packages as the Kaggle notebook (torch/torchvision come with the image), then every model and the
# PIE-Bench++ parquet files are downloaded once (scripts/kaggle/prefetch_models.py), so run times exclude downloads.
set -euo pipefail
FORK=${FORK:-abhinavrdy4/Falcon}
BRANCH=${BRANCH:-peccavi-baseline}
REPO=${REPO:-/workspace/Falcon}

if [ -d "$REPO/.git" ]; then
  git -C "$REPO" fetch -q origin "$BRANCH" && git -C "$REPO" reset -q --hard "origin/$BRANCH"
else
  git clone -q --depth 1 --branch "$BRANCH" "https://github.com/$FORK.git" "$REPO"
fi
git -C "$REPO" log -1 --format='code: %h %s'
command -v tmux >/dev/null || (apt-get update -qq && apt-get install -y -qq tmux >/dev/null)

cd "$REPO/research"
python -m pip install -q "diffusers>=0.35" "transformers>=4.50" accelerate safetensors huggingface_hub lpips \
  open_clip_torch saliency shapely pytorch-msssim omegaconf tabulate wandb pyyaml scipy pandas pyarrow pillow \
  opencv-python-headless matplotlib scikit-image
python - <<'EOF'
import sys, torch
sys.path.insert(0, "src")
from wmedit.utils import available_cpus
assert torch.cuda.is_available(), "no GPU visible"
p = torch.cuda.get_device_properties(0)
print(f"torch {torch.__version__}, {p.name} {p.total_memory / 2**30:.0f} GB, CPUs available: {available_cpus()}")
EOF
WANDB_MODE=disabled python scripts/kaggle/prefetch_models.py
df -h /workspace | tail -1
echo "setup done. Next: bash $REPO/research/scripts/vast/run.sh 1   (1-image test), then run.sh 90"
