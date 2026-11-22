#!/usr/bin/env bash
# Start or resume the PECCAVI run on one Vast.ai GPU, inside tmux so it survives SSH disconnects.
#   bash research/scripts/vast/run.sh 1     # 1-image test (check + full pipeline + extras + report)
#   python scripts/vast/xrai_check.py      # after the test: parallel XRAI vs the official sequential loop
#   bash research/scripts/vast/run.sh 90    # all 90; images already done (e.g. by the test) are skipped
# Watch: tmux attach -t peccavi (detach: Ctrl-b d) | tail -f $OUT/runner.log | per-stage logs in $OUT/logs/
# Outputs: $OUT/gpu0/peccavi_smoke/ (images, wm_state, results), $OUT/report/, $OUT/run_times.json
set -euo pipefail
N=${1:?usage: run.sh <max_images>}
REPO=${REPO:-/workspace/Falcon}
OUT=${OUT:-/workspace/peccavi_smoke}
CONFIG=${CONFIG:-configs/peccavi_full90_vast.yaml}

if tmux has-session -t peccavi 2>/dev/null; then
  echo "a run is already going: tmux attach -t peccavi"; exit 1
fi
mkdir -p "$OUT"
tmux new-session -d -s peccavi "cd $REPO/research && WANDB_MODE=disabled python scripts/kaggle/run_peccavi_smoke.py \
  --gpus 1 --no-gate --extras --config $CONFIG --out $OUT --set dataset.max_images=$N 2>&1 | tee -a $OUT/runner.log; \
  echo; echo 'run finished (exit above). Download $OUT, then destroy the instance on cloud.vast.ai.'; exec bash"
echo "started ($N images): tmux attach -t peccavi | tail -f $OUT/runner.log"
