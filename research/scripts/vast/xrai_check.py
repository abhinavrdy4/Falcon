#!/usr/bin/env python
"""Check that parallel XRAI (xrai_workers > 1) gives what the official sequential loop gives, on images the run
already processed. Exit 0 if identical.

  cd /workspace/Falcon/research && python scripts/vast/xrai_check.py --out /workspace/peccavi_smoke [--score-only]

full (default)  re-run split.SaliencyDetector().XRAI(folder) on a copy of each image's 5 paraphrases and compare
                every per-paraphrase XRAI_Analysis_data.json and final.json (minutes of CPU per image).
--score-only    the per-paraphrase XRAI outputs are already known to match; rebuild a folder the way the
                sequential loop would (paraphrases, then result folders in the loop's order) and re-run only the
                official process_and_score_boxes(), then compare final.json (seconds per image).
"""
import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from wmedit.peccavi import peccavi_env  # noqa: E402

IMG = (".png", ".jpg", ".jpeg")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("/workspace/peccavi_smoke"))
    ap.add_argument("--threads", type=int, default=2, help="torch threads, as each parallel worker (xrai_threads)")
    ap.add_argument("--score-only", action="store_true")
    ap.add_argument("--max-images", type=int)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    finals = sorted((a.out / "gpu0" / "peccavi_smoke" / "wm_state" / "peccavi").glob("*/paraphrases/*/final.json"))
    finals = finals[:a.max_images]
    if not finals:
        raise SystemExit(f"no final.json under {a.out}: run an image first")
    same = True
    for final in finals:
        par, sid = final.parent, final.parents[2].name
        seq = a.out / "xrai_check" / sid / par.name
        shutil.rmtree(seq, ignore_errors=True)
        seq.mkdir(parents=True)
        pngs = [f for f in os.listdir(par) if f.lower().endswith(IMG) and "Heatmap" not in f and "Top" not in f]
        for f in pngs:
            shutil.copy(par / f, seq / f)
        t = time.time()
        with peccavi_env("cpu"):
            from utils.split import SaliencyDetector

            if a.score_only:
                for f in pngs:  # result folders created in the sequential loop's order
                    shutil.copytree(par / os.path.splitext(f)[0], seq / os.path.splitext(f)[0])
                SaliencyDetector.__new__(SaliencyDetector).process_and_score_boxes(base_path=str(seq))
            else:
                SaliencyDetector().XRAI(str(seq))
        print(f"{sid}: sequential {'scoring' if a.score_only else 'XRAI'} {time.time() - t:.0f}s", flush=True)
        for f in sorted(seq.rglob("*.json")):
            g = par / f.relative_to(seq)
            eq = g.exists() and json.loads(f.read_text()) == json.loads(g.read_text())
            same &= eq
            print(("  SAME " if eq else "  DIFF ") + str(f.relative_to(seq)))
        print("  patches:", [e["patch"] for e in json.loads(final.read_text())])
    print("RESULT", "IDENTICAL" if same else "DIFFERENT")
    sys.exit(0 if same else 1)


if __name__ == "__main__":
    main()
