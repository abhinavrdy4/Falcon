#!/usr/bin/env python
"""After `run.sh 1`: re-run XRAI on the test image's 5 paraphrases with the official sequential loop
(split.SaliencyDetector().XRAI(folder), as with xrai_workers 0) and compare with what the parallel workers wrote:
every per-paraphrase XRAI_Analysis_data.json and final.json (the 4 patches). Exit 0 if identical.

  cd /workspace/Falcon/research && python scripts/vast/xrai_check.py --out /workspace/peccavi_smoke [--threads 2]
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from wmedit.peccavi import peccavi_env  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("/workspace/peccavi_smoke"))
    ap.add_argument("--threads", type=int, default=2, help="torch threads, as each parallel worker (xrai_threads)")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    finals = sorted((a.out / "gpu0" / "peccavi_smoke" / "wm_state" / "peccavi").glob("*/paraphrases/*/final.json"))
    if not finals:
        raise SystemExit(f"no final.json under {a.out}: run run.sh 1 first")
    same = True
    for final in finals:
        par = final.parent
        seq = a.out / "xrai_check" / par.parents[1].name / par.name
        shutil.rmtree(seq, ignore_errors=True)
        seq.mkdir(parents=True)
        for f in par.glob("*_gen_*.png"):
            shutil.copy(f, seq / f.name)
        t = time.time()
        with peccavi_env("cpu"):
            from utils.split import SaliencyDetector

            SaliencyDetector().XRAI(str(seq))
        print(f"{par.parents[1].name}: sequential XRAI {time.time() - t:.0f}s", flush=True)
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
