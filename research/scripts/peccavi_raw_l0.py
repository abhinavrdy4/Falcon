#!/usr/bin/env python
"""Diagnostic: PECCAVI detection on the watermarked image *before* adaptive enhancement (L0 raw), next to the
enhanced L0 that the pipeline scores. No change to the method: both images are written by Watermarker.forward()
(Image_<id>_wm_100.png, then Enhanced_<id>_100_wm.png); this only scores the first one too.

  python scripts/peccavi_raw_l0.py --config configs/peccavi_smoke.yaml --set output_dir=... dataset.shard=[0,2]

Writes <exp>/results/peccavi_raw_l0.csv: per image the raw-image Bonferroni p, WDP and per-patch p, plus
per-patch PSNR of raw and enhanced vs the original and the fraction of the raw residual the enhancement kept
(= 1 - theta).
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wmedit.pipeline import Experiment, load_config  # noqa: E402


def psnr(a, b):
    return float(10 * np.log10(1 / max(((a - b) ** 2).mean(), 1e-12)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    exp = Experiment(load_config(a.config, a.set))
    exp.prepare()
    wm = exp.watermark("peccavi")
    fpr = float(exp.cfg["evaluation"]["target_fpr"])
    out = exp.res_dir / "peccavi_raw_l0.csv"
    rows = pd.read_csv(out).to_dict("records") if out.exists() else []
    done = {str(r["id"]) for r in rows}
    for s in exp.samples:
        p = wm.paths(s.id)
        n = wm.file_id(s.id)
        raw_p = p["wm_img"].parent / f"Image_{n}_wm_100.png"
        if s.id in done or not (raw_p.exists() and p["state"].exists()):
            continue
        raw = Image.open(raw_p).convert("RGB")
        sc = wm.score(raw, None, fpr, sample=s)
        row = {"id": s.id, "category": s.category, **{f"raw_{k}": v for k, v in sc.items()
                                                      if not (isinstance(v, float) and np.isnan(v))}}
        o, w, e = (np.asarray(Image.open(x).convert("RGB"), np.float64) / 255
                   for x in (p["input"], raw_p, p["wm_img"]))
        for q in json.loads(p["state"].read_text())["patches"]:
            c, r = q["coords"]
            sl = np.s_[r * 64:(r + 1) * 64, c * 64:(c + 1) * 64]
            row[f"psnr_raw_patch{q['idx']}"] = psnr(w[sl], o[sl])
            row[f"psnr_enh_patch{q['idx']}"] = psnr(e[sl], o[sl])
            row[f"kept_patch{q['idx']}"] = float(np.abs(e[sl] - o[sl]).sum() / max(np.abs(w[sl] - o[sl]).sum(), 1e-12))
        rows.append(row)
        logging.info("raw L0 %s: p_bonf=%.3e detected=%s wdp=%.4f", s.id, sc["p_value"], sc["detected"], sc["wdp"])
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out, index=False)  # after every image, so a timeout keeps what was done


if __name__ == "__main__":
    main()
