#!/usr/bin/env python
"""One-image PECCAVI end-to-end check, run before the smoke test. Exits 1 if the unedited watermarked image is
not detected or the unedited clean image is (the smoke run should then not start).

  python scripts/peccavi_check.py --config configs/peccavi_smoke.yaml --set output_dir=... dataset.shard=[0,2]

Uses the first sample of the given (shard of the) config and the same output_dir as the smoke run, so its
paraphrases, patches, keys and L0 images are reused there. Prints ring-mask sizes, patch coordinates,
PSNR/SSIM vs the original, and WDP + Bonferroni p on both images; writes <exp>/peccavi_check.json.
"""
import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wmedit.metrics import image_quality  # noqa: E402
from wmedit.pipeline import Experiment, load_config  # noqa: E402
from wmedit.utils import write_json  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    exp = Experiment(load_config(a.config, a.set))
    exp.root.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(exp.root / "check.log")])
    exp.prepare()
    exp.samples = exp.samples[:1]
    s = exp.samples[0]
    t = time.time()
    exp.embed()  # L0 for the none and peccavi tracks (cached per stage)
    wm = exp.watermark("peccavi")
    fpr = float(exp.cfg["evaluation"]["target_fpr"])
    orig = exp.images[s.id]
    wm_img = exp.l0_path("peccavi", s.id)
    from PIL import Image

    wm_img = Image.open(wm_img).convert("RGB")
    state = json.loads(wm.paths(s.id)["state"].read_text())
    res = {"id": s.id, "caption": s.source_caption, "embed_wall_s": round(time.time() - t, 1),
           "image_info": wm.image_info(s.id), "patches": state["patches"],
           "quality": image_quality(orig, wm_img),
           "watermarked": wm.score(wm_img, None, fpr, sample=s), "clean": wm.score(orig, None, fpr, sample=s)}
    write_json(exp.root / "peccavi_check.json", res)

    print("\n" + "=" * 80)
    print(f"PECCAVI 1-image check  id={s.id}  caption={s.source_caption!r}")
    print(f"stage times: {res['image_info']}")
    for p in state["patches"]:
        print(f"  patch {p['idx']}: grid [col,row]={p['coords']}  channel={p['channel']}  "
              f"ring mask={p['mask_size']} coeffs  box={p['box']}  placed={p['placed']}")
    print(f"watermarked vs original: PSNR {res['quality']['psnr']:.2f} dB  SSIM {res['quality']['ssim']:.4f}")
    for name in ("watermarked", "clean"):
        r = res[name]
        per = " ".join(f"{r[f'p_patch{i}']:.2e}" for i in range(r["n_patches"]))
        print(f"{name:12s} WDP={r['wdp']:.6f}  p_bonf={r['p_value']:.3e}  detected={r['detected']}  per-patch p: {per}")
    ok = res["watermarked"]["detected"] and not res["clean"]["detected"]
    print("CHECK", "PASSED" if ok else "FAILED: stop here and report (see peccavi_check.json)")
    print("=" * 80)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
