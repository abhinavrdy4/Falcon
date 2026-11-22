#!/usr/bin/env python
"""Analysis extras for the PECCAVI run. Additive: reads the pipeline's images and keys, never rewrites its
images or rows.csv, and does not change embedding, editing or detection.

  python scripts/peccavi_extras.py --stage controls  --config configs/peccavi_full90.yaml [--set ...]
  python scripts/peccavi_extras.py --stage internals --config configs/peccavi_full90.yaml [--set ...]

controls   two no-semantic-change transforms of every L0 image (tracks none, peccavi, peccavi_raw), scored with
           PECCAVI's detector, to separate "the editor's machinery erases the watermark" from "the requested change
           does":
             vae            encode -> decode through InstructPix2Pix's own VAE (the latent bottleneck alone)
             ip2p_identity  InstructPix2Pix, same settings and seed as round 1, instruction "keep the image
                            unchanged"
           The clean (none) track is transformed too, with the same seed, so watermarked - clean differences stay
           paired; it is not scored. Images: <exp>/images/controls/<control>/<track>/<id>.png,
           rows: <exp>/results/peccavi_controls.csv.
internals  fills in <wm_state>/peccavi/<id>/internals/<image_key>.npz (see PECCAVI.detect_patches) for every
           pipeline and control image that has none, e.g. images scored before save_internals was on. Needs
           save_internals: true on the peccavi watermark in the config.
"""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wmedit.pipeline import Experiment, load_config  # noqa: E402
from wmedit.probes import identity_step  # noqa: E402
from wmedit.utils import pil_to_tensor, seed_everything, stable_seed, tensor_to_pil  # noqa: E402

log = logging.getLogger("wmedit")
TRACKS = ("none", "peccavi", "peccavi_raw")
CONTROLS = ("vae", "ip2p_identity")


def psnr(a: Image.Image, b: Image.Image) -> float:
    x, y = (np.asarray(i.convert("RGB"), np.float64) / 255 for i in (a, b))
    return float(10 * np.log10(1 / max(((x - y) ** 2).mean(), 1e-12)))


def control_path(exp: Experiment, control: str, track: str, sid: str) -> Path:
    return exp.img_dir / "controls" / control / track / f"{sid}.png"


@torch.no_grad()
def vae_roundtrip(editor, img: Image.Image) -> Image.Image:
    """InstructPix2Pix's image encode (latent_dist.mode(), as in InstructPix2Pix.edit) followed by its decode."""
    vae = editor.pipe.vae
    x = (pil_to_tensor(img, editor.device) * 2 - 1).to(editor.dtype)
    return tensor_to_pil((vae.decode(vae.encode(x).latent_dist.mode()).sample.float() + 1) / 2)


def run_controls(exp: Experiment) -> None:
    from wmedit.editors import build_editor

    tracks = [t for t in TRACKS if t in exp.wm_cfgs]
    out = exp.res_dir / "peccavi_controls.csv"
    rows = pd.read_csv(out, dtype={"id": str}).to_dict("records") if out.exists() else []
    done = {(r["id"], r["track"], r["control"]) for r in rows}
    ed_name, ed_cfg = next(iter(exp.ed_cfgs.items()))
    editor = None
    fpr = float(exp.cfg["evaluation"]["target_fpr"])
    for s in exp.samples:
        seed = stable_seed("edit", exp.seed, s.id, 1)  # the round-1 edit's seed: shared across tracks (paired)
        for track in tracks:
            l0_p = exp.l0_path(track, s.id)
            if not l0_p.exists():
                continue
            l0 = Image.open(l0_p).convert("RGB")
            for control in CONTROLS:
                p = control_path(exp, control, track, s.id)
                if not p.exists():
                    if editor is None:
                        editor = build_editor(ed_name, **{k: v for k, v in ed_cfg.items() if k != "name"})
                    seed_everything(seed)
                    img = (vae_roundtrip(editor, l0) if control == "vae"
                           else editor.edit(l0, identity_step(s.source_caption), seed))
                    p.parent.mkdir(parents=True, exist_ok=True)
                    img.save(p)
                if track == "none" or (s.id, track, control) in done:
                    continue
                img = Image.open(p).convert("RGB")
                sc = exp.watermark(track).score(img, None, fpr, sample=s)
                rows.append({"id": s.id, "category": s.category, "track": track, "control": control,
                             "psnr_vs_l0": psnr(img, l0), **{k: v for k, v in sc.items()
                                                             if not (isinstance(v, float) and np.isnan(v))}})
                log.info("control %s/%s %s: p=%.3e detected=%s", control, track, s.id, sc["p_value"], sc["detected"])
                out.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(rows).to_csv(out, index=False)  # after every row, so a timeout keeps what was done


def run_internals(exp: Experiment) -> None:
    det = exp.watermark("peccavi")  # keys and detector for every track, as in the pipeline's control track
    if not det.save_internals:
        raise SystemExit("set save_internals: true on the peccavi watermark in the config")
    n = 0
    for s in exp.samples:
        for track in TRACKS:
            if track not in exp.wm_cfgs:
                continue
            paths = [exp.level_path(m, e, track, s.id, r) for m in exp.methods for e in exp.ed_cfgs
                     if exp._applicable(m, track) for r in range(exp.rounds + 1)]
            paths += [control_path(exp, c, track, s.id) for c in CONTROLS]
            for p in paths:
                if not p.exists():
                    continue
                img = Image.open(p).convert("RGB")
                if not det.internals_path(img, s.id).exists():
                    det.detect_patches(img, s.id)
                    n += 1
    log.info("internals: %d images filled in", n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("controls", "internals"), required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    exp = Experiment(load_config(a.config, a.set))
    exp.prepare()
    (run_controls if a.stage == "controls" else run_internals)(exp)


if __name__ == "__main__":
    main()
