#!/usr/bin/env python
"""Merge PECCAVI shard runs and write rows.csv, results.md and the detection-vs-round figure.

  python scripts/peccavi_report.py --exp <shard0 exp dir> <shard1 exp dir> --out <dir> [--runner-log run_times.json]

Every number comes from the rows; the only hand-entered values are the Falcon-site reference numbers in
SITE_REF, which are labelled as such and were not re-run.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.analyze import wilson  # noqa: E402
from analysis.figures import GRID, INK, INK2, plt  # noqa: E402  (same rcParams as the site figures)

PECCAVI_COLOR = "#6250d6"  # categorical slot 7 (slots 1-3 are WAM / TrustMark / DWT-DCT-SVD on the site)
# Falcon site (prajak002.github.io/Falcon), InstructPix2Pix, L4 detection @ FPR 1e-3, 95% Wilson CI. Not re-run.
SITE_REF = [
    ("WAM", "post-hoc", 18.9, 12.1, 28.2, 90, "PIE-Bench++"),
    ("TrustMark", "post-hoc", 23.3, 15.8, 33.1, 90, "PIE-Bench++"),
    ("Tree-Ring", "generation-time", 22.2, 12.5, 36.3, 45, "SD-1.5 generations"),
    ("Gaussian Shading", "generation-time", 95.6, 85.2, 98.8, 45, "SD-1.5 generations"),
]
QUALITY = ["clip_t", "clip_t_gain", "clip_dir", "clip_i", "dino", "lpips", "ssim", "psnr", "psnr_vs_orig"]


def pct(k, n):
    lo, hi = wilson(k, n)
    return f"{100 * k / n:.1f}% ({max(0.0, 100 * lo):.1f}-{min(100.0, 100 * hi):.1f})" if n else "n/a"


def load_rows(exps: list[Path]) -> pd.DataFrame:
    rows = [json.loads(p.read_text()) for e in exps for p in sorted((e / "results" / "rows").glob("*.json"))]
    if not rows:
        raise SystemExit(f"no rows under {[str(e) for e in exps]}")
    df = pd.DataFrame(rows)
    return df.drop_duplicates(["method", "editor", "watermark", "id", "level"]).sort_values(
        ["watermark", "id", "level"]).reset_index(drop=True)


TRACKS = {"peccavi": ("PECCAVI (official)", "-", PECCAVI_COLOR),  # track -> label, line style, marker face
          "peccavi_raw": ("PECCAVI, no enhancement\n(ablation)", "--", "white")}


def detection_table(df: pd.DataFrame, track: str = "peccavi") -> pd.DataFrame:
    """Detection per round for `track`; false positives from the clean track. Both PECCAVI tracks use the same
    detector, patches and keys, so the clean track's peccavi_* columns are their shared control."""
    wm, clean = df[df.watermark == track], df[df.watermark == "none"]
    out = []
    for L in sorted(df.level.unique()):
        g, c = wm[wm.level == L], clean[clean.level == L]
        k, n = int(g.detected.sum()), len(g)
        fk, fn = int(c.peccavi_detected.sum()), len(c)
        lo, hi = wilson(k, n)
        flo, fhi = wilson(fk, fn)
        out.append({"level": int(L), "n": n, "detected": k, "det_rate": k / n if n else np.nan, "det_ci_lo": lo,
                    "det_ci_hi": hi, "mean_wdp": g.wdp.mean(), "median_p_bonf": g.p_value.median(),
                    "clean_n": fn, "clean_fp": fk, "fpr": fk / fn if fn else np.nan, "fpr_ci_lo": flo,
                    "fpr_ci_hi": fhi, "clean_mean_wdp": c.peccavi_wdp.mean(),
                    "clean_median_p_bonf": c.peccavi_p_value.median()})
    return pd.DataFrame(out)


def figure(dets: dict, out: Path):
    """dets: track -> detection_table(). One track: the original layout (label at L4). Two tracks: labels at L0
    (left of the axis, where the lines are furthest apart) and a legend below."""
    two = len(dets) > 1
    fig, ax = plt.subplots(figsize=(5.0, 3.0) if two else (4.2, 2.8))
    placed_l = []
    for track, det in sorted(dets.items(), key=lambda kv: -kv[1].det_rate.iloc[0]):
        label, ls, mfc = TRACKS[track]
        x = det.level.to_numpy()
        n = int(det.n.min())
        ax.fill_between(x, det.det_ci_lo, det.det_ci_hi, color=PECCAVI_COLOR, alpha=0.12 if ls == "-" else 0.06,
                        linewidth=0)
        ax.plot(x, det.det_rate, color=PECCAVI_COLOR, ls=ls, marker="o", mfc=mfc, mec=PECCAVI_COLOR, mew=1.5,
                label=f"{label.replace(chr(10), ' ')} (n={n})")
        if two:
            y = det.det_rate.iloc[0]
            yy = min(y, placed_l[-1] - 0.16) if placed_l else y
            placed_l.append(yy)
            ax.annotate(label, (x[0], y), xytext=(-0.2, yy), ha="right", va="center", fontsize=7.5, color=INK)
        else:
            ax.annotate(f"PECCAVI (n={n})", (x[-1], det.det_rate.iloc[-1]),
                        xytext=(x[-1] + 0.15, det.det_rate.iloc[-1]), va="center", fontsize=8, color=INK)
    x = next(iter(dets.values())).level.to_numpy()
    n = int(min(d.n.min() for d in dets.values()))
    # site reference: L4 only, neutral markers, direct labels (identity is in the text, not the colour)
    refs = sorted(SITE_REF, key=lambda r: r[2])
    placed = []
    for name, kind, v, lo, hi, nn, _ in refs:
        y = v / 100
        yy = max(y, placed[-1] + 0.085) if placed else y
        placed.append(yy)
        ax.errorbar(4.45, y, yerr=[[y - lo / 100], [hi / 100 - y]], fmt="D", ms=4.5, mfc="white", mec=INK2,
                    ecolor=GRID, elinewidth=1.2, capsize=0)
        ax.annotate(f"{name} {v:.0f}% (site, n={nn})", (4.45, y), xytext=(4.7, yy), va="center", fontsize=7,
                    color=INK2, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.5, relpos=(0, 0.5), shrinkA=1,
                                                shrinkB=4))
    ax.set_xticks(range(int(x.max()) + 1))
    ax.set_xlim(-2.6 if two else -0.2, 6.6)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("editing round (L0 = watermarked original)")
    ax.set_ylabel("detection rate @ FPR 1e-3")
    ax.set_title("InstructPix2Pix", fontsize=9, color=INK)
    if two:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, fontsize=7, frameon=False)
    fig.suptitle(f"PECCAVI retention under sequential editing (baseline, n={n} images; 95% Wilson CI)\n"
                 "smoke test: intervals are wide; diamonds = Falcon-site L4 values, not re-run",
                 fontsize=8, color=INK, y=1.08)
    fig.savefig(out / "peccavi_fig1_retention_det_rate.png")
    plt.close(fig)


def md_table(df: pd.DataFrame, floatfmt=".3g") -> str:
    return df.to_markdown(index=False, floatfmt=floatfmt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", nargs="+", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--runner-log", type=Path, help="run_times.json written by the Kaggle runner")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    df = load_rows(a.exp)
    df.to_csv(a.out / "rows.csv", index=False)
    det = detection_table(df)
    det.to_csv(a.out / "peccavi_detection_by_level.csv", index=False)
    dets = {"peccavi": det}
    rawt = df[df.watermark == "peccavi_raw"]
    if len(rawt):
        dets["peccavi_raw"] = det_raw = detection_table(df, "peccavi_raw")
        det_raw.to_csv(a.out / "peccavi_raw_detection_by_level.csv", index=False)
    figure(dets, a.out)

    wm, clean = df[df.watermark == "peccavi"], df[df.watermark == "none"]
    n = wm.id.nunique()
    L4 = det[det.level == det.level.max()].iloc[0]
    hi0 = 100 * wilson(0, n)[1]
    cats = ", ".join(f"{c} x{k}" for c, k in wm[wm.level == 0].category.value_counts(sort=False).items())
    s = [f"# PECCAVI under sequential InstructPix2Pix editing: smoke test (n={n})", "",
         f"**This is a smoke test with n={n} images. The 95% Wilson intervals are wide (with n={n} a 0/{n} result "
         f"is compatible with a true rate up to {hi0:.0f}%). Treat the rates as a pipeline check, not as estimates.**",
         "", f"Setup: PIE-Bench++ (first {n} of stage 2's 90 images, round-robin over categories: {cats}), "
         "InstructPix2Pix (50 steps, text cfg 7.5, image cfg 1.5), "
         "4 sequential edits as in stage 2 (round 1 = the image's own PIE-Bench++ edit; rounds 2-4 = global "
         "attribute, background, style), same seeds as stage 2. PECCAVI = official CVPR2026 code with the fixes in "
         "`external_repos/peccavi/FIXES.md`. Paraphrase captions = the PIE-Bench++ source prompt. Noisy burnishing "
         "(`utils/noisypatch.py`) is never called by the official `main.py` and is not used here.", "",
         "Detection: PECCAVI's per-patch Tree-Ring chi-squared p-value; image-level p = min(1, k x min_i p_i) over "
         "the k stored patches (Bonferroni); detected iff p <= 1e-3. WDP = max_i (1 - p_i) as in the paper "
         "(reported, not used for the decision).", "",
         "## Table 1 layout: L4 detection rate (watermark x editor), plain editing", "",
         "Values are % with 95% Wilson CI. Only the PECCAVI row was run here; the other rows are copied from the "
         "Falcon site for reference.", "",
         "| Watermark | InstructPix2Pix | SDEdit | FLUX.2-klein | n | Source |",
         "|---|---|---|---|---|---|",
         f"| **PECCAVI** (this run) | **{pct(int(L4.detected), int(L4.n))}** | not run | not run | {int(L4.n)} | "
         "this smoke test, PIE-Bench++ |"]
    if len(rawt):
        R4 = det_raw[det_raw.level == det_raw.level.max()].iloc[0]
        s.append(f"| PECCAVI, no adaptive enhancement (ablation, not the official method) | "
                 f"{pct(int(R4.detected), int(R4.n))} | not run | not run | {int(R4.n)} | this smoke test, PIE-Bench++ |")
    for name, kind, v, lo, hi, nn, data in SITE_REF:
        s.append(f"| {name} ({kind}) | {v}% ({lo}-{hi}) | see site | see site | {nn} | Falcon site, {data}, not re-run |")
    s += ["", "![PECCAVI detection vs round](peccavi_fig1_retention_det_rate.png)", "",
          "## Detection and mean WDP per round (watermarked track), false positives (clean track)", ""]
    t = det.copy()
    t["detection"] = [pct(int(k), int(m)) for k, m in zip(t.detected, t.n)]
    t["clean FP"] = [f"{int(k)}/{int(m)} " + pct(int(k), int(m)) for k, m in zip(t.clean_fp, t.clean_n)]
    s += [md_table(t[["level", "n", "detection", "mean_wdp", "median_p_bonf", "clean FP", "clean_mean_wdp",
                      "clean_median_p_bonf"]]), "",
          "The clean track runs the same PECCAVI detector, with each image's own stored patches and keys, on the "
          "unwatermarked image and its edits (same prompts and seeds).", ""]
    pcols = sorted([c for c in wm.columns if c.startswith("p_patch")])
    if len(rawt):
        r0 = rawt[rawt.level == 0]
        tr = det_raw.copy()
        tr["detection"] = [pct(int(k), int(m)) for k, m in zip(tr.detected, tr.n)]
        tr["clean FP"] = [f"{int(k)}/{int(m)} " + pct(int(k), int(m)) for k, m in zip(tr.clean_fp, tr.clean_n)]
        s += ["## Ablation: PECCAVI without adaptive enhancement (`peccavi_raw`)", "",
              "Not the official method. L0 is the image `Watermarker.forward()` writes *before* adaptive enhancement "
              "(`Image_<id>_wm_100.png`); the official L0 is the enhanced image (theta x original + (1 - theta) x "
              "raw, theta 0.4-0.7 with the released `ssim_threshold: 0.99`). Same images, patches, keys, detector, "
              "edit prompts and seeds as the official track; only the starting image differs. The paper reports "
              "PECCAVI (XRAI) at PSNR 29.6-29.9 dB / SSIM 0.93 (its Table 1), which is close to this un-enhanced "
              "image, not to the enhanced one. The clean track (false positives) is shared: same detector and keys.", "",
              f"Image quality at L0 vs original: PSNR {r0.wm_psnr.mean():.2f} dB (min {r0.wm_psnr.min():.2f}), "
              f"SSIM {r0.wm_ssim.mean():.4f} (min {r0.wm_ssim.min():.4f}); n={len(r0)}. Official (enhanced): "
              f"PSNR {wm[wm.level == 0].wm_psnr.mean():.2f} dB, SSIM {wm[wm.level == 0].wm_ssim.mean():.4f}.", "",
              md_table(tr[["level", "n", "detection", "mean_wdp", "median_p_bonf", "clean FP"]]), "",
              "Per image, per round:", "",
              md_table(rawt[["id", "category", "level", "detected", "p_value", "wdp"] + pcols], ".2e"), ""]
    s += ["## Per image, per round: Bonferroni p, WDP and per-patch p-values", ""]
    s += [md_table(wm[["id", "category", "level", "detected", "p_value", "wdp"] + pcols], ".2e"), "",
          "Clean track:", "", md_table(clean[["id", "level", "peccavi_detected", "peccavi_p_value", "peccavi_wdp"] +
                                             [f"peccavi_{c}" for c in pcols if f"peccavi_{c}" in clean]], ".2e"), ""]
    l0 = wm[wm.level == 0]
    raw = [pd.read_csv(e / "results" / "peccavi_raw_l0.csv", dtype={"id": str}) for e in a.exp
           if (e / "results" / "peccavi_raw_l0.csv").exists()]
    if raw:
        raw = pd.concat(raw).drop_duplicates("id")
        raw.to_csv(a.out / "peccavi_raw_l0.csv", index=False)
        m = l0.assign(id=l0.id.astype(str))[["id", "category", "detected", "p_value", "wdp"]].merge(
            raw, on=["id", "category"], how="inner")
        k_raw, k_enh, nn = int(m.raw_detected.sum()), int(m.detected.sum()), len(m)
        kept = m[[c for c in m if c.startswith("kept_patch")]].stack()
        pr = m[[c for c in m if c.startswith("psnr_raw_patch")]].stack()
        pe = m[[c for c in m if c.startswith("psnr_enh_patch")]].stack()
        s += ["## Diagnostic: L0 detection before vs after adaptive enhancement", "",
              "`Watermarker.forward()` writes the optimised image (raw) and then the adaptively enhanced one "
              "(= theta x original + (1 - theta) x raw), which is the image the pipeline uses as L0. Scoring the raw "
              "image with the same keys separates loss of signal in the enhancement step from loss in embedding / "
              "DDIM inversion. Method unchanged; this is an extra measurement.", "",
              f"- L0 detection, raw: **{pct(k_raw, nn)}**; enhanced (pipeline L0): **{pct(k_enh, nn)}**; n={nn}.",
              f"- Fraction of the raw residual kept by enhancement (1 - theta), per patch: mean {kept.mean():.2f} "
              f"(min {kept.min():.2f}, max {kept.max():.2f}).",
              f"- Watermarked-patch PSNR vs original: raw {pr.mean():.1f} dB (min {pr.min():.1f}), enhanced "
              f"{pe.mean():.1f} dB (min {pe.min():.1f}). Pixels outside the 4 patches are unchanged.", "",
              md_table(m[["id", "category", "raw_p_value", "raw_wdp", "raw_detected", "p_value", "wdp", "detected"]]
                       .rename(columns={"p_value": "enh_p_value", "wdp": "enh_wdp", "detected": "enh_detected"}),
                       ".2e"), ""]
    s += ["## Watermark imperceptibility (watermarked L0 vs original)", "",
          f"PSNR {l0.wm_psnr.mean():.2f} dB (min {l0.wm_psnr.min():.2f}, max {l0.wm_psnr.max():.2f}); "
          f"SSIM {l0.wm_ssim.mean():.4f} (min {l0.wm_ssim.min():.4f}, max {l0.wm_ssim.max():.4f}); n={len(l0)}.", "",
          "## Patches and NMP diagnostics (per image)", ""]
    diag = [c for c in ["id", "category", "wm_patches", "wm_nmp_paraphrases", "wm_nmp_boxes", "wm_nmp_boxes_in_all",
                        "wm_nmp_fallback"] if c in l0]
    s += [md_table(l0[diag]), "",
          "`nmp_fallback = quadrant` means every NMP group was filtered out by `split.py` (groups seen in all 5 "
          "paraphrases score 0.1 < 0.25) and the default box [256,256,512,512] was used; see FIXES.md.", "",
          "## Edit quality per round (metrics the pipeline computes; edit L(r-1) -> L(r))", ""]
    q = [c for c in QUALITY if c in df]
    if q:
        eq = df[df.level > 0].groupby(["watermark", "level"])[q].mean().reset_index()
        s += [md_table(eq), "", "`none` = the same edits on the unwatermarked image (B0).", ""]
    s += ["## Timings and GPU time per image", ""]
    tc = [c for c in l0.columns if c.startswith("wm_time_")]
    tim = l0[["id"] + tc].rename(columns=lambda c: c.replace("wm_time_", ""))
    s += [md_table(tim, ".1f"), ""]
    edit_s = df[df.level > 0].groupby("watermark").edit_seconds.mean() if "edit_seconds" in df else pd.Series(dtype=float)
    score_s = wm.score_s.mean() if "score_s" in wm else np.nan
    para, embed, xrai = (tim[c].mean() if c in tim else 0.0 for c in ("paraphrase_s", "embed_s", "xrai_cpu_s"))
    n_edits = 2 * int(det.level.max())  # both tracks
    per_img = para + embed + xrai + n_edits * float(edit_s.mean() if len(edit_s) else 0) + \
        2 * len(det) * float(score_s if score_s == score_s else 0)
    s += [f"Mean per image (s): paraphrase {para:.0f} (GPU), XRAI {xrai:.0f} (CPU, GPU idle), embed {embed:.0f} (GPU), "
          f"one IP2P edit {float(edit_s.mean()) if len(edit_s) else float('nan'):.1f} (GPU), one PECCAVI detection "
          f"{score_s:.1f} (GPU).", "",
          f"Wall time per image on one T4 (sequential stages, excluding model loading and CLIP/DINO/LPIPS metrics): "
          f"~{per_img / 60:.0f} min = paraphrase + XRAI + embed + {n_edits} edits + {2 * len(det)} detections.", ""]
    if a.runner_log and a.runner_log.exists():
        rl = json.loads(a.runner_log.read_text())
        s += ["Runner wall clock (from the Kaggle runner, includes model downloads/loads and metrics):", "",
              "```", json.dumps(rl, indent=2), "```", ""]
    for N in (30, 90):
        h = N * per_img / 3600
        s.append(f"- {N} images: ~{h:.1f} GPU-h of stage time, ~{h / 2:.1f} h wall on 2x T4 in parallel "
                 "(check Kaggle's quota meter after the smoke run to see how a 2x T4 session is charged).")
    s += ["", "XRAI runs on CPU in the official code; while it runs the GPU is idle, so it counts towards wall "
          "time and quota.", ""]
    (a.out / "results.md").write_text("\n".join(s))
    print((a.out / "results.md").read_text())


if __name__ == "__main__":
    main()
