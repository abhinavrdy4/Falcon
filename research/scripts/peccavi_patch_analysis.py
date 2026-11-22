#!/usr/bin/env python
"""Per-patch analysis of a PECCAVI run: where the watermark sits, what each edit changed there, and whether the
patch still carries a detectable signal. CPU only, after the run; reads the downloaded outputs and writes only
under --out.

  python scripts/peccavi_patch_analysis.py --config configs/peccavi_full90.yaml \
      --exp <dl>/gpu0/peccavi_smoke <dl>/gpu1/peccavi_smoke --out report/peccavi_full90/patches [--flow] [--semantic]

One row per (image, track, level, patch) in patches.csv, tracks peccavi / peccavi_raw / none (the clean image,
tested with the same patch keys). Columns:
  where       col,row (64 px grid cell), x0,y0, channel; nmp_score (split.py score in final.json);
              nmp_hits = paraphrases whose XRAI boxes cover >= half the patch; object_frac (PIE-Bench++ edit mask)
  detection   p, wdp, detected (p <= target_fpr / n_patches, the per-patch share of the image-level test);
              from the saved detector internals when present: ring_chi2 (statistic / df), ring_corr
              (correlation of the inverted ring with the key)
  change      per pixel CIELAB dE76; change_prev_T / change_l0_T = fraction of patch pixels with dE > T against the
              previous level / L0 of the same track, T in --thresholds; ssim_prev, lpips_prev (if lpips is installed)
  intended    intended_frac = share of the patch the instruction asked to change: round 1 the PIE-Bench++ mask,
              background edits 1 - mask, global attribute / style 1; outside_intended = intended_frac < 0.05
  signal      res_corr / res_energy: watermark residual left in the patch, (track Lk - clean Lk) against
              (track L0 - original); the clean chain is edited with the same seeds, so the difference is the mark
  texture     grad (mean Sobel magnitude of the original patch), entropy; label / label_frac with --semantic
              (SegFormer-B0 ADE20K on the original)
  motion      flow_mag (mean RAFT flow from the previous level, px) and flow_left (share of patch pixels whose
              content moves out of the patch) with --flow
  flags       lost = detected at the previous level and not now; lost_outside_unchanged = lost while outside the
              intended region and change_prev_<main T> < 0.05

Also patch_controls.csv (same per-patch fields for the vae / ip2p_identity controls against L0), summary.md,
fig_survival_vs_change.png, fig_lost_breakdown.png and patch_grids/<id>.png (patches outlined green = detected,
red dashed = not, with p).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import stats
from skimage.color import rgb2lab
from skimage.filters import sobel
from skimage.measure import shannon_entropy
from skimage.metrics import structural_similarity

RESEARCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RESEARCH / "src"))
sys.path.insert(0, str(RESEARCH))

from analysis.figures import GRID, INK, INK2, plt  # noqa: E402  (same rcParams as the site figures)
from wmedit.peccavi import PECCAVI  # noqa: E402

log = logging.getLogger("patches")
TRACKS = ("peccavi", "peccavi_raw", "none")
TRACK_LABEL = {"peccavi": "PECCAVI (official)", "peccavi_raw": "PECCAVI, no enhancement", "none": "clean"}
TRACK_COLOR = {"peccavi": "#2a78d6", "peccavi_raw": "#eb6834", "none": INK2}
OK, LOST = "#1baf7a", "#d62728"  # status colours; never colour alone (solid vs dashed outline + p label)
CONTROLS = ("vae", "ip2p_identity")
GLOBAL_ROUNDS = {"global_attribute", "style"}
P = 64  # patch size (px)


# ---------------------------------------------------------------- loading
def load(p: Path) -> np.ndarray:
    return np.asarray(Image.open(p).convert("RGB"))


class Run:
    """The files of one image in one shard's experiment folder."""

    def __init__(self, root: Path, sid: str):
        self.root, self.sid = root, sid
        st = root / "wm_state" / "peccavi" / sid
        n = PECCAVI.file_id(sid)
        self.state = json.loads((st / "wm" / n / "state.json").read_text())
        self.final = {tuple(e["patch"]): e["score"] for e in json.loads((st / "paraphrases" / n / "final.json").read_text())}
        self.boxes = [json.loads(f.read_text())["merged_boxes"]
                      for f in sorted((st / "paraphrases" / n).glob("*/XRAI_Analysis_data.json"))]
        self.internals = st / "internals"
        self.original = st / f"Image_{n}.png"  # the PIE-Bench++ image PECCAVI was given (RGB, lossless)

    def image(self, track: str, level: int) -> Path:
        if level == 0:
            return self.root / "images" / "_wm" / track / f"{self.sid}.png"
        return self.root / "images" / "baseline" / "ip2p" / track / self.sid / f"L{level}.png"

    def control(self, control: str, track: str) -> Path:
        return self.root / "images" / "controls" / control / track / f"{self.sid}.png"


def ring_stats(npz: Path, idx: int) -> dict:
    """tree_ring_p_value's statistic from the saved inverted latent, key and ring mask."""
    if not npz.exists():
        return {}
    z = np.load(npz)
    if f"latents{idx}" not in z:
        return {}
    m, key = z[f"mask{idx}"], z[f"key{idx}"]
    t = key[m].ravel()
    t = np.concatenate([t.real, t.imag])
    f = np.fft.fftshift(np.fft.fft2(z[f"latents{idx}"]), axes=(-1, -2))[m].ravel()
    f = np.concatenate([f.real, f.imag])
    s = f.std(ddof=1)
    return {"ring_chi2": float((((f - t) / s) ** 2).sum() / len(t)), "ring_corr": float(np.corrcoef(f, t)[0, 1])}


# ---------------------------------------------------------------- per-patch measurements
def patch_slice(c: int, r: int):
    return np.s_[r * P:(r + 1) * P, c * P:(c + 1) * P]


def nmp_hits(boxes: list, c: int, r: int) -> int:
    x0, y0, x1, y1 = c * P, r * P, (c + 1) * P, (r + 1) * P
    hits = 0
    for para in boxes:
        cover = max((max(0, min(x1, b["x2"]) - max(x0, b["x1"])) * max(0, min(y1, b["y2"]) - max(y0, b["y1"]))
                     for b in para), default=0) / P ** 2
        hits += cover >= 0.5
    return hits


def change_fracs(de: np.ndarray, sl, thresholds, tag: str) -> dict:
    d = de[sl]
    return {f"change_{tag}_{t:g}": float((d > t).mean()) for t in thresholds}


def residual(track_k, clean_k, track_0, orig, sl) -> dict:
    r0 = (track_0[sl].astype(np.float64) - orig[sl]).ravel()
    rk = (track_k[sl].astype(np.float64) - clean_k[sl]).ravel()
    e0 = (r0 ** 2).sum()
    corr = float(np.corrcoef(r0, rk)[0, 1]) if r0.std() > 0 and rk.std() > 0 else np.nan
    return {"res_corr": corr, "res_energy": float((rk ** 2).sum() / e0) if e0 > 0 else np.nan}


class Lpips:
    def __init__(self):
        try:
            import lpips
            import torch

            self.torch, self.net = torch, lpips.LPIPS(net="alex", verbose=False).eval()
        except ImportError:
            self.net = None
            log.warning("lpips not installed: lpips_prev left out")

    def __call__(self, a: np.ndarray, b: np.ndarray) -> float:
        if self.net is None:
            return np.nan
        t = lambda x: self.torch.tensor(x / 127.5 - 1, dtype=self.torch.float32).permute(2, 0, 1)[None]  # noqa: E731
        with self.torch.no_grad():
            return float(self.net(t(a), t(b)))


class Flow:
    def __init__(self):
        import torch
        from torchvision.models.optical_flow import Raft_Small_Weights, raft_small

        self.torch, w = torch, Raft_Small_Weights.DEFAULT
        self.net, self.tf = raft_small(weights=w).eval(), w.transforms()

    def __call__(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Flow a -> b, (H, W, 2) in px (x, y)."""
        t = lambda x: self.torch.tensor(x).permute(2, 0, 1)[None]  # noqa: E731
        with self.torch.no_grad():
            fa, fb = self.tf(t(a), t(b))
            return self.net(fa, fb)[-1][0].permute(1, 2, 0).numpy()


def flow_stats(flow: np.ndarray, c: int, r: int) -> dict:
    f = flow[patch_slice(c, r)]
    yy, xx = np.mgrid[r * P:(r + 1) * P, c * P:(c + 1) * P]
    nx, ny = xx + f[..., 0], yy + f[..., 1]
    inside = (nx >= c * P) & (nx < (c + 1) * P) & (ny >= r * P) & (ny < (r + 1) * P)
    return {"flow_mag": float(np.hypot(f[..., 0], f[..., 1]).mean()), "flow_left": float(1 - inside.mean())}


class Semantic:
    MODEL = "nvidia/segformer-b0-finetuned-ade-512-512"

    def __init__(self):
        import torch
        from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

        self.torch = torch
        self.proc = SegformerImageProcessor.from_pretrained(self.MODEL)
        self.net = SegformerForSemanticSegmentation.from_pretrained(self.MODEL).eval()

    def __call__(self, img: np.ndarray) -> tuple[np.ndarray, dict]:
        with self.torch.no_grad():
            logits = self.net(**self.proc(images=img, return_tensors="pt")).logits
            up = self.torch.nn.functional.interpolate(logits, size=img.shape[:2], mode="bilinear")
        return up.argmax(1)[0].numpy(), self.net.config.id2label


# ---------------------------------------------------------------- analysis
def analyze(exp_dirs: list[Path], masks: dict, fpr: float, out: Path, thresholds=(2, 5, 10, 15, 20),
            main_t: float = 10, flow: bool = False, semantic: bool = False, max_images: int | None = None) -> pd.DataFrame:
    out.mkdir(parents=True, exist_ok=True)
    rows = pd.concat([pd.read_csv(d / "results" / "rows.csv", dtype={"id": str}) for d in exp_dirs])
    ctrl_f = [d / "results" / "peccavi_controls.csv" for d in exp_dirs]
    ctrl = pd.concat([pd.read_csv(f, dtype={"id": str}) for f in ctrl_f if f.exists()]) if any(
        f.exists() for f in ctrl_f) else pd.DataFrame()
    where = {}  # image id -> the shard folder holding its keys and images
    for d in exp_dirs:
        for st in (d / "wm_state" / "peccavi").glob("*/wm/*/state.json"):
            where[st.parents[2].name] = d
    ids = sorted(set(rows.id) & set(where))[:max_images]
    lp, fl, sem = Lpips(), Flow() if flow else None, Semantic() if semantic else None
    recs, crecs = [], []
    for sid in ids:
        run = Run(where[sid], sid)
        orig = load(run.original)
        mask = masks.get(sid, np.zeros(orig.shape[:2], bool)).astype(bool)
        gray = orig.mean(2) / 255
        labels, id2label = sem(orig) if sem else (None, None)
        imgs = {t: {k: load(run.image(t, k)) for k in range(5) if run.image(t, k).exists()} for t in TRACKS}
        labs = {t: {k: rgb2lab(v) for k, v in d.items()} for t, d in imgs.items()}
        r_img = rows[rows.id == sid]
        for t in TRACKS:
            for k, img in imgs[t].items():
                row = r_img[(r_img.watermark == t) & (r_img.level == k)]
                if row.empty:
                    continue
                row = row.iloc[0]
                pre = "" if t != "none" else "peccavi_"
                n_p = int(row[f"{pre}n_patches"]) if pd.notna(row.get(f"{pre}n_patches")) else 4
                de_prev = np.linalg.norm(labs[t][k] - labs[t][k - 1], axis=2) if k else None
                de_l0 = np.linalg.norm(labs[t][k] - labs[t][0], axis=2)
                flow_f = fl(imgs[t][k - 1], img) if fl and k else None
                npz = run.internals / f"{PECCAVI.image_key(Image.fromarray(img))}.npz"
                for q in run.state["patches"]:
                    i, (c, rr) = q["idx"], q["coords"]
                    sl = patch_slice(c, rr)
                    p = row.get(f"{pre}p_patch{i}", np.nan)
                    rec = {"id": sid, "category": row["category"], "track": t, "level": k,
                           "round_category": row["round_category"], "patch": i, "col": c, "row": rr,
                           "x0": c * P, "y0": rr * P, "channel": q.get("channel", i % 4),
                           "nmp_score": run.final.get((c, rr), np.nan), "nmp_hits": nmp_hits(run.boxes, c, rr),
                           "object_frac": float(mask[sl].mean()), "grad": float(sobel(gray)[sl].mean()),
                           "entropy": float(shannon_entropy((gray[sl] * 255).astype(np.uint8))),
                           "p": p, "wdp": row.get(f"{pre}wdp_patch{i}", np.nan),
                           "detected": bool(pd.notna(p) and p <= fpr / n_p),
                           **ring_stats(npz, i), **change_fracs(de_l0, sl, thresholds, "l0")}
                    if k:
                        rec.update(change_fracs(de_prev, sl, thresholds, "prev"))
                        rec["ssim_prev"] = float(structural_similarity(imgs[t][k - 1][sl], img[sl], channel_axis=2,
                                                                       data_range=255))
                        rec["lpips_prev"] = lp(imgs[t][k - 1][sl], img[sl])
                        rc = row["round_category"]
                        m = float(mask[sl].mean())
                        rec["intended_frac"] = 1.0 if rc in GLOBAL_ROUNDS else (1 - m if rc == "background" else m)
                    if t != "none" and k in imgs["none"]:
                        rec.update(residual(img, imgs["none"][k], imgs[t][0], orig, sl))
                    if flow_f is not None:
                        rec.update(flow_stats(flow_f, c, rr))
                    if labels is not None:
                        lab, cnt = np.unique(labels[sl], return_counts=True)
                        rec["label"], rec["label_frac"] = id2label[int(lab[cnt.argmax()])], float(cnt.max() / P ** 2)
                    recs.append(rec)
            for cn in CONTROLS:
                cr = ctrl[(ctrl.id == sid) & (ctrl.track == t) & (ctrl.control == cn)] if len(ctrl) else ctrl
                if t == "none" or cr.empty or 0 not in imgs[t] or not run.control(cn, t).exists():
                    continue
                cr, cimg = cr.iloc[0], load(run.control(cn, t))
                de = np.linalg.norm(rgb2lab(cimg) - labs[t][0], axis=2)
                for q in run.state["patches"]:
                    i, (c, rr) = q["idx"], q["coords"]
                    sl, p = patch_slice(c, rr), cr.get(f"p_patch{i}", np.nan)
                    crec = {"id": sid, "track": t, "control": cn, "patch": i, "col": c, "row": rr, "p": p,
                            "detected": bool(pd.notna(p) and p <= fpr / int(cr["n_patches"])),
                            **change_fracs(de, sl, thresholds, "l0")}
                    if run.control(cn, "none").exists():
                        crec.update(residual(cimg, load(run.control(cn, "none")), imgs[t][0], orig, sl))
                    crecs.append(crec)
        log.info("patches: %s done", sid)
    df = pd.DataFrame(recs)
    df = flag(df, main_t)
    df.to_csv(out / "patches.csv", index=False)
    cdf = pd.DataFrame(crecs)
    if len(cdf):
        cdf.to_csv(out / "patch_controls.csv", index=False)
    summary(df, cdf, ctrl, main_t, out)
    figures(df, main_t, out)
    grids(df, where, out / "patch_grids")
    return df


def flag(df: pd.DataFrame, main_t: float) -> pd.DataFrame:
    key = ["id", "track", "patch"]
    df = df.sort_values(key + ["level"]).reset_index(drop=True)
    df["detected_prev"] = df.groupby(key)["detected"].shift(1)
    df["lost"] = (df.detected_prev == True) & ~df.detected  # noqa: E712
    if "intended_frac" in df:
        df["outside_intended"] = df.intended_frac < 0.05
        df["lost_outside_unchanged"] = df.lost & df.outside_intended & (df[f"change_prev_{main_t:g}"] < 0.05)
    return df


# ---------------------------------------------------------------- outputs
def summary(df: pd.DataFrame, cdf: pd.DataFrame, ctrl: pd.DataFrame, main_t: float, out: Path) -> None:
    c = f"change_prev_{main_t:g}"
    s = ["# PECCAVI per-patch analysis", "",
         f"{df.id.nunique()} images, {df.patch.nunique()} patches per image. A patch counts as detected at "
         f"p <= target FPR / number of patches; change = share of patch pixels with CIELAB dE > {main_t:g} against the "
         "previous level of the same track.", "", "## Patch detection rate and change by level", ""]
    g = df.groupby(["track", "level"]).agg(patches=("detected", "size"), detected=("detected", "mean"),
                                           change=(c, "mean") if c in df else ("detected", "size"),
                                           lost=("lost", "sum"))
    s.append(g.to_markdown(floatfmt=".3f"))
    wm = df[(df.track != "none") & (df.level > 0)]
    if "lost_outside_unchanged" in wm and len(wm):
        s += ["", "## Lost patches: was the patch inside the edit?", ""]
        lost = wm[wm.lost]
        b = lost.groupby(["track", "level"]).agg(lost=("lost", "size"),
                                                 outside_intended=("outside_intended", "sum"),
                                                 outside_and_unchanged=("lost_outside_unchanged", "sum"))
        s.append(b.to_markdown())
        alive = wm[wm.detected_prev == True]  # noqa: E712
        if len(alive) > 2 and c in alive:
            rho, pv = stats.spearmanr(alive[c], -np.log10(alive.p.clip(lower=1e-300)))
            s += ["", f"Spearman rho(change, -log10 p) over patches detected at the previous level: {rho:.3f} "
                      f"(p = {pv:.2g}, n = {len(alive)})."]
            bins = pd.cut(alive[c], [-0.001, 0.05, 0.2, 0.5, 1.0])
            s += ["", "Survival (still detected) of patches detected at the previous level, by change:", "",
                  alive.groupby(["track", bins], observed=True).detected.agg(["size", "mean"]).to_markdown(
                      floatfmt=".3f")]
    if len(cdf):
        s += ["", "## Controls (no requested change), per patch", "",
              cdf.groupby(["track", "control"]).agg(patches=("detected", "size"), detected=("detected", "mean"),
                                                    change=(f"change_l0_{main_t:g}", "mean")).to_markdown(floatfmt=".3f")]
    if len(ctrl):
        s += ["", "Image level (Bonferroni over patches):", "",
              ctrl.groupby(["track", "control"]).detected.agg(["size", "mean"]).to_markdown(floatfmt=".3f")]
    (out / "summary.md").write_text("\n".join(s) + "\n")


def figures(df: pd.DataFrame, main_t: float, out: Path) -> None:
    c = f"change_prev_{main_t:g}"
    wm = df[(df.track != "none") & (df.level > 0) & (df.detected_prev == True)]  # noqa: E712
    if wm.empty or c not in wm:
        return
    edges = [0, 0.05, 0.2, 0.5, 1.0]
    mids = ["<5%", "5-20%", "20-50%", ">50%"]
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    for t, g in wm.groupby("track"):
        b = pd.cut(g[c], [-0.001] + edges[1:], labels=mids)
        r = g.groupby(b, observed=False).detected.agg(["size", "mean"])
        x = np.arange(len(mids))
        ax.plot(x, r["mean"], marker="o", color=TRACK_COLOR[t], label=TRACK_LABEL[t])
        for xi, (n, m) in zip(x, r.values):
            if n:
                ax.annotate(f"n={int(n)}", (xi, m), xytext=(0, 6), textcoords="offset points", ha="center",
                            fontsize=6.5, color=INK2)
    ax.set_xticks(range(len(mids)), mids)
    ax.set_ylim(-0.05, 1.1)
    ax.set_xlabel(f"patch pixels changed by the edit (CIELAB dE > {main_t:g})")
    ax.set_ylabel("patches still detected")
    ax.set_title("Does a patch survive when its pixels barely change?", fontsize=9, color=INK)
    ax.legend(fontsize=7.5, loc="lower left")
    fig.savefig(out / "fig_survival_vs_change.png")
    plt.close(fig)

    lost = df[(df.track != "none") & df.lost]
    if lost.empty or "outside_intended" not in lost:
        return
    kinds = {"inside the edit": ~lost.outside_intended,
             "outside, pixels changed": lost.outside_intended & ~lost.lost_outside_unchanged,
             "outside, pixels unchanged": lost.lost_outside_unchanged}
    tracks = [t for t in TRACKS if t in set(lost.track)]
    fig, axes = plt.subplots(1, len(tracks), figsize=(3.2 * len(tracks), 3), sharey=True, squeeze=False)
    for ax, t in zip(axes[0], tracks):
        bottom = np.zeros(4)
        for (name, m), col in zip(kinds.items(), (INK2, "#9e9d98", LOST)):
            v = np.array([((lost.track == t) & (lost.level == lv) & m).sum() for lv in range(1, 5)])
            ax.bar(range(1, 5), v, bottom=bottom, color=col, label=name, edgecolor="white", linewidth=1)
            bottom += v
        ax.set_xticks(range(1, 5), [f"L{lv}" for lv in range(1, 5)])
        ax.set_title(TRACK_LABEL[t], fontsize=8.5, color=INK)
    axes[0][0].set_ylabel("patches lost at this level")
    axes[0][-1].legend(fontsize=7, loc="upper right")
    fig.savefig(out / "fig_lost_breakdown.png")
    plt.close(fig)


def grids(df: pd.DataFrame, where: dict, out: Path) -> None:
    import matplotlib.patches as mp

    out.mkdir(parents=True, exist_ok=True)
    for sid, g in df[df.track != "none"].groupby("id"):
        run = Run(where[sid], sid)
        tracks = [t for t in TRACKS if t != "none" and t in set(g.track)]
        levels = sorted(g.level.unique())
        fig, axes = plt.subplots(len(tracks), len(levels), figsize=(2.1 * len(levels), 2.3 * len(tracks)),
                                 squeeze=False)
        for ti, t in enumerate(tracks):
            for li, lv in enumerate(levels):
                ax = axes[ti][li]
                ax.imshow(load(run.image(t, lv)))
                ax.set_xticks([]), ax.set_yticks([]), ax.grid(False)
                for _, q in g[(g.track == t) & (g.level == lv)].iterrows():
                    ok = bool(q.detected)
                    ax.add_patch(mp.Rectangle((q.x0, q.y0), P, P, fill=False, lw=1.6, ec=OK if ok else LOST,
                                              ls="-" if ok else "--"))
                    ax.text(q.x0 + 2, q.y0 + 2, f"{q.p:.0e}" if pd.notna(q.p) else "-", fontsize=5, va="top",
                            color=INK, bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none", alpha=0.8))
                if ti == 0:
                    rc = g[g.level == lv].round_category.iloc[0]
                    ax.set_title(f"L{lv}" + ("" if lv == 0 else f" {rc}"), fontsize=7, color=INK)
                if li == 0:
                    ax.set_ylabel(TRACK_LABEL[t], fontsize=7, color=INK)
        fig.suptitle(f"{sid}: green solid = patch detected, red dashed = not (p shown)", fontsize=7.5, color=INK2)
        fig.savefig(out / f"{sid}.png", dpi=110)
        plt.close(fig)


def main():
    from wmedit.data import load_pie_bench
    from wmedit.pipeline import load_config

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--exp", nargs="+", type=Path, required=True, help="experiment folders (one per GPU shard)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--thresholds", nargs="+", type=float, default=[2, 5, 10, 15, 20])
    ap.add_argument("--main-threshold", type=float, default=10)
    ap.add_argument("--flow", action="store_true", help="RAFT-small optical flow per level (torchvision weights)")
    ap.add_argument("--semantic", action="store_true", help="SegFormer-B0 ADE20K label per patch (HF weights)")
    ap.add_argument("--max-images", type=int)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(a.config)
    d = cfg["dataset"]
    _, _, masks = load_pie_bench(RESEARCH / "datasets" / "hf_cache", d.get("subsets"), d.get("per_subset"),
                                 d.get("max_images"), int(d.get("rounds", 4)), int(cfg.get("seed", 42)),
                                 d.get("skip_per_subset", 0), d.get("followup_order"))
    analyze(a.exp, masks, float(cfg["evaluation"]["target_fpr"]), a.out, a.thresholds, a.main_threshold,
            a.flow, a.semantic, a.max_images)


if __name__ == "__main__":
    main()
