"""PECCAVI (Dixit et al., CVPR 2026): zero-bit, post-hoc, latent ring watermark in Non-Melting Points.

Runs the vendored official code (external_repos/peccavi = SRDdev/PECCAVI@CVPR2026 with the fixes listed in
external_repos/peccavi/FIXES.md). Per image:

  paraphrase  SDXL img2img x5 (strength 0.2, cfg 7.5), caption = PIE-Bench++ source prompt   Paraphraser
  xrai        XRAI saliency per paraphrase -> cross-paraphrase NMP boxes -> top-4 64 px patches  split.SaliencyDetector
  embed       per patch: DDIM-invert, ring key in channel idx%4, 100 Adam iters, adaptive enhancement
                                                                                               Watermarker.forward
  state       patches, channels, key seeds -> state.json (keys are regenerated from it at detection)

Detection is PECCAVI's own per-patch test (Watermarker.detect): DDIM-invert each stored patch and take the
Tree-Ring non-central chi-squared p-value of its ring. Image-level decision: Bonferroni over the k patches,
p = min(1, k * min_i p_i), detected iff p <= target_fpr. The paper's WDP = max_i (1 - p_i) is reported too.

Every stage is cached under <experiment>/wm_state/peccavi/<id>/ and skipped when its output exists, so an
interrupted run resumes. Paraphrasing runs in fp16 on CUDA (no gradients); embedding and detection stay in
fp32 as in the official code (Adam through the UNet; p-values from DDIM inversion).
"""
from __future__ import annotations

import gc
import hashlib
import importlib.machinery
import json
import logging
import os
import sys
import time
import types
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .utils import RESEARCH_ROOT, available_cpus, get_device, stable_seed, write_json
from .watermarks import Watermark, _cwd

PECCAVI_ROOT = RESEARCH_ROOT / "external_repos" / "peccavi"
log = logging.getLogger("wmedit")


@contextmanager
def peccavi_env(device: str):
    """The official code reads ./config.yaml and ./ZoDiac/loss/*.pth relative to the cwd, imports `utils` and
    `ZoDiac` as top-level packages, sets the root logger to DEBUG on import and writes debug plots to ./output."""
    if str(PECCAVI_ROOT) not in sys.path:
        sys.path.insert(0, str(PECCAVI_ROOT))
    if "tensorflow" not in sys.modules:  # split.py imports tensorflow but never uses it
        tf = types.ModuleType("tensorflow")
        tf.__spec__ = importlib.machinery.ModuleSpec("tensorflow", None)
        sys.modules["tensorflow"] = tf
    os.environ["PECCAVI_DEVICE"] = device
    (PECCAVI_ROOT / "output").mkdir(exist_ok=True)
    level = logging.getLogger().level
    with _cwd(PECCAVI_ROOT):
        try:
            yield
        finally:
            logging.getLogger().setLevel(level)
            import matplotlib.pyplot as plt

            plt.close("all")  # visualize_tensor() opens a figure per call and never closes it


class _OneFile:
    """Stands in for split.py's `os` module inside an XRAI worker: listdir() returns the one file, the rest is os."""

    def __init__(self, fname: str):
        self.fname = fname

    def listdir(self, _path):
        return [self.fname]

    def __getattr__(self, name):
        return getattr(os, name)


_XRAI_SAL = None  # one SaliencyDetector (InceptionV3) per worker process


def _xrai_worker_init(threads: int) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # split.py keeps InceptionV3 on the CPU; keep workers off the GPU
    os.environ["MPLBACKEND"] = "Agg"
    torch.set_num_threads(threads)


def _xrai_one(folder: str, fname: str) -> float:
    """split.SaliencyDetector.XRAI() on one paraphrase in `folder`, in a worker process: the official loop body,
    unchanged, with listdir narrowed to that file; the cross-paraphrase scoring runs once in the parent."""
    global _XRAI_SAL
    t = time.time()
    with peccavi_env("cpu"):
        import utils.split as split

        if _XRAI_SAL is None:
            _XRAI_SAL = split.SaliencyDetector()
            _XRAI_SAL.process_and_score_boxes = lambda *a, **k: None
        real, split.os = split.os, _OneFile(fname)
        try:
            _XRAI_SAL.XRAI(folder)
        finally:
            split.os = real
    return time.time() - t


class PECCAVI(Watermark):
    name = "peccavi"
    nbits = 0  # zero-bit
    per_image_state = True  # pipeline passes the Sample to embed/score and sets cache_dir
    state_name = "peccavi"  # cache folder under <experiment>/wm_state/ (shared by PECCAVIRaw)

    def __init__(self, device=None, num_paraphrases: int = 5, paraphrase_fp16: bool = True,
                 save_internals: bool = False, xrai_workers: int | str = 0, xrai_threads: int = 2, **_):
        import matplotlib

        matplotlib.use("Agg")
        self.device = torch.device(device or get_device())
        self.num_paraphrases = num_paraphrases
        self.paraphrase_fp16 = paraphrase_fp16 and self.device.type == "cuda"
        self.save_internals = save_internals  # analysis only: see detect_patches()
        # XRAI (CPU) per paraphrase in worker processes, same computation; 0/1 = official sequential loop
        self.xrai_threads = max(1, int(xrai_threads))
        self.xrai_workers = (max(1, (available_cpus() - 2) // self.xrai_threads) if xrai_workers == "auto"
                             else int(xrai_workers))
        self._xrai_pool, self._xrai_jobs = None, {}
        self.cache_dir: Path | None = None
        self._para = self._sal = self._wmk = None
        with peccavi_env(self.device.type):
            import wandb

            from utils.config import ConfigLoader

            wandb.init(mode="disabled")  # ZoDiac/loss/loss.py logs to wandb unconditionally
            self.cfg = ConfigLoader("config.yaml")

    # ------------------------------------------------------------------ paths
    @staticmethod
    def file_id(sid: str) -> str:
        """The official code finds the image id with r'Image_(\\d+)' and 'prefix_ID.ext', so it must be digits."""
        return sid if sid.isdigit() else str(stable_seed("peccavi_id", sid))

    def paths(self, sid: str) -> dict:
        root = Path(self.cache_dir) / sid
        n = self.file_id(sid)
        return {"root": root, "input": root / f"Image_{n}.png", "para_dir": root / "paraphrases" / n,
                "final": root / "paraphrases" / n / "final.json", "wm_dir": root / "wm",
                "wm_img": root / "wm" / n / f"Enhanced_{n}_100_wm.png", "raw_img": root / "wm" / n / f"Image_{n}_wm_100.png",
                "state": root / "wm" / n / "state.json",
                "timings": root / "timings.json"}

    def _record(self, sid: str, **seconds):
        p = self.paths(sid)["timings"]
        t = json.loads(p.read_text()) if p.exists() else {}
        t.update({k: round(v, 2) for k, v in seconds.items()})
        write_json(p, t)

    def image_info(self, sid: str) -> dict:
        """Per-image stage timings and NMP diagnostics (pipeline adds them to the L0 row as wm_*)."""
        p = self.paths(sid)
        info = {f"time_{k}": v for k, v in (json.loads(p["timings"].read_text()) if p["timings"].exists() else {}).items()}
        if p["final"].exists():
            entries = json.loads(p["final"].read_text())
            boxes = [{tuple(b.values()) for b in json.loads(f.read_text())["merged_boxes"]}
                     for f in sorted(p["para_dir"].glob("*/XRAI_Analysis_data.json"))]
            seen = [sum(b in bs for bs in boxes) for b in set().union(*boxes)] if boxes else []
            info.update({"nmp_paraphrases": len(boxes), "nmp_boxes": len(seen),
                         "nmp_boxes_in_all": sum(n == len(boxes) for n in seen),
                         "nmp_fallback": self.nmp_fallback(entries),
                         "patches": ";".join(f"{c},{r}" for c, r in (e["patch"] for e in entries))})
        return info

    @staticmethod
    def nmp_fallback(entries: list) -> str:
        """Which split.py fallback produced final.json, if any: 'quadrant' = every box was filtered out and the
        default box [256,256,512,512] was used (4 tied cells), 'centre' = no box overlapped the grid."""
        cells = [tuple(e["patch"]) for e in entries]
        if all(e["score"] == 1.0 for e in entries):
            if len(cells) == 4 and all(4 <= c <= 7 and 4 <= r <= 7 for c, r in cells):
                return "quadrant"
            if cells == [(4, 4)]:
                return "centre"
        return "none"

    # ------------------------------------------------------------------ models (one heavy model on GPU at a time)
    def _free(self, *names):
        for n in names:
            setattr(self, n, None)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _load_paraphraser(self):
        """utils.paraphrase.Paraphraser with its pipeline built here so it can be fp16 (its __init__ is fp32)."""
        from diffusers import AutoPipelineForImage2Image, AutoPipelineForText2Image

        from utils.paraphrase import Paraphraser

        kw = {"torch_dtype": torch.float16, "variant": "fp16"} if self.paraphrase_fp16 else {}
        t2i = AutoPipelineForText2Image.from_pretrained(self.cfg.paraphrasing_model, use_safetensors=True, **kw)
        para = Paraphraser.__new__(Paraphraser)
        para.model_id, para.device = self.cfg.paraphrasing_model, self.device
        para.pipeline_text2image = t2i.to(self.device)
        para.pipeline = AutoPipelineForImage2Image.from_pipe(para.pipeline_text2image).to(self.device)
        para.pipeline.set_progress_bar_config(disable=True)
        return para

    def _load_watermarker(self, final_json: Path):
        from utils.watermark import Watermarker

        return Watermarker(str(final_json))

    def _load_saliency(self):
        from utils.split import SaliencyDetector

        return SaliencyDetector()

    # ------------------------------------------------------------------ stages 1-2: paraphrase, XRAI -> patches
    def prepare(self, items: list) -> None:
        """Stage-major paraphrase then XRAI for [(sample, image)], so SDXL is loaded once and freed before SD-2."""
        todo = [(s, img) for s, img in items if not self.paths(s.id)["final"].exists()]
        if not todo:
            return
        with peccavi_env(self.device.type):
            for s, img in todo:
                p = self.paths(s.id)
                p["root"].mkdir(parents=True, exist_ok=True)
                if not p["input"].exists():
                    img.convert("RGB").save(p["input"])
                if len(list(p["para_dir"].glob("*_gen_*.png"))) >= self.num_paraphrases:
                    continue
                if self._para is None:
                    self._free("_wmk")
                    t = time.time()
                    self._para = self._load_paraphraser()
                    log.info("peccavi: paraphraser loaded in %.1fs", time.time() - t)
                torch.manual_seed(stable_seed("peccavi_paraphrase", s.id))
                t = time.time()
                self._para.paraphrase_images(str(p["input"]), str(p["para_dir"].parent), self.num_paraphrases,
                                             caption=[s.source_caption] * self.num_paraphrases)
                self._record(s.id, paraphrase_s=time.time() - t)
                n = len(list(p["para_dir"].glob("*_gen_*.png")))
                if n < self.num_paraphrases:
                    raise RuntimeError(f"peccavi: {s.id}: {n}/{self.num_paraphrases} paraphrases written")
            self._free("_para")
            for s, _ in todo:
                p = self.paths(s.id)
                if p["final"].exists() or s.id in self._xrai_jobs:
                    continue
                if self.xrai_workers > 1:  # CPU workers; embed() collects the result, so XRAI overlaps the GPU
                    self._submit_xrai(s.id)
                    continue
                self._sal = self._sal or self._load_saliency()
                t = time.time()
                self._sal.XRAI(str(p["para_dir"]))  # writes per-paraphrase boxes and final.json (top-4 patches)
                self._record(s.id, xrai_cpu_s=time.time() - t)

    # ------------------------------------------------------------------ parallel XRAI (opt-in: xrai_workers > 1)
    def _submit_xrai(self, sid: str) -> None:
        """One worker job per paraphrase, exactly the files split.XRAI() would loop over."""
        folder = self.paths(sid)["para_dir"].resolve()
        files = sorted(f for f in os.listdir(folder) if f.lower().endswith((".png", ".jpg", ".jpeg"))
                       and "Heatmap" not in f and "Top" not in f)
        if self._xrai_pool is None:
            import multiprocessing as mp
            from concurrent.futures import ProcessPoolExecutor

            self._xrai_pool = ProcessPoolExecutor(self.xrai_workers, mp_context=mp.get_context("spawn"),
                                                  initializer=_xrai_worker_init, initargs=(self.xrai_threads,))
            log.info("peccavi: XRAI on %d CPU workers x %d threads", self.xrai_workers, self.xrai_threads)
        self._xrai_jobs[sid] = [self._xrai_pool.submit(_xrai_one, str(folder), f) for f in files]

    def _finish_xrai(self, sid: str) -> None:
        """Wait for an image's XRAI jobs, then the official cross-paraphrase scoring (writes final.json)."""
        jobs = self._xrai_jobs.pop(sid, None)
        if jobs is None:
            return
        t = time.time()
        cpu = sum(j.result() for j in jobs)
        with peccavi_env(self.device.type):
            self._sal = self._sal or self._load_saliency()
            self._sal.process_and_score_boxes(base_path=str(self.paths(sid)["para_dir"].resolve()))
        self._record(sid, xrai_cpu_s=cpu, xrai_wait_s=time.time() - t)  # cpu: summed over the paraphrase jobs
        if not self._xrai_jobs:
            self._xrai_pool.shutdown()
            self._xrai_pool = None

    # ------------------------------------------------------------------ stage 3: embed
    def _watermarker(self, final_json: Path):
        if self._wmk is None:
            self._free("_para")
            t = time.time()
            self._wmk = self._load_watermarker(final_json)
            log.info("peccavi: watermark pipeline loaded in %.1fs", time.time() - t)
        return self._wmk

    def embed(self, img: Image.Image, bits, sample=None) -> Image.Image:
        if sample is None:
            raise ValueError("PECCAVI needs the sample (id, source caption): run it through the pipeline")
        p = self.paths(sample.id)
        self._finish_xrai(sample.id)
        if not p["final"].exists():
            self.prepare([(sample, img)])
            self._finish_xrai(sample.id)
        if not (p["wm_img"].exists() and p["state"].exists()):
            with peccavi_env(self.device.type):
                w = self._watermarker(p["final"])
                w.bbox_json_path = str(p["final"])
                w.patches = w.extract_patches()
                w.wm_pipes, w.patch_meta = {}, {}
                w.watermarked_path = str(p["wm_dir"])
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t = time.time()
                out = w.forward(str(p["input"]))
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                self._record(sample.id, embed_s=time.time() - t)
                assert Path(out) == p["wm_img"], (out, p["wm_img"])
            state = json.loads(p["state"].read_text())
            unplaced = [q["coords"] for q in state["patches"] if not q["placed"]]
            if unplaced:  # the official loop skips an iteration on RuntimeError; flag patches that never landed
                log.warning("peccavi: %s: patches %s were not written into the image", sample.id, unplaced)
        return Image.open(p["wm_img"]).convert("RGB")

    def decode(self, img):
        raise NotImplementedError("PECCAVI is zero-bit: use score()")

    # ------------------------------------------------------------------ detection
    def detect_patches(self, img: Image.Image, sid: str) -> list[dict]:
        """Watermarker.detect() on `img` with the keys stored for image `sid`; one dict per stored patch."""
        p = self.paths(sid)
        if not p["state"].exists():
            raise FileNotFoundError(f"no PECCAVI state for {sid}: embed it first ({p['state']})")
        tmp = p["root"] / "detect_input.png"  # detect() reads from disk (cv2), PNG keeps it lossless
        img.convert("RGB").save(tmp)
        with peccavi_env(self.device.type):
            w = self._watermarker(p["final"])
            w.load_state(str(p["state"]))
            seen = self._record_ring_inputs(w) if self.save_internals else None
            w.detect(str(tmp))
            if seen is not None:
                self._save_internals(img, sid, w, seen)
            return list(w.last_detection)

    @staticmethod
    def image_key(img: Image.Image) -> str:
        """Content hash that names an image's internals file (the pipeline's row keys never reach score())."""
        return hashlib.sha1(np.asarray(img.convert("RGB")).tobytes()).hexdigest()[:20]

    def internals_path(self, img: Image.Image, sid: str) -> Path:
        return self.paths(sid)["root"] / "internals" / f"{self.image_key(img)}.npz"

    @staticmethod
    def _record_ring_inputs(w) -> dict:
        """Wrap each patch key's official tree_ring_p_value so the inverted patch latent it tests is kept. The
        wrapper calls the original on the same tensor and returns its value, so detection is unchanged."""
        seen = {}
        for idx, gt in w.wm_pipes.items():
            def record(latents, _orig=gt.tree_ring_p_value, _idx=idx):
                seen[_idx] = latents.detach().float().cpu().numpy()
                return _orig(latents)
            gt.tree_ring_p_value = record
        return seen

    def _save_internals(self, img: Image.Image, sid: str, w, seen: dict) -> None:
        """<cache>/<id>/internals/<image_key>.npz: per patch the DDIM-inverted latent the ring test saw, the key
        (gt_patch, complex) and ring mask it compared against, and the p-value; enough to recompute the
        chi-squared statistic, ring correlation or per-frequency energy offline."""
        arrays = {}
        for d in w.last_detection:
            i = d["patch_idx"]
            gt = w.wm_pipes[i]
            arrays.update({f"latents{i}": seen[i], f"p{i}": np.float64(d["p_value"]),
                           f"coords{i}": np.asarray(d["coords"]), f"channel{i}": np.int64(d["channel"]),
                           f"key{i}": gt.gt_patch.detach().cpu().numpy(),
                           f"mask{i}": gt.watermarking_mask.detach().cpu().numpy()})
        out = self.internals_path(img, sid)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, **arrays)

    def score(self, img, bits, target_fpr, sample=None) -> dict:
        if sample is None:
            raise ValueError("PECCAVI needs the sample id to find the stored patches and keys")
        t = time.time()
        res = self.detect_patches(img, sample.id)
        ps = np.array([r["p_value"] for r in res], dtype=float)
        k = len(ps)
        p_img = float(min(1.0, k * ps.min())) if k else 1.0
        out = {"bit_acc": np.nan, "matches": np.nan, "p_value": p_img, "detected": bool(p_img <= target_fpr),
               "threshold_matches": np.nan, "exact_payload": np.nan,
               "wdp": float(max(r["wdp"] for r in res)) if k else 0.0, "p_min": float(ps.min()) if k else 1.0,
               "n_patches": k, "score_s": round(time.time() - t, 2)}
        for r in res:
            out[f"p_patch{r['patch_idx']}"] = r["p_value"]
            out[f"wdp_patch{r['patch_idx']}"] = r["wdp"]
        return out


class PECCAVIRaw(PECCAVI):
    """Ablation, not the official method: L0 is the optimised image *before* adaptive enhancement
    (Image_<id>_wm_100.png, written by the same Watermarker.forward() call as the enhanced image). Same patches,
    keys, detector and cache as `peccavi`, so it costs no extra paraphrasing, XRAI or embedding. Its image quality
    (PSNR ~29 dB, SSIM ~0.96 on the smoke images) is close to what the paper reports for PECCAVI (29.6-29.9 dB,
    0.93), while the enhanced image is at ~38.6 dB / 0.99."""
    name = "peccavi_raw"

    def embed(self, img: Image.Image, bits, sample=None) -> Image.Image:
        super().embed(img, bits, sample)  # makes sure the official outputs exist (cached after the first call)
        return Image.open(self.paths(sample.id)["raw_img"]).convert("RGB")
