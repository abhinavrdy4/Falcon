"""CPU tests for the analysis-only additions to the PECCAVI run: detector internals (save_internals) and
scripts/peccavi_extras.py (control transforms, internals backfill). Same stubs as test_peccavi.py."""
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))

from peccavi_stubs import textured_image  # noqa: E402
from test_peccavi import W, IdentityEditor, P, pec, sample  # noqa: E402,F401  (fixtures)
from wmedit.data import EditStep  # noqa: E402

RESEARCH = Path(__file__).resolve().parents[1]


def ring_p_value(latents, key, mask):
    """tree_ring_p_value (ZoDiac/main/wmpatch.py) recomputed from a saved internals file."""
    target = key[mask].ravel()
    target = np.concatenate([target.real, target.imag])
    f = np.fft.fftshift(np.fft.fft2(latents), axes=(-1, -2))[mask].ravel()
    f = np.concatenate([f.real, f.imag])
    sigma = f.std(ddof=1)  # torch.std is the sample std
    return stats.ncx2.cdf(x=(((f - target) / sigma) ** 2).sum(), df=len(target), nc=(target ** 2 / sigma ** 2).sum())


def test_save_internals_leaves_detection_unchanged(pec):
    s, img = sample(), textured_image(3)
    out = pec.embed(img, np.zeros(0, dtype=np.uint8), sample=s)
    drop = lambda d: {k: v for k, v in d.items() if k != "score_s"}  # noqa: E731
    before = {k: drop(pec.score(x, None, 1e-3, sample=s)) for k, x in (("wm", out), ("clean", img))}
    assert not (pec.paths(s.id)["root"] / "internals").exists()

    pec.save_internals = True
    for k, x in (("wm", out), ("clean", img)):
        after = drop(pec.score(x, None, 1e-3, sample=s))
        assert after == before[k]  # same p-values, decision and WDP
        z = np.load(pec.internals_path(x, s.id))
        for i in range(after["n_patches"]):
            assert float(z[f"p{i}"]) == after[f"p_patch{i}"]
            assert z[f"latents{i}"].shape == z[f"mask{i}"].shape and z[f"key{i}"].dtype.kind == "c"
            assert ring_p_value(z[f"latents{i}"], z[f"key{i}"], z[f"mask{i}"]) == pytest.approx(
                after[f"p_patch{i}"], rel=1e-4, abs=1e-12)
    assert pec.image_key(out) != pec.image_key(img)
    assert pec.image_key(Image.fromarray(np.asarray(out))) == pec.image_key(out)  # keyed by pixels, not object


class FakeVAE:
    class _D:
        def __init__(self, x):
            self.latent_dist = types.SimpleNamespace(mode=lambda: x)
            self.sample = x

    def encode(self, x):
        return self._D(x)

    def decode(self, z):
        return self._D(z)


class ControlEditor(IdentityEditor):
    """IdentityEditor plus what peccavi_extras.py touches: the VAE, device and dtype."""
    device, dtype = torch.device("cpu"), torch.float32
    pipe = types.SimpleNamespace(vae=FakeVAE())

    def __init__(self):
        self.steps = []

    def edit(self, img, step, seed, **kw):
        self.steps.append((step.instruction, seed))
        return super().edit(img, step, seed, **kw)


def load_extras():
    spec = importlib.util.spec_from_file_location("peccavi_extras", RESEARCH / "scripts" / "peccavi_extras.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def stub_run(tmp_path, monkeypatch, internals_during_evaluate=True):
    """peccavi_full90.yaml on 2 stub images, 2 rounds (round 2 mirrors the image); returns exp, samples, editor."""
    from wmedit import pipeline as PL

    samples = [sample("000000000001"), sample("000000000002")]
    for s in samples:
        s.plan = [EditStep(r, "x", f"edit {r}", "a", "b") for r in (1, 2)]
    images = {s.id: textured_image(i) for i, s in enumerate(samples)}
    monkeypatch.setattr(PL, "load_pie_bench", lambda *a, **k: (
        list(samples), dict(images), {k: np.zeros((512, 512), np.uint8) for k in images}))
    editor = ControlEditor()
    monkeypatch.setattr("wmedit.editors.build_editor", lambda name, **kw: editor)
    monkeypatch.setattr(P, "get_device", lambda: torch.device("cpu"))
    cfg = PL.load_config(RESEARCH / "configs" / "peccavi_full90.yaml", [f"output_dir={tmp_path}", "dataset.rounds=2"])
    assert cfg["name"] == "peccavi_smoke" and cfg["dataset"]["max_images"] == 90
    exp = PL.Experiment(cfg)
    exp.prepare()
    for t in ("peccavi", "peccavi_raw"):  # optionally score without internals first, to test the backfill
        exp.watermark(t).save_internals = internals_during_evaluate
    exp.embed()
    exp.edit()
    exp.evaluate(image_metrics=False)
    for t in ("peccavi", "peccavi_raw"):
        exp.watermark(t).save_internals = True
    return exp, samples, editor


def test_extras_controls_and_internals_backfill(pec, W, tmp_path, monkeypatch):
    exp, samples, editor = stub_run(tmp_path, monkeypatch, internals_during_evaluate=False)
    rows = pd.read_csv(exp.res_dir / "rows.csv")

    X = load_extras()
    X.run_controls(exp)
    ctrl = pd.read_csv(exp.res_dir / "peccavi_controls.csv", dtype={"id": str})
    assert len(ctrl) == 2 * 2 * 2  # images x scored tracks (peccavi, peccavi_raw) x controls
    assert set(ctrl.track) == {"peccavi", "peccavi_raw"} and set(ctrl.control) == set(X.CONTROLS)
    for c in X.CONTROLS:
        for t in X.TRACKS:
            assert all(X.control_path(exp, c, t, s.id).exists() for s in samples)
    # identity control: "keep the image unchanged" with the round-1 seed, the same for every track of an image
    ident = [st for st in editor.steps if st[0] == "keep the image unchanged"]
    assert len(ident) == 2 * 3 and len({seed for _, seed in ident}) == 2
    v = ctrl[(ctrl.control == "vae") & (ctrl.track == "peccavi")]
    assert v.detected.all()  # the stub VAE is lossless, so the watermark must survive it

    X.run_internals(exp)
    det = exp.watermark("peccavi")
    for s in samples:
        for t in X.TRACKS:
            for r in range(3):
                img = Image.open(exp.level_path("baseline", "ip2p", t, s.id, r)).convert("RGB")
                assert det.internals_path(img, s.id).exists()
    # the pipeline's own outputs are untouched
    assert rows.equals(pd.read_csv(exp.res_dir / "rows.csv"))
    n = len(list((exp.root / "wm_state" / "peccavi").glob("*/internals/*.npz")))
    X.run_controls(exp), X.run_internals(exp)  # resume: nothing redone
    assert len(pd.read_csv(exp.res_dir / "peccavi_controls.csv")) == len(ctrl)
    assert len(list((exp.root / "wm_state" / "peccavi").glob("*/internals/*.npz"))) == n


def test_extras_internals_needs_flag(pec, tmp_path, monkeypatch):
    from wmedit import pipeline as PL

    monkeypatch.setattr(PL, "load_pie_bench", lambda *a, **k: ([sample()], {sample().id: textured_image(1)},
                                                                {sample().id: np.zeros((512, 512), np.uint8)}))
    monkeypatch.setattr(P, "get_device", lambda: torch.device("cpu"))
    exp = PL.Experiment(PL.load_config(RESEARCH / "configs" / "peccavi_smoke.yaml", [f"output_dir={tmp_path}"]))
    exp.prepare()
    with pytest.raises(SystemExit):
        load_extras().run_internals(exp)


def test_patch_analysis_on_stub_run(pec, W, tmp_path, monkeypatch):
    exp, samples, _ = stub_run(tmp_path, monkeypatch)
    X = load_extras()
    X.run_controls(exp)
    spec = importlib.util.spec_from_file_location("pa", RESEARCH / "scripts" / "peccavi_patch_analysis.py")
    A = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(A)
    out = tmp_path / "patches"
    masks = {s.id: np.zeros((512, 512), bool) for s in samples}
    masks[samples[0].id][:, :256] = True  # left half is "the object" for the first image
    df = A.analyze([exp.root], masks, 1e-3, out)

    n_p = df.groupby("id").patch.nunique()
    assert len(df) == sum(n_p[s.id] * 3 * 3 for s in samples)  # tracks x levels L0..L2 x patches
    cols = {"nmp_score", "nmp_hits", "object_frac", "detected", "ring_chi2", "ring_corr", "change_prev_10",
            "change_l0_10", "ssim_prev", "intended_frac", "res_corr", "res_energy", "lost", "lost_outside_unchanged"}
    assert cols <= set(df.columns)
    l0 = df[df.level == 0]
    assert l0[l0.track == "peccavi"].detected.all() and not l0[l0.track == "none"].detected.any()
    assert l0.ring_chi2.notna().all()  # internals were saved during evaluate
    l1 = df[(df.level == 1) & (df.track == "peccavi")]  # stub round 1 = identity
    assert (l1.change_prev_10 == 0).all() and np.allclose(l1.res_corr, 1)
    l2 = df[(df.level == 2) & (df.track == "peccavi")]  # stub round 2 = mirror: pixels change
    assert (l2.change_prev_10 > 0).any()
    first = df[(df.id == samples[0].id) & (df.level == 1)]  # round category "x": intended = PIE mask share
    assert np.allclose(first.intended_frac, first.object_frac)
    assert ((df.lost & df.lost_outside_unchanged) <= df.lost).all()

    c = pd.read_csv(out / "patch_controls.csv", dtype={"id": str})
    assert len(c) == sum(n_p[s.id] * 2 * 2 for s in samples)  # tracks peccavi/raw x controls x patches
    assert c[c.control == "vae"].detected.all()  # stub VAE is lossless
    assert (out / "summary.md").read_text().startswith("# PECCAVI per-patch analysis")
    assert all((out / "patch_grids" / f"{s.id}.png").exists() for s in samples)
    assert (out / "fig_survival_vs_change.png").exists()


# ---------------------------------------------------------------- parallel XRAI (xrai_workers > 1)
def test_one_file_os_proxy():
    import os

    o = P._OneFile("a_gen_3.png")
    assert o.listdir("/anything") == ["a_gen_3.png"]
    assert o.path is os.path and o.makedirs is os.makedirs
    from wmedit.utils import available_cpus

    assert available_cpus() >= 1


def test_parallel_xrai_matches_sequential(pec, tmp_path, monkeypatch):
    """Orchestration: one job per paraphrase, embed() waits for its image, then the official scoring writes the
    same final.json as the sequential loop. Workers are threads running a stand-in for _xrai_one (the real one is
    checked against split.XRAI on real images separately; it needs InceptionV3)."""
    import json
    import shutil
    from concurrent.futures import ThreadPoolExecutor

    from test_peccavi import FakeSaliency

    jobs = []

    def fake_one(folder, fname):
        jobs.append(fname)
        d = Path(folder) / Path(fname).stem
        d.mkdir(exist_ok=True)
        (d / "XRAI_Analysis_data.json").write_text(json.dumps(
            {"grid_centers": [], "merged_boxes": [{"x1": 192, "y1": 128, "x2": 320, "y2": 256}]}))
        return 1.0

    def scorer(self):
        with P.peccavi_env("cpu"):
            from utils.split import SaliencyDetector
        return SaliencyDetector.__new__(SaliencyDetector)  # official process_and_score_boxes, no model

    monkeypatch.setattr(P, "_xrai_one", fake_one)
    monkeypatch.setattr(P.PECCAVI, "_load_saliency", scorer)
    pec.xrai_workers, pec._xrai_pool = 2, ThreadPoolExecutor(2)
    s, img = sample(), textured_image(3)
    pec.prepare([(s, img)])
    p = pec.paths(s.id)
    assert s.id in pec._xrai_jobs and not p["final"].exists()  # submitted, not waited for
    out = pec.embed(img, np.zeros(0, dtype=np.uint8), sample=s)
    assert sorted(jobs) == sorted(f.name for f in p["para_dir"].glob("*_gen_*.png")) and len(jobs) == 5
    assert p["final"].exists() and pec._xrai_pool is None and not pec._xrai_jobs
    t = json.loads(p["timings"].read_text())
    assert t["xrai_cpu_s"] == 5.0 and "xrai_wait_s" in t
    assert pec.score(out, None, 1e-3, sample=s)["detected"]

    seq = tmp_path / "seq" / p["para_dir"].name  # the same paraphrases through the sequential official loop
    seq.mkdir(parents=True)
    for f in p["para_dir"].glob("*_gen_*.png"):
        shutil.copy(f, seq / f.name)
    with P.peccavi_env("cpu"):
        FakeSaliency().XRAI(str(seq))
    assert json.loads((seq / "final.json").read_text()) == json.loads(p["final"].read_text())


def test_xrai_workers_auto(monkeypatch):
    monkeypatch.setattr(P, "available_cpus", lambda: 16)
    wm = P.PECCAVI(device="cpu", xrai_workers="auto", xrai_threads=2)
    assert wm.xrai_workers == 7  # (16 - 2) // 2
    assert P.PECCAVI(device="cpu").xrai_workers == 0  # default: official sequential loop
