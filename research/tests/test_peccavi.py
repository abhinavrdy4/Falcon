"""CPU tests for the PECCAVI integration: the vendored fixes and the wmedit wrapper, with stub models.

Diffusion models, SDXL, XRAI's InceptionV3 and the Watson-VGG loss are replaced by the stand-ins in
peccavi_stubs.py / below; everything else (GTWatermark, Watermarker control flow, split.py patch selection,
state files, detection statistics, pipeline wiring) is the real code.
"""
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from peccavi_stubs import FakeImg2Img, FakeScheduler, FakeSDPipe, textured_image  # noqa: E402
from wmedit import peccavi as P  # noqa: E402
from wmedit.data import EditStep, Sample  # noqa: E402

EPOCHS = 3  # official: 100; the stub "UNet" makes more iterations pointless here


class MSELoss(torch.nn.Module):  # stands in for LossProvider (Watson-VGG needs a 528 MB VGG download)
    def __init__(self, *a, **kw):
        super().__init__()

    def __call__(self, pred, gt, *a):
        return torch.nn.functional.mse_loss(pred, gt)


@pytest.fixture
def W(monkeypatch):
    """The vendored utils.watermark module with model loading stubbed out."""
    with P.peccavi_env("cpu"):
        import wandb

        wandb.init(mode="disabled")
        import utils.watermark as W
    monkeypatch.setattr(W, "WMDetectStableDiffusionPipeline", FakeSDPipe)
    monkeypatch.setattr(W, "DDIMScheduler", FakeScheduler)
    monkeypatch.setattr(W, "LossProvider", MSELoss)
    return W


def write_final(path: Path, patches):
    path.write_text(json.dumps([{"patch": list(p), "score": 1.0} for p in patches]))
    return path


def embed_one(W, tmp_path, patches=((3, 2), (4, 4), (0, 7), (7, 0)), image_id="42"):
    fj = write_final(tmp_path / "final.json", patches)
    ip = tmp_path / f"Image_{image_id}.png"
    textured_image(int(image_id) % 7).save(ip)
    with P.peccavi_env("cpu"):
        w = W.Watermarker(str(fj))
        w.watermarked_path, w.epochs = str(tmp_path / "wm"), EPOCHS
        out = w.forward(str(ip))
    return w, Path(out), ip


def bonferroni(res):
    return min(1.0, len(res) * min(r["p_value"] for r in res))


# ---------------------------------------------------------------- vendored fixes
def test_fix1_patch_box_and_nonempty_ring(W, tmp_path):
    w, _, _ = embed_one(W, tmp_path)
    assert w.patch_latent_box() == [4, 4, 8, 8]  # 64 px patch -> 8x8 latent, whole-latent box
    for i, wm_pipe in w.wm_pipes.items():
        m = wm_pipe.watermarking_mask
        assert m.shape == (1, 4, 8, 8)
        assert int(m[:, i % 4].sum()) == 47 and int(m.sum()) == 47  # ring only in the patch's channel


def test_fix2_seeded_keys_and_state_roundtrip(W, tmp_path):
    w, out, ip = embed_one(W, tmp_path)
    state = json.loads((out.parent / "state.json").read_text())
    assert [p["coords"] for p in state["patches"]] == [[3, 2], [4, 4], [0, 7], [7, 0]]
    assert [p["channel"] for p in state["patches"]] == [0, 1, 2, 3]
    assert all(p["placed"] and p["mask_size"] == 47 for p in state["patches"])
    assert state["patches"][0]["seed"] == W.Watermarker.key_seed("42", 0)
    with P.peccavi_env("cpu"):
        fresh = W.Watermarker(str(tmp_path / "final.json"))  # "new process": nothing in memory
        fresh.load_state(str(out.parent / "state.json"))
        for i in w.wm_pipes:
            assert torch.equal(fresh.wm_pipes[i].gt_patch, w.wm_pipes[i].gt_patch)
            assert torch.equal(fresh.wm_pipes[i].watermarking_mask, w.wm_pipes[i].watermarking_mask)
        fresh.detect(str(out))
        p_wm = bonferroni(fresh.last_detection)
        fresh.detect(str(ip))
        p_clean = bonferroni(fresh.last_detection)
    assert p_wm < 1e-6 < 1e-3 < p_clean


def test_fix2_keys_differ_across_images_and_patches(W):
    seeds = {W.Watermarker.key_seed(i, k) for i in ("1", "2") for k in range(4)}
    assert len(seeds) == 8


def test_fix4_detector_reads_rgb(W, tmp_path, monkeypatch):
    w, out, _ = embed_one(W, tmp_path, patches=((3, 2),))
    seen = []
    monkeypatch.setattr(W, "watermark_prob", lambda patch, *a, **k: seen.append(patch.clone()) or 0.5)
    with P.peccavi_env("cpu"):
        w.detect(str(out))
    rgb = torch.from_numpy(np.asarray(Image.open(out).convert("RGB"), dtype=np.float32) / 255).permute(2, 0, 1)
    assert torch.allclose(seen[0][0], rgb[:, 128:192, 192:256], atol=1e-6)  # patch [col 3, row 2]


def test_fix3_fix5_config(monkeypatch):
    with P.peccavi_env("cpu"):
        from utils.config import ConfigLoader

        c = ConfigLoader("config.yaml")
        assert c.device == "cpu"  # PECCAVI_DEVICE override
        assert c.model_id == "sd2-community/stable-diffusion-2-base"
        assert "xl" in c.paraphrasing_model
        monkeypatch.delenv("PECCAVI_DEVICE")
        assert ConfigLoader("config.yaml").device == "cuda"
    from wmedit.peccavi import PECCAVI_ROOT

    from ZoDiac.main.wmdiffusion import WMDetectStableDiffusionPipeline  # noqa: F401  (SD-1/2 class)
    assert (PECCAVI_ROOT / "FIXES.md").exists()


def test_method_constants_unchanged():
    with P.peccavi_env("cpu"):
        from utils.config import ConfigLoader

        c = ConfigLoader("config.yaml")
    assert (c.iters, c.num_inference_steps, c.learning_rate, c.milestones) == (100, 50, 0.25, [50, 80])
    assert c.loss_weights == [10.0, 0.1, 0.5, 0.0] and c.strengths == [1, 0.75, 0.5, 0.25, 0.1]
    assert c.ssim_threshold == 0.99


def run_split(folder):
    with P.peccavi_env("cpu"):
        from utils.split import SaliencyDetector

        SaliencyDetector.__new__(SaliencyDetector).process_and_score_boxes(str(folder))
    return json.loads((folder / "final.json").read_text())


def test_split_keeps_box_seen_in_two_paraphrases(tmp_path):
    folder = fake_xrai_outputs(tmp_path / "123", n=2)  # n=2 -> score 0.75 >= threshold 0.25
    assert [e["patch"] for e in run_split(folder)] == [[3, 2]]  # box (192,128)-(256,192) = cell col 3, row 2


def test_split_drops_box_seen_in_all_paraphrases(tmp_path):
    """Pins upstream behaviour (not changed, see FIXES.md): score = [1,.75,.5,.25,.1][n-1] for a box seen in n
    paraphrases, then boxes with score < 0.25 are dropped, so a box found in all 5 is discarded and, with nothing
    left, the default box [256,256,512,512] (bottom-right quadrant) is used."""
    entries = run_split(fake_xrai_outputs(tmp_path / "123", n=5))
    assert len(entries) == 4 and all(e["score"] == 1.0 for e in entries)
    assert all(4 <= c <= 7 and 4 <= r <= 7 for c, r in (e["patch"] for e in entries))
    assert P.PECCAVI.nmp_fallback(entries) == "quadrant"


# ---------------------------------------------------------------- wrapper
def fake_xrai_outputs(folder: Path, n=5, box=(192, 128, 256, 192)):
    """What split.SaliencyDetector.XRAI() leaves per paraphrase: <stem>/XRAI_Analysis_data.json."""
    folder.mkdir(parents=True, exist_ok=True)
    for k in range(n):
        d = folder / f"123_gen_{k}"
        d.mkdir(exist_ok=True)
        x1, y1, x2, y2 = box
        (d / "XRAI_Analysis_data.json").write_text(json.dumps(
            {"grid_centers": [], "merged_boxes": [{"x1": x1, "y1": y1, "x2": x2, "y2": y2}]}))
    return folder


class FakeSaliency:
    calls = 0

    def XRAI(self, folder):
        FakeSaliency.calls += 1
        folder = Path(folder)
        for k, png in enumerate(sorted(folder.glob("*_gen_*.png"))):
            d = folder / png.stem
            d.mkdir(exist_ok=True)
            (d / "XRAI_Analysis_data.json").write_text(json.dumps(
                {"grid_centers": [], "merged_boxes": [{"x1": 192, "y1": 128, "x2": 320, "y2": 256}]}))
        from utils.split import SaliencyDetector

        SaliencyDetector.__new__(SaliencyDetector).process_and_score_boxes(str(folder))


@pytest.fixture
def pec(W, tmp_path, monkeypatch):
    def paraphraser(self):
        from utils.paraphrase import Paraphraser

        para = Paraphraser.__new__(Paraphraser)
        para.pipeline = FakeImg2Img()
        return para

    def watermarker(self, final_json):
        w = W.Watermarker(str(final_json))
        w.epochs = EPOCHS
        return w

    monkeypatch.setattr(P.PECCAVI, "_load_paraphraser", paraphraser)
    monkeypatch.setattr(P.PECCAVI, "_load_saliency", lambda self: FakeSaliency())
    monkeypatch.setattr(P.PECCAVI, "_load_watermarker", watermarker)
    wm = P.PECCAVI(device="cpu")
    wm.cache_dir = tmp_path / "wm_state" / "peccavi"
    return wm


def sample(sid="000000000123", caption="a cat sitting on a bench"):
    return Sample(sid, "1_change_object_80", "change_object", caption,
                  [EditStep(1, "change_object", "make it a dog", caption, "a dog")])


def test_wrapper_embed_score_and_cache(pec):
    s, img = sample(), textured_image(3)
    out = pec.embed(img, np.zeros(0, dtype=np.uint8), sample=s)
    p = pec.paths(s.id)
    assert out.size == (512, 512) and p["state"].exists()
    assert len(list(p["para_dir"].glob("*_gen_*.png"))) == 5
    info = pec.image_info(s.id)
    assert {"time_paraphrase_s", "time_xrai_cpu_s", "time_embed_s"} <= set(info)
    assert info["nmp_paraphrases"] == 5 and info["nmp_fallback"] == "quadrant"  # FakeSaliency: same box in all 5

    sc_wm = pec.score(out, np.zeros(0), 1e-3, sample=s)
    sc_clean = pec.score(img, np.zeros(0), 1e-3, sample=s)
    assert sc_wm["detected"] and not sc_clean["detected"]
    k = sc_wm["n_patches"]
    assert 1 <= k <= 4 and sc_wm["p_value"] == pytest.approx(min(1, k * sc_wm["p_min"]))
    assert sc_wm["wdp"] == pytest.approx(max(sc_wm[f"wdp_patch{i}"] for i in range(k)))
    assert np.isnan(sc_wm["bit_acc"])  # zero-bit, as Tree-Ring

    # resume: nothing recomputed
    calls, t = FakeSaliency.calls, pec.image_info(s.id)
    pec.prepare([(s, img)])
    again = pec.embed(img, np.zeros(0), sample=s)
    assert FakeSaliency.calls == calls and pec.image_info(s.id) == t
    assert np.array_equal(np.asarray(again), np.asarray(out))


def test_wrapper_needs_sample(pec):
    with pytest.raises(ValueError):
        pec.embed(textured_image(1), np.zeros(0))
    with pytest.raises(FileNotFoundError):
        pec.score(textured_image(1), np.zeros(0), 1e-3, sample=sample("999"))


def test_file_id_is_numeric():
    assert P.PECCAVI.file_id("000000000123") == "000000000123"
    assert P.PECCAVI.file_id("abc").isdigit()


# ---------------------------------------------------------------- pipeline wiring
class IdentityEditor:
    name = "ip2p"

    def edit(self, img, step, seed, **_):
        return img.transpose(Image.FLIP_LEFT_RIGHT) if step.round == 2 else img


def test_pipeline_runs_peccavi_with_control_track(pec, tmp_path, monkeypatch):
    from wmedit import pipeline as PL

    samples = [sample("000000000001"), sample("000000000002"), sample("000000000003")]
    for s in samples:
        s.plan = [EditStep(r, "x", f"edit {r}", "a", "b") for r in (1, 2)]
    images = {s.id: textured_image(i) for i, s in enumerate(samples)}
    monkeypatch.setattr(PL, "load_pie_bench", lambda *a, **k: (
        list(samples), dict(images), {k: np.zeros((512, 512), np.uint8) for k in images}))
    monkeypatch.setattr("wmedit.editors.build_editor", lambda name, **kw: IdentityEditor())
    cfg = PL.load_config(Path(__file__).resolve().parents[1] / "configs" / "peccavi_smoke.yaml",
                         [f"output_dir={tmp_path}", "dataset.rounds=2", "dataset.shard=[0,2]"])
    monkeypatch.setattr(P, "get_device", lambda: torch.device("cpu"))
    exp = PL.Experiment(cfg)
    exp.prepare()
    assert [s.id for s in exp.samples] == ["000000000001", "000000000003"]  # shard 0 of 2
    exp.embed()
    exp.edit()
    exp.evaluate(image_metrics=False)
    assert exp.watermark("peccavi").cache_dir == exp.root / "wm_state" / "peccavi"
    df = __import__("pandas").read_csv(exp.res_dir / "rows.csv")
    assert len(df) == 2 * 2 * 3  # tracks x images x levels L0..L2
    wm, clean = df[df.watermark == "peccavi"], df[df.watermark == "none"]
    assert wm[wm.level == 0].detected.all() and not clean.peccavi_detected.any()
    assert {"p_patch0", "wdp_patch0", "wdp", "wm_psnr", "wm_time_embed_s", "wm_nmp_fallback"} <= set(df.columns)
    assert {"peccavi_p_value", "peccavi_wdp", "peccavi_p_patch0"} <= set(df.columns)


def test_pipeline_raw_track_reuses_official_embedding(pec, W, tmp_path, monkeypatch):
    """Ablation track peccavi_raw: L0 = the pre-enhancement image, from the same (single) embedding per image."""
    from wmedit import pipeline as PL

    samples = [sample("000000000001"), sample("000000000002")]
    for s in samples:
        s.plan = [EditStep(r, "x", f"edit {r}", "a", "b") for r in (1, 2)]
    images = {s.id: textured_image(i) for i, s in enumerate(samples)}
    monkeypatch.setattr(PL, "load_pie_bench", lambda *a, **k: (
        list(samples), dict(images), {k: np.zeros((512, 512), np.uint8) for k in images}))
    monkeypatch.setattr("wmedit.editors.build_editor", lambda name, **kw: IdentityEditor())
    monkeypatch.setattr(P, "get_device", lambda: torch.device("cpu"))
    forward, calls = W.Watermarker.forward, []
    monkeypatch.setattr(W.Watermarker, "forward", lambda self, *a: calls.append(a) or forward(self, *a))
    cfg = PL.load_config(Path(__file__).resolve().parents[1] / "configs" / "peccavi_smoke_raw.yaml",
                         [f"output_dir={tmp_path}", "dataset.rounds=2"])
    assert cfg["name"] == "peccavi_smoke"  # same experiment folders as the official run, so its cache is reused
    exp = PL.Experiment(cfg)
    exp.prepare()
    exp.embed()
    exp.edit()
    exp.evaluate(image_metrics=False)
    assert len(calls) == len(samples)  # one embedding per image, shared by both tracks
    raw = exp.watermark("peccavi_raw")
    assert raw.cache_dir == exp.watermark("peccavi").cache_dir == exp.root / "wm_state" / "peccavi"
    s = samples[0]
    l0_raw = np.asarray(Image.open(exp.l0_path("peccavi_raw", s.id)))
    assert np.array_equal(l0_raw, np.asarray(Image.open(raw.paths(s.id)["raw_img"]).convert("RGB")))
    assert not np.array_equal(l0_raw, np.asarray(Image.open(exp.l0_path("peccavi", s.id))))
    df = __import__("pandas").read_csv(exp.res_dir / "rows.csv")
    assert len(df) == 3 * 2 * 3  # tracks x images x levels
    r = df[df.watermark == "peccavi_raw"]
    assert r[r.level == 0].detected.all() and r[r.level == 0].wm_psnr.max() < df[
        (df.watermark == "peccavi") & (df.level == 0)].wm_psnr.min()  # un-enhanced = more distorted


def test_wmdiffusion_on_current_diffusers():
    """FIX 7: the official SD pipeline subclass loads and trains on the diffusers version Falcon uses."""
    with P.peccavi_env("cpu"):
        from diffusers import DDIMScheduler

        from ZoDiac.main.wmdiffusion import WMDetectStableDiffusionPipeline

        mid = "hf-internal-testing/tiny-stable-diffusion-torch"  # ~2 MB random-weight SD
        try:
            sch = DDIMScheduler.from_pretrained(mid, subfolder="scheduler")
            pipe = WMDetectStableDiffusionPipeline.from_pretrained(mid, scheduler=sch, safety_checker=None)
        except OSError as e:  # offline
            pytest.skip(f"tiny SD checkpoint unavailable: {e}")
        pipe.set_progress_bar_config(disable=True)
        lat = pipe.get_image_latents(torch.rand(1, 3, 64, 64), sample=False)
        z = pipe.forward_diffusion(latents=lat, text_embeddings=pipe.get_text_embedding(""), guidance_scale=1.0,
                                   num_inference_steps=3).detach().clone().requires_grad_(True)  # as Watermarker
        img = pipe(prompt="", guidance_scale=1.0, num_inference_steps=3, output_type="tensor",
                   use_trainable_latents=True, init_latents=z).images
        img.mean().backward()
    assert img.shape == (1, 3, 64, 64) and z.grad.abs().sum() > 0
