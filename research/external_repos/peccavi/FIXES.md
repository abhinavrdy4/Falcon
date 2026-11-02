# PECCAVI: vendored code and every change made to it

Upstream: https://github.com/SRDdev/PECCAVI, branch `CVPR2026`, commit `a429df0` ("CVPR2026 Split").
The repository has no licence file. The first commit on this branch, "Vendor PECCAVI ... unmodified", adds the
files below byte-for-byte. Diff against it with `git diff <that commit> -- research/external_repos/peccavi`.
Every edited line is tagged `PECCAVI-FIX n` in the source.

Vendored files: `config.yaml`, `main.py`, `requirements.txt`, `Readme.md`, `utils/{config, paraphrase, split,
salient, boundingbox, watermark, noisypatch}.py`, `ZoDiac/main/{utils, wmdiffusion, wmpatch}.py`,
`ZoDiac/loss/{loss, pytorch_ssim, watson_vgg}.py`, `ZoDiac/loss/rgb_watson_vgg_trial0.pth`.
Not vendored: datasets, W&B logs, the paper PDF, ZoDiac attack/FID code, `ZoDiac/main/watermarker.py`
(imwatermark baseline), `utils/{animation, clip}.py`.

## Method: unchanged

Strengths `[1, .75, .5, .25, .1]`, 100 Adam iterations (lr 0.25, milestones 50/80), 50 DDIM steps, loss weights
`[10, .1, .5, 0]` (L2, Watson-VGG, SSIM, watermark L1), adaptive enhancement (SSIM threshold 0.99), top-4
64 px patches from `split.py`, channel = patch index mod 4, Tree-Ring non-central chi-squared p-value per
patch, WDP = max over patches. `tests/test_peccavi.py::test_method_constants_unchanged` pins these values.

## Fixes

| # | File:line (after fix) | Problem (verified on the unmodified code with stub models) | Change |
|---|---|---|---|
| 1 | `utils/watermark.py:321-345`, `:423-437` | `Watermarker.watermarking()` passed `bounding_boxes=[[col, row]]`; `GTWatermark._get_watermarking_pattern()` unpacks `x, y, w, h = bbox` → `ValueError: not enough values to unpack (expected 4, got 2)` at `ZoDiac/main/wmpatch.py:98` on the first patch. `main.py` catches it with `except Exception` and moves on, so no image was ever watermarked. | The box is now `patch_latent_box()` = `(4, 4, 8, 8)`: centre and size in the patch's own latent. Each 64 px patch is DDIM-inverted on its own, which gives an 8×8 latent. `adjust_coordinates()` would clamp any 8×8 box to the same centre. `_make_wm_pipe()` raises if the ring mask is empty. Result: 47 ring coefficients in channel `idx % 4`. |
| 2 | `utils/watermark.py:330-372`, `:433-437`, `:509` | `_gen_gt_(latent_shape)` drew the ring key from an unseeded `randn_tensor`, and nothing was saved, so `detect()` only worked in the same process right after embedding. A new process got `RuntimeError: detect() called before watermarking()`, and a new key could never match. | The key is seeded by `key_seed(image_id, patch_idx)` (sha256 → 31-bit) through a CPU `torch.Generator`, so it is the same on CPU and GPU. After embedding, `save_state()` writes `state.json` next to the watermarked image with each patch's coords, channel, seed, box, latent shape, ring-mask size, and whether it was written into the image. `load_state()` rebuilds the `GTWatermark` instances from it. |
| 3 | `config.yaml:9-13` | `model_id` was SDXL, but `WMDetectStableDiffusionPipeline` subclasses the SD-1/2 `StableDiffusionPipeline` (one text encoder, no `text_encoder_2` or added conditioning), so it cannot load SDXL. | `model_id: sd2-community/stable-diffusion-2-base`. The `stabilityai/stable-diffusion-2-base` repo returns 401 since Stability removed SD-2 from the Hub. This mirror's `512-base-ema.ckpt` sha256 (`d635794c1fedfdfa…`) and UNet match `sd-research/` and `Manojb/`. `paraphrasing_model` stays SDXL. |
| 4 | `utils/watermark.py:587-588` | `detect()` read images with `cv2.imread` (BGR), while embedding used `get_img_tensor` (PIL, RGB). Test: cv2 channel 0 equals PIL channel 2. | `cv2.cvtColor(..., COLOR_BGR2RGB)` after the optional resize. |
| 5 | `config.yaml:26-28`, `utils/config.py:13,55-58` | `device: "cuda:1"` was hard-coded and failed on single-GPU or CPU hosts with `AssertionError: Torch not compiled with CUDA enabled`. | `device: "cuda"`. Select the GPU with `CUDA_VISIBLE_DEVICES`. `ConfigLoader` honours a `PECCAVI_DEVICE` override (`cpu` in tests). |
| 6 | `utils/watermark.py:227-230` | **Found after 1–5 were fixed:** `adaptive_enhancement()` probed detection by inverting the *full* 512 px image (64×64 latent) with the first patch's key, whose mask is 8×8. This gave `IndexError: The shape of the mask [1, 4, 8, 8] at index 2 does not match ... [1, 4, 64, 64]` on every image. | Probe skipped (`det_prob = nan`). It only fed a print and a W&B log, and the enhanced image does not depend on it. Detection of the enhanced image is measured by the pipeline at L0 with `detect()`. |

| 7 | `ZoDiac/main/wmdiffusion.py:29,39-40,368,378-379` | **Found when loading with Falcon's diffusers (≥0.35; upstream pinned 0.21.4):** diffusers 0.27 inserted an `image_encoder` argument before `requires_safety_checker` in `StableDiffusionPipeline.__init__`. The positional call put a bool into `image_encoder` (`AttributeError: 'bool' object has no attribute '__module__'`). After that was fixed, the subclass signature lacked the component and failed the pipeline consistency check. | Both subclasses accept `image_encoder=None` and pass it and `requires_safety_checker` by keyword. API compatibility only; `tests/test_peccavi.py::test_wmdiffusion_on_current_diffusers` loads a tiny SD checkpoint, inverts, and backprops through the trainable call. |

## Instrumentation (no behaviour change)

- `utils/watermark.py:64, 602, 616-620`: `detect()` also stores each patch's p-value and WDP in
  `self.last_detection`. The upstream line `det_prob = 1 - watermark_prob(...)` is split into two lines with
  the same arithmetic. Needed for per-patch p-values and the Bonferroni decision.
- `utils/watermark.py:491`: marks a patch `placed` when its last epoch writes it into the image. Upstream skips
  an epoch on `RuntimeError` and continues, so a failed last epoch silently leaves a patch unwatermarked. The
  wrapper logs such patches.

## Integration (in `src/wmedit/peccavi.py`, not in vendored files)

- The vendored code reads `./config.yaml` and `./ZoDiac/loss/*.pth` from the working directory, writes debug
  plots to `./output/` and `./watermark_mask.png`, and sets the root logger to DEBUG on import. The wrapper
  runs every call inside `peccavi_env()`: `cwd` = this directory, `output/` created (git-ignored), logger level
  restored, and matplotlib figures closed. `visualize_tensor()` never closes them.
- `utils/split.py` imports `tensorflow` but never uses it. The wrapper registers an empty `tensorflow`
  module, which avoids TF/torch CUDA and protobuf clashes on Kaggle.
- `ZoDiac/loss/loss.py` calls `wandb.log` unconditionally. The wrapper calls `wandb.init(mode="disabled")`.
- `Paraphraser.__init__` loads SDXL in fp32. The wrapper builds the same `AutoPipelineForText2Image` →
  `AutoPipelineForImage2Image` pair in fp16 on CUDA and calls the unchanged `paraphrase_images()`
  (strength 0.2, cfg 7.5, same negative prompt). Before the call it seeds the global RNG from the image id so
  the paraphrases are reproducible. Captions are the PIE-Bench++ source prompt repeated 5 times; upstream
  used 5 COCO captions.
- `main.py` is not used. It imports `utils.salient`, whose output has no `{"patch": [col, row]}` entries, so
  `extract_patches()` returns none. It also calls `SaliencyDetector.XRAI` with a file path where `split.py`
  expects a folder. The wrapper uses `utils.split.SaliencyDetector().XRAI(folder)`, which writes `final.json`
  with the top-4 patches.
- Noisy burnishing (`utils/noisypatch.py`) is described in the paper but never called by `main.py`. It is
  vendored for reference and not used.
- XRAI runs on CPU, as upstream (InceptionV3 is never moved to a GPU).

## Not changed, but flagged: NMP selection in `utils/split.py` differs from the paper

`process_and_score_boxes()` scores a box group seen in n paraphrases as `[1, .75, .5, .25, .1][n-1]`, which is
the paper's `Ws = max(0.1, 1 - 0.25(n-1))`. The paper (l. 288-292) says lower scores mean *more* stable. The
code, however, then **drops** groups scoring `< 0.25`, i.e. regions seen in all 5 paraphrases, and
`map_boxes_to_patches()` ranks patches by score-weighted coverage, which prefers *less* stable regions. If every
group is dropped, the default box `[256,256,512,512]` is used, and the top-4 of its 16 tied cells come from
`argsort` order. Also, every patch is embedded with strength 1, because `GTWatermark` zips its single box with
`strengths[0]`; the per-NMP `Ws` is not used. Pinned by
`test_split_drops_box_seen_in_all_paraphrases`. The wrapper records per image `nmp_boxes`, `nmp_boxes_in_all`
and `nmp_fallback` (`quadrant` / `centre` / `none`) so the smoke run shows how often this happens.
