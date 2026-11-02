![PECCAVI Banner](assets/peccavi_logo.png)

# PECCAVI: Overcoming the Brittleness of AI Image Watermarking Under Visual Paraphrasing Attacks

[![CVPR 2026 Main Track](https://img.shields.io/badge/CVPR-2026%20Main%20Track-0A66C2)](https://pragyaai.github.io/PECCAVI/)
[![arXiv](https://img.shields.io/badge/arXiv-2506.22960-b31b1b.svg)](https://arxiv.org/abs/2506.22960)
[![Project Page](https://img.shields.io/badge/Project-Website-1f8b4c)](https://pragyaai.github.io/PECCAVI/)

**🎉 Officially Accepted to the Main Track of CVPR 2026 🎉**

**Project page:** https://pragyaai.github.io/PECCAVI/

**Preprint:** https://arxiv.org/abs/2506.22960

---

## 📌 TL;DR

By 2026, up to 90% of online content may be synthetically generated, raising urgent concerns about the proliferation of AI-driven disinformation. In response, technology companies are turning to watermarking, but current methods remain brittle and vulnerable to attack. 

The **Visual Paraphrase Attack (VPA)** is a generative method that effectively removes both visible and invisible watermarks from AI-generated images while preserving their semantic content. It proceeds by generating a descriptive caption from the original image and feeding it into a diffusion-based text-to-image model to produce a visually similar, watermark-free image.

**PECCAVI** is the first watermarking technique explicitly designed to resist visual paraphrasing. 
* PECCAVI embeds robust, distortion-free watermarks in semantically stable image regions termed Non-Melting Points (NMPs). 
* It uses multi-channel frequency domain encoding and noisy burnishing to obfuscate watermark locations and hinder reverse engineering.
* The method is model-agnostic and achieves a reported detection accuracy of 98.8%.
* Alongside the model, we release the first benchmark dataset for visual paraphrasing attacks.

---

## ⚙️ Method at a Glance

The end-to-end implementation in `main.py` follows this comprehensive pipeline:

1. **Paraphrase Generation**: Generates five paraphrased versions of a given image to find stable visual representations.
2. **Saliency Detection & Bounding Box Aggregation**: Identifies salient regions using XRAI. It computes the Intersection over Union (IoU) to locate stable areas across paraphrases and refines them using Non-Maximum Suppression (NMS) to define Non-Melting Points (NMPs).
3. **Patch-wise Watermark Optimization**: Divides the image into patches, matching them with NMPs, and uses DDIM inversion to map patches to the latent space. The latent code is then transformed into the frequency domain using a Fourier transform, where concentric ring watermarks are embedded.
4. **Adaptive Enhancement**: Adjusts the watermarked output to preserve visual fidelity, optimizing for structural and perceptual metrics like PSNR and SSIM.
5. **Detection**: Systematically scans spatial patches and multiple latent feature channels, employing a statistical hypothesis test based on a non-central chi-squared distribution to calculate the final Watermark Detection Probability (WDP).

---

## 📂 Repository Structure

```text
PECCAVI/
|-- main.py                             # End-to-end pipeline entrypoint
|-- config.yaml                         # Experiment/configuration settings
|-- requirements.txt                    # Python dependencies
|-- captions_train2014_new_format.json  # Caption source keyed by image id
|-- assets/
|   |-- peccavi_logo.png
|   `-- Results.png
|-- utils/
|   |-- config.py       # Config loader
|   |-- paraphrase.py   # Image paraphrasing pipeline
|   |-- salient.py      # XRAI saliency + box aggregation
|   |-- watermark.py    # Watermark injection and detection wrapper
|   `-- ...
`-- ZoDiac/
    |-- main/           # Diffusion/watermark internals
    |-- loss/           # Loss implementations
    `-- old/            # Legacy variants
```



---

## 🚀 Environment Setup

```bash
conda create -n peccavi python=3.10 -y
conda activate peccavi
pip install -r requirements.txt
```

**Optional:** If using Weights & Biases logging, authenticate your environment:

```bash
wandb login
```

*(To execute locally without W&B syncing, simply set `wandb: False` in your `config.yaml`.)* 

---

## 📊 Data Preparation
Our current input contract expects data formatted as follows:

* **Input Directory**: Place source images in `data/Images/`.


* **File Naming**: Use the convention `Image_<id>.<ext>`.


* **Caption Keys**: The `<id>` must strictly match a key in `captions_train2014_new_format.json`.


* **Caption Source**: Captions are ingested via `annotations[<id>] -> list[str]`.


> **Example:**
> Image file: `data/Images/Image_492386.jpg` 
> 
> 
> Corresponding caption key: `"492386"` inside `captions_train2014_new_format.json` 
> 
> 

---

## 🛠 Configuration Guide

Adjust `config.yaml` to customize your experiments:

**Models & Device**:

* `model_id`: Base diffusion model for the watermarking process.
* `paraphrasing_model`: Generative model utilized for the image paraphrase attack.
* `device`: Hardware target (e.g., `cuda:0`, `cuda:1`, `cpu`).

**Paths**:

* `base_images_dir`: Input source folder.
* `paraphrased_images_dir`: Output for generated paraphrases.
* `WM_Paraphrased_dir`: Output for paraphrases derived from watermarked images.
* `watermarked_path`: Destination for successful PECCAVI embeddings.
* `json_data_path`: Pathway to the dataset captions.

**Watermark Optimization**:

* `iters`, `num_inference_steps`, `learning_rate`, `milestones`: Controls diffusion steps and optimizer logic.
* `loss_weights`: Balances L2, Watson-VGG, SSIM, and watermark retention loss functions.
* `w_channel`, `w_radius`, `w_type`: Controls the geometry and multi-channel distribution of the watermark rings.

**Detection & Quality Thresholds**:

* `detection`: Toggle the final extraction validation pass.
* `ssim_threshold`: Minimum accepted visual fidelity post-processing.
* `detect_threshold`: Cutoff for confirming watermark existence.

---

## 🏃 Running PECCAVI

Execute the main pipeline:

```bash
python main.py
```

**Runtime Behavior:** The entry point defaults to reading from `config.yaml`. It processes every file sequentially in `base_images_dir`, subjecting each to the full lifecycle: paraphrasing, saliency mapping, watermarking, re-paraphrasing, and final detection probability scoring.

### Expected Artifacts
Upon successful execution, the pipeline outputs:
* `data/Paraphrased/<id>/...`: Paraphrased variants and their corresponding saliency maps.
* `data/Paraphrased/<id>/final.json`: Aggregated NMP bounding boxes.
* `data/Watermarked/<id>/...`: The enhanced, PECCAVI-protected images.
* `data/WM_Paraphrased/<id>/...`: Re-paraphrased versions of the protected images for attack simulation.
* `output/<run_name>_results.json`: Final Watermark Detection Probability (WDP) scores.

`final.json` entries follow this structure:

```json
[
  {
    "box": [x1, y1, x2, y2],
    "score": 0.75,
    "patch": [patch_x, patch_y]
  }
]

```

---

## 📈 Results
PECCAVI consistently outperforms existing methods such as WAM and StableSignature, maintaining high detection probability and preserving visual quality even under intense semantic paraphrasing.

---

## 📑 Citation
If PECCAVI contributes to your research or if you utilize our benchmark dataset, please cite our CVPR 2026 paper:

```bibtex
@inproceedings{dixit2026peccavi,
  title={PECCAVI: Overcoming the Brittleness of AI Image Watermarking Under Visual Paraphrasing Attacks},
  author={Dixit, Shreyas and Aziz, Ashhar and Bajpai, Shashwat and Sharma, Vasu and Chadha, Aman and Jain, Vinija and Das, Amitava},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2026}
}

```

ArXiv preprint: https://arxiv.org/abs/2506.22960 

## 🤝 Acknowledgments
This implementation builds on diffusion watermarking and robustness ideas from prior work, including ZoDiac-style frequency-domain watermarking components and saliency-driven localization pipelines.

## ✉️ Contact
For issues, questions, or reproducibility discussions, please open an issue in this repository.
