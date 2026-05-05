# AIC-2026 Visual Object Tracking — MixFormer Submission

> **Model:** MixFormer CvT-Online (fine-tuned on contest data)  
> **Inference guarantee:** Every frame at original frame rate — no skips, no gaps  
> **TTA:** Horizontal-flip dual-tracker with score-weighted merge  
> **Post-processing:** EMA temporal smoothing + jump suppression  

---

## 📁 Repository Structure

```
aic-2026-tracker/
├── README.md               ← you are here
├── Dockerfile              ← inference-only reproducible container
├── requirements.txt        ← Python dependencies
├── config.py               ← ALL paths, hyperparams, tracker settings
├── train.py                ← fine-tuning pipeline (run once before inference)
├── inference.py            ← inference → submission.csv
└── docs/
    └── technical_report.pdf
```

---

## 📥 Model Download

### Pre-trained backbone (required for both training and inference)

Download **`mixformer_online_22k.pth.tar`** from the official MixFormer release:

```
https://drive.google.com/drive/folders/1wyeIs3ytYkmAtTXoVlLMkJ4aSTq5CBHq
```

Place it at:
```
checkpoints/mixformer_online_22k.pth.tar
```

### Fine-tuned checkpoint (inference-ready)

After running `train.py` (see below), the best checkpoint is saved to:
```
MixFormer/output/checkpoints/train/mixformer_cvt_online/finetune_contest/MIXFORMER_EP0005.pth.tar
```

Copy it to `checkpoints/best_model.pth.tar` for the Docker run command below.

> **Direct download link** (replace with your hosted URL after training):  
> `https://<your-storage>/best_model.pth.tar`  
> Place at: `checkpoints/best_model.pth.tar`

---

## 🛠 Prerequisites

- Python 3.9+
- PyTorch 2.2+ with CUDA 12.1
- NVIDIA GPU with ≥ 8 GB VRAM (16 GB recommended for training)
- Docker + nvidia-container-toolkit (for Docker inference)

### Clone this repo and MixFormer

```bash
git clone https://github.com/<your-username>/aic-2026-tracker.git
cd aic-2026-tracker

git clone https://github.com/MCG-NJU/MixFormer.git
```

### Install dependencies

```bash
# Install PyTorch first (if not already installed):
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121

# Then install project dependencies:
pip install -r requirements.txt
```

---

## ⚙️ Training

Fine-tunes MixFormer on the contest training split. All patches, dataset registration,
and config file generation are handled automatically by `train.py`.

```bash
python train.py \
  --pretrain_ckpt checkpoints/mixformer_online_22k.pth.tar \
  --data_root     /path/to/contest_release \
  --manifest      /path/to/contest_release/metadata/contestant_manifest.json \
  --mixformer_root MixFormer \
  --output_dir    MixFormer/output
```

| Argument | Default | Description |
|---|---|---|
| `--pretrain_ckpt` | `checkpoints/mixformer_online_22k.pth.tar` | ImageNet-22k backbone checkpoint |
| `--data_root` | `./data/contest_release` | Root of contest data |
| `--manifest` | `<data_root>/metadata/contestant_manifest.json` | Manifest JSON |
| `--mixformer_root` | `./MixFormer` | Path to cloned MixFormer repo |
| `--output_dir` | `MixFormer/output` | Checkpoint output directory |

All hyperparameters (LR, epochs, batch size, etc.) live in `config.py`.

### What `train.py` does automatically

1. Patches MixFormer source files for PyTorch 2.x compatibility
2. Writes `CustomContestDataset` into the MixFormer package
3. Registers `CUSTOMCONTEST` in MixFormer's `base_functions.py`
4. Health-checks every training video (skips unreadable files)
5. Writes the fine-tune YAML config from `config.py`
6. Injects a pretrained-checkpoint loader hook into `base_trainer.py`
7. Launches `run_training.py` as a subprocess

---

## 🔍 Inference (local)

```bash
python inference.py \
  --checkpoint  checkpoints/best_model.pth.tar \
  --manifest    /path/to/contest_release/metadata/contestant_manifest.json \
  --data_root   /path/to/contest_release \
  --output_csv  submission.csv \
  --split       public_lb
```

| Argument | Default | Description |
|---|---|---|
| `--checkpoint` | auto-resolved from `config.py` | Fine-tuned `.pth.tar` checkpoint |
| `--manifest` | `config.MANIFEST_PATH` | Path to manifest JSON |
| `--data_root` | `config.DATA_ROOT` | Contest data root |
| `--output_csv` | `./submission.csv` | Output CSV path |
| `--split` | `public_lb` | Manifest split to evaluate |
| `--no_tta` | off | Disable horizontal-flip TTA |

Output format: `id,x,y,w,h` — one row per frame, all frames, all sequences.

---

## 🐳 Docker Inference

The Docker image is inference-only and fully self-contained.

### Build

```bash
docker build -t aic2026-tracker .
```

### Run

```bash
docker run --gpus all \
  -v /absolute/path/to/contest_release:/app/data/contest_release:ro \
  -v /absolute/path/to/checkpoints:/app/checkpoints:ro \
  -v /absolute/path/to/output:/app/output \
  -e FINETUNED_CKPT=/app/checkpoints/best_model.pth.tar \
  -e MANIFEST_PATH=/app/data/contest_release/metadata/contestant_manifest.json \
  -e DATA_ROOT=/app/data/contest_release \
  -e OUTPUT_CSV=/app/output/submission.csv \
  aic2026-tracker
```

The CSV will appear at `/absolute/path/to/output/submission.csv` on the host.


---

## 🔧 Configuration

All settings are in `config.py`. Key overrides via environment variables:

| Env var | Purpose |
|---|---|
| `DATA_ROOT` | Contest data root directory |
| `MANIFEST_PATH` | Path to contestant_manifest.json |
| `OUTPUT_CSV` | Where to write submission.csv |
| `MIXFORMER_ROOT` | Path to cloned MixFormer |
| `FINETUNED_CKPT` | Fine-tuned checkpoint path |
| `PRETRAIN_CKPT` | Pretrained backbone checkpoint |
| `FILTERED_MANIFEST` | Path for filtered manifest cache |

---

## 📊 Method Summary

| Component | Choice | Reason |
|---|---|---|
| Architecture | MixFormer CvT-Online | Unified attention + online update; stable fine-tuning API |
| Pre-training | ImageNet-22k | Strong generalisation; domain-agnostic appearance features |
| Fine-tuning | 5 epochs, LR=2e-5 | Conservative to prevent catastrophic forgetting on small dataset |
| TTA | Horizontal flip, score-weighted | Reduces localisation variance on lateral motion |
| Smoothing | EMA α=0.7 | Removes jitter without perceivable lag |
| Jump guard | Centre/diagonal ratio > 0.6 | Rejects ID-switch artefacts and large re-detections |
| Frame guarantee | `range(total_frames)` loop + fallback | Every frame covered, no crash on video decode error |

See `docs/technical_report.pdf` for full details.

---

## 📄 Citation

```bibtex
@inproceedings{cui2022mixformer,
  title={MixFormer: End-to-End Tracking with Iterative Mixed Attention},
  author={Cui, Yutao and Jiang, Cheng and Wang, Limin and Wu, Gangshan},
  booktitle={CVPR},
  year={2022}
}
```
