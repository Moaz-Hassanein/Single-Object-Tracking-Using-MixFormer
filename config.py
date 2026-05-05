"""
config.py — Centralized configuration for the AIC-2026 MixFormer tracker.

All paths, hyperparameters, and tracker settings live here.
Override via environment variables or by editing this file directly.
"""

import os

# ─────────────────────────────────────────────
# Paths (all overridable via env vars)
# ─────────────────────────────────────────────

# Root of the contest data release (contains videos/, annotations/, metadata/)
DATA_ROOT = os.environ.get("DATA_ROOT", "./data/contest_release")

# Path to contestant_manifest.json
MANIFEST_PATH = os.environ.get(
    "MANIFEST_PATH",
    os.path.join(DATA_ROOT, "metadata", "contestant_manifest.json"),
)

# Where to write the final submission CSV
OUTPUT_CSV = os.environ.get("OUTPUT_CSV", "./submission.csv")

# MixFormer repo root (cloned during Docker build / setup)
MIXFORMER_ROOT = os.environ.get("MIXFORMER_ROOT", "./MixFormer")

# Directory that holds all checkpoints
CKPT_DIR = os.environ.get(
    "CKPT_DIR",
    os.path.join(MIXFORMER_ROOT, "output", "checkpoints", "train", "mixformer_cvt_online"),
)

# Pre-trained backbone checkpoint (MixFormer online, ImageNet-22k)
PRETRAIN_CKPT = os.environ.get(
    "PRETRAIN_CKPT",
    os.path.join(CKPT_DIR, "MixFormer", "models", "mixformer_online_22k.pth.tar"),
)

# Fine-tuned checkpoint produced by train.py (best model to use for inference)
FINETUNED_CKPT = os.environ.get(
    "FINETUNED_CKPT",
    os.path.join(CKPT_DIR, "finetune_contest", "MIXFORMER_EP0005.pth.tar"),
)

# Filtered manifest written by train.py after video health-check
FILTERED_MANIFEST = os.environ.get("FILTERED_MANIFEST", "./filtered_manifest.json")

# ─────────────────────────────────────────────
# Training hyperparameters
# ─────────────────────────────────────────────

TRAIN = {
    "lr": 2e-5,
    "weight_decay": 1e-4,
    "epochs": 5,
    "lr_drop_epoch": 4,
    "batch_size": 8,
    "num_workers": 0,           # set > 0 if your machine has enough RAM
    "optimizer": "ADAMW",
    "backbone_multiplier": 0.1,
    "iou_weight": 2.0,
    "l1_weight": 5.0,
    "grad_clip_norm": 0.1,
    "print_interval": 10,
    "val_epoch_interval": 5,
    "sample_per_epoch": 2000,
    "val_sample_per_epoch": 200,
    "data_fraction": None,      # set e.g. 0.2 to use 20 % of sequences
}

# ─────────────────────────────────────────────
# Data / augmentation settings
# ─────────────────────────────────────────────

DATA = {
    "mean": [0.485, 0.456, 0.406],
    "std":  [0.229, 0.224, 0.225],
    "search_size": 320,
    "search_factor": 5.0,
    "search_center_jitter": 4.5,
    "search_scale_jitter": 0.5,
    "template_size": 128,
    "max_sample_interval": 200,
    "sampler_mode": "trident_pro",
}

# ─────────────────────────────────────────────
# Tracker / inference settings
# ─────────────────────────────────────────────

TRACKER = {
    # Which split in the manifest to run inference on
    "eval_split": "public_lb",

    # MixFormerOnline runtime knobs
    "online_sizes": 1,
    "update_interval": 200,
    "online_score_th": 0.50,

    # Test-time augmentation: run a horizontally flipped copy in parallel
    "use_tta_flip": True,

    # Temporal smoothing (exponential moving average; higher = more weight on current frame)
    "ema_alpha": 0.7,

    # Jump suppression: relative centre-to-diagonal ratio above which we keep last box
    "jump_threshold": 0.6,

    # Search / template sizes at test time
    "test_search_factor": 4.5,
    "test_search_size": 288,
    "test_template_factor": 2.0,
    "test_template_size": 128,
}

# ─────────────────────────────────────────────
# Fine-tune YAML (written to disk by train.py)
# ─────────────────────────────────────────────

FINETUNE_YAML = """
MODEL:
  HEAD_TYPE: CORNER
  HIDDEN_DIM: 384
  NUM_OBJECT_QUERIES: 1
  POSITION_EMBEDDING: sine
  PREDICT_MASK: False
  PRETRAINED_STAGE1: True
  NLAYER_HEAD: 3
  HEAD_FREEZE_BN: False
  BACKBONE:
    PRETRAINED: False
    PRETRAINED_PATH: ''
    FREEZE_BN: True

TRAIN:
  TRAIN_SCORE: True
  SCORE_WEIGHT: 1.0
  LR: {lr}
  WEIGHT_DECAY: {weight_decay}
  EPOCH: {epochs}
  LR_DROP_EPOCH: {lr_drop_epoch}
  BATCH_SIZE: {batch_size}
  NUM_WORKER: {num_workers}
  OPTIMIZER: {optimizer}
  BACKBONE_MULTIPLIER: {backbone_multiplier}
  IOU_WEIGHT: {iou_weight}
  L1_WEIGHT: {l1_weight}
  DEEP_SUPERVISION: False
  FREEZE_STAGE0: True
  PRINT_INTERVAL: {print_interval}
  VAL_EPOCH_INTERVAL: {val_epoch_interval}
  GRAD_CLIP_NORM: {grad_clip_norm}
  SCHEDULER:
    TYPE: step
    DECAY_RATE: 0.1

DATA:
  SAMPLER_MODE: {sampler_mode}
  MEAN: {mean}
  STD: {std}
  MAX_SAMPLE_INTERVAL: [{max_sample_interval}]
  TRAIN:
    DATASETS_NAME: ['CUSTOMCONTEST']
    DATASETS_RATIO: [1]
    SAMPLE_PER_EPOCH: {sample_per_epoch}
  VAL:
    DATASETS_NAME: ['CUSTOMCONTEST']
    DATASETS_RATIO: [1]
    SAMPLE_PER_EPOCH: {val_sample_per_epoch}
  SEARCH:
    SIZE: {search_size}
    FACTOR: {search_factor}
    CENTER_JITTER: {search_center_jitter}
    SCALE_JITTER: {search_scale_jitter}
  TEMPLATE:
    SIZE: {template_size}

TEST:
  TEMPLATE_FACTOR: {test_template_factor}
  TEMPLATE_SIZE: {test_template_size}
  SEARCH_FACTOR: {test_search_factor}
  SEARCH_SIZE: {test_search_size}
  EPOCH: {epochs}
""".format(
    **{**TRAIN, **DATA, **TRACKER}
)
