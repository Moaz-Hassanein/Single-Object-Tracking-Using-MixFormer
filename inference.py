"""
inference.py — Frame-by-frame tracking inference with MixFormer (CvT-online).

Usage:
    python inference.py [--checkpoint PATH] [--manifest PATH]
                        [--data_root PATH] [--output_csv PATH]
                        [--split public_lb]

Guarantees
----------
* Every frame in every sequence receives a prediction row in the output CSV.
* If the tracker fails on a frame → last valid bbox is used (fallback).
* Test-time augmentation (horizontal flip) is merged via score-weighted average.
* Temporal smoothing (EMA) + jump suppression prevent wild bbox drift.
"""

import argparse
import csv
import glob
import json
import os
import sys

import cv2
import numpy as np

import config as cfg


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="MixFormer inference → submission CSV")
    p.add_argument("--checkpoint", default=None,
                   help="Path to .pth.tar checkpoint (fine-tuned or pretrained)")
    p.add_argument("--manifest", default=cfg.MANIFEST_PATH)
    p.add_argument("--data_root", default=cfg.DATA_ROOT)
    p.add_argument("--output_csv", default=cfg.OUTPUT_CSV)
    p.add_argument("--split", default=cfg.TRACKER["eval_split"],
                   help="Manifest split to evaluate (e.g. public_lb, test)")
    p.add_argument("--mixformer_root", default=cfg.MIXFORMER_ROOT)
    p.add_argument("--no_tta", action="store_true",
                   help="Disable horizontal-flip TTA")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Compatibility patch (same as train.py — must run before MixFormer import)
# ─────────────────────────────────────────────────────────────────────────────

def patch_torch_load():
    import torch
    if not getattr(torch.load, "_already_patched", False):
        _orig = torch.load
        def _patched(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return _orig(*args, **kwargs)
        _patched._already_patched = True
        torch.load = _patched


# ─────────────────────────────────────────────────────────────────────────────
# Tracker factory
# ─────────────────────────────────────────────────────────────────────────────

def build_tracker(mixformer_root: str, checkpoint_path: str):
    """Import MixFormer and return a factory function that creates fresh trackers."""
    if mixformer_root not in sys.path:
        sys.path.insert(0, mixformer_root)

    from lib.test.tracker.mixformer_cvt_online import MixFormerOnline
    from lib.test.parameter.mixformer_cvt_online import parameters

    tc = cfg.TRACKER
    params = parameters("baseline", model="mixformer_online_22k.pth.tar")
    params.checkpoint      = checkpoint_path
    params.debug           = False
    params.online_sizes    = tc["online_sizes"]
    params.update_interval = tc["update_interval"]
    params.online_score_th = tc["online_score_th"]

    def factory():
        return MixFormerOnline(params, dataset_name="lasot")

    return factory


# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────────────────────────────────────

def flip_box(x, y, w, h, frame_width: int):
    """Mirror bbox horizontally: used to unflip TTA predictions."""
    return frame_width - x - w, y, w, h


def ema_smooth(history: list, alpha: float = 0.7):
    """Exponential moving average between last two bboxes."""
    if len(history) < 2:
        return history[-1]
    prev = history[-2]
    curr = history[-1]
    return tuple(alpha * c + (1.0 - alpha) * p for c, p in zip(curr, prev))


def is_jump(prev_box, curr_box, threshold: float = 0.6) -> bool:
    """
    Return True when the centre displacement is large relative to the
    bounding-box diagonal — a heuristic for tracker drift / ID switches.
    """
    if prev_box is None:
        return False
    cx_p = prev_box[0] + prev_box[2] / 2
    cy_p = prev_box[1] + prev_box[3] / 2
    cx_c = curr_box[0] + curr_box[2] / 2
    cy_c = curr_box[1] + curr_box[3] / 2
    diag  = (prev_box[2] ** 2 + prev_box[3] ** 2) ** 0.5 + 1e-6
    dist  = ((cx_c - cx_p) ** 2 + (cy_c - cy_p) ** 2) ** 0.5
    return (dist / diag) > threshold


def clamp_box(x, y, w, h, frame_w: int, frame_h: int):
    """Ensure bbox stays within frame boundaries and has positive size."""
    x = max(0.0, min(float(x), frame_w - 1))
    y = max(0.0, min(float(y), frame_h - 1))
    w = max(1.0, min(float(w), frame_w - x))
    h = max(1.0, min(float(h), frame_h - y))
    return x, y, w, h


# ─────────────────────────────────────────────────────────────────────────────
# Annotation helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_first_bbox(annotation_path: str):
    """Read the first line of an annotation file → (x, y, w, h) floats."""
    with open(annotation_path) as f:
        line = f.readline().strip()
    return list(map(float, line.replace(",", " ").split()[:4]))


# ─────────────────────────────────────────────────────────────────────────────
# Per-sequence tracking
# ─────────────────────────────────────────────────────────────────────────────

def track_sequence(seq_id: str, entry: dict, data_root: str,
                   tracker_factory, use_tta: bool) -> list:
    """
    Run MixFormer on one video sequence.

    Returns a list of [row_id, x, y, w, h] for every frame (0-indexed).
    Fallback to last valid bbox on any failure — no frame is ever skipped.
    """
    tc = cfg.TRACKER
    video_path = os.path.join(data_root, entry["video_path"])
    ann_path   = os.path.join(data_root, entry["annotation_path"])

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [WARN] Cannot open video: {video_path}")
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ret, first_frame = cap.read()
    if not ret:
        cap.release()
        print(f"  [WARN] Cannot read first frame: {video_path}")
        return []

    H, W = first_frame.shape[:2]
    x0, y0, w0, h0 = load_first_bbox(ann_path)

    # Initialise primary tracker
    tracker = tracker_factory()
    tracker.initialize(first_frame, {"init_bbox": [x0, y0, w0, h0]})

    # Initialise flipped TTA tracker
    if use_tta:
        fx0, fy0, fw0, fh0 = flip_box(x0, y0, w0, h0, W)
        tracker_flp = tracker_factory()
        tracker_flp.initialize(cv2.flip(first_frame, 1), {"init_bbox": [fx0, fy0, fw0, fh0]})

    rows         = []
    box_history  = [(x0, y0, w0, h0)]
    last_valid   = (x0, y0, w0, h0)

    for frame_idx in range(total_frames):
        try:
            if frame_idx == 0:
                # Initialisation frame: use ground-truth bbox
                x, y, w, h = x0, y0, w0, h0

            else:
                ret, frame = cap.read()
                if not ret:
                    # Video ended early — hold last position for remaining frames
                    x, y, w, h = last_valid
                else:
                    # ── primary tracker ────────────────────────────────────
                    out    = tracker.track(frame)
                    bx, by, bw, bh = out["target_bbox"]
                    score  = out.get("best_score", 1.0)

                    if use_tta:
                        # ── flipped TTA tracker ────────────────────────────
                        out_flp = tracker_flp.track(cv2.flip(frame, 1))
                        bfx, bfy, bfw, bfh = out_flp["target_bbox"]
                        bfx, bfy, bfw, bfh = flip_box(bfx, bfy, bfw, bfh, W)
                        score_flp = out_flp.get("best_score", 1.0)

                        total_s = score + score_flp + 1e-6
                        bx = (bx * score + bfx * score_flp) / total_s
                        by = (by * score + bfy * score_flp) / total_s
                        bw = (bw * score + bfw * score_flp) / total_s
                        bh = (bh * score + bfh * score_flp) / total_s

                    curr_box = (bx, by, bw, bh)

                    # ── jump suppression ────────────────────────────────────
                    if is_jump(box_history[-1], curr_box, tc["jump_threshold"]):
                        # Discard this prediction; hold last smoothed box
                        x, y, w, h = box_history[-1]
                    else:
                        box_history.append(curr_box)
                        x, y, w, h = ema_smooth(box_history, tc["ema_alpha"])

                    x, y, w, h = clamp_box(x, y, w, h, W, H)

        except Exception as exc:
            # Fallback: any unexpected tracker error → last valid prediction
            print(f"  [fallback] frame {frame_idx} of {seq_id}: {exc}")
            x, y, w, h = last_valid

        last_valid = (x, y, w, h)
        rows.append([f"{seq_id}_{frame_idx}", float(x), float(y), float(w), float(h)])

    cap.release()
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Resolve checkpoint
# ─────────────────────────────────────────────────────────────────────────────

def resolve_checkpoint(cli_ckpt) -> str:
    """
    Priority:
      1. CLI --checkpoint argument
      2. FINETUNED_CKPT from config (if file exists)
      3. Any .pth.tar under CKPT_DIR (latest by name)
      4. PRETRAIN_CKPT from config (fallback)
    """
    if cli_ckpt and os.path.exists(cli_ckpt):
        return cli_ckpt

    if os.path.exists(cfg.FINETUNED_CKPT):
        print(f"  [ckpt] Using fine-tuned checkpoint: {cfg.FINETUNED_CKPT}")
        return cfg.FINETUNED_CKPT

    candidates = sorted(
        glob.glob(os.path.join(cfg.CKPT_DIR, "**", "*.pth.tar"), recursive=True)
    )
    if candidates:
        chosen = candidates[-1]
        print(f"  [ckpt] Auto-selected latest checkpoint: {chosen}")
        return chosen

    if os.path.exists(cfg.PRETRAIN_CKPT):
        print(f"  [ckpt] Falling back to pretrained checkpoint: {cfg.PRETRAIN_CKPT}")
        return cfg.PRETRAIN_CKPT

    print("[ERROR] No checkpoint found. "
          "Set FINETUNED_CKPT or pass --checkpoint.")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Entrypoint
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    print("\n══ AIC-2026 MixFormer Inference ══")

    # PyTorch compatibility patch (must happen before MixFormer import)
    patch_torch_load()

    checkpoint = resolve_checkpoint(args.checkpoint)
    print(f"  checkpoint : {checkpoint}")
    print(f"  split      : {args.split}")
    print(f"  output_csv : {args.output_csv}")

    # Load manifest
    with open(args.manifest) as f:
        manifest = json.load(f)

    if args.split not in manifest:
        print(f"[ERROR] Split '{args.split}' not found in manifest. "
              f"Available: {list(manifest.keys())}")
        sys.exit(1)

    sequences = manifest[args.split]
    print(f"  sequences  : {len(sequences)}\n")

    # Build tracker factory
    tracker_factory = build_tracker(args.mixformer_root, checkpoint)
    use_tta = cfg.TRACKER["use_tta_flip"] and not args.no_tta
    if use_tta:
        print("  TTA        : horizontal flip (score-weighted average)\n")

    # ── Frame-by-frame tracking ────────────────────────────────────────────
    all_rows = []
    for i, (seq_id, entry) in enumerate(sequences.items(), 1):
        print(f"[{i:3d}/{len(sequences)}] {seq_id} …", end=" ", flush=True)
        rows = track_sequence(seq_id, entry, args.data_root,
                              tracker_factory, use_tta)
        all_rows.extend(rows)
        print(f"{len(rows)} frames")

    # ── Write CSV ─────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "x", "y", "w", "h"])
        writer.writerows(all_rows)

    print(f"\n✔ Submission written → {args.output_csv}")
    print(f"  Total predictions : {len(all_rows)}")
    print(f"  Sequences covered : {len(sequences)}")


if __name__ == "__main__":
    main()
