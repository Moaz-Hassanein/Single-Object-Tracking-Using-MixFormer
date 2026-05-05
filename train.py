"""
train.py — End-to-end fine-tuning of MixFormer (CvT-online) on the contest dataset.

Usage:
    python train.py [--pretrain_ckpt PATH] [--data_root PATH] [--output_dir PATH]

All defaults are in config.py and can be overridden via env vars or CLI args.

Pipeline
--------
1. Patch MixFormer source files for PyTorch 2.x compatibility.
2. Write the custom dataset class into the MixFormer package.
3. Register CUSTOMCONTEST in MixFormer's base_functions.py.
4. Health-check every training video and write a filtered manifest.
5. Write the fine-tune YAML config from config.py.
6. Patch base_trainer.py to load the pretrained checkpoint at startup.
7. Launch MixFormer's run_training.py as a subprocess.
"""

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys

import cv2

import config as cfg


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune MixFormer on contest data")
    p.add_argument("--pretrain_ckpt", default=cfg.PRETRAIN_CKPT,
                   help="Path to mixformer_online_22k.pth.tar")
    p.add_argument("--data_root", default=cfg.DATA_ROOT,
                   help="Contest data root directory")
    p.add_argument("--manifest", default=cfg.MANIFEST_PATH,
                   help="Path to contestant_manifest.json")
    p.add_argument("--output_dir", default=os.path.join(cfg.MIXFORMER_ROOT, "output"),
                   help="Where MixFormer saves checkpoints")
    p.add_argument("--mixformer_root", default=cfg.MIXFORMER_ROOT,
                   help="Path to cloned MixFormer repo")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 – Compatibility patches (torch._six, torchvision, etc.)
# ─────────────────────────────────────────────────────────────────────────────

def patch_loader(mixformer_root: str):
    """Fix deprecated torch._six import in data/loader.py."""
    path = os.path.join(mixformer_root, "lib", "train", "data", "loader.py")
    if not os.path.exists(path):
        print(f"  [skip] loader.py not found: {path}")
        return
    with open(path) as f:
        lines = f.readlines()
    patched = False
    for i, line in enumerate(lines):
        if "torch._six" in line or ("string_classes" in line and "int_classes" in line):
            lines[i] = "string_classes = (str,)\nint_classes = (int,)\n"
            patched = True
            break
    if patched:
        with open(path, "w") as f:
            f.writelines(lines)
        print("  [patch] loader.py — removed torch._six import")


def patch_misc(mixformer_root: str):
    """Remove legacy torchvision import guard in lib/utils/misc.py."""
    path = os.path.join(mixformer_root, "lib", "utils", "misc.py")
    if not os.path.exists(path):
        print(f"  [skip] utils/misc.py not found: {path}")
        return
    with open(path) as f:
        src = f.read()
    old = (
        "if float(torchvision.__version__[:3]) < 0.7:\n"
        "    from torchvision.ops import _new_empty_tensor\n"
        "    from torchvision.ops.misc import _output_size"
    )
    new = "# torchvision >= 0.7 - legacy imports removed"
    if old in src:
        with open(path, "w") as f:
            f.write(src.replace(old, new))
        print("  [patch] utils/misc.py — removed torchvision version guard")


def patch_trainers_misc(mixformer_root: str):
    """Replace 'from torch._six import inf' with 'from math import inf'."""
    path = os.path.join(mixformer_root, "lib", "train", "trainers", "misc.py")
    if not os.path.exists(path):
        print(f"  [skip] trainers/misc.py not found: {path}")
        return
    with open(path) as f:
        src = f.read()
    if "from torch._six import inf" in src:
        with open(path, "w") as f:
            f.write(src.replace("from torch._six import inf", "from math import inf"))
        print("  [patch] trainers/misc.py — torch._six → math.inf")


def patch_torch_load():
    """Monkey-patch torch.load to default weights_only=False (PyTorch ≥ 2.6)."""
    import torch
    if not getattr(torch.load, "_already_patched", False):
        _orig = torch.load
        def _patched(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return _orig(*args, **kwargs)
        _patched._already_patched = True
        torch.load = _patched
        print("  [patch] torch.load — weights_only defaults to False")


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 – Write CustomContestDataset into MixFormer package
# ─────────────────────────────────────────────────────────────────────────────

CUSTOM_DATASET_CODE = '''import json
import os
import cv2
import random
import numpy as np
import torch
from lib.train.dataset.base_video_dataset import BaseVideoDataset


class CustomContestDataset(BaseVideoDataset):
    """
    Contest dataset — reads frames directly from MP4 video files.

    Each sequence maps to one entry in the manifest: a video file and a
    corresponding annotation file with one bbox (x y w h) per line.
    """

    def __init__(self, root=None, image_loader=None, split="train",
                 data_fraction=None, manifest_path=None):
        root = root or os.environ.get("DATA_ROOT", "./data/contest_release")
        super().__init__("CustomContest", root, image_loader)

        if manifest_path is None:
            manifest_path = os.environ.get(
                "FILTERED_MANIFEST",
                os.path.join(root, "metadata", "contestant_manifest.json"),
            )
        with open(manifest_path) as f:
            manifest = json.load(f)

        self.sequences = list(manifest["train"].values())

        if data_fraction is not None:
            k = max(1, int(len(self.sequences) * data_fraction))
            self.sequences = random.sample(self.sequences, k)

    # ── required interface ────────────────────────────────────────────────

    def get_num_sequences(self):
        return len(self.sequences)

    def get_sequence_info(self, seq_id):
        entry = self.sequences[seq_id]
        ann_path = os.path.join(self.root, entry["annotation_path"])
        bboxes = []
        with open(ann_path) as f:
            for line in f:
                parts = line.strip().replace(",", " ").split()
                if len(parts) >= 4:
                    bboxes.append(list(map(float, parts[:4])))
        bbox = torch.tensor(bboxes, dtype=torch.float32)
        valid   = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        visible = valid.clone()
        return {"bbox": bbox, "valid": valid, "visible": visible}

    def get_frames(self, seq_id, frame_ids, anno=None):
        entry = self.sequences[seq_id]
        video_path = os.path.join(self.root, entry["video_path"])
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            cap.release()
            raise ValueError(f"Cannot open video: {video_path}")

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total == 0:
            cap.release()
            raise ValueError(f"Empty video: {video_path}")

        cache = {}
        for fid in sorted(set(frame_ids)):
            idx = min(fid, total - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                cache[fid] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            elif cache:
                # fall back to last successfully read frame
                cache[fid] = list(cache.values())[-1].copy()
            else:
                cap.release()
                raise ValueError(f"Cannot read any frame from: {video_path}")
        cap.release()

        frames = [cache[fid] for fid in frame_ids]
        if anno is None:
            anno = self.get_sequence_info(seq_id)
        anno_frames = {k: v[frame_ids] for k, v in anno.items()}
        obj_meta = {
            "object_class_name": "object",
            "motion_class": None,
            "major_class": None,
            "root_class": None,
            "motion_adverb": None,
        }
        return frames, anno_frames, obj_meta
'''


def write_custom_dataset(mixformer_root: str):
    dataset_dir = os.path.join(mixformer_root, "lib", "train", "dataset")
    dst = os.path.join(dataset_dir, "custom_contest.py")
    with open(dst, "w") as f:
        f.write(CUSTOM_DATASET_CODE)

    # Register in __init__.py
    init_path = os.path.join(dataset_dir, "__init__.py")
    with open(init_path) as f:
        init_src = f.read()
    if "CustomContestDataset" not in init_src:
        with open(init_path, "a") as f:
            f.write("\nfrom .custom_contest import CustomContestDataset\n")
    print("  [write] CustomContestDataset registered in MixFormer package")


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 – Register CUSTOMCONTEST in base_functions.py
# ─────────────────────────────────────────────────────────────────────────────

def register_custom_dataset(mixformer_root: str):
    path = os.path.join(mixformer_root, "lib", "train", "base_functions.py")
    with open(path) as f:
        src = f.read()

    # Extend the assert-list
    old_list = (
        '["LASOT", "GOT10K_vottrain", "GOT10K_votval", "GOT10K_train_full", '
        '"COCO17", "VID", "TRACKINGNET", "TNL2k"]'
    )
    new_list = (
        '["LASOT", "GOT10K_vottrain", "GOT10K_votval", "GOT10K_train_full", '
        '"COCO17", "VID", "TRACKINGNET", "TNL2k", "CUSTOMCONTEST"]'
    )
    src = src.replace(old_list, new_list)

    # Inject dataset branch
    if "CUSTOMCONTEST" not in src:
        old = 'if name == "LASOT":'
        new = (
            'if name == "CUSTOMCONTEST":\n'
            '            from lib.train.dataset.custom_contest import CustomContestDataset\n'
            '            datasets.append(CustomContestDataset())\n'
            '        elif name == "LASOT":'
        )
        src = src.replace(old, new, 1)

    with open(path, "w") as f:
        f.write(src)
    print("  [patch] base_functions.py — CUSTOMCONTEST branch added")


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 – Health-check videos and write filtered manifest
# ─────────────────────────────────────────────────────────────────────────────

def build_filtered_manifest(manifest_path: str, data_root: str,
                             filtered_path: str) -> dict:
    with open(manifest_path) as f:
        manifest = json.load(f)

    good, bad = {}, []
    for seq_id, entry in manifest["train"].items():
        vp = os.path.join(data_root, entry["video_path"])
        cap = cv2.VideoCapture(vp)
        ok = cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) > 10
        cap.release()
        (good if ok else bad)[seq_id] = entry if ok else None
        if not ok:
            bad.append(seq_id)

    good = {k: v for k, v in manifest["train"].items() if k not in bad}
    print(f"  [manifest] {len(good)} good | {len(bad)} skipped: {bad or 'none'}")

    manifest["train"] = good
    with open(filtered_path, "w") as f:
        json.dump(manifest, f)
    print(f"  [manifest] Filtered manifest → {filtered_path}")
    return manifest


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 – Write fine-tune YAML
# ─────────────────────────────────────────────────────────────────────────────

def write_finetune_yaml(mixformer_root: str):
    yaml_dir = os.path.join(mixformer_root, "experiments", "mixformer_cvt_online")
    os.makedirs(yaml_dir, exist_ok=True)
    yaml_path = os.path.join(yaml_dir, "finetune_contest.yaml")
    with open(yaml_path, "w") as f:
        f.write(cfg.FINETUNE_YAML.strip())
    print(f"  [yaml] Fine-tune config → {yaml_path}")
    return yaml_path


# ─────────────────────────────────────────────────────────────────────────────
# Step 6 – Patch base_trainer to load pretrained weights at start
# ─────────────────────────────────────────────────────────────────────────────

TRAINER_INJECT = '''
        # ── AIC-2026: load pretrained checkpoint for fine-tuning ──────────
        import os as _os, torch as _torch
        _ckpt_env = _os.environ.get("MIXFORMER_PRETRAIN_CKPT", "")
        if _ckpt_env and _os.path.exists(_ckpt_env):
            _ckpt  = _torch.load(_ckpt_env, map_location="cpu", weights_only=False)
            _state = _ckpt.get("net", _ckpt.get("model", _ckpt))
            _miss, _unex = self.actor.net.load_state_dict(_state, strict=False)
            print(f"[FineTune] Loaded pretrained: {_ckpt_env}")
            print(f"[FineTune] Missing keys: {len(_miss)}  Unexpected: {len(_unex)}")
            _os.environ.pop("MIXFORMER_PRETRAIN_CKPT")
        # ─────────────────────────────────────────────────────────────────
'''


def patch_base_trainer(mixformer_root: str):
    path = os.path.join(mixformer_root, "lib", "train", "trainers", "base_trainer.py")
    with open(path) as f:
        src = f.read()
    if "MIXFORMER_PRETRAIN_CKPT" in src:
        print("  [skip] base_trainer.py already patched")
        return
    sig = "def train(self, max_epochs, load_latest=False, fail_safe=True, load_previous_ckpt=False, distill=False):"
    colon_idx = src.index(":", src.index(sig))
    src = src[: colon_idx + 1] + "\n" + TRAINER_INJECT + src[colon_idx + 1 :]
    with open(path, "w") as f:
        f.write(src)
    print("  [patch] base_trainer.py — pretrain-load hook injected")


# ─────────────────────────────────────────────────────────────────────────────
# Step 7 – Run MixFormer training
# ─────────────────────────────────────────────────────────────────────────────

def run_training(mixformer_root: str, pretrain_ckpt: str, output_dir: str,
                 filtered_manifest: str, data_root: str):
    os.makedirs(os.path.join(output_dir, "checkpoints", "train",
                             "mixformer_cvt_online", "finetune_contest"), exist_ok=True)

    env = os.environ.copy()
    env["MIXFORMER_PRETRAIN_CKPT"] = pretrain_ckpt
    env["FILTERED_MANIFEST"]       = filtered_manifest
    env["DATA_ROOT"]               = data_root

    cmd = [
        sys.executable,
        os.path.join(mixformer_root, "lib", "train", "run_training.py"),
        "--script", "mixformer_cvt_online",
        "--config", "finetune_contest",
        "--save_dir", output_dir,
    ]
    print("\n[train] Launching MixFormer training …")
    print("  CMD:", " ".join(cmd))
    result = subprocess.run(cmd, cwd=mixformer_root, env=env)
    if result.returncode != 0:
        print(f"\n[ERROR] Training exited with code {result.returncode}")
        sys.exit(result.returncode)
    print("\n[train] Training complete.")


# ─────────────────────────────────────────────────────────────────────────────
# Entrypoint
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    print("\n══ AIC-2026 MixFormer Fine-Tuning ══")

    # Resolve pretrain checkpoint
    pretrain_ckpt = args.pretrain_ckpt
    if not os.path.exists(pretrain_ckpt):
        # Try to find it anywhere under CKPT_DIR
        candidates = glob.glob(
            os.path.join(cfg.CKPT_DIR, "**", "mixformer_online_22k.pth.tar"),
            recursive=True,
        )
        if candidates:
            pretrain_ckpt = candidates[0]
            print(f"  [ckpt] Auto-located pretrain ckpt: {pretrain_ckpt}")
        else:
            print(f"  [ERROR] Pretrain checkpoint not found: {pretrain_ckpt}")
            print("          Place mixformer_online_22k.pth.tar at that path "
                  "or set PRETRAIN_CKPT env var.")
            sys.exit(1)

    print(f"\n1. Patching MixFormer for PyTorch 2.x compatibility …")
    patch_torch_load()
    patch_loader(args.mixformer_root)
    patch_misc(args.mixformer_root)
    patch_trainers_misc(args.mixformer_root)

    print(f"\n2. Writing CustomContestDataset …")
    write_custom_dataset(args.mixformer_root)
    register_custom_dataset(args.mixformer_root)

    # Make sure MixFormer is importable
    if args.mixformer_root not in sys.path:
        sys.path.insert(0, args.mixformer_root)

    # Set up local file (paths inside MixFormer)
    local_cfg_script = os.path.join(
        args.mixformer_root, "tracking", "create_default_local_file.py"
    )
    if os.path.exists(local_cfg_script):
        subprocess.run(
            [sys.executable, local_cfg_script,
             "--workspace_dir", ".",
             "--data_dir", "./data",
             "--save_dir", args.output_dir],
            cwd=args.mixformer_root,
        )

    print(f"\n3. Health-checking training videos …")
    filtered = build_filtered_manifest(
        args.manifest, args.data_root, cfg.FILTERED_MANIFEST
    )

    print(f"\n4. Writing fine-tune YAML …")
    write_finetune_yaml(args.mixformer_root)

    print(f"\n5. Patching base_trainer …")
    patch_base_trainer(args.mixformer_root)

    print(f"\n6. Launching training …")
    run_training(
        mixformer_root=args.mixformer_root,
        pretrain_ckpt=pretrain_ckpt,
        output_dir=args.output_dir,
        filtered_manifest=cfg.FILTERED_MANIFEST,
        data_root=args.data_root,
    )

    # Report where the final checkpoint lives
    ckpt_pattern = os.path.join(
        args.output_dir, "checkpoints", "train",
        "mixformer_cvt_online", "finetune_contest", "*.pth.tar",
    )
    ckpts = sorted(glob.glob(ckpt_pattern))
    if ckpts:
        print(f"\n✔ Fine-tuned checkpoint saved to: {ckpts[-1]}")
        print(f"  Pass it to inference.py with --checkpoint {ckpts[-1]}")
    else:
        print("\n⚠  No checkpoint found after training. Check training logs.")


if __name__ == "__main__":
    main()
