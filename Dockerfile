# ──────────────────────────────────────────────────────────────────────────
# AIC-2026 MixFormer Tracker — Inference Dockerfile
#
# Build:
#   docker build -t aic2026-tracker .
#
# Run:
#   docker run --gpus all \
#     -v /path/to/contest_release:/app/data/contest_release:ro \
#     -v /path/to/checkpoints:/app/checkpoints:ro \
#     -v /path/to/output:/app/output \
#     -e FINETUNED_CKPT=/app/checkpoints/best_model.pth.tar \
#     -e MANIFEST_PATH=/app/data/contest_release/metadata/contestant_manifest.json \
#     -e DATA_ROOT=/app/data/contest_release \
#     -e OUTPUT_CSV=/app/output/submission.csv \
#     aic2026-tracker
# ──────────────────────────────────────────────────────────────────────────

FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# ── System libraries required by OpenCV ──────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ───────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Clone MixFormer (pinned commit for reproducibility) ───────────────────
RUN git clone https://github.com/MCG-NJU/MixFormer.git /app/MixFormer

# ── Copy project files ────────────────────────────────────────────────────
COPY config.py    /app/config.py
COPY inference.py /app/inference.py
COPY train.py     /app/train.py

# ── Environment defaults (override at runtime via -e flags) ───────────────
ENV MIXFORMER_ROOT=/app/MixFormer
ENV DATA_ROOT=/app/data/contest_release
ENV MANIFEST_PATH=/app/data/contest_release/metadata/contestant_manifest.json
ENV OUTPUT_CSV=/app/output/submission.csv
ENV FINETUNED_CKPT=/app/checkpoints/best_model.pth.tar

# ── Initialise MixFormer workspace (creates local path config files) ───────
RUN python /app/MixFormer/tracking/create_default_local_file.py \
        --workspace_dir /app/MixFormer \
        --data_dir /app/MixFormer/data \
        --save_dir /app/MixFormer/output \
    || true   # non-fatal: script may not exist on all MixFormer versions

# ── Default command: run inference ────────────────────────────────────────
CMD ["python", "inference.py"]
