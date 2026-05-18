#!/usr/bin/env bash
set -euo pipefail

ENV_FILE="${ENV_FILE:-/workspace/runpod.env}"
REPO_URL="${REPO_URL:-https://github.com/sasmazt90/ai-virtual-tour-engine.git}"
REPO_DIR="${REPO_DIR:-/workspace/ai-virtual-tour-engine}"
OPEN_SPLAT_DIR="${OPEN_SPLAT_DIR:-/workspace/OpenSplat}"

if [ -z "${CUDA_ARCHITECTURES:-}" ]; then
  DETECTED_COMPUTE_CAP="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n 1 | tr -d ' .' || true)"
  if [ -n "$DETECTED_COMPUTE_CAP" ]; then
    CUDA_ARCHITECTURES="$DETECTED_COMPUTE_CAP"
  else
    CUDA_ARCHITECTURES="86"
  fi
fi

if [ -f "$ENV_FILE" ]; then
  set -a
  source "$ENV_FILE"
  set +a
elif [ -z "${DATABASE_URL:-}" ] || [ -z "${SUPABASE_URL:-}" ] || [ -z "${SUPABASE_SERVICE_ROLE_KEY:-}" ]; then
  echo "Missing env file: $ENV_FILE"
  echo "Create it from apps/video-worker/runpod.env.example and fill the real values, or pass DATABASE_URL, SUPABASE_URL, and SUPABASE_SERVICE_ROLE_KEY as environment variables."
  exit 1
fi

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends \
  build-essential \
  ca-certificates \
  cmake \
  colmap \
  ffmpeg \
  git \
  libopencv-dev \
  ninja-build \
  python3-pip \
  xvfb

if [ ! -d "$REPO_DIR/.git" ]; then
  git clone "$REPO_URL" "$REPO_DIR"
else
  git -C "$REPO_DIR" fetch origin main
  git -C "$REPO_DIR" reset --hard origin/main
fi

echo "Worker repository commit: $(git -C "$REPO_DIR" rev-parse --short HEAD)"

pip3 install --break-system-packages --no-cache-dir -r "$REPO_DIR/apps/video-worker/requirements.txt"

if [ ! -x "$OPEN_SPLAT_DIR/build/opensplat" ]; then
  if [ ! -d "$OPEN_SPLAT_DIR/.git" ]; then
    git clone --depth 1 https://github.com/pierotofy/OpenSplat "$OPEN_SPLAT_DIR"
  else
    git -C "$OPEN_SPLAT_DIR" pull --ff-only
  fi

  TORCH_CMAKE_PREFIX="$(python3 - <<'PY'
import torch
print(torch.utils.cmake_prefix_path)
PY
)"

  cmake -S "$OPEN_SPLAT_DIR" -B "$OPEN_SPLAT_DIR/build" -GNinja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH="$TORCH_CMAKE_PREFIX" \
    -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCHITECTURES"
  cmake --build "$OPEN_SPLAT_DIR/build" --config Release
fi

export QT_QPA_PLATFORM=offscreen
unset DISPLAY
export OPEN_SPLAT_BIN="$OPEN_SPLAT_DIR/build/opensplat"
python3 "$REPO_DIR/apps/video-worker/worker.py"
