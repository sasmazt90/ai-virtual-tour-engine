#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-360-estate-video-worker:latest}"
ENV_FILE="${ENV_FILE:-/workspace/runpod.env}"
REPO_URL="${REPO_URL:-https://github.com/sasmazt90/ai-virtual-tour-engine.git}"
REPO_DIR="${REPO_DIR:-/workspace/ai-virtual-tour-engine}"

if [ ! -d "$REPO_DIR/.git" ]; then
  git clone "$REPO_URL" "$REPO_DIR"
else
  git -C "$REPO_DIR" pull --ff-only
fi

if [ ! -f "$ENV_FILE" ]; then
  echo "Missing env file: $ENV_FILE"
  echo "Create it from apps/video-worker/runpod.env.example and fill the real values."
  exit 1
fi

docker build -t "$IMAGE_NAME" "$REPO_DIR/apps/video-worker"
docker run --rm --gpus all --env-file "$ENV_FILE" "$IMAGE_NAME"
