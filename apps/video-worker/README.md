# Video to 3D Tour Worker

This worker turns an uploaded iPhone walkthrough video into a Gaussian Splat virtual tour.

Pipeline:

1. Poll `ai_jobs` for `job_type = 'video_3d_tour'`.
2. Download the uploaded `.mp4/.mov`.
3. Extract frames with `ffmpeg`.
4. Reconstruct camera poses with `COLMAP`.
5. Train/export a Gaussian Splat with `OpenSplat`.
6. Upload the `.ply/.splat` result to Supabase Storage.
7. Save or replace the property's `splat3d` virtual tour.
8. Mark the job succeeded or failed.

Required environment variables:

```text
DATABASE_URL=postgresql://...
SUPABASE_URL=https://...
SUPABASE_SERVICE_ROLE_KEY=...
SUPABASE_STORAGE_BUCKET=uploads
VIDEO_UPLOAD_S3_ENDPOINT=https://s3api-DATACENTER.runpod.io
VIDEO_UPLOAD_S3_REGION=DATACENTER
VIDEO_UPLOAD_S3_BUCKET=RUNPOD_NETWORK_VOLUME_ID
VIDEO_UPLOAD_S3_ACCESS_KEY_ID=...
VIDEO_UPLOAD_S3_SECRET_ACCESS_KEY=...
OPEN_SPLAT_BIN=/usr/local/bin/opensplat
WORKER_POLL_SECONDS=10
FRAME_RATE=4
MAX_IMAGE_SIZE=1600
SPLAT_ITERATIONS=10000
SPLAT_DENSIFY_GRAD_THRESH=0.00012
MIN_REGISTERED_IMAGES=90
MIN_REGISTERED_IMAGE_RATIO=0.5
MIN_SPARSE_POINTS=14000
MIN_SPLAT_COUNT=70000
MAX_SCENE_FRAMES=360
EXHAUSTIVE_MATCH_MAX_FRAMES=220
RUN_EXHAUSTIVE_MATCHING=0
```

`OPEN_SPLAT_BIN` must point to an OpenSplat binary available in the container or host.
COLMAP and ffmpeg are installed in the Docker image.

This worker needs real compute. For good results use a GPU machine. CPU mode may work for small tests but can be slow.

## Cheapest practical start

Use a pay-as-you-go NVIDIA GPU pod only while a video is being processed, then stop it.
For the first production test, pick an RTX 4090/A10/L4 class machine with 16GB+ VRAM.

The Dockerfile builds OpenSplat into the worker image. On RunPod, choose a GPU pod with
Docker enabled and a persistent volume mounted at `/workspace`.

1. Upload an iPhone walkthrough video from the web app with `+ iPhone Video`.
2. Create `/workspace/runpod.env` from `runpod.env.example` and fill the real values.
3. If Docker is available on the pod, start the worker:

```bash
apt-get update && apt-get install -y git
bash <(curl -fsSL https://raw.githubusercontent.com/sasmazt90/ai-virtual-tour-engine/main/apps/video-worker/scripts/runpod-build-and-run.sh)
```

If the template does not include Docker, run it directly in the GPU pod:

```bash
apt-get update && apt-get install -y curl
bash <(curl -fsSL https://raw.githubusercontent.com/sasmazt90/ai-virtual-tour-engine/main/apps/video-worker/scripts/runpod-native-run.sh)
```

4. Watch the logs until the job status becomes `succeeded` or `failed`.
5. Stop the GPU pod after processing. This is what keeps the cost low.

OpenSplat expects COLMAP camera poses and the extracted images. The worker extracts
frames into `images/`, lets COLMAP create one or more `sparse/*` models, chooses the
model with the most reconstructed points, then runs:

```bash
opensplat /work/project/sparse/1 --colmap-image-path /work/project/images -o tour.ply -n 6000
```

After reconstruction, the worker validates the output before saving it. A job fails
instead of publishing a broken tour if too few video frames align, if the sparse
scene is weak, or if the exported splat has too few renderable points.

Sequential matching is the default because it is predictable for walkthrough
videos and finishes in a practical time on rented GPUs. Set
`RUN_EXHAUSTIVE_MATCHING=1` only for short clips when you intentionally want the
slower all-to-all matching pass.
