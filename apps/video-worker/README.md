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
OPEN_SPLAT_BIN=/usr/local/bin/opensplat
WORKER_POLL_SECONDS=10
FRAME_RATE=4
MAX_IMAGE_SIZE=1600
SPLAT_ITERATIONS=6000
SPLAT_DENSIFY_GRAD_THRESH=0.00015
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
