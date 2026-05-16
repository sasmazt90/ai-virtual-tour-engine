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
FRAME_RATE=2
MAX_IMAGE_SIZE=1600
SPLAT_ITERATIONS=2000
```

`OPEN_SPLAT_BIN` must point to an OpenSplat binary available in the container or host.
COLMAP and ffmpeg are installed in the Docker image.

This worker needs real compute. For good results use a GPU machine. CPU mode may work for small tests but can be slow.

## Cheapest practical start

Use a pay-as-you-go NVIDIA GPU pod only while a video is being processed, then stop it.
For the first production test, pick an RTX 4090/A10/L4 class machine with 16GB+ VRAM.

1. Build or start a container that has this worker, `ffmpeg`, `colmap`, and OpenSplat installed.
2. Set the environment variables above.
3. Upload an iPhone walkthrough video from the web app with `+ iPhone Video`.
4. Start the worker. It will pick the queued `video_3d_tour` job from Supabase.
5. Stop the GPU pod when the job status becomes `succeeded` or `failed`.

OpenSplat expects a COLMAP-style project folder. The worker extracts frames into `images/`,
creates `sparse/0` with COLMAP, then runs:

```bash
opensplat /work/project -o tour.ply -n 2000
```
