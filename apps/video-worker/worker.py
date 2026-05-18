import json
import os
import subprocess
import tempfile
import time
import uuid
from pathlib import Path

import psycopg
import requests
from supabase import create_client


DATABASE_URL = os.environ["DATABASE_URL"]
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_ROLE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
SUPABASE_STORAGE_BUCKET = os.getenv("SUPABASE_STORAGE_BUCKET", "uploads")
OPEN_SPLAT_BIN = os.getenv("OPEN_SPLAT_BIN", "opensplat")
WORKER_POLL_SECONDS = int(os.getenv("WORKER_POLL_SECONDS", "10"))
FRAME_RATE = float(os.getenv("FRAME_RATE", "4"))
MAX_IMAGE_SIZE = int(os.getenv("MAX_IMAGE_SIZE", "1600"))
SPLAT_ITERATIONS = int(os.getenv("SPLAT_ITERATIONS", "6000"))
SPLAT_DENSIFY_GRAD_THRESH = os.getenv("SPLAT_DENSIFY_GRAD_THRESH", "0.00015")
MIN_REGISTERED_IMAGES = int(os.getenv("MIN_REGISTERED_IMAGES", "80"))
MIN_REGISTERED_IMAGE_RATIO = float(os.getenv("MIN_REGISTERED_IMAGE_RATIO", "0.35"))
MIN_SPARSE_POINTS = int(os.getenv("MIN_SPARSE_POINTS", "8000"))
MIN_SPLAT_COUNT = int(os.getenv("MIN_SPLAT_COUNT", "25000"))
SIFT_MAX_NUM_FEATURES = int(os.getenv("SIFT_MAX_NUM_FEATURES", "16384"))
SIFT_PEAK_THRESHOLD = os.getenv("SIFT_PEAK_THRESHOLD", "0.002")
SIFT_EDGE_THRESHOLD = os.getenv("SIFT_EDGE_THRESHOLD", "10")
SEQUENTIAL_MATCH_OVERLAP = int(os.getenv("SEQUENTIAL_MATCH_OVERLAP", "40"))
EXHAUSTIVE_MATCH_MAX_FRAMES = int(os.getenv("EXHAUSTIVE_MATCH_MAX_FRAMES", "120"))
COLMAP_USE_GPU = os.getenv("COLMAP_USE_GPU", "0")


class ReconstructionQualityError(RuntimeError):
    pass


def run(cmd, cwd=None):
    print("$", " ".join(str(c) for c in cmd), flush=True)
    env = os.environ.copy()
    if cmd and cmd[0] == "colmap":
        env["QT_QPA_PLATFORM"] = "offscreen"
        env.setdefault("DISPLAY", "")
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def db():
    # Supabase pooler/transaction pooling can drop server-side prepared
    # statements between statements, which surfaces as:
    # "prepared statement _pg3_0 does not exist". Disable psycopg's automatic
    # prepared statements for the long-running worker connection.
    return psycopg.connect(DATABASE_URL, autocommit=True, prepare_threshold=None)


def update_job(conn, job_id, status=None, progress=None, error=None, result=None):
    sets = ["updated_at = NOW()", "last_heartbeat_at = NOW()"]
    values = []

    if status is not None:
        sets.append("job_status = %s")
        values.append(status)
        if status == "running":
            sets.append("started_at = COALESCE(started_at, NOW())")
            sets.append("error_message = NULL")
        if status in ("succeeded", "failed"):
            sets.append("completed_at = NOW()")
        if status == "succeeded":
            sets.append("error_message = NULL")
    if progress is not None:
        sets.append("progress = %s")
        values.append(progress)
    if error is not None:
        sets.append("error_message = %s")
        values.append(error)
    if result is not None:
        sets.append("result_payload = %s::jsonb")
        values.append(json.dumps(result))

    values.append(str(job_id))
    with conn.cursor() as cur:
        cur.execute(f"UPDATE ai_jobs SET {', '.join(sets)} WHERE id = %s", values)


def claim_job(conn, job_id=None):
    with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
        if job_id:
            cur.execute(
                """
                UPDATE ai_jobs
                SET job_status = 'running',
                    started_at = COALESCE(started_at, NOW()),
                    last_heartbeat_at = NOW(),
                    progress = GREATEST(progress, 1),
                    updated_at = NOW()
                WHERE id = %s
                  AND job_type = 'video_3d_tour'
                  AND job_status = 'queued'
                RETURNING *
                """,
                [str(job_id)],
            )
        else:
            cur.execute(
                """
                UPDATE ai_jobs
                SET job_status = 'running',
                    started_at = COALESCE(started_at, NOW()),
                    last_heartbeat_at = NOW(),
                    progress = GREATEST(progress, 1),
                    updated_at = NOW()
                WHERE id = (
                  SELECT id
                  FROM ai_jobs
                  WHERE job_type = 'video_3d_tour'
                    AND job_status = 'queued'
                  ORDER BY created_at ASC
                  LIMIT 1
                  FOR UPDATE SKIP LOCKED
                )
                RETURNING *
                """
            )
        return cur.fetchone()


def refund_credits_if_needed(conn, job, reason):
    credits = float(job.get("credits_reserved") or 0)
    if credits <= 0:
        return

    with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
        cur.execute(
            """
            SELECT id
            FROM credit_transactions
            WHERE user_id = %s
              AND transaction_type = 'refund'
              AND meta->>'jobId' = %s
            LIMIT 1
            """,
            [job["user_id"], str(job["id"])],
        )
        if cur.fetchone():
            return

        cur.execute(
            """
            WITH ensured_wallet AS (
              INSERT INTO credits_wallet (user_id, balance_credits)
              VALUES (%s, 0)
              ON CONFLICT (user_id) DO NOTHING
            ),
            updated_wallet AS (
              UPDATE credits_wallet
              SET balance_credits = balance_credits + %s,
                  updated_at = NOW()
              WHERE user_id = %s
              RETURNING balance_credits
            ),
            inserted_tx AS (
              INSERT INTO credit_transactions (
                user_id,
                transaction_type,
                credits_delta,
                provider,
                meta
              )
              VALUES (%s, 'refund', %s, 'video-worker', %s::jsonb)
              RETURNING id
            )
            SELECT 1 AS ok
            """,
            [
                job["user_id"],
                credits,
                job["user_id"],
                job["user_id"],
                credits,
                json.dumps({"jobId": str(job["id"]), "reason": reason}),
            ],
        )


def download_video(url, out_path):
    with requests.get(url, stream=True, timeout=120) as res:
        res.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in res.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def extract_frames(video_path, images_dir):
    images_dir.mkdir(parents=True, exist_ok=True)
    vf = f"fps={FRAME_RATE},scale='min({MAX_IMAGE_SIZE},iw)':-2,setsar=1"
    run([
        "ffmpeg",
        "-nostdin",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        vf,
        "-q:v",
        "2",
        str(images_dir / "frame_%06d.jpg"),
    ])


def run_colmap(images_dir, work_dir):
    db_path = work_dir / "colmap.db"
    sparse_dir = work_dir / "sparse"
    sparse_dir.mkdir(parents=True, exist_ok=True)
    frame_count = count_extracted_frames(images_dir)

    run([
        "colmap",
        "feature_extractor",
        "--database_path",
        str(db_path),
        "--image_path",
        str(images_dir),
        "--ImageReader.single_camera",
        "1",
        "--ImageReader.camera_model",
        "SIMPLE_RADIAL",
        "--SiftExtraction.use_gpu",
        COLMAP_USE_GPU,
        "--SiftExtraction.max_num_features",
        str(SIFT_MAX_NUM_FEATURES),
        "--SiftExtraction.peak_threshold",
        str(SIFT_PEAK_THRESHOLD),
        "--SiftExtraction.edge_threshold",
        str(SIFT_EDGE_THRESHOLD),
    ])
    run([
        "colmap",
        "sequential_matcher",
        "--database_path",
        str(db_path),
        "--SiftMatching.use_gpu",
        COLMAP_USE_GPU,
        "--SequentialMatching.overlap",
        str(SEQUENTIAL_MATCH_OVERLAP),
    ])
    if frame_count <= EXHAUSTIVE_MATCH_MAX_FRAMES:
        run([
            "colmap",
            "exhaustive_matcher",
            "--database_path",
            str(db_path),
            "--SiftMatching.use_gpu",
            COLMAP_USE_GPU,
            "--SiftMatching.guided_matching",
            "1",
        ])
    run([
        "colmap",
        "mapper",
        "--database_path",
        str(db_path),
        "--image_path",
        str(images_dir),
        "--output_path",
        str(sparse_dir),
        "--Mapper.min_num_matches",
        "12",
        "--Mapper.init_min_num_inliers",
        "50",
        "--Mapper.abs_pose_min_num_inliers",
        "15",
    ])

    model_dirs = [path for path in sparse_dir.iterdir() if path.is_dir()]
    if not model_dirs:
        raise RuntimeError("COLMAP did not produce a sparse model.")

    # COLMAP can split a walkthrough into multiple sparse reconstructions.
    # OpenSplat quality depends heavily on choosing the richest reconstruction,
    # and COLMAP does not guarantee that it is sparse/0.
    def model_weight(path):
        points = path / "points3D.bin"
        images = path / "images.bin"
        return (
            points.stat().st_size if points.exists() else 0,
            images.stat().st_size if images.exists() else 0,
        )

    return max(model_dirs, key=model_weight)


def count_extracted_frames(images_dir):
    return sum(1 for _ in images_dir.glob("*.jpg"))


def analyze_colmap_model(model_dir, work_dir):
    text_dir = work_dir / "colmap_text"
    text_dir.mkdir(parents=True, exist_ok=True)
    run([
        "colmap",
        "model_converter",
        "--input_path",
        str(model_dir),
        "--output_path",
        str(text_dir),
        "--output_type",
        "TXT",
    ])

    images_txt = text_dir / "images.txt"
    points_txt = text_dir / "points3D.txt"
    registered_images = 0
    sparse_points = 0

    if images_txt.exists():
        for line in images_txt.read_text(errors="ignore").splitlines():
            if line.startswith("#"):
                continue
            if ".jpg" in line.lower() or ".jpeg" in line.lower() or ".png" in line.lower():
                registered_images += 1

    if points_txt.exists():
        for line in points_txt.read_text(errors="ignore").splitlines():
            if line.strip() and not line.startswith("#"):
                sparse_points += 1

    return {
        "registeredImages": registered_images,
        "sparsePoints": sparse_points,
    }


def read_ply_vertex_count(path):
    with open(path, "rb") as f:
        for raw in f:
            line = raw.decode("utf-8", errors="ignore").strip()
            if line.startswith("element vertex "):
                try:
                    return int(line.split()[-1])
                except ValueError:
                    return 0
            if line == "end_header":
                break
    return 0


def assert_reconstruction_quality(frame_count, stats, splat_count):
    registered_images = int(stats.get("registeredImages") or 0)
    sparse_points = int(stats.get("sparsePoints") or 0)
    registered_ratio = registered_images / frame_count if frame_count else 0

    failures = []
    if registered_images < MIN_REGISTERED_IMAGES:
        failures.append(f"only {registered_images} usable frames")
    if registered_ratio < MIN_REGISTERED_IMAGE_RATIO:
        failures.append(f"only {registered_ratio:.0%} of frames aligned")
    if sparse_points < MIN_SPARSE_POINTS:
        failures.append(f"only {sparse_points} stable scene points")
    if splat_count < MIN_SPLAT_COUNT:
        failures.append(f"only {splat_count} render points")

    if failures:
        raise ReconstructionQualityError(
            "This video does not contain enough stable visual overlap for a reliable 3D tour. "
            "Please record a slower, brighter walkthrough with more overlap between views. "
            f"Quality checks failed: {', '.join(failures)}."
        )

    return {
        "frameCount": frame_count,
        "registeredImages": registered_images,
        "registeredImageRatio": registered_ratio,
        "sparsePoints": sparse_points,
        "splatCount": splat_count,
    }


def run_opensplat(model_dir, images_dir, output_path):
    run([
        OPEN_SPLAT_BIN,
        str(model_dir),
        "--colmap-image-path",
        str(images_dir),
        "-o",
        str(output_path),
        "-n",
        str(SPLAT_ITERATIONS),
        "--densify-grad-thresh",
        str(SPLAT_DENSIFY_GRAD_THRESH),
    ])
    if not output_path.exists():
        raise RuntimeError("OpenSplat did not produce an output file.")


def upload_result(path):
    supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
    object_path = f"video-3d-tours/{time.strftime('%Y-%m-%d')}/{uuid.uuid4()}{path.suffix}"
    with open(path, "rb") as f:
        supabase.storage.from_(SUPABASE_STORAGE_BUCKET).upload(
            object_path,
            f.read(),
            {"content-type": "model/vnd.ply" if path.suffix == ".ply" else "application/octet-stream"},
        )
    return supabase.storage.from_(SUPABASE_STORAGE_BUCKET).get_public_url(object_path)


def save_virtual_tour(conn, job, file_url, output_format, quality):
    payload = {
        "type": "splat3d",
        "fileUrl": file_url,
        "format": output_format,
        "sourceType": "original",
        "generatedFrom": "iphone_video",
        "jobId": str(job["id"]),
        "quality": {
            **quality,
            "profile": "validated",
        },
    }
    with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
        cur.execute(
            "SELECT id FROM virtual_tours WHERE property_id = %s AND source_type = 'original' LIMIT 1",
            [job["property_id"]],
        )
        existing = cur.fetchone()
        if existing:
            cur.execute(
                """
                UPDATE virtual_tours
                SET base_staging_id = NULL,
                    source_type = 'original',
                    staging_type = NULL,
                    tour_type = 'splat3d',
                    tour_payload = %s::jsonb
                WHERE id = %s
                RETURNING id
                """,
                [json.dumps(payload), existing["id"]],
            )
        else:
            cur.execute(
                """
                INSERT INTO virtual_tours (
                  property_id, base_staging_id, source_type, staging_type, tour_type, tour_payload
                )
                VALUES (%s, NULL, 'original', NULL, 'splat3d', %s::jsonb)
                RETURNING id
                """,
                [job["property_id"], json.dumps(payload)],
            )
        return cur.fetchone()["id"]


def process_job(conn, job):
    request_payload = job["request_payload"] or {}
    video_url = request_payload.get("videoUrl")
    if not video_url:
        raise RuntimeError("Job is missing request_payload.videoUrl.")

    with tempfile.TemporaryDirectory(prefix="video-3d-tour-") as tmp:
        work_dir = Path(tmp)
        video_path = work_dir / "input_video"
        images_dir = work_dir / "images"
        output_path = work_dir / "tour.ply"

        update_job(conn, job["id"], progress=5)
        download_video(video_url, video_path)

        update_job(conn, job["id"], progress=15)
        extract_frames(video_path, images_dir)
        frame_count = count_extracted_frames(images_dir)

        update_job(conn, job["id"], progress=35)
        model_dir = run_colmap(images_dir, work_dir)
        model_stats = analyze_colmap_model(model_dir, work_dir)

        update_job(conn, job["id"], progress=70)
        run_opensplat(model_dir, images_dir, output_path)
        splat_count = read_ply_vertex_count(output_path)
        quality = assert_reconstruction_quality(frame_count, model_stats, splat_count)

        update_job(conn, job["id"], progress=88)
        file_url = upload_result(output_path)

        update_job(conn, job["id"], progress=95)
        tour_id = save_virtual_tour(conn, job, file_url, "ply", quality)

        update_job(
            conn,
            job["id"],
            status="succeeded",
            progress=100,
            result={"tourId": str(tour_id), "fileUrl": file_url, "quality": quality},
        )


def main():
    once_job_id = os.getenv("PROCESS_JOB_ID")
    with db() as conn:
      while True:
          job = claim_job(conn, once_job_id)
          if not job:
              if once_job_id:
                  print(f"No claimable job: {once_job_id}", flush=True)
                  return
              time.sleep(WORKER_POLL_SECONDS)
              continue

          print(f"Processing video_3d_tour job {job['id']}", flush=True)
          try:
              process_job(conn, job)
          except Exception as exc:
              print(f"Job failed: {exc}", flush=True)
              try:
                  refund_credits_if_needed(conn, job, str(exc))
              except Exception as refund_exc:
                  print(f"Credit refund failed: {refund_exc}", flush=True)
              update_job(conn, job["id"], status="failed", error=str(exc))

          if once_job_id:
              return


if __name__ == "__main__":
    main()
