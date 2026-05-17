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
FRAME_RATE = float(os.getenv("FRAME_RATE", "2"))
MAX_IMAGE_SIZE = int(os.getenv("MAX_IMAGE_SIZE", "1600"))
SPLAT_ITERATIONS = int(os.getenv("SPLAT_ITERATIONS", "2000"))


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
                  AND job_status IN ('queued', 'failed')
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


def download_video(url, out_path):
    with requests.get(url, stream=True, timeout=120) as res:
        res.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in res.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def extract_frames(video_path, images_dir):
    images_dir.mkdir(parents=True, exist_ok=True)
    vf = f"fps={FRAME_RATE},scale='min({MAX_IMAGE_SIZE},iw)':-2"
    run([
        "ffmpeg",
        "-nostdin",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        vf,
        str(images_dir / "frame_%06d.jpg"),
    ])


def run_colmap(images_dir, work_dir):
    db_path = work_dir / "colmap.db"
    sparse_dir = work_dir / "sparse"
    sparse_dir.mkdir(parents=True, exist_ok=True)

    run([
        "colmap",
        "feature_extractor",
        "--database_path",
        str(db_path),
        "--image_path",
        str(images_dir),
        "--ImageReader.single_camera",
        "1",
        "--SiftExtraction.use_gpu",
        "0",
    ])
    run([
        "colmap",
        "exhaustive_matcher",
        "--database_path",
        str(db_path),
        "--SiftMatching.use_gpu",
        "0",
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
    ])

    model_dir = sparse_dir / "0"
    if not model_dir.exists():
        raise RuntimeError("COLMAP did not produce a sparse model.")
    return model_dir


def run_opensplat(project_dir, output_path):
    run([
        OPEN_SPLAT_BIN,
        str(project_dir),
        "-o",
        str(output_path),
        "-n",
        str(SPLAT_ITERATIONS),
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


def save_virtual_tour(conn, job, file_url, output_format):
    payload = {
        "type": "splat3d",
        "fileUrl": file_url,
        "format": output_format,
        "sourceType": "original",
        "generatedFrom": "iphone_video",
        "jobId": str(job["id"]),
        "camera": {
            "up": [0, -1, -0.6],
            "position": [-1, -4, 6],
            "lookAt": [0, 0, 0],
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

        update_job(conn, job["id"], progress=35)
        run_colmap(images_dir, work_dir)

        update_job(conn, job["id"], progress=70)
        run_opensplat(work_dir, output_path)

        update_job(conn, job["id"], progress=88)
        file_url = upload_result(output_path)

        update_job(conn, job["id"], progress=95)
        tour_id = save_virtual_tour(conn, job, file_url, "ply")

        update_job(
            conn,
            job["id"],
            status="succeeded",
            progress=100,
            result={"tourId": str(tour_id), "fileUrl": file_url},
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
              update_job(conn, job["id"], status="failed", error=str(exc))

          if once_job_id:
              return


if __name__ == "__main__":
    main()
