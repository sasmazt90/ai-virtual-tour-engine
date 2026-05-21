import json
import hashlib
import math
import os
import re
import shutil
import struct
import subprocess
import tempfile
import time
import uuid
from pathlib import Path

import psycopg
import requests


DATABASE_URL = os.environ["DATABASE_URL"]
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_ROLE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
SUPABASE_STORAGE_BUCKET = os.getenv("SUPABASE_STORAGE_BUCKET", "uploads")
OPEN_SPLAT_BIN = os.getenv("OPEN_SPLAT_BIN", "opensplat")
WORKER_POLL_SECONDS = int(os.getenv("WORKER_POLL_SECONDS", "10"))
FRAME_RATE = float(os.getenv("FRAME_RATE", "4"))
MAX_IMAGE_SIZE = int(os.getenv("MAX_IMAGE_SIZE", "1600"))
SPLAT_ITERATIONS = int(os.getenv("SPLAT_ITERATIONS", "10000"))
SPLAT_DENSIFY_GRAD_THRESH = os.getenv("SPLAT_DENSIFY_GRAD_THRESH", "0.00012")
MIN_REGISTERED_IMAGES = int(os.getenv("MIN_REGISTERED_IMAGES", "90"))
MIN_REGISTERED_IMAGE_RATIO = float(os.getenv("MIN_REGISTERED_IMAGE_RATIO", "0.5"))
MIN_SPARSE_POINTS = int(os.getenv("MIN_SPARSE_POINTS", "14000"))
MIN_SPLAT_COUNT = int(os.getenv("MIN_SPLAT_COUNT", "70000"))
MIN_CAMERA_BASELINE = float(os.getenv("MIN_CAMERA_BASELINE", "0.18"))
MIN_CAMERA_SPREAD_RATIO = float(os.getenv("MIN_CAMERA_SPREAD_RATIO", "0.04"))
MIN_OPAQUE_SPLAT_RATIO = float(os.getenv("MIN_OPAQUE_SPLAT_RATIO", "0.18"))
MAX_SPLAT_SCALE_P95 = float(os.getenv("MAX_SPLAT_SCALE_P95", "0.065"))
MAX_SPLAT_OUTLIER_RATIO = float(os.getenv("MAX_SPLAT_OUTLIER_RATIO", "3.5"))
MAX_OUTPUT_SPLATS = int(os.getenv("MAX_OUTPUT_SPLATS", "220000"))
MAX_OUTPUT_FILE_MB = float(os.getenv("MAX_OUTPUT_FILE_MB", "45"))
PLY_MIN_OPACITY = float(os.getenv("PLY_MIN_OPACITY", "0.32"))
PLY_MAX_SCALE = float(os.getenv("PLY_MAX_SCALE", "0.065"))
PLY_OUTLIER_QUANTILE_LOW = float(os.getenv("PLY_OUTLIER_QUANTILE_LOW", "0.02"))
PLY_OUTLIER_QUANTILE_HIGH = float(os.getenv("PLY_OUTLIER_QUANTILE_HIGH", "0.98"))
MIN_SCENE_FRAMES = int(os.getenv("MIN_SCENE_FRAMES", "45"))
MAX_SCENE_FRAMES = int(os.getenv("MAX_SCENE_FRAMES", "220"))
FRAME_QUALITY_DROP_RATIO = float(os.getenv("FRAME_QUALITY_DROP_RATIO", "0.12"))
MAX_SCENES_PER_TOUR = int(os.getenv("MAX_SCENES_PER_TOUR", "8"))
SIFT_MAX_NUM_FEATURES = int(os.getenv("SIFT_MAX_NUM_FEATURES", "8192"))
SIFT_PEAK_THRESHOLD = os.getenv("SIFT_PEAK_THRESHOLD", "0.002")
SIFT_EDGE_THRESHOLD = os.getenv("SIFT_EDGE_THRESHOLD", "10")
SEQUENTIAL_MATCH_OVERLAP = int(os.getenv("SEQUENTIAL_MATCH_OVERLAP", "20"))
EXHAUSTIVE_MATCH_MAX_FRAMES = int(os.getenv("EXHAUSTIVE_MATCH_MAX_FRAMES", "220"))
RUN_EXHAUSTIVE_MATCHING = os.getenv("RUN_EXHAUSTIVE_MATCHING", "0").lower() in ("1", "true", "yes")
COLMAP_USE_GPU = os.getenv("COLMAP_USE_GPU", "0")


class ReconstructionQualityError(RuntimeError):
    pass


def run(cmd, cwd=None, heartbeat=None):
    run_cmd = [str(c) for c in cmd]
    env = os.environ.copy()
    if run_cmd and run_cmd[0] == "colmap":
        if shutil.which("xvfb-run"):
            env["QT_QPA_PLATFORM"] = "xcb"
            run_cmd = ["xvfb-run", "-a", "--server-args=-screen 0 1280x1024x24", *run_cmd]
        else:
            env["QT_QPA_PLATFORM"] = "offscreen"
            env.pop("DISPLAY", None)
    print("$", " ".join(str(c) for c in run_cmd), flush=True)
    process = subprocess.Popen(run_cmd, cwd=cwd, env=env)
    last_heartbeat = time.monotonic()
    while True:
        return_code = process.poll()
        if return_code is not None:
            if return_code:
                raise subprocess.CalledProcessError(return_code, run_cmd)
            return
        if heartbeat and time.monotonic() - last_heartbeat >= 15:
            heartbeat()
            last_heartbeat = time.monotonic()
        time.sleep(1)


def quantile(values, q, fallback=0.0):
    clean = sorted(v for v in values if isinstance(v, (int, float)) and math.isfinite(v))
    if not clean:
        return fallback
    index = max(0, min(len(clean) - 1, int(len(clean) * q)))
    return clean[index]


def vector_norm(values):
    return math.sqrt(sum(float(v) * float(v) for v in values))


def matrix_transpose_multiply(matrix, vector):
    return [
        sum(matrix[row][col] * vector[row] for row in range(3))
        for col in range(3)
    ]


def vector_add(left, right):
    return [float(left[i]) + float(right[i]) for i in range(3)]


def vector_scale(values, scale):
    return [float(value) * scale for value in values]


def quaternion_to_rotation(qw, qx, qy, qz):
    norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if norm <= 0:
        return None
    qw, qx, qy, qz = qw / norm, qx / norm, qy / norm, qz / norm
    return [
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ]


def camera_center(rotation, translation):
    # COLMAP stores world-to-camera pose. The camera center is -R^T * t.
    return [
        -sum(rotation[row][col] * translation[row] for row in range(3))
        for col in range(3)
    ]


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


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_frames(video_path, images_dir, pattern="frame_%06d.jpg"):
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
        str(images_dir / pattern),
    ])


def extract_video_set(video_items, work_dir, images_dir):
    video_count = len(video_items)
    seen_hashes = set()
    extracted = 0
    for index, item in enumerate(video_items, start=1):
        video_url = item["videoUrl"]
        video_path = work_dir / f"input_video_{index:03d}"
        print(f"Downloading clip {index}/{video_count}", flush=True)
        download_video(video_url, video_path)
        content_hash = file_sha256(video_path)
        if content_hash in seen_hashes:
            print(f"Skipping duplicate clip {index}/{video_count}", flush=True)
            continue
        seen_hashes.add(content_hash)
        extracted += 1
        update_pattern = f"clip{extracted:03d}_frame_%06d.jpg"
        print(f"Extracting frames from clip {index}/{video_count}", flush=True)
        extract_frames(video_path, images_dir, update_pattern)


def scene_key_from_video(item):
    name = str(item.get("originalName") or item.get("name") or "").strip()
    if not name:
        return "main"

    stem = Path(name).stem.lower()
    stem = re.sub(r"[-_\s]*angle[-_\s]*\d+$", "", stem)
    stem = re.sub(r"[-_\s]*(clip|video|take)[-_\s]*\d+$", "", stem)
    stem = re.sub(r"[^a-z0-9]+", "-", stem).strip("-")
    return stem or "main"


def title_from_scene_key(key):
    return " ".join(part.capitalize() for part in str(key or "Area").split("-") if part) or "Area"


def group_video_items(video_items):
    groups = []
    by_key = {}

    for item in video_items:
        key = scene_key_from_video(item)
        if key not in by_key:
            group = {
                "key": key,
                "title": title_from_scene_key(key),
                "videos": [],
            }
            by_key[key] = group
            groups.append(group)
        by_key[key]["videos"].append(item)

    # A single clip rarely has enough parallax for a reliable 3D reconstruction.
    # If every clip has a different name, keep them together instead of creating
    # many weak one-video scenes.
    if len(groups) > 1 and all(len(group["videos"]) == 1 for group in groups):
        return [{"key": "main", "title": "Property", "videos": video_items}]

    reliable_groups = [group for group in groups if len(group["videos"]) > 1]
    if reliable_groups:
        return reliable_groups[:MAX_SCENES_PER_TOUR]

    return groups[:MAX_SCENES_PER_TOUR]


def run_colmap(images_dir, work_dir, heartbeat=None):
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
    ], heartbeat=heartbeat)
    run([
        "colmap",
        "sequential_matcher",
        "--database_path",
        str(db_path),
        "--SiftMatching.use_gpu",
        COLMAP_USE_GPU,
        "--SequentialMatching.overlap",
        str(SEQUENTIAL_MATCH_OVERLAP),
    ], heartbeat=heartbeat)
    if RUN_EXHAUSTIVE_MATCHING and frame_count <= EXHAUSTIVE_MATCH_MAX_FRAMES:
        run([
            "colmap",
            "exhaustive_matcher",
            "--database_path",
            str(db_path),
            "--SiftMatching.use_gpu",
            COLMAP_USE_GPU,
            "--SiftMatching.guided_matching",
            "1",
        ], heartbeat=heartbeat)
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
        "--Mapper.ba_refine_focal_length",
        "1",
        "--Mapper.ba_refine_extra_params",
        "1",
    ], heartbeat=heartbeat)

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


def clip_key_from_frame(frame):
    match = re.match(r"clip(\d+)_frame_\d+\.jpg$", frame.name)
    return match.group(1) if match else "main"


def select_representative_frames(frames, target_count):
    if target_count <= 0 or len(frames) <= target_count:
        return set(frames)

    weighted_frames = [
        (frame, frame.stat().st_size if frame.exists() else 0)
        for frame in frames
    ]
    weighted_frames.sort(key=lambda item: item[1])

    drop_count = int(len(weighted_frames) * FRAME_QUALITY_DROP_RATIO)
    candidates = [frame for frame, _ in weighted_frames[drop_count:]] or frames
    candidates = sorted(candidates)

    keep = set()
    for index in range(target_count):
        start = round(index * len(candidates) / target_count)
        end = round((index + 1) * len(candidates) / target_count)
        bucket = candidates[start:max(start + 1, end)]
        keep.add(max(bucket, key=lambda path: path.stat().st_size if path.exists() else 0))
    return keep


def limit_scene_frames(images_dir, max_frames=MAX_SCENE_FRAMES):
    frames = sorted(images_dir.glob("*.jpg"))
    frame_count = len(frames)
    if max_frames <= 0 or frame_count <= max_frames:
        return frame_count

    by_clip = {}
    for frame in frames:
        by_clip.setdefault(clip_key_from_frame(frame), []).append(frame)

    clip_count = max(1, len(by_clip))
    base_per_clip = max(1, max_frames // clip_count)
    remainder = max_frames % clip_count

    keep = set()
    for clip_index, clip_key in enumerate(sorted(by_clip.keys())):
        clip_frames = sorted(by_clip[clip_key])
        target = min(len(clip_frames), base_per_clip + (1 if clip_index < remainder else 0))
        keep.update(select_representative_frames(clip_frames, target))

    for frame in frames:
        if frame not in keep:
            frame.unlink(missing_ok=True)

    kept_count = len(keep)
    print(
        f"Selected {kept_count} balanced frames from {frame_count} extracted frames across {clip_count} clips",
        flush=True,
    )
    return kept_count


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
    camera_centers = []
    camera_poses = []
    point_errors = []
    track_lengths = []

    if images_txt.exists():
        for line in images_txt.read_text(errors="ignore").splitlines():
            if line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 10 and (".jpg" in line.lower() or ".jpeg" in line.lower() or ".png" in line.lower()):
                registered_images += 1
                try:
                    qw, qx, qy, qz = (float(parts[i]) for i in range(1, 5))
                    translation = [float(parts[i]) for i in range(5, 8)]
                    rotation = quaternion_to_rotation(qw, qx, qy, qz)
                    if rotation:
                        center = camera_center(rotation, translation)
                        forward = matrix_transpose_multiply(rotation, [0, 0, 1])
                        up = matrix_transpose_multiply(rotation, [0, -1, 0])
                        camera_centers.append(center)
                        camera_poses.append({
                            "position": center,
                            "forward": forward,
                            "up": up,
                            "image": parts[9] if len(parts) > 9 else "",
                        })
                except ValueError:
                    pass

    if points_txt.exists():
        for line in points_txt.read_text(errors="ignore").splitlines():
            if line.strip() and not line.startswith("#"):
                sparse_points += 1
                parts = line.split()
                if len(parts) >= 8:
                    try:
                        point_errors.append(float(parts[7]))
                        # Tracks are stored as IMAGE_ID POINT2D_IDX pairs after
                        # the first 8 fields.
                        track_lengths.append(max(0, (len(parts) - 8) // 2))
                    except ValueError:
                        pass

    camera_spans = [0.0, 0.0, 0.0]
    camera_baseline_p95 = 0.0
    camera_baseline_max = 0.0
    if camera_centers:
        axes = list(zip(*camera_centers))
        camera_spans = [
            quantile(list(axis), 0.95) - quantile(list(axis), 0.05)
            for axis in axes
        ]
        first = camera_centers[0]
        distances = [vector_norm([center[i] - first[i] for i in range(3)]) for center in camera_centers]
        camera_baseline_p95 = quantile(distances, 0.95)
        camera_baseline_max = max(distances) if distances else 0.0

    viewer_camera = None
    if camera_poses:
        representative = camera_poses[len(camera_poses) // 2]
        target_distance = max(0.35, min(4.0, camera_baseline_p95 * 0.75 or camera_baseline_max * 0.5 or 1.0))
        viewer_camera = {
            "position": representative["position"],
            "lookAt": vector_add(representative["position"], vector_scale(representative["forward"], target_distance)),
            "up": representative["up"],
            "sourceImage": representative["image"],
        }

    return {
        "registeredImages": registered_images,
        "sparsePoints": sparse_points,
        "cameraSpan": camera_spans,
        "cameraBaselineP95": camera_baseline_p95,
        "cameraBaselineMax": camera_baseline_max,
        "viewerCamera": viewer_camera,
        "meanTrackLength": sum(track_lengths) / len(track_lengths) if track_lengths else 0,
        "pointErrorP75": quantile(point_errors, 0.75),
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


def read_ply_header(f):
    header_lines = []
    properties = []
    vertex_count = 0
    is_binary_little_endian = False
    in_vertex_element = False

    while True:
        raw = f.readline()
        if not raw:
            break
        line = raw.decode("utf-8", errors="ignore").strip()
        header_lines.append(line)

        if line == "format binary_little_endian 1.0":
            is_binary_little_endian = True
        elif line.startswith("element vertex "):
            in_vertex_element = True
            try:
                vertex_count = int(line.split()[-1])
            except ValueError:
                vertex_count = 0
        elif line.startswith("element "):
            in_vertex_element = False
        elif in_vertex_element and line.startswith("property "):
            parts = line.split()
            if len(parts) >= 3:
                properties.append((parts[1], parts[-1]))
        elif line == "end_header":
            break

    return {
        "headerLines": header_lines,
        "vertexCount": vertex_count,
        "properties": properties,
        "isBinaryLittleEndian": is_binary_little_endian,
        "headerSize": f.tell(),
    }


def sigmoid(value):
    if value > 80:
        return 1.0
    if value < -80:
        return 0.0
    return 1 / (1 + math.exp(-value))


def analyze_ply(path, sample_limit=80000):
    with open(path, "rb") as f:
        header = read_ply_header(f)
        vertex_count = header["vertexCount"]
        properties = [name for prop_type, name in header["properties"] if prop_type == "float"]

        if not header["isBinaryLittleEndian"] or not vertex_count or not properties:
            return {"splatCount": vertex_count}

        required = {"x", "y", "z", "opacity", "scale_0", "scale_1", "scale_2"}
        if not required.issubset(set(properties)):
            return {"splatCount": vertex_count}

        prop_index = {name: index for index, name in enumerate(properties)}
        stride = len(properties) * 4
        step = max(1, vertex_count // sample_limit)
        xs, ys, zs, opacities, max_scales = [], [], [], [], []

        for index in range(vertex_count):
            data = f.read(stride)
            if len(data) != stride:
                break
            if index % step:
                continue
            values = struct.unpack("<" + "f" * len(properties), data)
            x, y, z = values[prop_index["x"]], values[prop_index["y"]], values[prop_index["z"]]
            if not all(math.isfinite(v) for v in (x, y, z)):
                continue
            xs.append(x)
            ys.append(y)
            zs.append(z)
            opacities.append(sigmoid(values[prop_index["opacity"]]))
            try:
                max_scales.append(
                    max(
                        math.exp(values[prop_index["scale_0"]]),
                        math.exp(values[prop_index["scale_1"]]),
                        math.exp(values[prop_index["scale_2"]]),
                    )
                )
            except OverflowError:
                max_scales.append(float("inf"))

    central_spans = [
        quantile(xs, 0.95) - quantile(xs, 0.05),
        quantile(ys, 0.95) - quantile(ys, 0.05),
        quantile(zs, 0.95) - quantile(zs, 0.05),
    ]
    broad_spans = [
        quantile(xs, 0.99) - quantile(xs, 0.01),
        quantile(ys, 0.99) - quantile(ys, 0.01),
        quantile(zs, 0.99) - quantile(zs, 0.01),
    ]
    central_diag = vector_norm(central_spans)
    broad_diag = vector_norm(broad_spans)
    opaque_ratio = (
        sum(1 for value in opacities if value >= 0.5) / len(opacities)
        if opacities
        else 0
    )

    return {
        "splatCount": vertex_count,
        "opaqueSplatRatio": opaque_ratio,
        "scaleP95": quantile(max_scales, 0.95),
        "scaleP99": quantile(max_scales, 0.99),
        "centralDiagonal": central_diag,
        "broadDiagonal": broad_diag,
        "outlierRatio": broad_diag / max(central_diag, 0.001),
    }


def optimize_ply_for_web(input_path, output_path):
    with open(input_path, "rb") as f:
        header = read_ply_header(f)
        properties = header["properties"]
        vertex_count = header["vertexCount"]

        if not header["isBinaryLittleEndian"] or not vertex_count or not properties:
            shutil.copyfile(input_path, output_path)
            return {"sourceSplatCount": vertex_count, "optimizedSplatCount": vertex_count, "optimized": False}

        if any(prop_type != "float" for prop_type, _ in properties):
            shutil.copyfile(input_path, output_path)
            return {"sourceSplatCount": vertex_count, "optimizedSplatCount": vertex_count, "optimized": False}

        prop_names = [name for _, name in properties]
        required = {"x", "y", "z", "opacity", "scale_0", "scale_1", "scale_2"}
        if not required.issubset(set(prop_names)):
            shutil.copyfile(input_path, output_path)
            return {"sourceSplatCount": vertex_count, "optimizedSplatCount": vertex_count, "optimized": False}

        prop_index = {name: index for index, name in enumerate(prop_names)}
        stride = len(prop_names) * 4
        fmt = "<" + "f" * len(prop_names)
        records = []
        xs, ys, zs = [], [], []

        for index in range(vertex_count):
            data = f.read(stride)
            if len(data) != stride:
                break
            values = struct.unpack(fmt, data)
            x, y, z = values[prop_index["x"]], values[prop_index["y"]], values[prop_index["z"]]
            if not all(math.isfinite(v) for v in (x, y, z)):
                continue
            try:
                max_scale = max(
                    math.exp(values[prop_index["scale_0"]]),
                    math.exp(values[prop_index["scale_1"]]),
                    math.exp(values[prop_index["scale_2"]]),
                )
            except OverflowError:
                max_scale = float("inf")
            opacity = sigmoid(values[prop_index["opacity"]])
            xs.append(x)
            ys.append(y)
            zs.append(z)
            records.append({
                "index": index,
                "data": data,
                "x": x,
                "y": y,
                "z": z,
                "opacity": opacity,
                "maxScale": max_scale,
            })

    if not records:
        raise ReconstructionQualityError("The 3D reconstruction did not contain usable render points.")

    low = max(0.0, min(0.45, PLY_OUTLIER_QUANTILE_LOW))
    high = min(1.0, max(0.55, PLY_OUTLIER_QUANTILE_HIGH))
    bounds = {
        "x": (quantile(xs, low), quantile(xs, high)),
        "y": (quantile(ys, low), quantile(ys, high)),
        "z": (quantile(zs, low), quantile(zs, high)),
    }

    def within_bounds(record):
        return (
            bounds["x"][0] <= record["x"] <= bounds["x"][1]
            and bounds["y"][0] <= record["y"] <= bounds["y"][1]
            and bounds["z"][0] <= record["z"] <= bounds["z"][1]
        )

    filtered = [
        record
        for record in records
        if record["opacity"] >= PLY_MIN_OPACITY
        and record["maxScale"] <= PLY_MAX_SCALE
        and within_bounds(record)
    ]

    min_after_filter = max(12000, min(MIN_SPLAT_COUNT, int(vertex_count * 0.08)))
    if len(filtered) < min_after_filter:
        relaxed_opacity = max(0.18, PLY_MIN_OPACITY * 0.65)
        relaxed_scale = min(0.14, PLY_MAX_SCALE * 1.6)
        filtered = [
            record
            for record in records
            if record["opacity"] >= relaxed_opacity
            and record["maxScale"] <= relaxed_scale
            and within_bounds(record)
        ]

    if len(filtered) < min_after_filter:
        raise ReconstructionQualityError(
            "The uploaded video set produced too few clean 3D points after quality filtering."
        )

    header_text = "\n".join(header["headerLines"]) + "\n"
    max_by_file_size = int(max(1, (MAX_OUTPUT_FILE_MB * 1024 * 1024 - len(header_text.encode("utf-8"))) // stride))
    max_output_splats = max(1, min(MAX_OUTPUT_SPLATS, max_by_file_size))

    if len(filtered) > max_output_splats:
        filtered = sorted(
            filtered,
            key=lambda record: (record["opacity"], -record["maxScale"]),
            reverse=True,
        )[:max_output_splats]

    filtered.sort(key=lambda record: record["index"])
    output_header_lines = [
        f"element vertex {len(filtered)}" if line.startswith("element vertex ") else line
        for line in header["headerLines"]
    ]
    output_header = ("\n".join(output_header_lines) + "\n").encode("utf-8")

    with open(output_path, "wb") as f:
        f.write(output_header)
        for record in filtered:
            f.write(record["data"])

    return {
        "sourceSplatCount": vertex_count,
        "optimizedSplatCount": len(filtered),
        "optimized": True,
        "outputFileMb": output_path.stat().st_size / (1024 * 1024),
    }


def assert_geometry_quality(frame_count, stats):
    registered_images = int(stats.get("registeredImages") or 0)
    sparse_points = int(stats.get("sparsePoints") or 0)
    camera_baseline = float(stats.get("cameraBaselineP95") or 0)
    camera_span = vector_norm(stats.get("cameraSpan") or [0, 0, 0])
    camera_spread_ratio = camera_baseline / max(camera_span, 0.001)
    registered_ratio = registered_images / frame_count if frame_count else 0
    min_registered = min(
        frame_count,
        max(MIN_REGISTERED_IMAGES, int(frame_count * MIN_REGISTERED_IMAGE_RATIO)),
    )
    min_sparse_points = max(MIN_SPARSE_POINTS, int(frame_count * 80))

    failures = []
    if registered_images < min_registered:
        failures.append(f"only {registered_images} usable frames")
    if registered_ratio < MIN_REGISTERED_IMAGE_RATIO:
        failures.append(f"only {registered_ratio:.0%} of frames aligned")
    if sparse_points < min_sparse_points:
        failures.append(f"only {sparse_points} stable scene points")
    if camera_baseline < MIN_CAMERA_BASELINE:
        failures.append("not enough real camera movement")
    if camera_spread_ratio < MIN_CAMERA_SPREAD_RATIO:
        failures.append("camera movement is too concentrated")

    if failures:
        raise ReconstructionQualityError(
            "The uploaded video set is not reliable enough for a sellable 3D tour. "
            "Please record slower, brighter landscape clips with more overlap and real side-to-side movement. "
            f"Quality checks failed: {', '.join(failures)}."
        )

    return {
        "frameCount": frame_count,
        "registeredImages": registered_images,
        "registeredImageRatio": registered_ratio,
        "sparsePoints": sparse_points,
        "cameraBaselineP95": camera_baseline,
    }


def assert_reconstruction_quality(frame_count, stats, splat_stats):
    quality = assert_geometry_quality(frame_count, stats)
    splat_count = int(splat_stats.get("splatCount") or 0)
    opaque_splat_ratio = float(splat_stats.get("opaqueSplatRatio") or 0)
    scale_p95 = float(splat_stats.get("scaleP95") or 0)
    outlier_ratio = float(splat_stats.get("outlierRatio") or 0)
    min_splat_count = max(MIN_SPLAT_COUNT, int(frame_count * 220))

    failures = []
    if splat_count < min_splat_count:
        failures.append(f"only {splat_count} render points")
    if opaque_splat_ratio < MIN_OPAQUE_SPLAT_RATIO:
        failures.append("not enough visible 3D detail")
    if scale_p95 > MAX_SPLAT_SCALE_P95:
        failures.append("3D detail is too blurred")
    if outlier_ratio > MAX_SPLAT_OUTLIER_RATIO:
        failures.append("3D structure has too many outliers")

    if failures:
        raise ReconstructionQualityError(
            "The uploaded video set is not reliable enough for a sellable 3D tour. "
            "Please record slower, brighter landscape clips with more overlap and real side-to-side movement. "
            f"Quality checks failed: {', '.join(failures)}."
        )

    return {
        **quality,
        "splatCount": splat_count,
        "opaqueSplatRatio": opaque_splat_ratio,
        "scaleP95": scale_p95,
        "outlierRatio": outlier_ratio,
    }


def process_scene(conn, job, scene, base_work_dir, progress_start, progress_end):
    scene_dir = base_work_dir / f"scene-{scene['key']}"
    images_dir = scene_dir / "images"
    output_path = scene_dir / "tour.ply"
    scene_dir.mkdir(parents=True, exist_ok=True)

    update_job(conn, job["id"], progress=progress_start)
    extract_video_set(scene["videos"], scene_dir, images_dir)
    frame_count = limit_scene_frames(images_dir)
    if frame_count < MIN_SCENE_FRAMES:
        raise ReconstructionQualityError(
            f"{scene['title']} does not have enough usable video frames for a reliable 3D tour."
        )

    update_job(conn, job["id"], progress=progress_start + int((progress_end - progress_start) * 0.25))
    def heartbeat():
        update_job(conn, job["id"], progress=progress_start + int((progress_end - progress_start) * 0.25))

    model_dir = run_colmap(images_dir, scene_dir, heartbeat=heartbeat)
    model_stats = analyze_colmap_model(model_dir, scene_dir)
    assert_geometry_quality(frame_count, model_stats)

    update_job(conn, job["id"], progress=progress_start + int((progress_end - progress_start) * 0.7))
    def training_heartbeat():
        update_job(conn, job["id"], progress=progress_start + int((progress_end - progress_start) * 0.7))

    run_opensplat(model_dir, images_dir, output_path, heartbeat=training_heartbeat)
    optimized_path = scene_dir / "tour-optimized.ply"
    optimization_stats = optimize_ply_for_web(output_path, optimized_path)
    splat_stats = analyze_ply(optimized_path)
    quality = assert_reconstruction_quality(frame_count, model_stats, splat_stats)
    quality = {**quality, **optimization_stats}

    update_job(conn, job["id"], progress=progress_end)
    file_url = upload_result(optimized_path)

    return {
        "key": scene["key"],
        "title": scene["title"],
        "fileUrl": file_url,
        "format": "ply",
        "camera": model_stats.get("viewerCamera") or {},
        "videoCount": len(scene["videos"]),
        "quality": quality,
    }


def run_opensplat(model_dir, images_dir, output_path, heartbeat=None):
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
    ], heartbeat=heartbeat)
    if not output_path.exists():
        raise RuntimeError("OpenSplat did not produce an output file.")


def upload_result(path):
    object_path = f"video-3d-tours/{time.strftime('%Y-%m-%d')}/{uuid.uuid4()}{path.suffix}"
    file_mb = path.stat().st_size / (1024 * 1024)
    if file_mb > MAX_OUTPUT_FILE_MB + 1:
        raise ReconstructionQualityError(
            f"The generated 3D tour is still too large to upload safely ({file_mb:.1f} MB)."
        )

    content_type = "model/vnd.ply" if path.suffix == ".ply" else "application/octet-stream"
    upload_url = (
        f"{SUPABASE_URL.rstrip('/')}/storage/v1/object/"
        f"{SUPABASE_STORAGE_BUCKET}/{object_path}"
    )
    headers = {
        "apikey": SUPABASE_SERVICE_ROLE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
        "Content-Type": content_type,
        "x-upsert": "false",
    }

    with open(path, "rb") as f:
        response = requests.post(upload_url, headers=headers, data=f, timeout=180)

    if not response.ok:
        detail = response.text[:500]
        if response.status_code == 413:
            raise ReconstructionQualityError(
                f"The generated 3D tour file is too large for storage upload ({file_mb:.1f} MB). "
                "The worker should retry with a smaller web export."
            )
        raise RuntimeError(
            f"3D tour upload failed: {response.status_code} {response.reason}"
            f"{f' - {detail}' if detail else ''}"
        )

    public_path = "/".join(requests.utils.quote(part, safe="") for part in object_path.split("/"))
    return (
        f"{SUPABASE_URL.rstrip('/')}/storage/v1/object/public/"
        f"{requests.utils.quote(SUPABASE_STORAGE_BUCKET, safe='')}/{public_path}"
    )


def save_virtual_tour(conn, job, scenes, skipped_scenes=None):
    request_payload = job.get("request_payload") or {}
    primary_scene = scenes[0]
    payload = {
        "type": "splat3d",
        "fileUrl": primary_scene["fileUrl"],
        "format": primary_scene["format"],
        "alphaRemovalThreshold": 8,
        "sourceType": "original",
        "generatedFrom": request_payload.get("captureType") or "iphone_video",
        "jobId": str(job["id"]),
        "videoCount": request_payload.get("videoCount") or 1,
        "sceneCount": len(scenes),
        "scenes": scenes,
        "skippedScenes": skipped_scenes or [],
        "quality": {
            **primary_scene["quality"],
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
    videos = request_payload.get("videos")
    if isinstance(videos, list) and videos:
        video_items = []
        for item in sorted(
            [item for item in videos if isinstance(item, dict)],
            key=lambda item: int(item.get("index") or 0),
        ):
            video_url = item.get("videoUrl") or item.get("url")
            if not video_url:
                continue
            video_items.append({
                "videoUrl": video_url,
                "originalName": item.get("originalName") or item.get("name") or "",
            })
    else:
        video_url = request_payload.get("videoUrl")
        video_items = [{
            "videoUrl": video_url,
            "originalName": request_payload.get("originalName") or "",
        }] if video_url else []

    if not video_items:
        raise RuntimeError("Job is missing request_payload.videoUrl or videos[].videoUrl.")

    with tempfile.TemporaryDirectory(prefix="video-3d-tour-") as tmp:
        work_dir = Path(tmp)
        scene_groups = group_video_items(video_items)
        scenes = []
        skipped_scenes = []
        span = max(1, int(82 / max(1, len(scene_groups))))

        for index, scene in enumerate(scene_groups):
            start = 5 + index * span
            end = min(87, start + span)
            print(
                f"Processing scene {index + 1}/{len(scene_groups)}: {scene['title']}",
                flush=True,
            )
            try:
                scenes.append(process_scene(conn, job, scene, work_dir, start, end))
            except Exception as exc:
                skipped_scenes.append({
                    "key": scene["key"],
                    "title": scene["title"],
                    "error": str(exc),
                })
                print(f"Scene skipped: {scene['title']}: {exc}", flush=True)

        if not scenes:
            update_job(
                conn,
                job["id"],
                result={
                    "sceneCount": 0,
                    "skippedScenes": skipped_scenes,
                    "quality": {
                        "profile": "rejected",
                        "reason": "no_reliable_scenes",
                    },
                },
            )
            raise ReconstructionQualityError("No reliable 3D scenes could be created from the uploaded videos.")

        update_job(conn, job["id"], progress=95)
        tour_id = save_virtual_tour(conn, job, scenes, skipped_scenes)

        update_job(
            conn,
            job["id"],
            status="succeeded",
            progress=100,
            result={
                "tourId": str(tour_id),
                "fileUrl": scenes[0]["fileUrl"],
                "sceneCount": len(scenes),
                "scenes": scenes,
                "skippedScenes": skipped_scenes,
            },
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
