import sql from "@/app/api/utils/sql";
import { auth } from "@/auth";
import { getDbUserIdFromSession } from "@/app/api/utils/dbUser";

const SUPPORTED_VIDEO_EXTENSIONS = new Set(["mp4", "mov", "m4v"]);

function getExtension(value) {
  try {
    const url = new URL(String(value || ""));
    return url.pathname.split(".").pop()?.toLowerCase().trim() || "";
  } catch {
    return String(value || "").split(".").pop()?.toLowerCase().trim() || "";
  }
}

async function notifyWorker({ jobId }) {
  const workerUrl = process.env.VIDEO_TO_SPLAT_WORKER_URL;
  if (!workerUrl) return { notified: false, reason: "worker_url_missing" };

  try {
    const res = await fetch(workerUrl.replace(/\/+$/, "") + "/jobs", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...(process.env.VIDEO_TO_SPLAT_WORKER_SECRET
          ? { Authorization: `Bearer ${process.env.VIDEO_TO_SPLAT_WORKER_SECRET}` }
          : {}),
      },
      body: JSON.stringify({ jobId }),
    });

    return {
      notified: res.ok,
      status: res.status,
    };
  } catch (error) {
    console.warn("video-to-splat worker notification failed", error);
    return { notified: false, reason: "request_failed" };
  }
}

export async function POST(request) {
  try {
    const session = await auth();
    if (!session || !session.user?.id) {
      return Response.json({ error: "Unauthorized" }, { status: 401 });
    }

    const userId = await getDbUserIdFromSession(session);
    if (!userId) {
      return Response.json({ error: "Unauthorized" }, { status: 401 });
    }

    const body = await request.json().catch(() => ({}));
    const propertyId =
      typeof body?.propertyId === "string" ? body.propertyId.trim() : "";
    const videoUrl =
      typeof body?.videoUrl === "string" ? body.videoUrl.trim() : "";
    const originalName =
      typeof body?.originalName === "string" ? body.originalName.trim() : "";

    if (!propertyId) {
      return Response.json({ error: "Missing property ID." }, { status: 400 });
    }

    if (!videoUrl || !/^https?:\/\//i.test(videoUrl)) {
      return Response.json(
        { error: "Please upload an iPhone video first." },
        { status: 400 },
      );
    }

    const ext = getExtension(originalName) || getExtension(videoUrl);
    if (!SUPPORTED_VIDEO_EXTENSIONS.has(ext)) {
      return Response.json(
        { error: "Supported video formats are .mp4, .mov and .m4v." },
        { status: 400 },
      );
    }

    const props = await sql(
      "SELECT id FROM properties WHERE id = $1 AND user_id = $2 LIMIT 1",
      [propertyId, userId],
    );

    if (props.length === 0) {
      return Response.json({ error: "Property not found." }, { status: 404 });
    }

    const requestPayload = {
      videoUrl,
      originalName: originalName || null,
      captureType: "iphone_video",
      outputType: "gaussian_splat",
      tourType: "splat3d",
    };

    const rows = await sql(
      `
      INSERT INTO ai_jobs (
        user_id,
        property_id,
        job_type,
        job_status,
        progress,
        credits_reserved,
        request_payload,
        last_heartbeat_at
      )
      VALUES ($1, $2, 'video_3d_tour', 'queued', 0, 0, $3::jsonb, NOW())
      RETURNING id
      `,
      [userId, propertyId, JSON.stringify(requestPayload)],
    );

    const jobId = rows?.[0]?.id;
    const worker = jobId ? await notifyWorker({ jobId }) : null;

    return Response.json({
      jobId,
      status: "queued",
      worker,
    });
  } catch (error) {
    console.error("POST /api/ai/video-3d-tour/create error:", error);
    return Response.json(
      { error: "Could not start the 3D video tour job." },
      { status: 500 },
    );
  }
}
