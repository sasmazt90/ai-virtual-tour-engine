import sql from "@/app/api/utils/sql";
import { auth } from "@/auth";
import { getDbUserIdFromSession } from "@/app/api/utils/dbUser";
import {
  AI_VIDEO_3D_MAX_BYTES,
  AI_VIDEO_3D_MAX_MB,
  AI_VIDEO_3D_MAX_FILES,
  calculateVideo3DTourCreditCost,
  getVideo3DTourCreditTier,
} from "@/app/api/utils/pricing";

const SUPPORTED_VIDEO_EXTENSIONS = new Set(["mp4", "mov", "m4v", "webm"]);

function getExtension(value) {
  try {
    const url = new URL(String(value || ""));
    return url.pathname.split(".").pop()?.toLowerCase().trim() || "";
  } catch {
    return String(value || "").split(".").pop()?.toLowerCase().trim() || "";
  }
}

async function getRemoteContentLength(url) {
  try {
    const res = await fetch(url, { method: "HEAD" });
    if (!res.ok) return 0;
    const length = Number(res.headers.get("content-length") || 0);
    return Number.isFinite(length) && length > 0 ? length : 0;
  } catch {
    return 0;
  }
}

function normalizeVideoInputs(body) {
  const rawVideos = Array.isArray(body?.videos) ? body.videos : null;
  if (rawVideos) {
    return rawVideos
      .map((item, index) => ({
        videoUrl:
          typeof item?.videoUrl === "string"
            ? item.videoUrl.trim()
            : typeof item?.url === "string"
              ? item.url.trim()
              : "",
        originalName:
          typeof item?.originalName === "string"
            ? item.originalName.trim()
            : typeof item?.name === "string"
              ? item.name.trim()
              : "",
        fileSizeBytes: Number(item?.fileSizeBytes || item?.sizeBytes || 0) || 0,
        index,
      }))
      .filter((item) => item.videoUrl);
  }

  const videoUrl =
    typeof body?.videoUrl === "string" ? body.videoUrl.trim() : "";
  return videoUrl
    ? [
        {
          videoUrl,
          originalName:
            typeof body?.originalName === "string"
              ? body.originalName.trim()
              : "",
          fileSizeBytes: Number(body?.fileSizeBytes || 0) || 0,
          index: 0,
        },
      ]
    : [];
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
    const videoInputs = normalizeVideoInputs(body);

    if (!propertyId) {
      return Response.json({ error: "Missing property ID." }, { status: 400 });
    }

    if (videoInputs.length === 0) {
      return Response.json(
        { error: "Please upload at least one iPhone video first." },
        { status: 400 },
      );
    }

    if (videoInputs.length > AI_VIDEO_3D_MAX_FILES) {
      return Response.json(
        { error: `Please upload ${AI_VIDEO_3D_MAX_FILES} videos or fewer.` },
        { status: 400 },
      );
    }

    for (const input of videoInputs) {
      if (!/^https?:\/\//i.test(input.videoUrl)) {
        return Response.json(
          { error: "One of the uploaded videos is invalid." },
          { status: 400 },
        );
      }

      const ext = getExtension(input.originalName) || getExtension(input.videoUrl);
      if (!SUPPORTED_VIDEO_EXTENSIONS.has(ext)) {
        return Response.json(
          { error: "Supported video formats are .mp4, .mov, .m4v and .webm." },
          { status: 400 },
        );
      }
    }

    const verifiedVideos = [];
    for (const input of videoInputs) {
      const remoteFileSizeBytes = await getRemoteContentLength(input.videoUrl);
      const fileSizeBytes = Math.max(input.fileSizeBytes, remoteFileSizeBytes);
      if (!Number.isFinite(fileSizeBytes) || fileSizeBytes <= 0) {
        return Response.json(
          { error: "Could not verify one of the uploaded video sizes." },
          { status: 400 },
        );
      }
      verifiedVideos.push({
        videoUrl: input.videoUrl,
        originalName: input.originalName || null,
        fileSizeBytes,
        index: input.index,
      });
    }

    const totalFileSizeBytes = verifiedVideos.reduce(
      (sum, item) => sum + item.fileSizeBytes,
      0,
    );

    if (
      Number.isFinite(AI_VIDEO_3D_MAX_BYTES) &&
      totalFileSizeBytes > AI_VIDEO_3D_MAX_BYTES
    ) {
      return Response.json(
        {
          error: `Videos are too large. Please upload ${AI_VIDEO_3D_MAX_MB} MB total or less.`,
        },
        { status: 413 },
      );
    }

    const props = await sql(
      "SELECT id FROM properties WHERE id = $1 AND user_id = $2 LIMIT 1",
      [propertyId, userId],
    );

    if (props.length === 0) {
      return Response.json({ error: "Property not found." }, { status: 404 });
    }

    const creditsReserved = calculateVideo3DTourCreditCost(totalFileSizeBytes);
    const pricingTier = getVideo3DTourCreditTier(totalFileSizeBytes);

    const requestPayload = {
      videoUrl: verifiedVideos[0]?.videoUrl || null,
      originalName: verifiedVideos[0]?.originalName || null,
      fileSizeBytes: totalFileSizeBytes,
      videos: verifiedVideos,
      videoCount: verifiedVideos.length,
      captureType: verifiedVideos.length > 1 ? "iphone_video_set" : "iphone_video",
      outputType: "gaussian_splat",
      tourType: "splat3d",
      pricing: {
        creditCost: creditsReserved,
        tierLabel: pricingTier?.label || null,
      },
    };

    const rows = await sql(
      `
      WITH existing_job AS (
        SELECT id, job_status
        FROM ai_jobs
        WHERE user_id = $1
          AND property_id = $2
          AND job_type = 'video_3d_tour'
          AND job_status IN ('queued','running')
          AND request_payload = $4::jsonb
        ORDER BY created_at DESC
        LIMIT 1
      ),
      ensured_wallet AS (
        INSERT INTO credits_wallet (user_id, balance_credits)
        VALUES ($1, 0)
        ON CONFLICT (user_id) DO NOTHING
      ),
      deducted AS (
        UPDATE credits_wallet
        SET balance_credits = balance_credits - $3
        WHERE user_id = $1
          AND balance_credits >= $3
          AND NOT EXISTS (SELECT 1 FROM existing_job)
        RETURNING balance_credits
      ),
      created_job AS (
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
        SELECT $1, $2, 'video_3d_tour', 'queued', 0, $3, $4::jsonb, NOW()
        WHERE NOT EXISTS (SELECT 1 FROM existing_job)
          AND EXISTS (SELECT 1 FROM deducted)
        RETURNING id
      ),
      spend_tx AS (
        INSERT INTO credit_transactions (user_id, transaction_type, credits_delta, meta)
        SELECT $1, 'spend', -$3, $5::jsonb
        WHERE EXISTS (SELECT 1 FROM created_job)
        RETURNING id
      )
      SELECT
        COALESCE((SELECT id FROM existing_job), (SELECT id FROM created_job)) AS job_id,
        COALESCE((SELECT job_status FROM existing_job), 'queued') AS job_status,
        CASE
          WHEN EXISTS (SELECT 1 FROM existing_job) THEN 'deduped'
          WHEN EXISTS (SELECT 1 FROM created_job) THEN 'created'
          ELSE 'insufficient'
        END AS outcome
      `,
      [
        userId,
        propertyId,
        creditsReserved,
        JSON.stringify(requestPayload),
        JSON.stringify({
          kind: "reserve",
          jobType: "video_3d_tour",
          creditCost: creditsReserved,
          fileSizeBytes: totalFileSizeBytes,
          videoCount: verifiedVideos.length,
          pricingTier: pricingTier?.label || null,
        }),
      ],
    );

    const created = rows?.[0] || null;
    if (!created || !created.job_id) {
      return Response.json({ error: "Insufficient credits" }, { status: 402 });
    }

    if (created.outcome === "insufficient") {
      return Response.json({ error: "Insufficient credits" }, { status: 402 });
    }

    const jobId = created.job_id;
    const worker =
      created.outcome === "created" ? await notifyWorker({ jobId }) : null;

    return Response.json({
      jobId,
      status: created.outcome === "deduped" ? created.job_status : "queued",
      creditsReserved,
      creditCost: creditsReserved,
      pricingTier,
      deduped: created.outcome === "deduped",
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
