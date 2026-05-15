import sql from "@/app/api/utils/sql";
import { auth } from "@/auth";
import { getDbUserIdFromSession } from "@/app/api/utils/dbUser";

const HEARTBEAT_TIMEOUT_MS = 10 * 60 * 1000; // 10 minutes
const VIDEO_3D_HEARTBEAT_TIMEOUT_MS = 6 * 60 * 60 * 1000; // 6 hours

async function refundCreditsIfNeeded({ userId, jobId }) {
  const jobs = await sql(
    "SELECT id, credits_reserved FROM ai_jobs WHERE id = $1 AND user_id = $2 LIMIT 1",
    [jobId, userId],
  );
  if (jobs.length === 0) return;

  const credits = Number(jobs[0].credits_reserved || 0);
  if (!credits || credits <= 0) return;

  const existingRefund = await sql(
    "SELECT id FROM credit_transactions WHERE user_id = $1 AND transaction_type = 'refund' AND meta->>'jobId' = $2 LIMIT 1",
    [userId, String(jobId)],
  );

  if (existingRefund.length > 0) {
    return;
  }

  // IMPORTANT: Anything sql.transaction() does NOT accept an async callback.
  // Use a single atomic query instead.
  await sql(
    `
      WITH ensured_wallet AS (
        INSERT INTO credits_wallet (user_id, balance_credits)
        VALUES ($1, 0)
        ON CONFLICT (user_id) DO NOTHING
      ),
      updated_wallet AS (
        UPDATE credits_wallet
        SET balance_credits = balance_credits + $2
        WHERE user_id = $1
        RETURNING balance_credits
      ),
      inserted_tx AS (
        INSERT INTO credit_transactions (user_id, transaction_type, credits_delta, meta)
        VALUES ($1, 'refund', $2, $3::jsonb)
        RETURNING id
      )
      SELECT 1 as ok;
    `,
    [userId, credits, JSON.stringify({ jobId })],
  );
}

async function failIfTimedOut({ userId, jobRow }) {
  if (!jobRow) return false;
  if (jobRow.job_status !== "running") return false;

  const last =
    jobRow.last_heartbeat_at || jobRow.started_at || jobRow.updated_at;
  if (!last) return false;

  const lastMs = new Date(last).getTime();
  if (!Number.isFinite(lastMs)) return false;

  const timeoutMs =
    jobRow.job_type === "video_3d_tour"
      ? VIDEO_3D_HEARTBEAT_TIMEOUT_MS
      : HEARTBEAT_TIMEOUT_MS;

  if (Date.now() - lastMs <= timeoutMs) {
    return false;
  }

  // Mark failed + refund (best-effort)
  await refundCreditsIfNeeded({ userId, jobId: jobRow.id });
  await sql(
    "UPDATE ai_jobs SET job_status = 'failed', error_message = $1 WHERE id = $2 AND user_id = $3 AND job_status = 'running'",
    [
      `Timed out: no heartbeat for more than ${Math.round(timeoutMs / 60000)} minutes`,
      jobRow.id,
      userId,
    ],
  );

  return true;
}

export async function GET(request, { params }) {
  try {
    const session = await auth();
    if (!session || !session.user?.id) {
      return Response.json({ error: "Unauthorized" }, { status: 401 });
    }

    const userId = await getDbUserIdFromSession(session);
    if (!userId) {
      return Response.json({ error: "Unauthorized" }, { status: 401 });
    }
    const jobId = params.jobId;

    const rows = await sql(
      "SELECT id, job_type, job_status, progress, result_payload, error_message, started_at, last_heartbeat_at, updated_at FROM ai_jobs WHERE id = $1 AND user_id = $2 LIMIT 1",
      [jobId, userId],
    );

    if (rows.length === 0) {
      return Response.json({ error: "Job not found" }, { status: 404 });
    }

    const j = rows[0];

    // Soft-timeout safety: if running but heartbeat is stale, auto-fail & refund.
    const timedOut = await failIfTimedOut({ userId, jobRow: j });
    const fresh = timedOut
      ? (
          await sql(
            "SELECT id, job_type, job_status, progress, result_payload, error_message, started_at, last_heartbeat_at FROM ai_jobs WHERE id = $1 AND user_id = $2 LIMIT 1",
            [jobId, userId],
          )
        )?.[0] || j
      : j;

    return Response.json({
      jobId: fresh.id,
      status: fresh.job_status,
      progress: fresh.progress || 0,
      result: fresh.result_payload || null,
      error: fresh.error_message || null,
      startedAt: fresh.started_at || null,
      lastHeartbeatAt: fresh.last_heartbeat_at || null,
    });
  } catch (error) {
    console.error("GET /api/ai/jobs/[jobId] error:", error);
    return Response.json({ error: "Internal Server Error" }, { status: 500 });
  }
}
