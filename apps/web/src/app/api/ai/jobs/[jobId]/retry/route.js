import sql from "@/app/api/utils/sql";
import { auth } from "@/auth";
import { processStagingJob } from "@/app/api/ai/staging/create/route";
import { processVirtualTourJob } from "@/app/api/ai/virtual-tour/create/route";
import { getDbUserIdFromSession } from "@/app/api/utils/dbUser";

export async function POST(request, { params }) {
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
      "SELECT id, user_id, property_id, job_type, job_status, credits_reserved, request_payload FROM ai_jobs WHERE id = $1 AND user_id = $2 LIMIT 1",
      [jobId, userId],
    );

    if (rows.length === 0) {
      return Response.json({ error: "Job not found" }, { status: 404 });
    }

    const job = rows[0];

    if (job.job_status !== "failed") {
      return Response.json(
        { error: "Only failed jobs can be retried" },
        { status: 400 },
      );
    }

    const creditsReserved = Number(job.credits_reserved || 0);
    if (!creditsReserved || creditsReserved <= 0) {
      return Response.json(
        { error: "This job cannot be retried (missing credits_reserved)" },
        { status: 400 },
      );
    }

    const requestPayload = job.request_payload || {};
    const requestPayloadJson = JSON.stringify(requestPayload);

    // Best-effort dedupe: if a retry with the same request_payload is already queued/running, return it.
    const existing = await sql(
      "SELECT id, job_status FROM ai_jobs WHERE user_id = $1 AND property_id = $2 AND job_type = $3 AND job_status IN ('queued','running') AND request_payload = $4::jsonb ORDER BY created_at DESC LIMIT 1",
      [userId, job.property_id, job.job_type, requestPayloadJson],
    );

    if (existing.length > 0) {
      return Response.json({
        jobId: existing[0].id,
        status: existing[0].job_status,
        creditsReserved,
        creditCost: creditsReserved,
        deduped: true,
      });
    }

    const spendMeta = {
      kind: "reserve",
      jobType: job.job_type,
      retryOfJobId: jobId,
    };

    // IMPORTANT: avoid sql.transaction(async ...). Use one atomic statement.
    const createdRows = await sql(
      `
        WITH ensured_wallet AS (
          INSERT INTO credits_wallet (user_id, balance_credits)
          VALUES ($1, 0)
          ON CONFLICT (user_id) DO NOTHING
        ),
        deducted AS (
          UPDATE credits_wallet
          SET balance_credits = balance_credits - $2
          WHERE user_id = $1
            AND balance_credits >= $2
          RETURNING balance_credits
        ),
        created_job AS (
          INSERT INTO ai_jobs (user_id, property_id, job_type, job_status, progress, credits_reserved, request_payload, started_at, last_heartbeat_at)
          SELECT $1, $3, $4, 'queued', 0, $2, $5::jsonb, NULL, NOW()
          WHERE EXISTS (SELECT 1 FROM deducted)
          RETURNING id
        ),
        spend_tx AS (
          INSERT INTO credit_transactions (user_id, transaction_type, credits_delta, meta)
          SELECT $1, 'spend', -$2, $6::jsonb
          WHERE EXISTS (SELECT 1 FROM created_job)
          RETURNING id
        )
        SELECT
          (SELECT id FROM created_job) AS new_job_id,
          CASE WHEN EXISTS (SELECT 1 FROM created_job) THEN 'created' ELSE 'insufficient' END AS outcome;
      `,
      [
        userId,
        creditsReserved,
        job.property_id,
        job.job_type,
        requestPayloadJson,
        JSON.stringify(spendMeta),
      ],
    );

    const created = createdRows?.[0] || null;
    if (!created || !created.new_job_id) {
      return Response.json({ error: "Insufficient credits" }, { status: 402 });
    }

    const newJobId = created.new_job_id;

    const start = () => {
      if (job.job_type === "staging") {
        processStagingJob({ jobId: newJobId }).catch((e) => console.error(e));
        return;
      }
      if (job.job_type === "virtual_tour") {
        processVirtualTourJob({ jobId: newJobId }).catch((e) =>
          console.error(e),
        );
        return;
      }
    };

    if (typeof queueMicrotask === "function") {
      queueMicrotask(start);
    } else {
      setTimeout(start, 0);
    }

    return Response.json({
      jobId: newJobId,
      status: "queued",
      creditsReserved,
      creditCost: creditsReserved,
      deduped: false,
    });
  } catch (error) {
    console.error("POST /api/ai/jobs/[jobId]/retry error:", error);
    return Response.json(
      { error: error?.message || "Internal Server Error" },
      { status: 500 },
    );
  }
}
