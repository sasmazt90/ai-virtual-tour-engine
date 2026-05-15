import sql from "@/app/api/utils/sql";
import { auth } from "@/auth";
import { getDbUserIdFromSession } from "@/app/api/utils/dbUser";

export async function GET() {
  try {
    const session = await auth();
    if (!session || !session.user?.id) {
      return Response.json({ error: "Unauthorized" }, { status: 401 });
    }

    const userId = await getDbUserIdFromSession(session);
    if (!userId) {
      return Response.json({ error: "Unauthorized" }, { status: 401 });
    }

    // Include best-effort job status for reserve transactions so the UI can
    // conservatively compute "finalized" usage totals (succeeded only).
    const transactions = await sql(
      "SELECT ct.*, aj.job_status AS ai_job_status FROM credit_transactions ct LEFT JOIN ai_jobs aj ON aj.id = (CASE WHEN (ct.meta ? 'jobId') AND (ct.meta->>'jobId') ~ '^[0-9a-fA-F-]{36}$' THEN (ct.meta->>'jobId')::uuid ELSE NULL END) WHERE ct.user_id = $1 ORDER BY ct.created_at DESC LIMIT 50",
      [userId],
    );

    return Response.json(transactions);
  } catch (error) {
    console.error("GET /api/credits/transactions error:", error);
    return Response.json({ error: "Internal Server Error" }, { status: 500 });
  }
}
