import sql from "@/app/api/utils/sql";
import { auth } from "@/auth";
import { getDbUserIdFromSession } from "@/app/api/utils/dbUser";

export async function GET(request) {
  try {
    const session = await auth();
    if (!session || !session.user?.id) {
      return Response.json({ error: "Unauthorized" }, { status: 401 });
    }

    const userId = await getDbUserIdFromSession(session);
    if (!userId) {
      return Response.json({ error: "Unauthorized" }, { status: 401 });
    }
    const url = new URL(request.url);
    const limitRaw = url.searchParams.get("limit");
    const limit = Math.max(
      1,
      Math.min(50, Number.isFinite(Number(limitRaw)) ? Number(limitRaw) : 20),
    );

    const rows = await sql(
      `
      SELECT
        j.id,
        j.job_type,
        j.job_status,
        j.progress,
        j.credits_reserved,
        j.created_at,
        j.started_at,
        j.updated_at,
        j.error_message,
        j.result_payload,
        j.property_id,
        p.title AS property_title
      FROM ai_jobs j
      LEFT JOIN properties p ON p.id = j.property_id
      WHERE j.user_id = $1
      ORDER BY j.created_at DESC
      LIMIT $2
      `,
      [userId, limit],
    );

    return Response.json(rows);
  } catch (error) {
    console.error("GET /api/ai/jobs error:", error);
    return Response.json({ error: "Internal Server Error" }, { status: 500 });
  }
}
