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

    // Must be fast: counts only.
    // Never block UI if something goes wrong.
    try {
      const rows = await sql(
        "SELECT COUNT(*) FILTER (WHERE job_status = 'queued') AS queued, COUNT(*) FILTER (WHERE job_status = 'running') AS running FROM ai_jobs WHERE user_id = $1 AND job_status IN ('queued','running')",
        [userId],
      );

      const queued = Number(rows?.[0]?.queued || 0);
      const running = Number(rows?.[0]?.running || 0);
      const busy = queued + running > 0;

      return Response.json({ busy, queued, running });
    } catch (innerError) {
      console.error("GET /api/ai/busy partial error:", innerError);
      return Response.json(
        {
          busy: false,
          queued: 0,
          running: 0,
          partial: true,
        },
        { status: 200 },
      );
    }
  } catch (error) {
    console.error("GET /api/ai/busy error:", error);
    return Response.json(
      { busy: false, queued: 0, running: 0, partial: true },
      { status: 200 },
    );
  }
}
