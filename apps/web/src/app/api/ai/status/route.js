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

    // Performance/safety guard: keep this endpoint fast and small.
    // If something goes wrong (DB hiccup / slow query), return partial "unknown" values
    // instead of failing /profile/tools.
    const fallback = {
      lastStagingJob: null,
      lastVirtualTourJob: null,
      lastCompletedAt: null,
      partial: true,
    };

    try {
      const [_, lastStaging, lastTour, lastCompleted] = await sql.transaction(
        (txn) => [
          // Hard limit to avoid hanging UI (best-effort; if unsupported, it will just error and we fall back)
          txn("SET LOCAL statement_timeout = '1500ms'"),
          txn(
            "SELECT job_status, created_at, updated_at FROM ai_jobs WHERE user_id = $1 AND job_type = 'staging' ORDER BY created_at DESC LIMIT 1",
            [userId],
          ),
          txn(
            "SELECT job_status, created_at, updated_at FROM ai_jobs WHERE user_id = $1 AND job_type = 'virtual_tour' ORDER BY created_at DESC LIMIT 1",
            [userId],
          ),
          txn(
            "SELECT updated_at FROM ai_jobs WHERE user_id = $1 AND job_status IN ('succeeded','failed') ORDER BY updated_at DESC LIMIT 1",
            [userId],
          ),
        ],
      );

      const staging = lastStaging?.[0] || null;
      const tour = lastTour?.[0] || null;
      const completed = lastCompleted?.[0] || null;

      return Response.json({
        lastStagingJob: staging,
        lastVirtualTourJob: tour,
        lastCompletedAt: completed?.updated_at || null,
        partial: false,
      });
    } catch (innerError) {
      console.error("GET /api/ai/status partial error:", innerError);
      return Response.json({
        ...fallback,
        error: "AI status temporarily unavailable",
      });
    }
  } catch (error) {
    console.error("GET /api/ai/status error:", error);
    // Defensive: do not break /profile/tools for signed-in users.
    return Response.json({
      lastStagingJob: null,
      lastVirtualTourJob: null,
      lastCompletedAt: null,
      partial: true,
      error: "AI status temporarily unavailable",
    });
  }
}
