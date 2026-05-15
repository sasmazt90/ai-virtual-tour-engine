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

    // Performance/safety guard: this endpoint should never return large payloads.
    // If queries fail or are slow, return partial "unknown" values rather than breaking /profile/tools.
    const fallback = {
      activeCount: null,
      expiredCount: null,
      oldestActiveExpiresAt: null,
      partial: true,
    };

    try {
      const [_, counts, oldest] = await sql.transaction((txn) => [
        txn("SET LOCAL statement_timeout = '1500ms'"),
        txn(
          "SELECT COUNT(*) FILTER (WHERE expires_at IS NULL OR expires_at > NOW()) AS active_count, COUNT(*) FILTER (WHERE expires_at IS NOT NULL AND expires_at <= NOW()) AS expired_count FROM share_links WHERE user_id = $1",
          [userId],
        ),
        txn(
          "SELECT MIN(expires_at) AS oldest_active_expires_at FROM share_links WHERE user_id = $1 AND (expires_at IS NULL OR expires_at > NOW()) AND expires_at IS NOT NULL",
          [userId],
        ),
      ]);

      const row = counts?.[0] || {};
      const oldestRow = oldest?.[0] || {};

      return Response.json({
        activeCount: Number(row.active_count || 0),
        expiredCount: Number(row.expired_count || 0),
        oldestActiveExpiresAt: oldestRow.oldest_active_expires_at || null,
        partial: false,
      });
    } catch (innerError) {
      console.error("GET /api/share-links/status partial error:", innerError);
      return Response.json({
        ...fallback,
        error: "Share status temporarily unavailable",
      });
    }
  } catch (error) {
    console.error("GET /api/share-links/status error:", error);
    return Response.json({
      activeCount: null,
      expiredCount: null,
      oldestActiveExpiresAt: null,
      partial: true,
      error: "Share status temporarily unavailable",
    });
  }
}
