import sql from "@/app/api/utils/sql";
import { auth } from "@/auth";

export async function POST(request, { params }) {
  try {
    const session = await auth();
    if (!session || !session.user?.id) {
      return Response.json({ error: "Unauthorized" }, { status: 401 });
    }

    const userId = session.user.id;
    const id = params?.id;

    if (!id) {
      return Response.json({ error: "Bad request" }, { status: 400 });
    }

    const body = await request.json().catch(() => ({}));
    const extendDaysRaw = body?.extendDays;
    const extendDays = Number(extendDaysRaw);

    if (!Number.isFinite(extendDays) || extendDays <= 0) {
      return Response.json(
        { error: "extendDays (positive number) is required" },
        { status: 400 },
      );
    }

    const clamped = Math.min(extendDays, 365);

    const rows = await sql(
      `
      UPDATE share_links
      SET expires_at = (
        CASE
          WHEN expires_at IS NULL THEN NOW() + ($3::int * INTERVAL '1 day')
          WHEN expires_at < NOW() THEN NOW() + ($3::int * INTERVAL '1 day')
          ELSE expires_at + ($3::int * INTERVAL '1 day')
        END
      )
      WHERE id = $1 AND user_id = $2
      RETURNING *
      `,
      [id, userId, clamped],
    );

    if (rows.length === 0) {
      return Response.json({ error: "Share link not found" }, { status: 404 });
    }

    return Response.json(rows[0]);
  } catch (error) {
    console.error("POST /api/share-links/[id]/extend error:", error);
    return Response.json({ error: "Internal Server Error" }, { status: 500 });
  }
}
