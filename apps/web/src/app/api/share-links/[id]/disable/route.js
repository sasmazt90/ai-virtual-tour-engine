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

    const rows = await sql(
      `
      UPDATE share_links
      SET expires_at = NOW()
      WHERE id = $1 AND user_id = $2
      RETURNING *
      `,
      [id, userId],
    );

    if (rows.length === 0) {
      return Response.json({ error: "Share link not found" }, { status: 404 });
    }

    return Response.json(rows[0]);
  } catch (error) {
    console.error("POST /api/share-links/[id]/disable error:", error);
    return Response.json({ error: "Internal Server Error" }, { status: 500 });
  }
}
