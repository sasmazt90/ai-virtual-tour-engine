import sql from "@/app/api/utils/sql";
import { auth } from "@/auth";
import { getDbUserIdFromSession } from "@/app/api/utils/dbUser";

export async function DELETE(request, { params }) {
  try {
    const session = await auth();
    if (!session || !session.user?.id) {
      return Response.json({ error: "Unauthorized" }, { status: 401 });
    }

    const userId = await getDbUserIdFromSession(session);
    if (!userId) {
      return Response.json({ error: "Unauthorized" }, { status: 401 });
    }

    const tourId = params?.id;
    if (!tourId || typeof tourId !== "string") {
      return Response.json({ error: "Tour id is required" }, { status: 400 });
    }

    const owned = await sql(
      `SELECT vt.id
       FROM virtual_tours vt
       JOIN properties p ON p.id = vt.property_id
       WHERE vt.id = $1 AND p.user_id = $2
       LIMIT 1`,
      [tourId, userId],
    );

    if (owned.length === 0) {
      return Response.json({ error: "Not found" }, { status: 404 });
    }

    await sql("DELETE FROM virtual_tours WHERE id = $1", [tourId]);

    return Response.json({ success: true });
  } catch (error) {
    console.error("DELETE /api/virtual-tours/[id] error:", error);
    return Response.json({ error: "Internal Server Error" }, { status: 500 });
  }
}
