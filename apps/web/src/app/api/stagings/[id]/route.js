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

    const stagingId = params?.id;
    if (!stagingId || typeof stagingId !== "string") {
      return Response.json(
        { error: "Staging id is required" },
        { status: 400 },
      );
    }

    // Verify ownership + get staging slot info
    const rows = await sql(
      `SELECT s.id, s.property_id, s.staging_type
       FROM stagings s
       JOIN properties p ON p.id = s.property_id
       WHERE s.id = $1 AND p.user_id = $2
       LIMIT 1`,
      [stagingId, userId],
    );

    if (rows.length === 0) {
      return Response.json({ error: "Not found" }, { status: 404 });
    }

    const propertyId = rows[0].property_id;
    const stagingType = rows[0].staging_type;

    // Delete associated staging tour(s) and the staging itself.
    // Also remove the staging id from any share links so old links don't break.
    await sql.transaction((txn) => [
      txn(
        "UPDATE share_links SET include_staging_ids = array_remove(include_staging_ids, $1::uuid) WHERE include_staging_ids IS NOT NULL AND $1::uuid = ANY(include_staging_ids)",
        [stagingId],
      ),
      txn(
        "DELETE FROM virtual_tours WHERE property_id = $1 AND source_type = 'staging' AND staging_type = $2",
        [propertyId, stagingType],
      ),
      txn("DELETE FROM stagings WHERE id = $1", [stagingId]),
    ]);

    return Response.json({ success: true, stagingId });
  } catch (error) {
    console.error("DELETE /api/stagings/[id] error:", error);
    return Response.json({ error: "Internal Server Error" }, { status: 500 });
  }
}
