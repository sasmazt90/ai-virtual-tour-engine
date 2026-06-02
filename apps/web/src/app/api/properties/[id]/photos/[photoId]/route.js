import sql from "@/app/api/utils/sql";
import { auth } from "@/auth";
import { getDbUserIdFromSession } from "@/app/api/utils/dbUser";
import { deleteSupabaseStorageObjects } from "@/app/api/utils/storageCleanup";

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

    const propertyId = params?.id;
    const photoId = params?.photoId;

    if (!propertyId || !photoId) {
      return Response.json({ error: "Bad request" }, { status: 400 });
    }

    const rows = await sql(
      `
      DELETE FROM property_photos pp
      USING properties p
      WHERE pp.id = $1
        AND pp.property_id = $2
        AND p.id = pp.property_id
        AND p.user_id = $3
      RETURNING pp.id, pp.storage_path
      `,
      [photoId, propertyId, userId],
    );

    if (rows.length === 0) {
      return Response.json({ error: "Not found" }, { status: 404 });
    }

    const cleanup = await deleteSupabaseStorageObjects([rows[0].storage_path]);

    return Response.json({ success: true, id: rows[0].id, cleanup });
  } catch (error) {
    console.error("DELETE /api/properties/[id]/photos/[photoId] error:", error);
    return Response.json({ error: "Internal Server Error" }, { status: 500 });
  }
}
