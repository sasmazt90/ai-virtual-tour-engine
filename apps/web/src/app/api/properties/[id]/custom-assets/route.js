import sql from "@/app/api/utils/sql";
import { auth } from "@/auth";
import { getDbUserIdFromSession } from "@/app/api/utils/dbUser";

export async function GET(request, { params }) {
  try {
    const session = await auth();
    if (!session || !session.user?.id) {
      return Response.json({ error: "Unauthorized" }, { status: 401 });
    }

    const userId = await getDbUserIdFromSession(session);
    if (!userId) {
      return Response.json({ error: "Unauthorized" }, { status: 401 });
    }
    const propertyId = params.id;

    const props = await sql(
      "SELECT id FROM properties WHERE id = $1 AND user_id = $2 LIMIT 1",
      [propertyId, userId],
    );

    if (props.length === 0) {
      return Response.json({ error: "Property not found" }, { status: 404 });
    }

    const rows = await sql(
      "SELECT * FROM custom_assets WHERE property_id = $1 ORDER BY created_at DESC",
      [propertyId],
    );

    return Response.json(rows);
  } catch (error) {
    console.error("GET /api/properties/[id]/custom-assets error:", error);
    return Response.json({ error: "Internal Server Error" }, { status: 500 });
  }
}

export async function POST(request, { params }) {
  try {
    const session = await auth();
    if (!session || !session.user?.id) {
      return Response.json({ error: "Unauthorized" }, { status: 401 });
    }

    const userId = await getDbUserIdFromSession(session);
    if (!userId) {
      return Response.json({ error: "Unauthorized" }, { status: 401 });
    }
    const propertyId = params.id;
    const body = await request.json();

    const storage_path = body?.storage_path;
    const label = body?.label || null;

    if (!storage_path) {
      return Response.json(
        { error: "storage_path is required" },
        { status: 400 },
      );
    }

    const props = await sql(
      "SELECT id FROM properties WHERE id = $1 AND user_id = $2 LIMIT 1",
      [propertyId, userId],
    );

    if (props.length === 0) {
      return Response.json({ error: "Property not found" }, { status: 404 });
    }

    const rows = await sql(
      "INSERT INTO custom_assets (property_id, storage_path, label) VALUES ($1, $2, $3) RETURNING *",
      [propertyId, storage_path, label],
    );

    return Response.json(rows[0], { status: 201 });
  } catch (error) {
    console.error("POST /api/properties/[id]/custom-assets error:", error);
    return Response.json({ error: "Internal Server Error" }, { status: 500 });
  }
}
