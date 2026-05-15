import sql from "@/app/api/utils/sql";
import { auth } from "@/auth";
import { getDbUserIdFromSession } from "@/app/api/utils/dbUser";

export async function PUT(request, { params }) {
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
    const clientIds = Array.isArray(body?.clientIds) ? body.clientIds : null;

    if (!clientIds) {
      return Response.json(
        { error: "clientIds must be an array" },
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

    // Only allow attaching clients owned by this user
    const valid = await sql(
      "SELECT id FROM clients WHERE user_id = $1 AND id = ANY($2::uuid[])",
      [userId, clientIds],
    );

    const validIds = valid.map((r) => r.id);

    // IMPORTANT: avoid unsupported sql.transaction(async ...). Use a single atomic statement.
    await sql(
      `
        WITH deleted AS (
          DELETE FROM property_interested_clients
          WHERE property_id = $1
        )
        INSERT INTO property_interested_clients (property_id, client_id)
        SELECT $1, unnest($2::uuid[]);
      `,
      [propertyId, validIds],
    );

    return Response.json({ success: true, clientIds: validIds });
  } catch (error) {
    console.error("PUT /api/properties/[id]/interested-clients error:", error);
    return Response.json({ error: "Internal Server Error" }, { status: 500 });
  }
}
