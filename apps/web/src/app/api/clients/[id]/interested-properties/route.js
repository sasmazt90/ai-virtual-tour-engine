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

    const clientId = params?.id;
    if (!clientId) {
      return Response.json({ error: "Bad request" }, { status: 400 });
    }

    const rows = await sql(
      `
      SELECT
        p.id,
        p.title,
        p.property_status,
        p.address_line,
        p.city,
        p.postal_code,
        p.country,
        p.price,
        p.currency,
        p.housing_type,
        p.size_sqm,
        p.rooms,
        p.created_at,
        pic.created_at as linked_at
      FROM property_interested_clients pic
      JOIN properties p ON p.id = pic.property_id
      WHERE pic.client_id = $1
        AND p.user_id = $2
      ORDER BY pic.created_at DESC
      `,
      [clientId, userId],
    );

    return Response.json(rows);
  } catch (error) {
    console.error("GET /api/clients/[id]/interested-properties error:", error);
    return Response.json({ error: "Internal Server Error" }, { status: 500 });
  }
}
