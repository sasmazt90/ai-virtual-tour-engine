import sql from "@/app/api/utils/sql";
import { auth } from "@/auth";
import { getDbUserIdFromSession } from "@/app/api/utils/dbUser";
import { normalizePhoneToE164 } from "@/app/api/utils/phone";

export async function GET(request) {
  try {
    const session = await auth();
    if (!session || !session.user?.id) {
      return Response.json({ error: "Unauthorized" }, { status: 401 });
    }

    const userId = await getDbUserIdFromSession(session);
    if (!userId) {
      return Response.json({ error: "Unauthorized" }, { status: 401 });
    }

    const { searchParams } = new URL(request.url);
    const id = searchParams.get("id");
    const type = searchParams.get("type");
    const search = searchParams.get("search");

    if (id) {
      const rows = await sql(
        `SELECT * FROM clients WHERE user_id = $1 AND id = $2 LIMIT 1`,
        [userId, id],
      );
      return Response.json(rows[0] || null);
    }

    let queryStr = `SELECT * FROM clients WHERE user_id = $1`;
    const values = [userId];
    let paramIndex = 2;

    if (type && type !== "all") {
      queryStr += ` AND client_type = $${paramIndex}`;
      values.push(type);
      paramIndex++;
    }

    if (search) {
      queryStr += ` AND (
        LOWER(full_name) LIKE LOWER($${paramIndex}) OR
        LOWER(email) LIKE LOWER($${paramIndex}) OR
        LOWER(phone) LIKE LOWER($${paramIndex})
      )`;
      values.push(`%${search}%`);
      paramIndex++;
    }

    queryStr += ` ORDER BY full_name`;

    const clients = await sql(queryStr, values);
    return Response.json(clients);
  } catch (error) {
    console.error("GET /api/clients error:", error);
    return Response.json({ error: "Internal Server Error" }, { status: 500 });
  }
}

export async function POST(request) {
  try {
    const session = await auth();
    if (!session || !session.user?.id) {
      return Response.json({ error: "Unauthorized" }, { status: 401 });
    }

    const userId = await getDbUserIdFromSession(session);
    if (!userId) {
      return Response.json({ error: "Unauthorized" }, { status: 401 });
    }

    const body = await request.json();
    const { client_type, full_name, phone, email, notes, country, city } = body;

    if (!client_type || !full_name) {
      return Response.json(
        { error: "Type and name are required" },
        { status: 400 },
      );
    }

    const normalizedPhone = phone ? normalizePhoneToE164(phone) : null;

    // If this client already exists (by email or phone), update and reuse it.
    const matchRows = await sql(
      `
      SELECT *
      FROM clients
      WHERE user_id = $1
        AND (
          ($2::text IS NOT NULL AND email = $2)
          OR ($3::text IS NOT NULL AND phone = $3)
        )
      LIMIT 1
      `,
      [userId, email || null, normalizedPhone || null],
    );

    if (matchRows.length > 0) {
      const existing = matchRows[0];
      const updatedRows = await sql(
        `
        UPDATE clients
        SET
          client_type = $1,
          full_name = $2,
          email = $3,
          phone = $4,
          notes = $5,
          country = $6,
          city = $7
        WHERE id = $8
        RETURNING *
        `,
        [
          client_type,
          full_name,
          email || null,
          normalizedPhone || null,
          notes || existing.notes || null,
          country || existing.country || null,
          city || existing.city || null,
          existing.id,
        ],
      );
      return Response.json(updatedRows[0]);
    }

    const result = await sql`
      INSERT INTO clients (user_id, client_type, full_name, phone, email, notes, country, city)
      VALUES (
        ${userId},
        ${client_type},
        ${full_name},
        ${normalizedPhone || null},
        ${email || null},
        ${notes || null},
        ${country || null},
        ${city || null}
      )
      RETURNING *
    `;

    return Response.json(result[0], { status: 201 });
  } catch (error) {
    console.error("POST /api/clients error:", error);
    return Response.json({ error: "Internal Server Error" }, { status: 500 });
  }
}
