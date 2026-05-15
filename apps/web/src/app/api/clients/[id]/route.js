import sql from "@/app/api/utils/sql";
import { auth } from "@/auth";
import { getDbUserIdFromSession } from "@/app/api/utils/dbUser";
import { normalizePhoneToE164 } from "@/app/api/utils/phone";

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

    const clientId = params?.id;
    if (!clientId) {
      return Response.json({ error: "Client id is required" }, { status: 400 });
    }

    const body = await request.json().catch(() => ({}));

    const fields = {
      client_type: body?.client_type,
      full_name: body?.full_name,
      email: body?.email,
      phone: body?.phone,
      notes: body?.notes,
      country: body?.country,
      city: body?.city,
    };

    if (fields.phone) {
      fields.phone = normalizePhoneToE164(fields.phone);
    }

    // Ensure ownership
    const existing = await sql(
      "SELECT id FROM clients WHERE id = $1 AND user_id = $2 LIMIT 1",
      [clientId, userId],
    );

    if (existing.length === 0) {
      return Response.json({ error: "Client not found" }, { status: 404 });
    }

    const setClauses = [];
    const values = [];
    let idx = 1;

    for (const [key, val] of Object.entries(fields)) {
      if (val === undefined) continue;
      setClauses.push(`${key} = $${idx}`);
      values.push(val === "" ? null : val);
      idx++;
    }

    if (setClauses.length === 0) {
      return Response.json({ error: "No fields to update" }, { status: 400 });
    }

    values.push(clientId, userId);

    const whereIdParam = idx;
    const whereUserParam = idx + 1;
    const query = `UPDATE clients SET ${setClauses.join(", ")} WHERE id = $${whereIdParam} AND user_id = $${whereUserParam} RETURNING *`;

    const rows = await sql(query, values);
    return Response.json(rows[0]);
  } catch (error) {
    console.error("PUT /api/clients/[id] error:", error);
    return Response.json({ error: "Internal Server Error" }, { status: 500 });
  }
}

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

    const clientId = params?.id;
    if (!clientId) {
      return Response.json({ error: "Client id is required" }, { status: 400 });
    }

    const rows = await sql(
      "DELETE FROM clients WHERE id = $1 AND user_id = $2 RETURNING id",
      [clientId, userId],
    );

    if (rows.length === 0) {
      return Response.json({ error: "Client not found" }, { status: 404 });
    }

    return Response.json({ success: true, id: rows[0].id });
  } catch (error) {
    console.error("DELETE /api/clients/[id] error:", error);
    return Response.json({ error: "Internal Server Error" }, { status: 500 });
  }
}
