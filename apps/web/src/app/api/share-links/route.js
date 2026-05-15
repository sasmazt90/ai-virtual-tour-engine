import sql from "@/app/api/utils/sql";
import { auth } from "@/auth";
import { getDbUserIdFromSession } from "@/app/api/utils/dbUser";
import {
  SHARE_LINK_DEFAULT_EXPIRY_DAYS,
  normalizeExpiryDays,
} from "@/utils/shareLinksConfig";

function randomSlug(len = 12) {
  const chars = "abcdefghijklmnopqrstuvwxyz0123456789";
  let out = "";
  for (let i = 0; i < len; i++) {
    out += chars[Math.floor(Math.random() * chars.length)];
  }
  return out;
}

function computeExpiresAt({ expiresAt, expiresInDays }) {
  if (expiresAt) {
    const d = new Date(expiresAt);
    if (!Number.isNaN(d.getTime())) return d.toISOString();
  }

  const days = normalizeExpiryDays(expiresInDays);
  const d = new Date(Date.now() + days * 24 * 60 * 60 * 1000);
  return d.toISOString();
}

function isSingleId(value) {
  if (!value) return false;
  if (Array.isArray(value)) return false;
  if (typeof value !== "string") return false;
  const trimmed = value.trim();
  if (!trimmed) return false;
  return true;
}

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

    const url = new URL(request.url);
    const propertyId = url.searchParams.get("propertyId");
    const clientId = url.searchParams.get("clientId");
    const activeOnly = url.searchParams.get("activeOnly") === "1";
    const limitRaw = url.searchParams.get("limit");
    const limit = limitRaw
      ? Math.max(1, Math.min(Number(limitRaw) || 0, 50))
      : null;

    const where = ["sl.user_id = $1"];
    const values = [userId];
    let idx = 2;

    if (propertyId && isSingleId(propertyId)) {
      where.push(`sl.property_id = $${idx}`);
      values.push(propertyId);
      idx += 1;
    }

    if (clientId && isSingleId(clientId)) {
      where.push(`sl.client_id = $${idx}`);
      values.push(clientId);
      idx += 1;
    }

    if (activeOnly) {
      where.push("(sl.expires_at IS NULL OR sl.expires_at > NOW())");
    }

    const whereClause = where.length ? `WHERE ${where.join(" AND ")}` : "";
    const limitClause = limit ? `LIMIT ${limit}` : "";

    const query = `
      SELECT
        sl.*,
        p.title as property_title,
        c.full_name as client_name
      FROM share_links sl
      LEFT JOIN properties p ON sl.property_id = p.id
      LEFT JOIN clients c ON sl.client_id = c.id
      ${whereClause}
      ORDER BY sl.created_at DESC
      ${limitClause}
    `;

    const rows = await sql(query, values);

    return Response.json(rows);
  } catch (error) {
    console.error("GET /api/share-links error:", error);
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

    const propertyId = body?.propertyId;
    const clientId = body?.clientId;

    const includeStagingIds = Array.isArray(body?.includeStagingIds)
      ? body.includeStagingIds
      : null;
    const includeTourId = body?.includeTourId || null;
    const includeContractIds = Array.isArray(body?.includeContractIds)
      ? body.includeContractIds
      : null;

    const expiresAt = computeExpiresAt({
      expiresAt: body?.expiresAt || null,
      expiresInDays: body?.expiresInDays,
    });

    // Client selection is mandatory and must be unambiguous (exactly one id)
    if (!isSingleId(propertyId) || !isSingleId(clientId)) {
      return Response.json(
        { error: "propertyId and clientId are required (single values)" },
        { status: 400 },
      );
    }

    // Ensure property and client belong to this user
    const props = await sql(
      "SELECT id FROM properties WHERE id = $1 AND user_id = $2 LIMIT 1",
      [propertyId, userId],
    );
    if (props.length === 0) {
      return Response.json({ error: "Property not found" }, { status: 404 });
    }

    const clients = await sql(
      "SELECT id FROM clients WHERE id = $1 AND user_id = $2 LIMIT 1",
      [clientId, userId],
    );
    if (clients.length === 0) {
      return Response.json({ error: "Client not found" }, { status: 404 });
    }

    // Generate unique slug with a couple retries
    let slug = null;
    for (let i = 0; i < 5; i++) {
      const candidate = randomSlug(12);
      const existing = await sql(
        "SELECT id FROM share_links WHERE slug = $1 LIMIT 1",
        [candidate],
      );
      if (existing.length === 0) {
        slug = candidate;
        break;
      }
    }

    if (!slug) {
      return Response.json(
        { error: "Could not generate share link" },
        { status: 500 },
      );
    }

    // Enforce single active link per (user, property, client) in a transaction-safe way.
    // We use an advisory transaction lock to prevent races when two links are created quickly.
    const key = `share-link:${userId}:${propertyId}:${clientId}`;

    // IMPORTANT: Anything sql.transaction() expects an array of queries (or a function returning an array).
    const [, , createdRows] = await sql.transaction((txn) => [
      txn("SELECT pg_advisory_xact_lock(hashtext($1))", [key]),
      txn(
        `
          UPDATE share_links
          SET expires_at = NOW()
          WHERE user_id = $1
            AND property_id = $2
            AND client_id = $3
            AND (expires_at IS NULL OR expires_at > NOW())
        `,
        [userId, propertyId, clientId],
      ),
      txn(
        `
          INSERT INTO share_links (
            user_id,
            slug,
            property_id,
            client_id,
            include_staging_ids,
            include_tour_id,
            include_contract_ids,
            expires_at
          )
          VALUES ($1,$2,$3,$4,$5,$6,$7,$8)
          RETURNING *
        `,
        [
          userId,
          slug,
          propertyId,
          clientId,
          includeStagingIds,
          includeTourId,
          includeContractIds,
          expiresAt,
        ],
      ),
    ]);

    return Response.json(createdRows[0], { status: 201 });
  } catch (error) {
    console.error("POST /api/share-links error:", error);
    return Response.json({ error: "Internal Server Error" }, { status: 500 });
  }
}
