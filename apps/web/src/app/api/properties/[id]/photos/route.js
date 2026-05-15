import sql from "@/app/api/utils/sql";
import { auth } from "@/auth";
import { getDbUserIdFromSession } from "@/app/api/utils/dbUser";

function normalizeUrls(raw) {
  const arr = Array.isArray(raw) ? raw : [];
  const out = [];
  for (const v of arr) {
    if (typeof v !== "string") continue;
    const s = v.trim();
    if (!s) continue;
    out.push(s);
  }
  return out;
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

    const propertyId = params?.id;
    if (!propertyId) {
      return Response.json({ error: "Bad request" }, { status: 400 });
    }

    const body = await request.json().catch(() => ({}));
    const urls = normalizeUrls(body?.photoUrls);

    if (urls.length === 0) {
      return Response.json({ error: "photoUrls is required" }, { status: 400 });
    }

    const owned = await sql(
      "SELECT id FROM properties WHERE id = $1 AND user_id = $2 LIMIT 1",
      [propertyId, userId],
    );

    if (owned.length === 0) {
      return Response.json({ error: "Property not found" }, { status: 404 });
    }

    const maxRows = await sql(
      "SELECT COALESCE(MAX(sort_order), 0) as max_sort FROM property_photos WHERE property_id = $1",
      [propertyId],
    );

    let sort = Number(maxRows?.[0]?.max_sort || 0);

    const created = [];
    for (const u of urls) {
      sort += 1;
      const rows = await sql(
        "INSERT INTO property_photos (property_id, storage_path, sort_order) VALUES ($1, $2, $3) RETURNING *",
        [propertyId, u, sort],
      );
      if (rows[0]) created.push(rows[0]);
    }

    return Response.json({ photos: created }, { status: 201 });
  } catch (error) {
    console.error("POST /api/properties/[id]/photos error:", error);
    return Response.json({ error: "Internal Server Error" }, { status: 500 });
  }
}
