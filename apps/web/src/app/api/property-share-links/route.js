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

function toUuidArray(value) {
  if (!Array.isArray(value)) return [];
  return Array.from(
    new Set(
      value
        .map((v) => (v === null || v === undefined ? "" : String(v).trim()))
        .filter(Boolean),
    ),
  );
}

function normalizeStagingType(raw) {
  const s = typeof raw === "string" ? raw.trim() : "";
  if (!s) return "";
  const normalized = s.toLowerCase().replace(/\s+/g, "_").replace(/-+/g, "_");

  // Keep aligned with DB enum public.staging_type
  const allowed = new Set([
    "default",
    "vacant",
    "minimalist",
    "luxury",
    "scandinavian",
    "classic",
    "modern",
    "custom",
  ]);

  return allowed.has(normalized) ? normalized : "";
}

function normalizeTourSlot(raw) {
  const sourceType =
    raw?.sourceType === "staging" || raw?.sourceType === "original"
      ? raw.sourceType
      : null;
  if (!sourceType) return null;

  if (sourceType === "original") {
    return { sourceType: "original", stagingType: null };
  }

  const st = normalizeStagingType(raw?.stagingType);
  if (!st) return null;
  return { sourceType: "staging", stagingType: st };
}

function tourSlotKey(slot) {
  if (slot?.sourceType === "original") return "original";
  if (slot?.sourceType === "staging" && slot?.stagingType)
    return `staging:${slot.stagingType}`;
  return "";
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

    const propertyId = body?.propertyId ? String(body.propertyId).trim() : "";
    const customerId = body?.customerId ? String(body.customerId).trim() : "";

    const stagingIds = toUuidArray(body?.stagingIds);
    const contractIds = toUuidArray(body?.contractIds);

    // NEW: type-based tour selection (share stores slot descriptors, not tour ids)
    const tourSlotsRaw = Array.isArray(body?.virtualTourSlots)
      ? body.virtualTourSlots
      : null;

    const tourSlots = tourSlotsRaw
      ? Array.from(
          new Map(
            tourSlotsRaw
              .map((s) => normalizeTourSlot(s))
              .filter(Boolean)
              .map((s) => [tourSlotKey(s), s]),
          ).values(),
        )
      : [];

    // Backward-compat: older clients may still send ids
    const virtualTourIds = toUuidArray(body?.virtualTourIds);

    // Expiry support (consistent with /api/share-links)
    const expiresInDays = normalizeExpiryDays(
      body?.expiresInDays ?? SHARE_LINK_DEFAULT_EXPIRY_DAYS,
    );
    const expiresAt = new Date(
      Date.now() + expiresInDays * 24 * 60 * 60 * 1000,
    ).toISOString();

    if (!propertyId || !customerId) {
      return Response.json(
        { error: "Please select a customer." },
        { status: 400 },
      );
    }

    // Validate property belongs to agent
    const props = await sql(
      "SELECT id FROM properties WHERE id = $1 AND user_id = $2 LIMIT 1",
      [propertyId, userId],
    );
    if (props.length === 0) {
      return Response.json(
        { error: "This property is not available." },
        { status: 404 },
      );
    }

    // Validate customer belongs to agent
    const customers = await sql(
      "SELECT id FROM clients WHERE id = $1 AND user_id = $2 LIMIT 1",
      [customerId, userId],
    );
    if (customers.length === 0) {
      return Response.json(
        { error: "This customer is not available." },
        { status: 404 },
      );
    }

    // Validate selected stagings belong to property
    if (stagingIds.length > 0) {
      const rows = await sql(
        "SELECT id FROM stagings WHERE property_id = $1 AND id = ANY($2::uuid[])",
        [propertyId, stagingIds],
      );
      if (rows.length !== stagingIds.length) {
        return Response.json(
          { error: "Some selected stagings are not available." },
          { status: 400 },
        );
      }
    }

    // Validate selected contracts belong to property
    if (contractIds.length > 0) {
      const rows = await sql(
        "SELECT id FROM contracts WHERE property_id = $1 AND id = ANY($2::uuid[])",
        [propertyId, contractIds],
      );
      if (rows.length !== contractIds.length) {
        return Response.json(
          { error: "Some selected contracts are not available." },
          { status: 400 },
        );
      }
    }

    // Validate selected tours belong to property.
    // New flow: validate by slot (property_id + source_type + staging_type)
    // Old flow: validate by id (kept for existing share links)
    if (tourSlots.length > 0) {
      const tourRows = await sql(
        "SELECT id, source_type, staging_type, tour_payload FROM virtual_tours WHERE property_id = $1",
        [propertyId],
      );

      const existingKeys = new Set();
      for (const t of tourRows) {
        const sourceType = t?.source_type;
        if (sourceType === "original") {
          existingKeys.add("original");
        } else if (sourceType === "staging") {
          const st =
            typeof t?.staging_type === "string"
              ? String(t.staging_type).trim()
              : "";
          if (st) existingKeys.add(`staging:${st}`);
        }
      }

      for (const slot of tourSlots) {
        const key = tourSlotKey(slot);
        if (!key || !existingKeys.has(key)) {
          return Response.json(
            { error: "Some selected virtual tours are not available." },
            { status: 400 },
          );
        }
      }
    } else if (virtualTourIds.length > 0) {
      const rows = await sql(
        "SELECT id FROM virtual_tours WHERE property_id = $1 AND id = ANY($2::uuid[])",
        [propertyId, virtualTourIds],
      );
      if (rows.length !== virtualTourIds.length) {
        return Response.json(
          { error: "Some selected virtual tours are not available." },
          { status: 400 },
        );
      }
    }

    // Generate unique slug with retries
    let slug = null;
    for (let i = 0; i < 6; i++) {
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
        { error: "Could not generate the share link. Please try again." },
        { status: 500 },
      );
    }

    const meta = tourSlots.length
      ? { include_tour_slots: tourSlots }
      : { include_tour_ids: virtualTourIds };

    // Expire any existing active links for the same property+client,
    // then create the new one (consistent with /api/share-links endpoint).
    const [, , createdRows] = await sql.transaction((txn) => [
      txn("SELECT pg_advisory_xact_lock(hashtext($1))", [
        `share-link:${userId}:${propertyId}:${customerId}`,
      ]),
      txn(
        `UPDATE share_links
         SET expires_at = NOW()
         WHERE user_id = $1
           AND property_id = $2
           AND client_id = $3
           AND (expires_at IS NULL OR expires_at > NOW())`,
        [userId, propertyId, customerId],
      ),
      txn(
        `INSERT INTO share_links (
           user_id, slug, property_id, client_id,
           include_staging_ids, include_contract_ids,
           meta, expires_at
         ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8)
         RETURNING *`,
        [
          userId,
          slug,
          propertyId,
          customerId,
          stagingIds,
          contractIds,
          meta,
          expiresAt,
        ],
      ),
    ]);

    const created = createdRows[0];

    return Response.json(
      {
        slug: created?.slug || slug,
        url: `/share/${created?.slug || slug}`,
      },
      { status: 201 },
    );
  } catch (error) {
    console.error("POST /api/property-share-links error:", error);
    return Response.json({ error: "Internal Server Error" }, { status: 500 });
  }
}
