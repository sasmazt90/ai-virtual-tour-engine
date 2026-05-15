import sql from "@/app/api/utils/sql";
import { auth } from "@/auth";
import {
  appendAudit,
  getMissingFields,
  withSignatureDefaults,
} from "@/app/api/utils/contractTemplates";
import { getEditableFieldSet } from "@/utils/contractSchema.js";
import { getDbUserIdFromSession } from "@/app/api/utils/dbUser";

async function loadAgentInfo(userId) {
  const rows = await sql(
    `
    SELECT
      p.id,
      p.full_name as agent_name,
      p.company as company_name,
      au.email as agent_email,
      COALESCE(p.company_logo_url, au.image) as company_logo_url
    FROM profiles p
    LEFT JOIN auth_users au
      ON au.id = (
        CASE
          WHEN p.id::text LIKE '00000000-0000-0000-0000-%'
          THEN (right(p.id::text, 12))::int
          ELSE NULL
        END
      )
    WHERE p.id = $1
    LIMIT 1
    `,
    [userId],
  );
  return rows[0] || null;
}

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

    const id = params.id;

    const rows = await sql(
      `
      SELECT
        co.*,
        p.title AS property_title,
        p.address_line,
        p.city,
        p.postal_code,
        p.country,
        p.currency,
        p.price,
        p.housing_type,
        p.size_sqm,
        p.gross_area_sqm,
        p.rooms,
        p.floor_number,
        p.total_floors,
        p.title_deed_status,
        p.furnished_status,
        p.deposit,

        -- Contract customer (buyer/tenant)
        c.full_name AS client_name,
        c.email AS client_email,
        c.phone AS client_phone,
        c.city AS client_city,
        c.country AS client_country,

        -- Property owner (seller/landlord)
        oc.id AS owner_client_id,
        oc.full_name AS owner_name,
        oc.email AS owner_email,
        oc.phone AS owner_phone,
        oc.city AS owner_city,
        oc.country AS owner_country
      FROM contracts co
      JOIN properties p ON co.property_id = p.id
      LEFT JOIN clients c ON co.client_id = c.id
      LEFT JOIN clients oc ON p.owner_client_id = oc.id
      WHERE co.id = $1 AND p.user_id = $2
      LIMIT 1
      `,
      [id, userId],
    );

    if (rows.length === 0) {
      return Response.json({ error: "Contract not found" }, { status: 404 });
    }

    const agent = await loadAgentInfo(userId);

    // Enum-like suggestions sourced from existing DB values (per-agent)
    const housingRows = await sql(
      `
      SELECT DISTINCT housing_type
      FROM properties
      WHERE user_id = $1 AND housing_type IS NOT NULL AND housing_type <> ''
      ORDER BY housing_type
      LIMIT 50
      `,
      [userId],
    );

    const furnishedRows = await sql(
      `
      SELECT DISTINCT furnished_status
      FROM properties
      WHERE user_id = $1 AND furnished_status IS NOT NULL AND furnished_status <> ''
      ORDER BY furnished_status
      LIMIT 50
      `,
      [userId],
    );

    const enumOptions = {
      PROPERTY_TYPE: housingRows
        .map((r) => r?.housing_type)
        .filter((v) => typeof v === "string" && v.trim().length > 0),
      FURNISHED_STATUS: furnishedRows
        .map((r) => r?.furnished_status)
        .filter((v) => typeof v === "string" && v.trim().length > 0),
    };

    // Extend response with agent info + enum options (useful for autocomplete; no UI decisions).
    return Response.json({
      ...rows[0],
      agent,
      enumOptions,
    });
  } catch (error) {
    console.error("GET /api/contracts/[id] error:", error);
    return Response.json({ error: "Internal Server Error" }, { status: 500 });
  }
}

// NEW: allow editing contract fields only while unsigned.
// This updates the same contract record (no revisions) and preserves signature + _system state.
export async function PATCH(request, { params }) {
  try {
    const session = await auth();
    if (!session || !session.user?.id) {
      return Response.json({ error: "Unauthorized" }, { status: 401 });
    }

    const userId = await getDbUserIdFromSession(session);
    if (!userId) {
      return Response.json({ error: "Unauthorized" }, { status: 401 });
    }

    const id = params.id;

    const body = await request.json().catch(() => ({}));
    const incomingFields =
      body?.filledFields && typeof body.filledFields === "object"
        ? body.filledFields
        : null;

    if (!incomingFields) {
      return Response.json(
        { error: "filledFields (object) is required" },
        { status: 400 },
      );
    }

    const rows = await sql(
      `
      SELECT
        co.*,
        p.user_id AS owner_user_id
      FROM contracts co
      JOIN properties p ON co.property_id = p.id
      WHERE co.id = $1 AND p.user_id = $2
      LIMIT 1
      `,
      [id, userId],
    );

    if (rows.length === 0) {
      return Response.json({ error: "Contract not found" }, { status: 404 });
    }

    const contract = rows[0];
    const existing = withSignatureDefaults(contract.filled_fields || {});

    if (String(existing.signed_status || "unsigned") === "signed") {
      return Response.json(
        {
          error:
            "This contract is signed and can no longer be edited. Mark it as unsigned first if you need to correct it.",
        },
        { status: 409 },
      );
    }

    // Legal-field immutability (pre-sign, add-only)
    const editableSet = getEditableFieldSet(contract.template_type);
    const signatureKeys = new Set([
      "signature_method",
      "signed_status",
      "signed_at",
      "signed_by_agent_name",
      "signed_by_client_name",
    ]);

    const incomingKeys = Object.keys(incomingFields || {});
    const forbiddenKeys = incomingKeys.filter((k) => {
      if (!k || typeof k !== "string") return true;
      if (k === "_system" || k.startsWith("_system.")) return true;
      if (k.startsWith("_")) return true;
      if (signatureKeys.has(k)) return true;
      if (!editableSet.has(k)) return true;
      return false;
    });

    if (forbiddenKeys.length > 0) {
      return Response.json(
        {
          error: `Some fields are immutable and cannot be edited (pre-sign): ${forbiddenKeys.join(", ")}`,
          forbiddenKeys,
        },
        { status: 400 },
      );
    }

    const safeIncoming = {};
    for (const k of incomingKeys) {
      safeIncoming[k] = incomingFields[k];
    }

    const nowIso = new Date().toISOString();

    // Merge changes into existing fields.
    const merged = {
      ...existing,
      ...safeIncoming,
      // Always preserve signature tracking fields (not editable via PATCH)
      signature_method: existing.signature_method,
      signed_status: existing.signed_status,
      signed_at: existing.signed_at,
      signed_by_agent_name: existing.signed_by_agent_name,
      signed_by_client_name: existing.signed_by_client_name,
      // Preserve system state
      _system: existing._system,
    };

    const changes = {};
    for (const k of incomingKeys) {
      const fromVal = existing?.[k];
      const toVal = merged?.[k];
      const fromStr =
        fromVal === null || fromVal === undefined ? "" : String(fromVal);
      const toStr = toVal === null || toVal === undefined ? "" : String(toVal);
      if (fromStr !== toStr) {
        changes[k] = {
          from: fromVal === undefined ? null : fromVal,
          to: toVal === undefined ? null : toVal,
        };
      }
    }

    const mergedWithAudit =
      Object.keys(changes).length > 0
        ? appendAudit(merged, {
            action: "edited",
            timestamp: nowIso,
            actor: "agent",
            changes,
          })
        : merged;

    const missing = getMissingFields(contract.template_type, mergedWithAudit);
    if (missing.length > 0) {
      return Response.json(
        {
          error: `Missing required fields for template: ${missing.join(", ")}`,
          missingFields: missing,
        },
        { status: 400 },
      );
    }

    const updated = await sql(
      "UPDATE contracts SET filled_fields = $1 WHERE id = $2 RETURNING *",
      [mergedWithAudit, id],
    );

    return Response.json(updated[0]);
  } catch (error) {
    console.error("PATCH /api/contracts/[id] error:", error);
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

    const id = params?.id;
    if (!id) {
      return Response.json({ error: "Bad request" }, { status: 400 });
    }

    // Enforce ownership via property
    const rows = await sql(
      `
      DELETE FROM contracts co
      USING properties p
      WHERE co.property_id = p.id
        AND co.id = $1
        AND p.user_id = $2
      RETURNING co.id
      `,
      [id, userId],
    );

    if (rows.length === 0) {
      return Response.json({ error: "Contract not found" }, { status: 404 });
    }

    return Response.json({ success: true, id: rows[0].id });
  } catch (error) {
    console.error("DELETE /api/contracts/[id] error:", error);
    return Response.json({ error: "Internal Server Error" }, { status: 500 });
  }
}
