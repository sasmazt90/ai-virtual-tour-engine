import sql from "@/app/api/utils/sql";
import { auth } from "@/auth";
import {
  appendAudit,
  buildContractHtml,
  getMissingFields,
  tryGenerateFillablePdfFromContractData,
  withPdfSystemState,
  withSignatureDefaults,
} from "@/app/api/utils/contractTemplates";
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

function stripPdfRegenLock(fields) {
  if (!fields || typeof fields !== "object") return fields;
  const sys =
    fields._system && typeof fields._system === "object"
      ? fields._system
      : null;
  if (!sys) return fields;
  if (!("pdfRegenLock" in sys)) return fields;

  const { pdfRegenLock, ...restSys } = sys;
  return {
    ...fields,
    _system: restSys,
  };
}

export async function POST(request, { params }) {
  const contractId = params?.id;

  // Lock is stored in filled_fields._system.pdfRegenLock (ISO timestamp)
  const lockNowIso = new Date().toISOString();

  let lockAcquired = false;

  try {
    const session = await auth();
    if (!session || !session.user?.id) {
      return Response.json({ error: "Unauthorized" }, { status: 401 });
    }

    const userId = await getDbUserIdFromSession(session);
    if (!userId) {
      return Response.json({ error: "Unauthorized" }, { status: 401 });
    }

    if (!contractId) {
      return Response.json({ error: "Bad request" }, { status: 400 });
    }

    // Acquire a per-contract lock (best-effort) to avoid concurrent regeneration.
    // Allow retry if lock is stale (>5 minutes).
    const lockRows = await sql(
      `
      UPDATE contracts co
      SET filled_fields = jsonb_set(
        COALESCE(co.filled_fields, '{}'::jsonb),
        '{_system,pdfRegenLock}',
        to_jsonb($3::text),
        true
      )
      FROM properties p
      WHERE co.property_id = p.id
        AND co.id = $1
        AND p.user_id = $2
        AND (
          (co.filled_fields #>> '{_system,pdfRegenLock}') IS NULL
          OR (co.filled_fields #>> '{_system,pdfRegenLock}')::timestamptz < now() - interval '5 minutes'
        )
      RETURNING co.*
      `,
      [contractId, userId, lockNowIso],
    );

    if (lockRows.length === 0) {
      return Response.json(
        {
          error:
            "PDF regeneration already in progress for this contract. Please try again in a moment.",
        },
        { status: 409 },
      );
    }

    lockAcquired = true;

    // Load full contract + related data (ownership already checked above)
    const rows = await sql(
      `
      SELECT
        co.*,
        p.title AS property_title,
        p.address_line,
        p.city,
        p.postal_code,
        p.country,
        p.price AS property_price,
        p.size_sqm,
        p.rooms,
        c.full_name AS client_name,
        c.email AS client_email,
        c.phone AS client_phone
      FROM contracts co
      JOIN properties p ON co.property_id = p.id
      LEFT JOIN clients c ON co.client_id = c.id
      WHERE co.id = $1 AND p.user_id = $2
      LIMIT 1
      `,
      [contractId, userId],
    );

    if (rows.length === 0) {
      return Response.json({ error: "Contract not found" }, { status: 404 });
    }

    const row = rows[0];

    const property = {
      id: row.property_id,
      title: row.property_title,
      address_line: row.address_line,
      city: row.city,
      postal_code: row.postal_code,
      country: row.country,
      price: row.property_price,
      size_sqm: row.size_sqm,
      rooms: row.rooms,
    };

    const client = {
      id: row.client_id,
      full_name: row.client_name,
      email: row.client_email,
      phone: row.client_phone,
    };

    const agent = await loadAgentInfo(userId);

    // Ensure signature tracking keys exist (legacy-safe)
    const filledFields = withSignatureDefaults(row.filled_fields || {});

    // NOTE: We do not support in-platform signing. Contracts may always be regenerated
    // to produce an editable PDF for off-platform signing.
    // (Any existing signed_status values are treated as informational only.)

    const missing = getMissingFields(row.template_type, filledFields);
    if (missing.length > 0) {
      return Response.json(
        {
          error: `Missing required fields for template: ${missing.join(", ")}`,
          missingFields: missing,
        },
        { status: 400 },
      );
    }

    // Legacy HTML (kept for backward compatibility)
    buildContractHtml({
      templateType: row.template_type,
      property,
      client,
      fields: filledFields,
      agent,
      contractMeta: {
        generatedAt: new Date().toISOString(),
        version: row?.version || 1,
      },
    });

    const existingPdfUrl = row.storage_path_pdf || null;

    // NEW: regeneration must produce a fillable (AcroForm) PDF.
    const pdfResultRaw = await tryGenerateFillablePdfFromContractData({
      templateType: row.template_type,
      property,
      client,
      fields: filledFields,
      agent,
      contractMeta: {
        generatedAt: new Date().toISOString(),
        version: row?.version || 1,
      },
    });

    // Safety requirement: only advance _system.pdf.updatedAt after successful upload.
    const pdfResult =
      pdfResultRaw.status === "succeeded"
        ? { ...pdfResultRaw, updatedAt: new Date().toISOString() }
        : { ...pdfResultRaw, updatedAt: null };

    // If regeneration fails/disabled, keep prior PDF URL (do not wipe it).
    const pdfStateForRecord =
      pdfResult.status === "succeeded"
        ? pdfResult
        : {
            status: pdfResult.status,
            error: pdfResult.error,
            updatedAt: null,
            storagePath: existingPdfUrl,
          };

    let fieldsWithState = withPdfSystemState(filledFields, pdfStateForRecord);

    // NEW: audit success only (pdf_regenerated)
    if (pdfResult.status === "succeeded" && pdfResult.storagePath) {
      fieldsWithState = appendAudit(fieldsWithState, {
        action: "pdf_regenerated",
        timestamp: pdfResult.updatedAt || new Date().toISOString(),
        actor: "agent",
        changes: null,
      });
    }

    const cleanFields = stripPdfRegenLock(fieldsWithState);

    if (pdfResult.status === "succeeded" && pdfResult.storagePath) {
      const updated = await sql(
        "UPDATE contracts SET storage_path_pdf = $1, filled_fields = $2 WHERE id = $3 RETURNING *",
        [pdfResult.storagePath, cleanFields, contractId],
      );
      return Response.json(updated[0]);
    }

    const updated = await sql(
      "UPDATE contracts SET filled_fields = $1 WHERE id = $2 RETURNING *",
      [cleanFields, contractId],
    );

    return Response.json(updated[0]);
  } catch (error) {
    console.error("POST /api/contracts/[id]/regenerate-pdf error:", error);
    return Response.json({ error: "Internal Server Error" }, { status: 500 });
  } finally {
    // Best-effort unlock so a transient error doesn't leave the contract stuck.
    // IMPORTANT: only clear if we actually acquired the lock (prevents unauthorized unlock attempts).
    try {
      if (lockAcquired && contractId) {
        await sql(
          "UPDATE contracts SET filled_fields = (COALESCE(filled_fields, '{}'::jsonb) #- '{_system,pdfRegenLock}') WHERE id = $1",
          [contractId],
        );
      }
    } catch (e) {
      console.error("Failed to clear pdfRegenLock", e);
    }
  }
}
