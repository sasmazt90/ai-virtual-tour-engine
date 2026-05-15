import sql from "@/app/api/utils/sql";
import { auth } from "@/auth";
import {
  appendAudit,
  withSignatureDefaults,
} from "@/app/api/utils/contractTemplates";
import { getDbUserIdFromSession } from "@/app/api/utils/dbUser";

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

    const contractId = params?.id;

    if (!contractId) {
      return Response.json({ error: "Bad request" }, { status: 400 });
    }

    // Load + verify ownership via properties.user_id
    const rows = await sql(
      `
      SELECT co.id, co.filled_fields
      FROM contracts co
      JOIN properties p ON co.property_id = p.id
      WHERE co.id = $1 AND p.user_id = $2
      LIMIT 1
      `,
      [contractId, userId],
    );

    if (rows.length === 0) {
      return Response.json({ error: "Contract not found" }, { status: 404 });
    }

    const contract = rows[0];
    const existing = withSignatureDefaults(contract.filled_fields || {});

    if (
      existing.signature_method &&
      existing.signature_method !== "off_platform"
    ) {
      return Response.json(
        {
          error:
            "Cannot mark this contract as unsigned because signature_method is not off_platform",
        },
        { status: 400 },
      );
    }

    const alreadyUnsigned =
      String(existing.signed_status || "unsigned") !== "signed";

    if (alreadyUnsigned && !existing.signed_at) {
      // Return as-is
      const full = await sql("SELECT * FROM contracts WHERE id = $1 LIMIT 1", [
        contractId,
      ]);
      return Response.json(full[0]);
    }

    const nowIso = new Date().toISOString();

    let nextFields = {
      ...existing,
      signature_method: "off_platform",
      signed_status: "unsigned",
      signed_at: null,
      // Preserve signed_by_* names for correction purposes
      signed_by_agent_name: existing.signed_by_agent_name || null,
      signed_by_client_name: existing.signed_by_client_name || null,
    };

    if (!alreadyUnsigned) {
      nextFields = appendAudit(nextFields, {
        action: "marked_unsigned",
        timestamp: nowIso,
        actor: "agent",
        changes: null,
      });
    }

    const updated = await sql(
      "UPDATE contracts SET filled_fields = $1 WHERE id = $2 RETURNING *",
      [nextFields, contractId],
    );

    return Response.json(updated[0]);
  } catch (error) {
    console.error("POST /api/contracts/[id]/mark-unsigned error:", error);
    return Response.json({ error: "Internal Server Error" }, { status: 500 });
  }
}
