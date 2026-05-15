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

    const body = await request.json().catch(() => ({}));

    // Load + verify ownership via properties.user_id
    const rows = await sql(
      `
      SELECT co.id, co.filled_fields, co.property_id
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

    // Guard: this endpoint only supports off-platform signature tracking.
    if (
      existing.signature_method &&
      existing.signature_method !== "off_platform"
    ) {
      return Response.json(
        {
          error:
            "Cannot mark this contract as signed because signature_method is not off_platform",
        },
        { status: 400 },
      );
    }

    const alreadySigned =
      String(existing.signed_status || "unsigned") === "signed";

    const agentNameFromBody =
      typeof body?.signed_by_agent_name === "string"
        ? body.signed_by_agent_name.trim()
        : "";
    const clientNameFromBody =
      typeof body?.signed_by_client_name === "string"
        ? body.signed_by_client_name.trim()
        : "";

    // Do not wipe names on re-mark; only update if a non-empty value is provided.
    const agentName = agentNameFromBody
      ? agentNameFromBody
      : existing.signed_by_agent_name || existing.agentName || null;

    const clientName = clientNameFromBody
      ? clientNameFromBody
      : existing.signed_by_client_name || existing.clientName || null;

    // Validate required names before allowing signed_status = 'signed'
    if (!agentName || !String(agentName).trim()) {
      return Response.json(
        { error: "signed_by_agent_name is required to mark as signed" },
        { status: 400 },
      );
    }

    if (!clientName || !String(clientName).trim()) {
      return Response.json(
        { error: "signed_by_client_name is required to mark as signed" },
        { status: 400 },
      );
    }

    // Idempotency: keep the original signed_at if already signed.
    const nowIso = new Date().toISOString();
    const signedAt =
      alreadySigned && existing.signed_at ? existing.signed_at : nowIso;

    // If this is a repeat call and nothing would change, return as-is.
    const noChanges =
      alreadySigned &&
      signedAt === existing.signed_at &&
      String(agentName) === String(existing.signed_by_agent_name || "") &&
      String(clientName) === String(existing.signed_by_client_name || "");

    if (noChanges) {
      // Return the full contract shape for consistency.
      const full = await sql("SELECT * FROM contracts WHERE id = $1 LIMIT 1", [
        contractId,
      ]);
      return Response.json(full[0]);
    }

    let nextFields = {
      ...existing,
      signature_method: "off_platform",
      signed_status: "signed",
      signed_at: signedAt,
      signed_by_agent_name: agentName,
      signed_by_client_name: clientName,
    };

    // Audit: only record the state change when transitioning to signed.
    if (!alreadySigned) {
      nextFields = appendAudit(nextFields, {
        action: "marked_signed",
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
    console.error("POST /api/contracts/[id]/mark-signed error:", error);
    return Response.json({ error: "Internal Server Error" }, { status: 500 });
  }
}
