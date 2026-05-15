import sql from "@/app/api/utils/sql";
import { auth } from "@/auth";

function isConfigured(val) {
  if (!val) return false;
  const str = String(val).trim();
  return !!str;
}

export async function GET() {
  // This endpoint is internal-only (agent-visible pages).
  try {
    const session = await auth();
    if (!session || !session.user?.id) {
      return Response.json({ error: "Unauthorized" }, { status: 401 });
    }

    const stripeConfigured =
      isConfigured(process.env.STRIPE_PUBLISHABLE_KEY) &&
      isConfigured(process.env.STRIPE_SECRET_KEY) &&
      isConfigured(process.env.STRIPE_WEBHOOK_KEY);

    const openAiConfigured = isConfigured(process.env.OPEN_AI_API_KEY);

    // Our PDF generation path uses Anything's PDF integration endpoint.
    // PDF generation uses PDF_SERVICE_URL when configured, otherwise the local PDF fallback.
    const pdfGenerationConfigured = isConfigured(process.env.APP_URL);

    // Counts only, single-row aggregate. Keep it fast + safe.
    let dataIntegrity = {
      contractsWithoutProperty: null,
      stagingsWithoutProperty: null,
      virtualToursWithoutProperty: null,
    };

    let partial = false;

    try {
      const [, rows] = await sql.transaction((txn) => [
        txn`SET LOCAL statement_timeout = '1500ms'`,
        txn`
          SELECT
            (
              SELECT COUNT(*)
              FROM contracts co
              LEFT JOIN properties p ON co.property_id = p.id
              WHERE co.property_id IS NULL OR p.id IS NULL
            ) AS contracts_without_property,
            (
              SELECT COUNT(*)
              FROM stagings s
              LEFT JOIN properties p2 ON s.property_id = p2.id
              WHERE s.property_id IS NULL OR p2.id IS NULL
            ) AS stagings_without_property,
            (
              SELECT COUNT(*)
              FROM virtual_tours vt
              LEFT JOIN properties p3 ON vt.property_id = p3.id
              WHERE vt.property_id IS NULL OR p3.id IS NULL
            ) AS virtual_tours_without_property
        `,
      ]);

      const row = rows?.[0] || {};

      dataIntegrity = {
        contractsWithoutProperty: Number(row.contracts_without_property || 0),
        stagingsWithoutProperty: Number(row.stagings_without_property || 0),
        virtualToursWithoutProperty: Number(
          row.virtual_tours_without_property || 0,
        ),
      };
    } catch (e) {
      partial = true;
      console.error("GET /api/tools/overview data integrity query failed:", e);
    }

    return Response.json({
      configurationStatus: {
        stripe: stripeConfigured ? "configured" : "missing",
        openai: openAiConfigured ? "configured" : "missing",
        pdfGeneration: pdfGenerationConfigured ? "configured" : "missing",
      },
      dataIntegrity,
      partial,
    });
  } catch (error) {
    console.error("GET /api/tools/overview error:", error);

    // Best-effort response: still do not expose any secret values.
    return Response.json(
      {
        configurationStatus: {
          stripe: "unknown",
          openai: "unknown",
          pdfGeneration: "unknown",
        },
        dataIntegrity: {
          contractsWithoutProperty: null,
          stagingsWithoutProperty: null,
          virtualToursWithoutProperty: null,
        },
        partial: true,
      },
      { status: 200 },
    );
  }
}
