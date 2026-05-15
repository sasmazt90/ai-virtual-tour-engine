import sql from "@/app/api/utils/sql";
import { auth } from "@/auth";
import { getDbUserIdFromSession } from "@/app/api/utils/dbUser";
import { withSignatureDefaults } from "@/app/api/utils/contractTemplates";

function safeString(v) {
  return typeof v === "string" ? v.trim() : "";
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

    const body = await request.json().catch(() => ({}));

    const propertyId = safeString(body?.propertyId);
    const pdfUrl = safeString(body?.pdfUrl);
    const fileName = safeString(body?.fileName);
    const requestedClientId = safeString(body?.clientId);

    if (!propertyId) {
      return Response.json({ error: "Missing property." }, { status: 400 });
    }

    if (!pdfUrl) {
      return Response.json(
        { error: "Please upload a PDF file." },
        { status: 400 },
      );
    }

    const props = await sql(
      "SELECT id, owner_client_id FROM properties WHERE id = $1 AND user_id = $2 LIMIT 1",
      [propertyId, userId],
    );

    if (props.length === 0) {
      return Response.json({ error: "Property not found." }, { status: 404 });
    }

    const property = props[0];

    // NEW: allow attaching to a specific client card.
    let finalClientId = property.owner_client_id || null;
    if (requestedClientId) {
      const clientRows = await sql(
        "SELECT id FROM clients WHERE id = $1 AND user_id = $2 LIMIT 1",
        [requestedClientId, userId],
      );
      if (clientRows.length === 0) {
        return Response.json({ error: "Client not found." }, { status: 404 });
      }
      finalClientId = clientRows[0].id;
    }

    const metadata = {
      display_name: fileName || "Uploaded contract",
      uploaded_at: new Date().toISOString(),
    };

    const inserted = await sql(
      `
      INSERT INTO contracts (
        property_id,
        client_id,
        template_type,
        filled_fields,
        storage_path_pdf,
        source_type,
        pdf_url,
        metadata
      )
      VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
      RETURNING *
      `,
      [
        propertyId,
        finalClientId,
        "uploaded_pdf",
        withSignatureDefaults({}),
        pdfUrl,
        "upload",
        pdfUrl,
        metadata,
      ],
    );

    return Response.json(inserted[0], { status: 201 });
  } catch (error) {
    console.error("POST /api/contracts/upload error:", error);
    return Response.json({ error: "Internal Server Error" }, { status: 500 });
  }
}
