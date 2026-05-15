import sql from "@/app/api/utils/sql";
import { auth } from "@/auth";
import { getDbUserIdFromSession } from "@/app/api/utils/dbUser";

function safeFilenamePart(str) {
  return String(str || "")
    .replaceAll(/[^a-zA-Z0-9._-]+/g, "-")
    .slice(0, 80);
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

    const contractId = params?.id;
    if (!contractId) {
      return Response.json({ error: "Bad request" }, { status: 400 });
    }

    const rows = await sql(
      `
      SELECT co.id, co.template_type, co.storage_path_pdf, co.pdf_url, co.filled_fields
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

    const pdfUrl =
      contract.storage_path_pdf ||
      contract.pdf_url ||
      contract?.filled_fields?._system?.pdf?.storagePath ||
      null;

    if (!pdfUrl) {
      return Response.json(
        { error: "PDF not available yet." },
        { status: 404 },
      );
    }

    const upstream = await fetch(pdfUrl);
    if (!upstream.ok) {
      const text = await upstream.text().catch(() => "");
      console.error("Contract PDF fetch failed:", upstream.status, text);
      return Response.json(
        { error: "This PDF cannot be accessed at the moment." },
        { status: 500 },
      );
    }

    const buffer = await upstream.arrayBuffer();

    const templatePart = safeFilenamePart(contract.template_type || "contract");
    const filename = `${templatePart}-${safeFilenamePart(contract.id)}.pdf`;

    const { searchParams } = new URL(request.url);
    const disposition = String(
      searchParams.get("disposition") || "",
    ).toLowerCase();
    const contentDisposition =
      disposition === "inline" ? "inline" : "attachment";

    return new Response(buffer, {
      status: 200,
      headers: {
        "Content-Type": "application/pdf",
        "Content-Disposition": `${contentDisposition}; filename="${filename}"`,
        "Cache-Control": "private, max-age=60",
      },
    });
  } catch (error) {
    console.error("GET /api/contracts/[id]/download error:", error);
    return Response.json({ error: "Internal Server Error" }, { status: 500 });
  }
}
