import sql from "@/app/api/utils/sql";

function safeFilenamePart(str) {
  return String(str || "")
    .replaceAll(/[^a-zA-Z0-9._-]+/g, "-")
    .slice(0, 80);
}

export async function GET(request, { params }) {
  try {
    const slug = params?.slug;
    const contractId = params?.contractId;

    if (!slug || !contractId) {
      return Response.json(
        { error: "This file cannot be accessed at the moment." },
        { status: 400 },
      );
    }

    const links = await sql(
      `
      SELECT *
      FROM share_links
      WHERE slug = $1
      LIMIT 1
      `,
      [slug],
    );

    if (links.length === 0) {
      return Response.json(
        { error: "This file is no longer available." },
        { status: 404 },
      );
    }

    const link = links[0];

    if (link.expires_at) {
      const expires = new Date(link.expires_at);
      if (expires.getTime() < Date.now()) {
        return Response.json(
          { error: "This file is no longer available." },
          { status: 410 },
        );
      }
    }

    const includeContractIds = Array.isArray(link.include_contract_ids)
      ? link.include_contract_ids
      : [];

    const allowed = includeContractIds.some(
      (id) => String(id) === String(contractId),
    );
    if (!allowed) {
      return Response.json(
        { error: "This file is no longer available." },
        { status: 404 },
      );
    }

    // Also ensure the contract belongs to the same property as the share link.
    const rows = await sql(
      `
      SELECT co.id, co.template_type, co.storage_path_pdf, co.filled_fields
      FROM contracts co
      WHERE co.id = $1 AND co.property_id = $2
      LIMIT 1
      `,
      [contractId, link.property_id],
    );

    if (rows.length === 0) {
      return Response.json(
        { error: "This file is no longer available." },
        { status: 404 },
      );
    }

    const contract = rows[0];

    // Prefer the explicit column; fall back to standardized _system.pdf.storagePath (legacy-safe)
    const pdfUrl =
      contract.storage_path_pdf ||
      contract?.filled_fields?._system?.pdf?.storagePath ||
      null;

    if (!pdfUrl) {
      return Response.json(
        { error: "This file is no longer available." },
        { status: 404 },
      );
    }

    // Fetch and stream the PDF without exposing the underlying storage URL.
    const upstream = await fetch(pdfUrl);
    if (!upstream.ok) {
      const text = await upstream.text().catch(() => "");
      console.error("Share contract PDF fetch failed:", upstream.status, text);
      return Response.json(
        { error: "This file cannot be accessed at the moment." },
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
        // Keep it short-lived; the slug itself is already an access token.
        "Cache-Control": "private, max-age=60",
      },
    });
  } catch (error) {
    console.error(
      "GET /api/share/[slug]/contracts/[contractId]/download error:",
      error,
    );
    return Response.json(
      { error: "This file cannot be accessed at the moment." },
      { status: 500 },
    );
  }
}
