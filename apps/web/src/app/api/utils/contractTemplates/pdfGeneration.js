import { upload } from "@/app/api/utils/upload";
import { generateFillablePdfBytes } from "./pdfGenerator";
import { buildContractHtml } from "./htmlTemplates";

const HTML_TEMPLATE_TYPES = new Set([
  "agency_authorization",
  "buyer_representation",
  "handover_protocol",
  "offer_letter",
  "rental_agreement",
  "sale_agreement",
  "seller_listing_agreement",
  "tenant_representation",
  "viewing_report",
]);

// NEW: Generate a fillable PDF (AcroForm) from structured contract data.
export async function tryGenerateFillablePdfFromContractData({
  templateType,
  property,
  client,
  fields,
  agent,
  contractMeta,
}) {
  const updatedAt = new Date().toISOString();

  try {
    // Prefer HTML-to-PDF for our public templates (multi-page and closer to the provided templates).
    // Fall back to the in-house fillable PDF generator if the integration is unavailable.
    const t = String(templateType || "");
    if (HTML_TEMPLATE_TYPES.has(t)) {
      const html = buildContractHtml({
        templateType,
        property,
        client,
        fields,
        agent,
        contractMeta,
      });

      const pdfRes = await tryGeneratePdfFromHtml({ html });
      if (pdfRes.status === "succeeded") {
        return pdfRes;
      }
      // If integration failed/disabled, proceed to fillable fallback below.
    }

    const buffer = generateFillablePdfBytes({
      templateType,
      fields,
      agent,
      contractMeta,
    });

    const base64 = buffer.toString("base64");
    const pdfDataUri = `data:application/pdf;base64,${base64}`;

    const uploaded = await upload({ base64: pdfDataUri });

    if (!uploaded || !uploaded.url) {
      return {
        status: "failed",
        error: "We could not upload the generated PDF. Please try again.",
        updatedAt,
        storagePath: null,
      };
    }

    return {
      status: "succeeded",
      error: null,
      updatedAt,
      storagePath: uploaded.url,
    };
  } catch (e) {
    console.error("Fillable PDF generation failed:", e);
    return {
      status: "failed",
      error:
        "We could not generate an editable PDF for this contract. Please try again in a moment.",
      updatedAt,
      storagePath: null,
    };
  }
}

export async function tryGeneratePdfFromHtml({ html }) {
  // LEGACY: HTML-to-PDF output (not fillable). Kept for backwards compatibility.
  // New contract generation should use tryGenerateFillablePdfFromContractData.
  const updatedAt = new Date().toISOString();

  const pdfServiceUrl = process.env.PDF_SERVICE_URL;
  if (!pdfServiceUrl) {
    return {
      status: "disabled",
      error: "PDF_SERVICE_URL is not set; using local fillable PDF fallback",
      updatedAt,
      storagePath: null,
    };
  }

  const pdfUrl = new URL(pdfServiceUrl);

  let pdfRes;
  try {
    pdfRes = await fetch(pdfUrl.toString(), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        source: { html },
        styles: [
          {
            content: `
              body { font-family: Arial, Helvetica, sans-serif; font-size: 12px; }
              .page { padding: 24px; }
              h1 { font-size: 18px; margin: 0 0 10px; }
              h2 { font-size: 13px; margin: 16px 0 6px; }
              p { margin: 0 0 8px; line-height: 1.4; }
              .meta { margin-top: 10px; padding: 10px; border: 1px solid #d4d4d4; border-radius: 8px; }
              .meta div { margin: 4px 0; }
              .box { margin-top: 12px; border: 1px solid #d4d4d4; border-radius: 8px; padding: 10px; }
              .box .label { font-weight: bold; margin-bottom: 6px; }

              .doc-header { display: flex; align-items: center; gap: 10px; padding-bottom: 10px; border-bottom: 1px solid #e5e5e5; margin-bottom: 14px; }
              .doc-header .logo { width: 40px; height: 40px; object-fit: cover; border-radius: 6px; border: 1px solid #e5e5e5; }
              .doc-header .company { font-size: 14px; font-weight: 700; }

              .signature { margin-top: 18px; }
              .sigmeta { font-size: 11px; margin-bottom: 10px; }
              .sigrow { display: flex; gap: 20px; }
              .sig { flex: 1; }
              .line { border-bottom: 1px solid #000; height: 20px; margin-bottom: 6px; }
              .siglabel { font-size: 11px; }

              .doc-footer { margin-top: 18px; padding-top: 12px; border-top: 1px solid #e5e5e5; font-size: 11px; }
              .doc-footer div { margin: 2px 0; }
            `,
          },
        ],
      }),
    });
  } catch (e) {
    return {
      status: "failed",
      error: e?.message || "Could not reach PDF integration",
      updatedAt,
      storagePath: null,
    };
  }

  if (!pdfRes.ok) {
    const errText = await pdfRes.text().catch(() => "");
    console.error("PDF generation failed", pdfRes.status, errText);

    if (pdfRes.status === 404) {
      return {
        status: "disabled",
        error: `PDF integration unavailable (404)`,
        updatedAt,
        storagePath: null,
      };
    }

    return {
      status: "failed",
      error: `PDF integration returned [${pdfRes.status}] ${pdfRes.statusText}`,
      updatedAt,
      storagePath: null,
    };
  }

  try {
    const buffer = await pdfRes.arrayBuffer();
    const base64 = Buffer.from(buffer).toString("base64");
    const pdfDataUri = `data:application/pdf;base64,${base64}`;

    const uploaded = await upload({ base64: pdfDataUri });
    if (!uploaded || !uploaded.url) {
      return {
        status: "failed",
        error: "We could not upload the generated PDF. Please try again.",
        updatedAt,
        storagePath: null,
      };
    }

    return {
      status: "succeeded",
      error: null,
      updatedAt,
      storagePath: uploaded.url,
    };
  } catch (e) {
    return {
      status: "failed",
      error: e?.message || "Could not process PDF response",
      updatedAt,
      storagePath: null,
    };
  }
}
