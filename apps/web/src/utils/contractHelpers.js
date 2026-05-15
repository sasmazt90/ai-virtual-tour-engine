export function getPdfMetadata(contract) {
  const rawPdfState = contract?.filled_fields?._system?.pdf || null;
  const pdfStatus = rawPdfState?.status || null;
  const pdfError = rawPdfState?.error || null;
  const pdfStoragePath = rawPdfState?.storagePath || null;
  const pdfUpdatedAt = rawPdfState?.updatedAt || null;

  const pdfRawUrl = contract?.storage_path_pdf || pdfStoragePath || null;

  // IMPORTANT: Prefer our authenticated download proxy so:
  // - downloads work reliably
  // - filenames are correct
  // - we don't depend on the upstream storage URL behavior
  const contractId = contract?.id ? String(contract.id) : null;
  const pdfInlineUrl =
    contractId && pdfRawUrl
      ? `/api/contracts/${contractId}/download?disposition=inline`
      : null;
  const pdfDownloadUrl =
    contractId && pdfRawUrl
      ? `/api/contracts/${contractId}/download?disposition=attachment`
      : null;

  let pdfMessage = null;
  if (!pdfRawUrl) {
    if (pdfStatus === "failed") {
      pdfMessage = pdfError
        ? `PDF not available: ${pdfError}`
        : "PDF generation failed. You can still use this contract.";
    } else if (pdfStatus === "disabled") {
      pdfMessage = pdfError
        ? `PDF generation is disabled: ${pdfError}`
        : "PDF generation is disabled.";
    } else {
      pdfMessage = "PDF not available yet.";
    }
  }

  return {
    pdfStatus,
    pdfError,
    pdfStoragePath,
    pdfUpdatedAt,
    pdfRawUrl,
    pdfInlineUrl,
    pdfDownloadUrl,
    pdfMessage,
  };
}

export function getSignatureMetadata(contract) {
  const signatureMethod =
    contract?.filled_fields?.signature_method || "off_platform";
  const signedStatus = contract?.filled_fields?.signed_status || "unsigned";
  const signedAt = contract?.filled_fields?.signed_at || null;

  const isSigned = signedStatus === "signed";

  return {
    signatureMethod,
    signedStatus,
    signedAt,
    isSigned,
  };
}

export function getAuditEntries(contract) {
  const raw = Array.isArray(contract?.filled_fields?._system?.audit)
    ? contract.filled_fields._system.audit
    : [];

  // Normalize + sort chronologically (oldest -> newest)
  const normalized = raw
    .map((a, idx) => {
      const action = a?.action ? String(a.action) : "";
      const timestamp = a?.timestamp ? String(a.timestamp) : "";
      const actor = a?.actor ? String(a.actor) : "";
      const changesRaw = a?.changes;
      const changes =
        changesRaw &&
        typeof changesRaw === "object" &&
        !Array.isArray(changesRaw)
          ? changesRaw
          : null;
      // Preserve legacy notes if present (read-only UI can show it)
      const notes = a?.notes ? String(a.notes) : null;
      return { action, timestamp, actor, changes, notes, _idx: idx };
    })
    .filter((a) => a.action);

  const toTime = (t) => {
    const ms = new Date(t).getTime();
    return Number.isFinite(ms) ? ms : null;
  };

  return normalized.slice().sort((a, b) => {
    const ta = toTime(a.timestamp);
    const tb = toTime(b.timestamp);
    if (ta === null && tb === null) return a._idx - b._idx;
    if (ta === null) return 1;
    if (tb === null) return -1;
    return ta - tb;
  });
}
