// Standardized contract PDF system metadata.
// {
//   status: 'succeeded'|'failed'|'disabled',
//   error: string|null,
//   updatedAt: ISO string,
//   storagePath: string|null
// }
export function withPdfSystemState(fields, pdfState) {
  const safeFields = fields && typeof fields === "object" ? fields : {};
  const existingSystem =
    safeFields._system && typeof safeFields._system === "object"
      ? safeFields._system
      : {};

  const existingPdf =
    existingSystem.pdf && typeof existingSystem.pdf === "object"
      ? existingSystem.pdf
      : {};

  const nextPdf = {
    status: pdfState?.status || existingPdf.status || null,
    error:
      pdfState?.error !== undefined
        ? pdfState.error
        : existingPdf.error !== undefined
          ? existingPdf.error
          : null,
    updatedAt: pdfState?.updatedAt || existingPdf.updatedAt || null,
    storagePath:
      pdfState?.storagePath !== undefined
        ? pdfState.storagePath
        : existingPdf.storagePath !== undefined
          ? existingPdf.storagePath
          : null,
  };

  return {
    ...safeFields,
    _system: {
      ...existingSystem,
      pdf: nextPdf,
    },
  };
}

export function withSignatureDefaults(fields) {
  const safe = fields && typeof fields === "object" ? fields : {};

  return {
    ...safe,
    signature_method: safe.signature_method || "off_platform",
    signed_status: safe.signed_status || "unsigned",
    signed_at: safe.signed_at || null,
    signed_by_agent_name: safe.signed_by_agent_name || null,
    signed_by_client_name: safe.signed_by_client_name || null,
  };
}

// NEW: lightweight audit trail helper (stored in filled_fields._system.audit[])
export function appendAudit(fields, entry) {
  const safeFields = fields && typeof fields === "object" ? fields : {};
  const existingSystem =
    safeFields._system && typeof safeFields._system === "object"
      ? safeFields._system
      : {};

  const existingAudit = Array.isArray(existingSystem.audit)
    ? existingSystem.audit
    : [];

  const action = String(entry?.action || "").trim();
  if (!action) {
    return safeFields;
  }

  const timestamp = String(entry?.timestamp || new Date().toISOString());
  const actor = String(entry?.actor || "agent");

  const changesRaw = entry?.changes;
  const changes =
    changesRaw && typeof changesRaw === "object" && !Array.isArray(changesRaw)
      ? changesRaw
      : null;

  // Normalized audit schema:
  // {
  //   action: 'marked_signed'|'marked_unsigned'|'pdf_regenerated'|'edited',
  //   timestamp: ISO string,
  //   actor: 'agent',
  //   changes: { field: { from, to } } | null
  // }
  const nextEntry = {
    action,
    timestamp,
    actor,
    changes,
  };

  return {
    ...safeFields,
    _system: {
      ...existingSystem,
      audit: [...existingAudit, nextEntry],
    },
  };
}
