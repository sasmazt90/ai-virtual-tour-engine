export function escapeHtml(str) {
  return String(str || "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

export function buildPropertyAddress({ property, fields }) {
  const fallback =
    [
      property?.address_line,
      property?.city,
      property?.postal_code,
      property?.country,
    ]
      .filter(Boolean)
      .join(", ") || "";

  return fields?.propertyAddress || fallback;
}

export function resolveField(fields, key, fallback) {
  const val = fields?.[key];
  if (val === null || val === undefined) return fallback;
  const str = typeof val === "string" ? val.trim() : String(val).trim();
  return str || fallback;
}

export function paragraph(num, html) {
  return `<p><strong>${escapeHtml(num)}</strong> ${html}</p>`;
}

export function sectionTitle(num, title) {
  return `<h2>${escapeHtml(String(num))}. ${escapeHtml(title)}</h2>`;
}

export function signatureBlock({ fields }) {
  const status = String(fields?.signed_status || "unsigned");
  const signedAt = fields?.signed_at ? String(fields.signed_at) : null;
  const agentSigned = fields?.signed_by_agent_name || null;
  const clientSigned = fields?.signed_by_client_name || null;

  const statusLine =
    status === "signed"
      ? `Signed (tracked off-platform)${signedAt ? ` on ${escapeHtml(signedAt)}` : ""}.`
      : "Unsigned (tracked off-platform).";

  const whoLine =
    status === "signed"
      ? `Client: ${escapeHtml(clientSigned || "—")}. Agent: ${escapeHtml(agentSigned || "—")}.`
      : "";

  return `
    <div class="signature">
      <div class="sigmeta">
        <div><strong>Signature method:</strong> Off-platform (print & sign)</div>
        <div><strong>Status:</strong> ${statusLine}</div>
        ${whoLine ? `<div>${whoLine}</div>` : ""}
      </div>

      <div class="sigrow">
        <div class="sig">
          <div class="line"></div>
          <div class="siglabel">Client Signature</div>
        </div>
        <div class="sig">
          <div class="line"></div>
          <div class="siglabel">Agent Signature</div>
        </div>
      </div>
    </div>
  `;
}

export function headerBlock({ agent }) {
  const companyName = agent?.company_name ? String(agent.company_name) : "";
  const logoUrl = agent?.company_logo_url ? String(agent.company_logo_url) : "";

  if (!companyName && !logoUrl) {
    return "";
  }

  const logoHtml = logoUrl
    ? `<img class="logo" src="${escapeHtml(logoUrl)}" alt="${escapeHtml(companyName || "Company")}" />`
    : "";

  const nameHtml = companyName
    ? `<div class="company">${escapeHtml(companyName)}</div>`
    : "";

  return `
    <div class="doc-header">
      ${logoHtml}
      ${nameHtml}
    </div>
  `;
}

export function footerBlock({ agent, contractMeta }) {
  const agentName = agent?.agent_name ? String(agent.agent_name) : "";
  const companyName = agent?.company_name ? String(agent.company_name) : "";

  const generatedAtIso = contractMeta?.generatedAt
    ? String(contractMeta.generatedAt)
    : new Date().toISOString();

  const genDate = generatedAtIso.slice(0, 10);
  const version =
    typeof contractMeta?.version === "number" && contractMeta.version > 0
      ? contractMeta.version
      : 1;

  const lines = [];
  lines.push(
    `<div><strong>Agent:</strong> ${escapeHtml(agentName || "—")}</div>`,
  );
  if (companyName) {
    lines.push(
      `<div><strong>Company:</strong> ${escapeHtml(companyName)}</div>`,
    );
  }
  lines.push(`<div><strong>Date:</strong> ${escapeHtml(genDate)}</div>`);
  lines.push(
    `<div><strong>Version:</strong> ${escapeHtml(String(version))}</div>`,
  );

  return `
    <div class="doc-footer">
      ${lines.join("\n")}
    </div>
  `;
}
