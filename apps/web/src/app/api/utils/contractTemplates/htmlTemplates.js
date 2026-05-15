import { getTemplateDef } from "./templateDefinitions";
import {
  escapeHtml,
  buildPropertyAddress,
  resolveField,
  paragraph,
  sectionTitle,
  signatureBlock,
  headerBlock,
  footerBlock,
} from "./htmlHelpers";

function resolveAny(fields, keys, fallback) {
  for (const k of keys) {
    const v = resolveField(fields, k, "");
    if (v) return v;
  }
  return fallback;
}

function baseDocumentStyle() {
  // NOTE: pdfGeneration.js already injects a base stylesheet.
  // We still include a template-aware style block so multi-page layouts work.
  // IMPORTANT: Keep CSS very simple (no CSS grid) because some PDF renderers
  // handle grid poorly and can cause overlapping text.
  return `
    <style>
      * { box-sizing: border-box; }
      body { font-family: Arial, Helvetica, sans-serif; font-size: 12px; color: #111; }
      img { max-width: 100%; }

      .page { page-break-after: always; padding: 28px; }
      .page:last-child { page-break-after: auto; }

      h1 { font-size: 18px; margin: 0 0 12px; }
      h2 { font-size: 13px; margin: 14px 0 6px; }
      p { margin: 0 0 8px; line-height: 1.35; }

      /* Two-column layout without CSS grid (more PDF-renderer friendly) */
      .grid2 { width: 100%; }
      .grid2:after { content: ""; display: table; clear: both; }
      .grid2 .field { float: left; width: calc(50% - 6px); }
      .grid2 .field:nth-child(2n) { margin-left: 12px; }

      .field { border: 1px solid #d4d4d4; border-radius: 8px; padding: 10px; margin-bottom: 12px; }
      .label { font-size: 11px; color: #444; margin-bottom: 4px; }
      .value { font-size: 12px; min-height: 18px; line-height: 1.35; word-break: break-word; overflow-wrap: anywhere; }
      .muted { color: #666; }
      .small { font-size: 11px; }
      .hr { height: 1px; background: #e5e5e5; margin: 14px 0; }

      /* Signature row without CSS grid */
      .sigrow { width: 100%; margin-top: 18px; }
      .sigrow:after { content: ""; display: table; clear: both; }
      .sigrow > div { float: left; width: calc(50% - 9px); }
      .sigrow > div:nth-child(2) { margin-left: 18px; }

      .sigline { border-bottom: 1px solid #111; height: 20px; margin-bottom: 6px; }
      .siglabel { font-size: 11px; color: #444; }

      .pageMeta { font-size: 11px; color: #666; margin-bottom: 8px; }

      /* Keep existing header/footer classnames used by injected styles */
    </style>
  `;
}

// ====== NEW templates (requested) ======
function buildLeaseAgreementHtml({
  fields,
  property,
  client,
  agent,
  contractMeta,
}) {
  const title = getTemplateDef("rental_agreement").title;

  const effectiveDate = resolveAny(
    fields,
    ["RENT_START_DATE", "effectiveDate"],
    new Date().toISOString().slice(0, 10),
  );

  const landlordName = resolveAny(fields, ["OWNER_NAME"], "");
  const tenantName = resolveAny(
    fields,
    ["CUSTOMER_NAME"],
    client?.full_name || "",
  );

  const propertyAddress = resolveAny(
    fields,
    ["PROPERTY_ADDRESS"],
    buildPropertyAddress({ property, fields }),
  );

  const rent = resolveAny(fields, ["RENT_PRICE", "price"], "");
  const deposit = resolveAny(fields, ["DEPOSIT_AMOUNT"], "");
  const currency = resolveAny(
    fields,
    ["CURRENCY"],
    property?.currency ? String(property.currency) : "",
  );
  const duration = resolveAny(fields, ["RENT_DURATION"], "");

  const notes = resolveAny(fields, ["additionalTerms"], "");

  const resolvedAgent = {
    agent_name:
      (agent?.agent_name ? String(agent.agent_name) : "") ||
      resolveAny(fields, ["AGENT_NAME", "agentName"], ""),
    company_name:
      (agent?.company_name ? String(agent.company_name) : "") ||
      resolveAny(fields, ["AGENCY_NAME", "company"], ""),
    company_logo_url: agent?.company_logo_url
      ? String(agent.company_logo_url)
      : "",
  };

  const headerHtml = headerBlock({ agent: resolvedAgent });
  const footerHtml = footerBlock({
    agent: resolvedAgent,
    contractMeta: {
      generatedAt: contractMeta?.generatedAt || new Date().toISOString(),
      version:
        typeof contractMeta?.version === "number" ? contractMeta.version : 1,
    },
  });

  const rentText = [rent, currency].filter(Boolean).join(" ").trim();
  const depositText = [deposit, currency].filter(Boolean).join(" ").trim();

  return `
    <div class="page">
      ${headerHtml}
      <h1>${escapeHtml(title)}</h1>

      <div class="grid2">
        <div class="field">
          <div class="label">Effective date</div>
          <div class="value">${escapeHtml(effectiveDate || "—")}</div>
        </div>
        <div class="field">
          <div class="label">Lease term</div>
          <div class="value">${escapeHtml(duration || "—")}</div>
        </div>
      </div>

      <div class="hr"></div>

      <div class="grid2">
        <div class="field">
          <div class="label">Landlord</div>
          <div class="value">${escapeHtml(landlordName || "—")}</div>
        </div>
        <div class="field">
          <div class="label">Tenant</div>
          <div class="value">${escapeHtml(tenantName || "—")}</div>
        </div>
      </div>

      <div class="field" style="margin-top: 12px;">
        <div class="label">Rental property address</div>
        <div class="value">${escapeHtml(propertyAddress || "—")}</div>
      </div>

      <div class="grid2" style="margin-top: 12px;">
        <div class="field">
          <div class="label">Monthly rent</div>
          <div class="value">${escapeHtml(rentText || "—")}</div>
        </div>
        <div class="field">
          <div class="label">Security deposit</div>
          <div class="value">${escapeHtml(depositText || "—")}</div>
        </div>
      </div>

      <h2>Terms</h2>
      <p class="small muted">
        This is a simplified one-page lease layout. Please confirm compliance
        with local laws.
      </p>
      <p>1) Tenant pays rent and complies with rules and laws.</p>
      <p>2) Landlord maintains essential systems and provides habitable premises.</p>
      <p>3) Utilities, pets, and any special terms should be written below.</p>

      <div class="field">
        <div class="label">Additional terms (notes)</div>
        <div class="value">${escapeHtml(notes).replaceAll("\n", "<br/>")}</div>
      </div>

      <div class="sigrow">
        <div>
          <div class="sigline"></div>
          <div class="siglabel">Tenant signature</div>
        </div>
        <div>
          <div class="sigline"></div>
          <div class="siglabel">Landlord signature</div>
        </div>
      </div>

      ${footerHtml}
    </div>
  `;
}

function buildPurchaseAgreement10PagesHtml({
  fields,
  property,
  client,
  agent,
  contractMeta,
}) {
  const title = getTemplateDef("sale_agreement").title;

  const buyerName = resolveAny(
    fields,
    ["CUSTOMER_NAME"],
    client?.full_name || "",
  );
  const sellerName = resolveAny(fields, ["OWNER_NAME"], "");
  const propertyAddress = resolveAny(
    fields,
    ["PROPERTY_ADDRESS"],
    buildPropertyAddress({ property, fields }),
  );
  const price = resolveAny(
    fields,
    ["SALE_PRICE", "price"],
    property?.price !== null && property?.price !== undefined
      ? String(property.price)
      : "",
  );
  const currency = resolveAny(
    fields,
    ["CURRENCY"],
    property?.currency ? String(property.currency) : "",
  );

  const paymentMethod = resolveAny(fields, ["PAYMENT_METHOD"], "");
  const notes = resolveAny(fields, ["additionalTerms"], "");

  const resolvedAgent = {
    agent_name:
      (agent?.agent_name ? String(agent.agent_name) : "") ||
      resolveAny(fields, ["AGENT_NAME", "agentName"], ""),
    company_name:
      (agent?.company_name ? String(agent.company_name) : "") ||
      resolveAny(fields, ["AGENCY_NAME", "company"], ""),
    company_logo_url: agent?.company_logo_url
      ? String(agent.company_logo_url)
      : "",
  };

  const headerHtml = headerBlock({ agent: resolvedAgent });
  const footerHtml = footerBlock({
    agent: resolvedAgent,
    contractMeta: {
      generatedAt: contractMeta?.generatedAt || new Date().toISOString(),
      version:
        typeof contractMeta?.version === "number" ? contractMeta.version : 1,
    },
  });

  const moneyText = [price, currency].filter(Boolean).join(" ").trim();

  const page = (n, innerHtml) => `
    <div class="page">
      ${headerHtml}
      <div class="pageMeta">Page ${n} of 10</div>
      <h1>${escapeHtml(title)}</h1>
      ${innerHtml}
      ${footerHtml}
    </div>
  `;

  const summary = `
    <div class="grid2">
      <div class="field"><div class="label">Buyer</div><div class="value">${escapeHtml(buyerName || "—")}</div></div>
      <div class="field"><div class="label">Seller</div><div class="value">${escapeHtml(sellerName || "—")}</div></div>
    </div>

    <div class="field" style="margin-top: 12px;">
      <div class="label">Property address</div>
      <div class="value">${escapeHtml(propertyAddress || "—")}</div>
    </div>

    <div class="grid2" style="margin-top: 12px;">
      <div class="field"><div class="label">Purchase price</div><div class="value">${escapeHtml(moneyText || "—")}</div></div>
      <div class="field"><div class="label">Payment method</div><div class="value">${escapeHtml(paymentMethod || "—")}</div></div>
    </div>

    <h2>Agreement</h2>
    <p>
      This Purchase Agreement ("Agreement") sets forth the terms under which
      the Buyer will purchase and the Seller will sell the Property described
      above.
    </p>

    <p class="small muted">
      Note: This is a structured template for internal use. Please have legal
      counsel review before use in production.
    </p>
  `;

  const section = (idx, titleLine, a, b, c) => {
    return `
      <h2>Section ${idx}: ${escapeHtml(titleLine)}</h2>
      <p><strong>${idx}.1</strong> ${escapeHtml(a)}</p>
      <p><strong>${idx}.2</strong> ${escapeHtml(b)}</p>
      <p><strong>${idx}.3</strong> ${escapeHtml(c)}</p>
    `;
  };

  const p2 = section(
    2,
    "Definitions and interpretation",
    "Definitions used in this Agreement will be interpreted reasonably and in good faith.",
    "Headings are for convenience only.",
    "If a provision is found invalid, the remainder stays effective.",
  );

  const p3 = section(
    3,
    "Purchase price, deposits, and payment terms",
    "The purchase price is as stated on Page 1 unless amended in writing.",
    "Any deposits or escrow arrangements should be documented by the parties.",
    "Payment timing and method will be agreed before closing.",
  );

  const p4 = section(
    4,
    "Financing and appraisal",
    "If financing is required, Buyer should obtain approval as soon as possible.",
    "If appraisal is required, parties will cooperate to schedule it.",
    "Failure to obtain financing may be handled as agreed in writing.",
  );

  const p5 = section(
    5,
    "Inspection and disclosures",
    "Buyer may conduct inspections during an agreed due diligence period.",
    "Seller will provide legally required disclosures if applicable.",
    "Any negotiated repairs should be documented in writing.",
  );

  const p6 = section(
    6,
    "Title, deed, and closing",
    "Seller will convey marketable title unless otherwise agreed.",
    "Closing will take place at a mutually agreed location/date.",
    "Costs and prorations will be allocated as agreed by the parties.",
  );

  const p7 = section(
    7,
    "Possession and handover",
    "Possession will be delivered at closing unless otherwise agreed.",
    "Keys and access devices will be delivered to Buyer upon handover.",
    "A final walkthrough may be completed before closing.",
  );

  const p8 = section(
    8,
    "Default and remedies",
    "If either party defaults, the non-defaulting party may pursue legal remedies.",
    "Parties may agree to mediation/arbitration where permitted.",
    "Attorney fees and costs are as permitted by law or agreement.",
  );

  const p9 = `
    <h2>Section 9: Additional terms</h2>
    <p class="small muted">Write any custom terms here (if needed).</p>
    <div class="field">
      <div class="label">Additional terms (notes)</div>
      <div class="value">${escapeHtml(notes).replaceAll("\n", "<br/>")}</div>
    </div>
  `;

  const p10 = `
    <h2>Section 10: Signatures</h2>
    <p class="small muted">This Agreement is intended to be signed off-platform (print & sign).</p>

    <div class="sigrow">
      <div>
        <div class="sigline"></div>
        <div class="siglabel">Buyer signature</div>
      </div>
      <div>
        <div class="sigline"></div>
        <div class="siglabel">Seller signature</div>
      </div>
    </div>

    ${signatureBlock({ fields })}
  `;

  return (
    page(1, summary) +
    page(2, p2) +
    page(3, p3) +
    page(4, p4) +
    page(5, p5) +
    page(6, p6) +
    page(7, p7) +
    page(8, p8) +
    page(9, p9) +
    page(10, p10)
  );
}

// ====== Legacy templates (kept) ======
function buildSaleAgreementBody({
  effectiveDate,
  address,
  agentName,
  company,
  clientName,
  clientEmail,
  clientPhone,
  closingDate,
  price,
  earnestMoney,
  governingLaw,
}) {
  let body = "";

  body += sectionTitle(1, "Parties");
  body += paragraph(
    "1.1",
    `This Sale Agreement ("Agreement") is made effective on ${escapeHtml(effectiveDate)} between ${escapeHtml(agentName)}${company ? ` (${escapeHtml(company)})` : ""} ("Agent") and ${escapeHtml(clientName)} ("Client").`,
  );
  body += paragraph(
    "1.2",
    `Client contact details: Email ${escapeHtml(clientEmail || "—")}; Phone ${escapeHtml(clientPhone || "—")}.`,
  );

  body += sectionTitle(2, "Property Description");
  body += paragraph(
    "2.1",
    `The property subject to this Agreement ("Property") is located at ${escapeHtml(address)}.`,
  );

  body += sectionTitle(3, "Term & Dates");
  body += paragraph(
    "3.1",
    `The intended closing date is ${escapeHtml(closingDate || "[to be agreed]")}.`,
  );

  body += sectionTitle(4, "Financial Terms");
  body += paragraph(
    "4.1",
    `The purchase price is ${escapeHtml(price)} (the "Purchase Price").`,
  );
  body += paragraph(
    "4.2",
    `Earnest money deposit (if any): ${escapeHtml(earnestMoney || "—")}.`,
  );

  body += sectionTitle(5, "Governing Law");
  body += paragraph(
    "5.1",
    `This Agreement is governed by ${escapeHtml(governingLaw)}.`,
  );

  body += sectionTitle(6, "Signatures");
  body += paragraph(
    "6.1",
    `This Agreement is intended to be signed off-platform (print & sign).`,
  );

  return body;
}

function buildRentalAgreementBody({
  effectiveDate,
  address,
  agentName,
  company,
  clientName,
  clientEmail,
  clientPhone,
  leaseStartDate,
  leaseEndDate,
  price,
  rentDueDay,
  securityDeposit,
  governingLaw,
}) {
  let body = "";

  body += sectionTitle(1, "Parties");
  body += paragraph(
    "1.1",
    `This Rental Agreement ("Agreement") is made effective on ${escapeHtml(effectiveDate)} between ${escapeHtml(agentName)}${company ? ` (${escapeHtml(company)})` : ""} ("Agent") and ${escapeHtml(clientName)} ("Tenant").`,
  );
  body += paragraph(
    "1.2",
    `Tenant contact details: Email ${escapeHtml(clientEmail || "—")}; Phone ${escapeHtml(clientPhone || "—")}.`,
  );

  body += sectionTitle(2, "Property Description");
  body += paragraph(
    "2.1",
    `The rental property subject to this Agreement ("Property") is located at ${escapeHtml(address)}.`,
  );

  body += sectionTitle(3, "Term & Duration");
  body += paragraph(
    "3.1",
    `The lease term begins on ${escapeHtml(leaseStartDate || "[to be agreed]")}.`,
  );
  body += paragraph(
    "3.2",
    `The lease term ends on ${escapeHtml(leaseEndDate || "[to be agreed]")}.`,
  );

  body += sectionTitle(4, "Financial Terms");
  body += paragraph(
    "4.1",
    `The rent is ${escapeHtml(price)} per month unless otherwise stated in the Additional Terms.`,
  );
  body += paragraph(
    "4.2",
    `Rent due day (if applicable): ${escapeHtml(rentDueDay || "—")}.`,
  );
  body += paragraph(
    "4.3",
    `Security deposit (if any): ${escapeHtml(securityDeposit || "—")}.`,
  );

  body += sectionTitle(5, "Governing Law");
  body += paragraph(
    "5.1",
    `This Agreement is governed by ${escapeHtml(governingLaw)}.`,
  );

  body += sectionTitle(6, "Signatures");
  body += paragraph(
    "6.1",
    `This Agreement is intended to be signed off-platform (print & sign).`,
  );

  return body;
}

function buildLegacyContractBody({
  effectiveDate,
  address,
  agentName,
  company,
  clientName,
  price,
  governingLaw,
}) {
  let body = "";

  body += sectionTitle(1, "Parties");
  body += paragraph(
    "1.1",
    `This Agreement is made effective on ${escapeHtml(effectiveDate)} between ${escapeHtml(agentName)}${company ? ` (${escapeHtml(company)})` : ""} ("Agent") and ${escapeHtml(clientName)} ("Client").`,
  );

  body += sectionTitle(2, "Property Description");
  body += paragraph(
    "2.1",
    `The property subject to this Agreement is located at ${escapeHtml(address)}.`,
  );

  body += sectionTitle(3, "Financial Terms");
  body += paragraph("3.1", `Price / amount: ${escapeHtml(price || "—")}.`);

  body += sectionTitle(4, "Governing Law");
  body += paragraph(
    "4.1",
    `This Agreement is governed by ${escapeHtml(governingLaw)}.`,
  );

  body += sectionTitle(5, "Signatures");
  body += paragraph(
    "5.1",
    `This Agreement is intended to be signed off-platform (print & sign).`,
  );

  return body;
}

export function buildContractHtml({
  templateType,
  property,
  client,
  fields,
  agent,
  contractMeta,
}) {
  const def = getTemplateDef(templateType);
  const title = def.title;

  // Use the new layouts for the 2 main templates.
  if (String(templateType) === "rental_agreement") {
    return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>${escapeHtml(title)}</title>
  ${baseDocumentStyle()}
</head>
<body>
  ${buildLeaseAgreementHtml({ fields, property, client, agent, contractMeta })}
</body>
</html>`;
  }

  if (String(templateType) === "sale_agreement") {
    return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>${escapeHtml(title)}</title>
  ${baseDocumentStyle()}
</head>
<body>
  ${buildPurchaseAgreement10PagesHtml({ fields, property, client, agent, contractMeta })}
</body>
</html>`;
  }

  // ===== Legacy behavior (kept for backward compatibility) =====
  const effectiveDate = resolveField(
    fields,
    "effectiveDate",
    new Date().toISOString().slice(0, 10),
  );
  const address = buildPropertyAddress({ property, fields });

  const size = resolveField(
    fields,
    "propertySizeSqm",
    property?.size_sqm ?? "",
  );
  const rooms = resolveField(fields, "propertyRooms", property?.rooms ?? "");

  const clientName = resolveField(
    fields,
    "clientName",
    client?.full_name || "",
  );
  const clientEmail = resolveField(fields, "clientEmail", client?.email || "");
  const clientPhone = resolveField(fields, "clientPhone", client?.phone || "");

  const agentName = resolveField(fields, "agentName", "");
  const company = resolveField(fields, "company", "");

  const governingLaw = resolveField(
    fields,
    "governingLaw",
    "the applicable laws of the jurisdiction where the Property is located",
  );

  const additionalTerms = resolveField(fields, "additionalTerms", "");

  const price = resolveField(fields, "price", property?.price ?? "");

  const leaseStartDate = resolveField(fields, "leaseStartDate", "");
  const leaseEndDate = resolveField(fields, "leaseEndDate", "");
  const closingDate = resolveField(fields, "closingDate", "");

  const earnestMoney = resolveField(fields, "earnestMoney", "");
  const securityDeposit = resolveField(fields, "securityDeposit", "");
  const rentDueDay = resolveField(fields, "rentDueDay", "");

  const resolvedAgent = {
    agent_name:
      (agent?.agent_name ? String(agent.agent_name) : "") ||
      resolveField(fields, "agentName", ""),
    company_name:
      (agent?.company_name ? String(agent.company_name) : "") ||
      resolveField(fields, "company", ""),
    company_logo_url: agent?.company_logo_url
      ? String(agent.company_logo_url)
      : "",
  };

  const headerHtml = headerBlock({ agent: resolvedAgent });
  const footerHtml = footerBlock({
    agent: resolvedAgent,
    contractMeta: {
      generatedAt: contractMeta?.generatedAt || new Date().toISOString(),
      version:
        typeof contractMeta?.version === "number" ? contractMeta.version : 1,
    },
  });

  const headerMetaLines = [
    `<div><strong>Effective date:</strong> ${escapeHtml(effectiveDate)}</div>`,
    `<div><strong>Property:</strong> ${escapeHtml(property?.title || "")}</div>`,
    `<div><strong>Address:</strong> ${escapeHtml(address)}</div>`,
  ];
  if (size)
    headerMetaLines.push(
      `<div><strong>Size (m²):</strong> ${escapeHtml(size)}</div>`,
    );
  if (rooms)
    headerMetaLines.push(
      `<div><strong>Rooms:</strong> ${escapeHtml(rooms)}</div>`,
    );

  let body = "";

  if (String(templateType) === "purchase_agreement") {
    body = buildSaleAgreementBody({
      effectiveDate,
      address,
      agentName,
      company,
      clientName,
      clientEmail,
      clientPhone,
      closingDate,
      price,
      earnestMoney,
      governingLaw,
    });
  } else if (String(templateType) === "listing_agreement") {
    body = buildRentalAgreementBody({
      effectiveDate,
      address,
      agentName,
      company,
      clientName,
      clientEmail,
      clientPhone,
      leaseStartDate,
      leaseEndDate,
      price,
      rentDueDay,
      securityDeposit,
      governingLaw,
    });
  } else {
    body = buildLegacyContractBody({
      effectiveDate,
      address,
      agentName,
      company,
      clientName,
      price,
      governingLaw,
    });
  }

  const additionalTermsHtml = additionalTerms
    ? `<div class="box"><div class="label">Additional Terms</div><div class="value">${escapeHtml(additionalTerms).replaceAll("\n", "<br/>")}</div></div>`
    : "";

  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>${escapeHtml(title)}</title>
</head>
<body>
  <div class="page">
    ${headerHtml}

    <h1>${escapeHtml(title)}</h1>

    <div class="meta">
      ${headerMetaLines.join("\n")}
    </div>

    ${body}

    ${additionalTermsHtml}

    ${signatureBlock({ fields })}

    ${footerHtml}
  </div>
</body>
</html>`;
}
