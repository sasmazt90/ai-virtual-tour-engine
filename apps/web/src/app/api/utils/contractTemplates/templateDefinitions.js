export const TEMPLATE_DEFS = {
  // NOTE: These are the public-facing labels in the app.
  // The detailed template layout is implemented in htmlTemplates.js (and used for PDF generation).
  sale_agreement: {
    title: "Residential Purchase Agreement",
    // Draft-first: do not block generation on missing fields.
    required: [],
  },
  rental_agreement: {
    title: "Residential Lease Agreement",
    required: [],
  },
  buyer_representation: {
    title: "Buyer Representation Agreement",
    required: [],
  },
  seller_listing_agreement: {
    title: "Seller Listing Agreement",
    required: [],
  },
  tenant_representation: {
    title: "Tenant Representation Agreement",
    required: [],
  },
  offer_letter: {
    title: "Property Offer Letter",
    required: [],
  },
  viewing_report: {
    title: "Viewing and Inspection Report",
    required: [],
  },
  handover_protocol: {
    title: "Property Handover Protocol",
    required: [],
  },
  agency_authorization: {
    title: "Agency Authorization",
    required: [],
  },
  // Legacy templates kept for backwards compatibility.
  purchase_agreement: {
    title: "Purchase Agreement (Legacy)",
    required: [
      "effectiveDate",
      "propertyAddress",
      "price",
      "clientName",
      "agentName",
    ],
  },
  listing_agreement: {
    title: "Listing Agreement (Legacy)",
    required: ["effectiveDate", "propertyAddress", "clientName", "agentName"],
  },
};

export function getTemplateDef(templateType) {
  const key = String(templateType || "");
  return (
    TEMPLATE_DEFS[key] || {
      title: "Contract (Legacy)",
      required: ["effectiveDate", "propertyAddress", "clientName"],
    }
  );
}

export function getMissingFields(templateType, fields) {
  const def = getTemplateDef(templateType);
  const required = Array.isArray(def.required) ? def.required : [];

  const missing = [];
  for (const key of required) {
    const val = fields?.[key];
    const str = typeof val === "string" ? val.trim() : String(val || "").trim();
    if (!str) missing.push(key);
  }
  return missing;
}
