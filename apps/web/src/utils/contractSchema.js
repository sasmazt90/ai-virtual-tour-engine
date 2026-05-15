export const TEMPLATE_SCHEMA = {
  sale_agreement: {
    // Fillable template: do not hard-block generation on missing fields.
    required: [],
    // Agents can prepare/adjust any field before sharing/downloading.
    editable: [
      "OWNER_NAME",
      "OWNER_ID",
      "OWNER_ADDRESS",
      "CUSTOMER_NAME",
      "CUSTOMER_ID",
      "CUSTOMER_ADDRESS",
      "PROPERTY_ADDRESS",
      "PROPERTY_TYPE",
      "PROPERTY_SIZE",
      "ROOM_COUNT",
      "FLOOR_INFO",
      "TITLE_DEED_INFO",
      "SALE_PRICE",
      "CURRENCY",
      "PAYMENT_METHOD",
      "AGENCY_NAME",
      "AGENT_NAME",
      "AGENT_CONTACT",
      // NEW: allow adding custom notes to match the multi-page purchase agreement
      "additionalTerms",
    ],
    sections: [
      {
        title: "Seller (Owner)",
        fields: ["OWNER_NAME", "OWNER_ID", "OWNER_ADDRESS"],
      },
      {
        title: "Buyer (Client)",
        fields: ["CUSTOMER_NAME", "CUSTOMER_ID", "CUSTOMER_ADDRESS"],
      },
      {
        title: "Property details",
        fields: [
          "PROPERTY_ADDRESS",
          "PROPERTY_TYPE",
          "PROPERTY_SIZE",
          "ROOM_COUNT",
          "FLOOR_INFO",
          "TITLE_DEED_INFO",
        ],
      },
      {
        title: "Purchase details",
        fields: ["SALE_PRICE", "CURRENCY", "PAYMENT_METHOD"],
      },
      {
        title: "Agent / Office",
        fields: ["AGENCY_NAME", "AGENT_NAME", "AGENT_CONTACT"],
      },
      {
        title: "Additional terms",
        fields: ["additionalTerms"],
      },
    ],
  },
  rental_agreement: {
    required: [],
    editable: [
      "OWNER_NAME",
      "OWNER_ID",
      "OWNER_ADDRESS",
      "CUSTOMER_NAME",
      "CUSTOMER_ID",
      "CUSTOMER_ADDRESS",
      "PROPERTY_ADDRESS",
      "PROPERTY_TYPE",
      "PROPERTY_SIZE",
      "FURNISHED_STATUS",
      "RENT_PRICE",
      "CURRENCY",
      "DEPOSIT_AMOUNT",
      "RENT_START_DATE",
      "RENT_DURATION",
      "AGENCY_NAME",
      "AGENT_NAME",
      "AGENT_CONTACT",
      // NEW: allow adding custom notes to match the one-page lease
      "additionalTerms",
    ],
    sections: [
      {
        title: "Landlord (Owner)",
        fields: ["OWNER_NAME", "OWNER_ID", "OWNER_ADDRESS"],
      },
      {
        title: "Tenant (Client)",
        fields: ["CUSTOMER_NAME", "CUSTOMER_ID", "CUSTOMER_ADDRESS"],
      },
      {
        title: "Property details",
        fields: [
          "PROPERTY_ADDRESS",
          "PROPERTY_TYPE",
          "PROPERTY_SIZE",
          "FURNISHED_STATUS",
        ],
      },
      {
        title: "Lease terms",
        fields: [
          "RENT_PRICE",
          "CURRENCY",
          "DEPOSIT_AMOUNT",
          "RENT_START_DATE",
          "RENT_DURATION",
        ],
      },
      {
        title: "Agent / Office",
        fields: ["AGENCY_NAME", "AGENT_NAME", "AGENT_CONTACT"],
      },
      {
        title: "Additional terms",
        fields: ["additionalTerms"],
      },
    ],
  },
  legacy: {
    required: ["effectiveDate", "propertyAddress", "clientName", "agentName"],
    // Conservative default for legacy templates.
    editable: ["effectiveDate", "price", "governingLaw", "additionalTerms"],
    sections: [
      {
        title: "Parties",
        fields: [
          "clientName",
          "clientEmail",
          "clientPhone",
          "agentName",
          "company",
        ],
      },
      {
        title: "Property Description",
        fields: ["propertyAddress"],
      },
      {
        title: "Dates & Financial Terms",
        fields: ["effectiveDate", "price"],
      },
      {
        title: "Additional Terms",
        fields: ["additionalTerms"],
      },
    ],
  },
};

export function getEditableFieldSet(templateType) {
  const t = String(templateType || "");
  const key =
    t === "sale_agreement"
      ? "sale_agreement"
      : t === "rental_agreement"
        ? "rental_agreement"
        : "legacy";
  const schema = TEMPLATE_SCHEMA[key] || TEMPLATE_SCHEMA.legacy;
  const editable = Array.isArray(schema.editable) ? schema.editable : [];
  return new Set(editable);
}

export const FIELD_META = {
  // ===== Fillable templates (EN) =====
  OWNER_NAME: { label: "Owner full name", type: "text" },
  OWNER_ID: { label: "Owner ID / Passport no.", type: "text" },
  OWNER_ADDRESS: { label: "Owner address", type: "text" },

  CUSTOMER_NAME: { label: "Client full name", type: "text" },
  CUSTOMER_ID: { label: "Client ID / Passport no.", type: "text" },
  CUSTOMER_ADDRESS: { label: "Client address", type: "text" },

  PROPERTY_ADDRESS: { label: "Property address", type: "text" },
  PROPERTY_TYPE: { label: "Property type", type: "text" },
  PROPERTY_SIZE: { label: "Property size (sqm)", type: "text" },
  ROOM_COUNT: { label: "Room count", type: "text" },
  FLOOR_INFO: { label: "Floor info", type: "text" },
  TITLE_DEED_INFO: { label: "Title deed info", type: "text" },

  SALE_PRICE: { label: "Sale price", type: "text" },
  RENT_PRICE: { label: "Rent price", type: "text" },
  CURRENCY: { label: "Currency", type: "text" },
  PAYMENT_METHOD: { label: "Payment method", type: "text" },

  FURNISHED_STATUS: { label: "Furnished status", type: "text" },
  DEPOSIT_AMOUNT: { label: "Deposit", type: "text" },
  RENT_START_DATE: { label: "Rent start date", type: "date" },
  RENT_DURATION: { label: "Rent duration", type: "text" },

  AGENCY_NAME: { label: "Agency / Company", type: "text" },
  AGENT_NAME: { label: "Agent name", type: "text" },
  AGENT_CONTACT: { label: "Agent contact", type: "text" },

  // ===== Legacy fields (kept) =====
  effectiveDate: { label: "Effective Date", type: "date" },
  closingDate: { label: "Closing Date", type: "date" },
  leaseStartDate: { label: "Lease Start Date", type: "date" },
  leaseEndDate: { label: "Lease End Date", type: "date" },

  propertyAddress: {
    label: "Property Address",
    type: "text",
    placeholder: "Address",
  },
  propertySizeSqm: { label: "Size (m²)", type: "text", placeholder: "e.g. 85" },
  propertyRooms: { label: "Rooms", type: "text", placeholder: "e.g. 3" },

  price: { label: "Price / Rent", type: "text", placeholder: "Amount" },
  earnestMoney: {
    label: "Earnest Money (optional)",
    type: "text",
    placeholder: "Amount",
  },
  securityDeposit: {
    label: "Security Deposit (optional)",
    type: "text",
    placeholder: "Amount",
  },
  rentDueDay: {
    label: "Rent Due Day (optional)",
    type: "text",
    placeholder: "e.g. 1",
  },

  clientName: {
    label: "Client / Tenant Name",
    type: "text",
    placeholder: "Full name",
  },
  clientEmail: { label: "Client Email", type: "email", placeholder: "Email" },
  clientPhone: { label: "Client Phone", type: "text", placeholder: "Phone" },

  agentName: { label: "Agent Name", type: "text", placeholder: "Agent name" },
  company: {
    label: "Company (optional)",
    type: "text",
    placeholder: "Company",
  },

  governingLaw: {
    label: "Governing Law (optional)",
    type: "text",
    placeholder: "e.g. laws of the jurisdiction where the property is located",
  },

  additionalTerms: {
    label: "Additional terms (notes)",
    type: "textarea",
    placeholder: "Optional notes / special terms...",
  },
};
