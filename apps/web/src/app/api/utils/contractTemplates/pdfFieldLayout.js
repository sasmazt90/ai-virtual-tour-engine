import { getTemplateDef } from "./templateDefinitions";
import { resolveField } from "./htmlHelpers";

function resolveAny(fields, keys, fallback) {
  for (const k of keys) {
    const val = resolveField(fields, k, "");
    if (val) return val;
  }
  return fallback;
}

export function buildFillableContractFieldLayout({
  templateType,
  fields,
  agent,
  contractMeta,
}) {
  const agentName =
    (agent?.agent_name ? String(agent.agent_name) : "") ||
    resolveField(fields, "AGENT_NAME", "") ||
    resolveField(fields, "agentName", "");

  const companyName =
    (agent?.company_name ? String(agent.company_name) : "") ||
    resolveField(fields, "AGENCY_NAME", "") ||
    resolveField(fields, "company", "");

  const version =
    typeof contractMeta?.version === "number" && contractMeta.version > 0
      ? contractMeta.version
      : 1;

  const genDate = (contractMeta?.generatedAt || new Date().toISOString()).slice(
    0,
    10,
  );

  const title = getTemplateDef(templateType).title;

  const header = {
    title,
    companyName,
  };

  const footer = {
    agentName,
    companyName,
    genDate,
    version,
    // English footer lines
    lines: [
      `Agent: ${agentName || "—"}`,
      companyName ? `Company: ${companyName}` : null,
      `Generated date: ${genDate}`,
      `Version: ${String(version || 1)}`,
    ].filter(Boolean),
  };

  // ====== Fillable templates (EN) ======
  const t = String(templateType || "");

  if (t === "sale_agreement" || t === "rental_agreement") {
    // Field values (support legacy keys as fallback so older records don't break)
    const PROPERTY_ADDRESS = resolveAny(
      fields,
      ["PROPERTY_ADDRESS", "propertyAddress"],
      "",
    );

    const PROPERTY_TYPE = resolveAny(
      fields,
      ["PROPERTY_TYPE"],
      resolveField(fields, "housingType", ""),
    );

    const bodyLines =
      t === "sale_agreement"
        ? [
            "This sale agreement is made between the parties listed below.",
            PROPERTY_ADDRESS
              ? `Property address: ${PROPERTY_ADDRESS}`
              : "Property address: {{PROPERTY_ADDRESS}}",
            PROPERTY_TYPE
              ? `Property type: ${PROPERTY_TYPE}`
              : "Property type: {{PROPERTY_TYPE}}",
          ]
        : [
            "This rental agreement is made between the parties listed below.",
            PROPERTY_ADDRESS
              ? `Property address: ${PROPERTY_ADDRESS}`
              : "Property address: {{PROPERTY_ADDRESS}}",
            PROPERTY_TYPE
              ? `Property type: ${PROPERTY_TYPE}`
              : "Property type: {{PROPERTY_TYPE}}",
          ];

    const contentBlocks = [];

    // Body text (below title/company)
    let by = companyName ? 768 : 780;
    for (const line of bodyLines) {
      contentBlocks.push({ type: "text", x: 50, y: by, size: 10, text: line });
      by -= 14;
    }

    const fieldsLayout = [];

    const xLeft = 50;
    const xRight = 315;
    const w = 230;
    const h = 18;
    const rowGap = 26;

    let yTop = 700; // start of form area (increased from 690 for heading spacing)

    const addHeading = (text) => {
      yTop -= 8; // gap before new section heading
      contentBlocks.push({
        type: "text",
        x: xLeft,
        y: yTop,
        size: 11,
        text,
      });
      yTop -= 34; // space consumed by heading + gap before first field
    };

    const addText = (name, label, value, col, wide) => {
      const x = col === "right" ? xRight : xLeft;
      const rect = wide
        ? [xLeft, yTop - 4, xLeft + 495, yTop - 4 + h]
        : [x, yTop - 4, x + w, yTop - 4 + h];

      fieldsLayout.push({
        kind: "text",
        name,
        label,
        value,
        rect,
        wide: !!wide,
      });
    };

    const nextRow = () => {
      yTop -= rowGap;
    };

    // ---- Owner / Seller / Landlord ----
    addHeading(t === "sale_agreement" ? "Seller (Owner)" : "Landlord (Owner)");
    addText(
      "OWNER_NAME",
      "Owner full name",
      resolveAny(fields, ["OWNER_NAME"], ""),
      "left",
    );
    addText(
      "OWNER_ID",
      "Owner ID / Passport no.",
      resolveAny(fields, ["OWNER_ID"], ""),
      "right",
    );
    nextRow();
    addText(
      "OWNER_ADDRESS",
      "Owner address",
      resolveAny(fields, ["OWNER_ADDRESS"], ""),
      "left",
      true,
    );
    nextRow();

    // ---- Buyer / Tenant ----
    addHeading(t === "sale_agreement" ? "Buyer (Client)" : "Tenant (Client)");
    addText(
      "CUSTOMER_NAME",
      "Client full name",
      resolveAny(fields, ["CUSTOMER_NAME", "clientName"], ""),
      "left",
    );
    addText(
      "CUSTOMER_ID",
      "Client ID / Passport no.",
      resolveAny(fields, ["CUSTOMER_ID"], ""),
      "right",
    );
    nextRow();
    addText(
      "CUSTOMER_ADDRESS",
      "Client address",
      resolveAny(fields, ["CUSTOMER_ADDRESS"], ""),
      "left",
      true,
    );
    nextRow();

    // ---- Property ----
    addHeading("Property details");
    addText(
      "PROPERTY_ADDRESS",
      "Property address",
      resolveAny(fields, ["PROPERTY_ADDRESS", "propertyAddress"], ""),
      "left",
      true,
    );
    nextRow();
    addText(
      "PROPERTY_TYPE",
      "Property type",
      resolveAny(fields, ["PROPERTY_TYPE"], ""),
      "left",
    );
    addText(
      "PROPERTY_SIZE",
      "Property size (sqm)",
      resolveAny(fields, ["PROPERTY_SIZE"], ""),
      "right",
    );
    nextRow();

    if (t === "sale_agreement") {
      addText(
        "ROOM_COUNT",
        "Room count",
        resolveAny(fields, ["ROOM_COUNT"], ""),
        "left",
      );
      addText(
        "FLOOR_INFO",
        "Floor info",
        resolveAny(fields, ["FLOOR_INFO"], ""),
        "right",
      );
      nextRow();
      addText(
        "TITLE_DEED_INFO",
        "Title deed info",
        resolveAny(fields, ["TITLE_DEED_INFO"], ""),
        "left",
        true,
      );
      nextRow();

      // ---- Sale details ----
      addHeading("Sale details");
      addText(
        "SALE_PRICE",
        "Sale price",
        resolveAny(fields, ["SALE_PRICE", "price"], ""),
        "left",
      );
      addText(
        "CURRENCY",
        "Currency",
        resolveAny(fields, ["CURRENCY"], ""),
        "right",
      );
      nextRow();
      addText(
        "PAYMENT_METHOD",
        "Payment method",
        resolveAny(fields, ["PAYMENT_METHOD"], ""),
        "left",
        true,
      );
      nextRow();
    } else {
      addText(
        "FURNISHED_STATUS",
        "Furnished status",
        resolveAny(fields, ["FURNISHED_STATUS"], ""),
        "left",
      );
      addText(
        "CURRENCY",
        "Currency",
        resolveAny(fields, ["CURRENCY"], ""),
        "right",
      );
      nextRow();

      // ---- Rental terms ----
      addHeading("Rental terms");
      addText(
        "RENT_PRICE",
        "Rent price",
        resolveAny(fields, ["RENT_PRICE", "price"], ""),
        "left",
      );
      addText(
        "DEPOSIT_AMOUNT",
        "Deposit",
        resolveAny(fields, ["DEPOSIT_AMOUNT"], ""),
        "right",
      );
      nextRow();
      addText(
        "RENT_START_DATE",
        "Rent start date",
        resolveAny(fields, ["RENT_START_DATE"], genDate),
        "left",
      );
      addText(
        "RENT_DURATION",
        "Rent duration",
        resolveAny(fields, ["RENT_DURATION"], ""),
        "right",
      );
      nextRow();
    }

    // ---- Agent / Agency ----
    addHeading("Agent / Office");
    addText(
      "AGENCY_NAME",
      "Agency / Company",
      resolveAny(fields, ["AGENCY_NAME", "company"], companyName || ""),
      "left",
    );
    addText(
      "AGENT_NAME",
      "Agent name",
      resolveAny(fields, ["AGENT_NAME", "agentName"], agentName || ""),
      "right",
    );
    nextRow();
    addText(
      "AGENT_CONTACT",
      "Agent contact",
      resolveAny(fields, ["AGENT_CONTACT"], ""),
      "left",
      true,
    );

    // Signature placeholders (lines only; no signing logic)
    contentBlocks.push({
      type: "text",
      x: 50,
      y: 120,
      size: 10,
      text: "Signatures",
    });
    // Buyer/Tenant line
    contentBlocks.push({
      type: "line",
      x1: 50,
      y1: 95,
      x2: 270,
      y2: 95,
      width: 1,
    });
    contentBlocks.push({
      type: "text",
      x: 50,
      y: 80,
      size: 9,
      text: t === "sale_agreement" ? "Buyer signature" : "Tenant signature",
    });
    // Seller/Landlord line
    contentBlocks.push({
      type: "line",
      x1: 315,
      y1: 95,
      x2: 545,
      y2: 95,
      width: 1,
    });
    contentBlocks.push({
      type: "text",
      x: 315,
      y: 80,
      size: 9,
      text: t === "sale_agreement" ? "Seller signature" : "Landlord signature",
    });

    return { header, footer, fieldsLayout, contentBlocks };
  }

  // ====== Legacy (existing simple layout) ======
  const effectiveDate = resolveField(
    fields,
    "effectiveDate",
    new Date().toISOString().slice(0, 10),
  );
  const propertyAddress = resolveField(fields, "propertyAddress", "");
  const price = resolveField(fields, "price", "");

  const clientName = resolveField(fields, "clientName", "");
  const clientEmail = resolveField(fields, "clientEmail", "");
  const clientPhone = resolveField(fields, "clientPhone", "");

  const yTop = 760;
  const xLeft = 50;
  const xRight = 320;
  const w = 225;
  const h = 18;
  const rowGap = 28;

  // Note: coordinates are from bottom-left, so y values are "baseline" for the field rect.
  const fieldsLayout = [];

  let row = 0;
  const addText = (name, label, value, col) => {
    const x = col === "right" ? xRight : xLeft;
    const y = yTop - row * rowGap;
    fieldsLayout.push({
      kind: "text",
      name,
      label,
      value,
      rect: [x, y - 4, x + w, y - 4 + h],
    });
  };

  // top grid
  addText("effectiveDate", "Effective date", effectiveDate, "left");
  addText("price", "Price", price, "right");
  row++;

  if (String(templateType) === "sale_agreement") {
    const closingDate = resolveField(fields, "closingDate", "");
    addText("closingDate", "Closing date", closingDate, "left");
  } else if (String(templateType) === "rental_agreement") {
    const leaseStartDate = resolveField(fields, "leaseStartDate", "");
    const leaseEndDate = resolveField(fields, "leaseEndDate", "");
    addText("leaseStartDate", "Lease start", leaseStartDate, "left");
    addText("leaseEndDate", "Lease end", leaseEndDate, "right");
    row++;
  }

  addText("clientName", "Client name", clientName, "left");
  addText("agentName", "Agent name", agentName, "right");
  row++;

  addText("clientEmail", "Client email", clientEmail, "left");
  addText("clientPhone", "Client phone", clientPhone, "right");
  row++;

  // Address spans full width (use left column with larger width)
  fieldsLayout.push({
    kind: "text",
    name: "propertyAddress",
    label: "Property address",
    value: propertyAddress,
    rect: [
      xLeft,
      yTop - row * rowGap - 4,
      xLeft + 495,
      yTop - row * rowGap - 4 + h,
    ],
    wide: true,
  });
  row++;

  // Acknowledgement checkboxes (to guarantee checkbox support in generated PDFs)
  fieldsLayout.push({
    kind: "checkbox",
    name: "clientAck",
    label: "Client confirms details above",
    value: false,
    rect: [
      xLeft,
      yTop - row * rowGap - 2,
      xLeft + 12,
      yTop - row * rowGap + 10,
    ],
  });
  fieldsLayout.push({
    kind: "checkbox",
    name: "agentAck",
    label: "Agent confirms details above",
    value: false,
    rect: [
      xRight,
      yTop - row * rowGap - 2,
      xRight + 12,
      yTop - row * rowGap + 10,
    ],
  });

  return { header, footer, fieldsLayout, contentBlocks: [] };
}
