import sql from "@/app/api/utils/sql";
import { auth } from "@/auth";
import {
  buildContractHtml,
  getTemplateDef,
  getMissingFields,
  tryGenerateFillablePdfFromContractData,
  withPdfSystemState,
  withSignatureDefaults,
} from "@/app/api/utils/contractTemplates";
import { getDbUserIdFromSession } from "@/app/api/utils/dbUser";

function safeString(v) {
  return typeof v === "string" ? v.trim() : "";
}

function addDaysIso(dateIso, days) {
  const dt = new Date(dateIso);
  if (Number.isNaN(dt.getTime())) return dateIso;
  dt.setDate(dt.getDate() + days);
  return dt.toISOString().slice(0, 10);
}

async function loadAgentInfo(userId) {
  const rows = await sql(
    `
    SELECT
      p.id,
      p.full_name as agent_name,
      p.company as company_name,
      au.email as agent_email,
      COALESCE(p.company_logo_url, au.image) as company_logo_url
    FROM profiles p
    LEFT JOIN auth_users au
      ON au.id = (
        CASE
          WHEN p.id::text LIKE '00000000-0000-0000-0000-%'
          THEN (right(p.id::text, 12))::int
          ELSE NULL
        END
      )
    WHERE p.id = $1
    LIMIT 1
    `,
    [userId],
  );
  return rows[0] || null;
}

function buildAddressFromProperty(p) {
  return [p?.address_line, p?.city, p?.postal_code, p?.country]
    .filter(Boolean)
    .join(", ");
}

function buildClientAddress(c) {
  return [c?.city, c?.country].filter(Boolean).join(", ");
}

function buildFloorInfo(p) {
  const floor =
    p?.floor_number === null || p?.floor_number === undefined
      ? ""
      : String(p.floor_number);
  const total =
    p?.total_floors === null || p?.total_floors === undefined
      ? ""
      : String(p.total_floors);
  if (floor && total) return `${floor} / ${total}`;
  return floor || total || "";
}

function normalizeContractValue(val) {
  const s = typeof val === "string" ? val.trim() : "";
  if (!s) return "";

  // Lightweight TR -> EN normalization for common real-estate terms.
  // This keeps the generated PDF English even if the property form stored Turkish values.
  const map = {
    daire: "Apartment",
    dubleks: "Duplex",
    villa: "Villa",
    mustakil: "Detached house",
    müstakil: "Detached house",

    esyali: "Furnished",
    eşyalı: "Furnished",
    esyasiz: "Unfurnished",
    eşyasız: "Unfurnished",

    "kat mulkiyeti": "Condominium title deed",
    "kat mülkiyeti": "Condominium title deed",
    "kat irtifaki": "Construction servitude title deed",
    hisseli: "Shared title deed",

    iskanli: "Habitation permit (Iskan)",
    iskanlı: "Habitation permit (Iskan)",
  };

  const key = s.toLowerCase();
  return map[key] || s;
}

export async function POST(request) {
  try {
    const session = await auth();
    if (!session || !session.user?.id) {
      return Response.json({ error: "Unauthorized" }, { status: 401 });
    }

    const userId = await getDbUserIdFromSession(session);
    if (!userId) {
      return Response.json({ error: "Unauthorized" }, { status: 401 });
    }

    const body = await request.json().catch(() => ({}));

    const propertyId = safeString(body?.propertyId);
    const templateType = safeString(body?.templateType);
    const requestedClientId = safeString(body?.clientId);
    const fieldOverrides =
      body?.fieldOverrides && typeof body.fieldOverrides === "object"
        ? body.fieldOverrides
        : {};

    if (!propertyId || !templateType) {
      return Response.json(
        { error: "propertyId and templateType are required." },
        { status: 400 },
      );
    }

    const props = await sql(
      `
      SELECT
        p.*,
        oc.id as owner_id,
        oc.full_name as owner_name,
        oc.email as owner_email,
        oc.phone as owner_phone,
        oc.city as owner_city,
        oc.country as owner_country
      FROM properties p
      LEFT JOIN clients oc ON p.owner_client_id = oc.id
      WHERE p.id = $1 AND p.user_id = $2
      LIMIT 1
      `,
      [propertyId, userId],
    );

    if (props.length === 0) {
      return Response.json({ error: "Property not found." }, { status: 404 });
    }

    const property = props[0];

    // Note: For the quick-generate flow from Property Detail, we use the linked owner client
    // as the default party record (agents can adjust fields later in the contract editor).
    if (!property.owner_client_id) {
      return Response.json(
        {
          error: "Please link a property owner before generating a contract.",
        },
        { status: 400 },
      );
    }

    const ownerClient = {
      id: property.owner_id,
      full_name: property.owner_name,
      email: property.owner_email,
      phone: property.owner_phone,
      city: property.owner_city,
      country: property.owner_country,
    };

    // NEW: optionally attach the generated contract to a specific client card (buyer/tenant).
    let customerClient = ownerClient;
    if (requestedClientId) {
      const rows = await sql(
        "SELECT id, full_name, email, phone, city, country FROM clients WHERE id = $1 AND user_id = $2 LIMIT 1",
        [requestedClientId, userId],
      );
      if (rows.length === 0) {
        return Response.json({ error: "Client not found." }, { status: 404 });
      }
      customerClient = rows[0];
    }

    if (property.price === null || property.price === undefined) {
      return Response.json(
        { error: "Please set a price on the property before generating." },
        { status: 400 },
      );
    }

    const agent = await loadAgentInfo(userId);

    const today = new Date().toISOString().slice(0, 10);

    const address = buildAddressFromProperty(property);
    const currency = property.currency ? String(property.currency) : "";

    // ===== Fillable templates =====
    const isRichTemplate = [
      "agency_authorization",
      "buyer_representation",
      "handover_protocol",
      "offer_letter",
      "rental_agreement",
      "sale_agreement",
      "seller_listing_agreement",
      "tenant_representation",
      "viewing_report",
    ].includes(templateType);

    // Keep legacy fields for backward compatibility (older UI may expect them).
    const baseLegacyFields = {
      effectiveDate: today,
      propertyAddress: address,
      price: currency
        ? `${property.price} ${currency}`.trim()
        : String(property.price),
      clientName: customerClient.full_name || "",
      clientEmail: customerClient.email || "",
      clientPhone: customerClient.phone || "",
      agentName: agent?.agent_name || session.user?.name || "",
      company: agent?.company_name || "",
    };

    const baseTrFields = isRichTemplate
      ? {
          OWNER_NAME: ownerClient.full_name || "",
          OWNER_EMAIL: ownerClient.email || "",
          OWNER_PHONE: ownerClient.phone || "",
          OWNER_ID: "",
          OWNER_ADDRESS: normalizeContractValue(
            buildClientAddress(ownerClient),
          ),

          // Buyer/Tenant side (saved as contract.client_id)
          CUSTOMER_NAME: customerClient.full_name || "",
          CUSTOMER_EMAIL: customerClient.email || "",
          CUSTOMER_PHONE: customerClient.phone || "",
          CUSTOMER_ID: "",
          CUSTOMER_ADDRESS: normalizeContractValue(
            buildClientAddress(customerClient),
          ),

          PROPERTY_TITLE: property.title || "",
          PROPERTY_ADDRESS: normalizeContractValue(address),
          PROPERTY_TYPE: property.housing_type
            ? normalizeContractValue(String(property.housing_type))
            : "",
          PROPERTY_SIZE:
            property.gross_area_sqm !== null &&
            property.gross_area_sqm !== undefined
              ? String(property.gross_area_sqm)
              : property.size_sqm !== null && property.size_sqm !== undefined
                ? String(property.size_sqm)
                : "",
          ROOM_COUNT:
            property.rooms !== null && property.rooms !== undefined
              ? String(property.rooms)
              : "",
          FLOOR_INFO: buildFloorInfo(property),
          TITLE_DEED_INFO: property.title_deed_status
            ? normalizeContractValue(String(property.title_deed_status))
            : "",

          SALE_PRICE: String(property.price),
          RENT_PRICE: String(property.price),
          CURRENCY: currency,
          PAYMENT_METHOD: "",

          FURNISHED_STATUS: property.furnished_status
            ? normalizeContractValue(String(property.furnished_status))
            : "",
          DEPOSIT_AMOUNT:
            property.deposit !== null && property.deposit !== undefined
              ? String(property.deposit)
              : "",
          RENT_START_DATE: today,
          RENT_DURATION: "",

          AGENCY_NAME: agent?.company_name || "",
          AGENT_NAME: agent?.agent_name || session.user?.name || "",
          AGENT_CONTACT: agent?.agent_email || "",
        }
      : null;

    const cleanOverrides = Object.fromEntries(
      Object.entries(fieldOverrides)
        .filter(([key]) => typeof key === "string" && key.length <= 80)
        .map(([key, value]) => [
          key,
          value === null || value === undefined ? "" : String(value).trim(),
        ]),
    );

    const mergedBase = isRichTemplate
      ? { ...baseLegacyFields, ...baseTrFields }
      : baseLegacyFields;

    const filledFields = withSignatureDefaults({
      ...mergedBase,
      ...cleanOverrides,
    });

    const missing = getMissingFields(templateType, filledFields);
    if (missing.length > 0) {
      return Response.json(
        {
          error:
            "We could not generate this contract because some required details are missing.",
          missingFields: missing,
        },
        { status: 400 },
      );
    }

    const meta = {
      template_type: templateType,
      generated_at: new Date().toISOString(),
      display_name: getTemplateDef(templateType).title,
    };

    // Insert first so we have an ID + version.
    const inserted = await sql(
      `
      INSERT INTO contracts (
        property_id,
        client_id,
        template_type,
        filled_fields,
        storage_path_pdf,
        source_type,
        pdf_url,
        metadata
      )
      VALUES ($1, $2, $3, $4, NULL, $5, NULL, $6)
      RETURNING *
      `,
      [
        propertyId,
        customerClient.id,
        templateType,
        filledFields,
        "generated",
        meta,
      ],
    );

    let contract = inserted[0];

    const html = buildContractHtml({
      templateType,
      property,
      client: customerClient,
      fields: filledFields,
      agent,
      contractMeta: {
        generatedAt: new Date().toISOString(),
        version: contract?.version || 1,
      },
    });

    // NEW: generated PDFs must be fillable (AcroForm). Use the structured generator.
    const pdfResult = await tryGenerateFillablePdfFromContractData({
      templateType,
      property,
      client: customerClient,
      fields: filledFields,
      agent,
      contractMeta: {
        generatedAt: new Date().toISOString(),
        version: contract?.version || 1,
      },
    });

    const fieldsWithState = withPdfSystemState(filledFields, pdfResult);

    if (pdfResult.status === "succeeded" && pdfResult.storagePath) {
      const updated = await sql(
        "UPDATE contracts SET storage_path_pdf = $1, pdf_url = $2, filled_fields = $3 WHERE id = $4 RETURNING *",
        [
          pdfResult.storagePath,
          pdfResult.storagePath,
          fieldsWithState,
          contract.id,
        ],
      );
      contract = updated[0];
      return Response.json(contract, { status: 201 });
    }

    // Non-blocking: keep contract saved even if PDF generation is unavailable.
    const updated = await sql(
      "UPDATE contracts SET filled_fields = $1 WHERE id = $2 RETURNING *",
      [fieldsWithState, contract.id],
    );

    contract = updated[0];
    return Response.json(contract, { status: 201 });
  } catch (error) {
    console.error("POST /api/contracts/generate error:", error);
    return Response.json(
      {
        error:
          error?.message ||
          "We could not create this contract. Please review the details and try again.",
      },
      { status: 500 },
    );
  }
}
