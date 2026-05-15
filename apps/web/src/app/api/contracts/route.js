import sql from "@/app/api/utils/sql";
import { auth } from "@/auth";
import {
  buildContractHtml,
  getMissingFields,
  tryGenerateFillablePdfFromContractData,
  withPdfSystemState,
  withSignatureDefaults,
} from "@/app/api/utils/contractTemplates";
import { getDbUserIdFromSession } from "@/app/api/utils/dbUser";

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

function buildPropertyAddress(p) {
  if (!p) return "";
  return [p.address_line, p.city, p.postal_code, p.country]
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

export async function GET(request) {
  try {
    const session = await auth();
    if (!session || !session.user?.id) {
      return Response.json({ error: "Unauthorized" }, { status: 401 });
    }

    const userId = await getDbUserIdFromSession(session);
    if (!userId) {
      return Response.json({ error: "Unauthorized" }, { status: 401 });
    }

    const { searchParams } = new URL(request.url);
    const propertyId = searchParams.get("propertyId");
    const clientId = searchParams.get("clientId");

    let query = `
      SELECT
        co.*,
        p.title AS property_title,
        c.full_name AS client_name
      FROM contracts co
      JOIN properties p ON co.property_id = p.id
      LEFT JOIN clients c ON co.client_id = c.id
      WHERE p.user_id = $1
    `;

    const values = [userId];
    let idx = 2;

    if (propertyId) {
      query += ` AND co.property_id = $${idx}`;
      values.push(propertyId);
      idx++;
    }

    if (clientId) {
      query += ` AND co.client_id = $${idx}`;
      values.push(clientId);
      idx++;
    }

    query += ` ORDER BY co.created_at DESC`;

    const rows = await sql(query, values);
    return Response.json(rows);
  } catch (error) {
    console.error("GET /api/contracts error:", error);
    return Response.json({ error: "Internal Server Error" }, { status: 500 });
  }
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

    const body = await request.json();

    // NEW: propertyId is optional. If omitted, we auto-pick from the client's interested properties.
    const propertyId = body?.propertyId || null;
    const clientId = body?.clientId;
    const templateType = body?.templateType;

    // NEW: allow creating a draft contract without generating a PDF yet.
    const generatePdf = body?.generatePdf === true;

    const incoming =
      body?.filledFields && typeof body.filledFields === "object"
        ? body.filledFields
        : {};
    const filledFieldsRaw = withSignatureDefaults(incoming);

    if (!clientId || !templateType) {
      return Response.json(
        { error: "clientId and templateType are required" },
        { status: 400 },
      );
    }

    // Resolve property:
    // - explicit propertyId wins
    // - otherwise pick the most recently-linked interested property for this client
    let resolvedPropertyId = propertyId;
    if (!resolvedPropertyId) {
      const interestRows = await sql(
        `
        SELECT pic.property_id
        FROM property_interested_clients pic
        JOIN properties p ON p.id = pic.property_id
        WHERE pic.client_id = $1
          AND p.user_id = $2
        ORDER BY pic.created_at DESC
        LIMIT 1
        `,
        [clientId, userId],
      );

      resolvedPropertyId = interestRows?.[0]?.property_id || null;
    }

    if (!resolvedPropertyId) {
      return Response.json(
        {
          error:
            "This client is not linked to any property yet. Please mark the client as 'interested' in a property first.",
        },
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
      [resolvedPropertyId, userId],
    );
    if (props.length === 0) {
      return Response.json({ error: "Property not found" }, { status: 404 });
    }

    const clients = await sql(
      "SELECT * FROM clients WHERE id = $1 AND user_id = $2 LIMIT 1",
      [clientId, userId],
    );
    if (clients.length === 0) {
      return Response.json({ error: "Client not found" }, { status: 404 });
    }

    const property = props[0];
    const customer = clients[0];

    const owner = property.owner_id
      ? {
          id: property.owner_id,
          full_name: property.owner_name,
          email: property.owner_email,
          phone: property.owner_phone,
          city: property.owner_city,
          country: property.owner_country,
        }
      : null;

    const today = new Date().toISOString().slice(0, 10);
    const address = buildPropertyAddress(property);
    const currency = property.currency ? String(property.currency) : "";

    const isTrTemplate =
      templateType === "sale_agreement" || templateType === "rental_agreement";

    // Merge sane defaults from DB into whatever the agent typed.
    const defaultsLegacy = {
      effectiveDate: filledFieldsRaw.effectiveDate || today,
      propertyAddress: filledFieldsRaw.propertyAddress || address,
      price:
        filledFieldsRaw.price ||
        (property.price !== null && property.price !== undefined
          ? currency
            ? `${property.price} ${currency}`.trim()
            : String(property.price)
          : ""),
      clientName: filledFieldsRaw.clientName || customer.full_name || "",
      clientEmail: filledFieldsRaw.clientEmail || customer.email || "",
      clientPhone: filledFieldsRaw.clientPhone || customer.phone || "",
      agentName: filledFieldsRaw.agentName || session.user?.name || "",
      company: filledFieldsRaw.company || "",
    };

    const defaultsTr = isTrTemplate
      ? {
          OWNER_NAME: filledFieldsRaw.OWNER_NAME || owner?.full_name || "",
          OWNER_ID: filledFieldsRaw.OWNER_ID || "",
          OWNER_ADDRESS:
            filledFieldsRaw.OWNER_ADDRESS ||
            (owner ? buildClientAddress(owner) : ""),

          CUSTOMER_NAME:
            filledFieldsRaw.CUSTOMER_NAME || customer.full_name || "",
          CUSTOMER_ID: filledFieldsRaw.CUSTOMER_ID || "",
          CUSTOMER_ADDRESS:
            filledFieldsRaw.CUSTOMER_ADDRESS || buildClientAddress(customer),

          PROPERTY_ADDRESS: filledFieldsRaw.PROPERTY_ADDRESS || address,
          PROPERTY_TYPE:
            filledFieldsRaw.PROPERTY_TYPE ||
            (property.housing_type ? String(property.housing_type) : ""),
          PROPERTY_SIZE:
            filledFieldsRaw.PROPERTY_SIZE ||
            (property.gross_area_sqm !== null &&
            property.gross_area_sqm !== undefined
              ? String(property.gross_area_sqm)
              : property.size_sqm !== null && property.size_sqm !== undefined
                ? String(property.size_sqm)
                : ""),
          ROOM_COUNT:
            filledFieldsRaw.ROOM_COUNT ||
            (property.rooms !== null && property.rooms !== undefined
              ? String(property.rooms)
              : ""),
          FLOOR_INFO: filledFieldsRaw.FLOOR_INFO || buildFloorInfo(property),
          TITLE_DEED_INFO:
            filledFieldsRaw.TITLE_DEED_INFO ||
            (property.title_deed_status
              ? String(property.title_deed_status)
              : ""),

          SALE_PRICE:
            filledFieldsRaw.SALE_PRICE ||
            (property.price !== null && property.price !== undefined
              ? String(property.price)
              : ""),
          RENT_PRICE:
            filledFieldsRaw.RENT_PRICE ||
            (property.price !== null && property.price !== undefined
              ? String(property.price)
              : ""),
          CURRENCY: filledFieldsRaw.CURRENCY || currency,
          PAYMENT_METHOD: filledFieldsRaw.PAYMENT_METHOD || "",

          FURNISHED_STATUS:
            filledFieldsRaw.FURNISHED_STATUS ||
            (property.furnished_status
              ? String(property.furnished_status)
              : ""),
          DEPOSIT_AMOUNT:
            filledFieldsRaw.DEPOSIT_AMOUNT ||
            (property.deposit !== null && property.deposit !== undefined
              ? String(property.deposit)
              : ""),
          RENT_START_DATE: filledFieldsRaw.RENT_START_DATE || today,
          RENT_DURATION: filledFieldsRaw.RENT_DURATION || "",

          AGENCY_NAME: filledFieldsRaw.AGENCY_NAME || "",
          AGENT_NAME: filledFieldsRaw.AGENT_NAME || session.user?.name || "",
          AGENT_CONTACT: filledFieldsRaw.AGENT_CONTACT || "",
        }
      : null;

    const merged = isTrTemplate
      ? { ...defaultsLegacy, ...defaultsTr, ...filledFieldsRaw }
      : { ...defaultsLegacy, ...filledFieldsRaw };

    const filledFields = withSignatureDefaults(merged);

    const missing = getMissingFields(templateType, filledFields);
    if (missing.length > 0) {
      return Response.json(
        {
          error: `Missing required fields for template: ${missing.join(", ")}`,
          missingFields: missing,
        },
        { status: 400 },
      );
    }

    const meta = {
      template_type: templateType,
      created_from: "contracts_new",
      auto_property_from_interest: propertyId ? false : true,
      display_name:
        templateType === "sale_agreement"
          ? "Purchase Agreement"
          : templateType === "rental_agreement"
            ? "Lease Agreement"
            : null,
    };

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
      [resolvedPropertyId, clientId, templateType, filledFields, "draft", meta],
    );

    let contract = inserted[0];

    // NEW: Draft-first flow. Only generate PDF if explicitly requested.
    if (!generatePdf) {
      return Response.json(contract, { status: 201 });
    }

    const agent = await loadAgentInfo(userId);

    // Build HTML (kept for debugging/compat)
    buildContractHtml({
      templateType,
      property,
      client: customer,
      fields: filledFields,
      agent,
      contractMeta: {
        generatedAt: new Date().toISOString(),
        version: contract?.version || 1,
      },
    });

    const pdfResult = await tryGenerateFillablePdfFromContractData({
      templateType,
      property,
      client: customer,
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

    const updated = await sql(
      "UPDATE contracts SET filled_fields = $1 WHERE id = $2 RETURNING *",
      [fieldsWithState, contract.id],
    );

    contract = updated[0];
    return Response.json(contract, { status: 201 });
  } catch (error) {
    console.error("POST /api/contracts error:", error);
    return Response.json({ error: "Internal Server Error" }, { status: 500 });
  }
}
