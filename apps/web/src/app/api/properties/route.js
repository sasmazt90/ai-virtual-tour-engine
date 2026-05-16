import sql from "@/app/api/utils/sql";
import { auth } from "@/auth";
import { getDbUserIdFromSession } from "@/app/api/utils/dbUser";
import {
  buildNearbyPlaces,
  geocodeAddress,
} from "@/app/api/utils/googlePlaces";

const UUID_RE =
  /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

function sanitizeStringArray(value) {
  if (!Array.isArray(value)) {
    return [];
  }

  return value
    .map((v) => (v === null || v === undefined ? "" : String(v).trim()))
    .filter(Boolean);
}

function toJsonbParam(value) {
  if (value === null || value === undefined) {
    return null;
  }

  try {
    return JSON.stringify(value);
  } catch (error) {
    console.warn("toJsonbParam: failed to stringify", error);
    return null;
  }
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
    const status = searchParams.get("status");
    const search = searchParams.get("search");

    let queryStr = `
      SELECT 
        p.*,
        json_agg(
          json_build_object('id', pp.id, 'storage_path', pp.storage_path, 'sort_order', pp.sort_order)
          ORDER BY pp.sort_order
        ) FILTER (WHERE pp.id IS NOT NULL) as photos
      FROM properties p
      LEFT JOIN property_photos pp ON p.id = pp.property_id
      WHERE p.user_id = $1
    `;
    const values = [userId];
    let paramIndex = 2;

    if (status && status !== "all") {
      queryStr += ` AND p.property_status = $${paramIndex}`;
      values.push(status);
      paramIndex++;
    }

    if (search) {
      queryStr += ` AND (
        LOWER(p.title) LIKE LOWER($${paramIndex}) OR
        LOWER(p.address_line) LIKE LOWER($${paramIndex}) OR
        LOWER(p.city) LIKE LOWER($${paramIndex})
      )`;
      values.push(`%${search}%`);
      paramIndex++;
    }

    queryStr += ` GROUP BY p.id ORDER BY p.created_at DESC`;

    const properties = await sql(queryStr, values);
    return Response.json(properties);
  } catch (error) {
    console.error("GET /api/properties error:", error);
    return Response.json(
      {
        error: "Internal Server Error",
        message: error instanceof Error ? error.message : String(error),
      },
      { status: 500 },
    );
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
    const {
      title,
      property_status,
      address_line,
      city,
      postal_code,
      country,
      price,
      currency,
      size_sqm,
      rooms,
      description,
      owner_client_id,
      photos = [],

      housing_type,
      housing_shape,
      bedrooms,
      living_rooms,
      bathrooms,
      gross_area_sqm,
      net_area_sqm,
      total_floors,
      floor_number,
      building_age,
      heating_type,
      elevator,
      parking_type,
      title_deed_status,
      furnished_status,
      mortgage_eligible,
      construction_type,
      usage_status,
      facade,
      deposit,
      dues,
      features_interior,
      features_exterior,
      features_location,
    } = body;

    if (!title || !property_status) {
      return Response.json(
        { error: "Title and status are required" },
        { status: 400 },
      );
    }

    // Owner is required for this app flow.
    const ownerClientId = owner_client_id ? String(owner_client_id).trim() : "";
    if (!ownerClientId) {
      return Response.json(
        { error: "Owner client is required" },
        { status: 400 },
      );
    }

    if (!UUID_RE.test(ownerClientId)) {
      return Response.json(
        { error: "Invalid owner client id" },
        { status: 400 },
      );
    }

    // IMPORTANT:
    // Neon/pg will serialize JS arrays as Postgres arrays (e.g. {"a","b"}).
    // But our DB columns are jsonb, so we must send JSON strings and cast to jsonb.
    const safeFeaturesInterior = sanitizeStringArray(features_interior);
    const safeFeaturesExterior = sanitizeStringArray(features_exterior);
    const safeFeaturesLocation = sanitizeStringArray(features_location);

    // Best-effort: geocode & compute nearby places from address.
    // IMPORTANT: this must never block property creation.
    let geoLat = null;
    let geoLng = null;
    let formattedAddress = null;
    let nearbyPlaces = {};

    try {
      const geo = await geocodeAddress({
        addressLine: address_line,
        city,
        postalCode: postal_code,
        country,
      });

      if (geo?.ok) {
        geoLat = geo.lat;
        geoLng = geo.lng;
        formattedAddress = geo.formattedAddress || null;

        try {
          nearbyPlaces = await buildNearbyPlaces({
            lat: geo.lat,
            lng: geo.lng,
          });
        } catch (error) {
          console.warn("POST /api/properties: buildNearbyPlaces failed", error);
        }
      }
    } catch (error) {
      console.warn("POST /api/properties: geocodeAddress failed", error);
    }

    const result = await sql`
      INSERT INTO properties (
        user_id,
        title,
        property_status,
        address_line,
        city,
        postal_code,
        country,
        price,
        currency,
        size_sqm,
        rooms,
        description,
        owner_client_id,

        housing_type,
        housing_shape,
        bedrooms,
        living_rooms,
        bathrooms,
        gross_area_sqm,
        net_area_sqm,
        total_floors,
        floor_number,
        building_age,
        heating_type,
        elevator,
        parking_type,
        title_deed_status,
        furnished_status,
        mortgage_eligible,
        construction_type,
        usage_status,
        facade,
        deposit,
        dues,
        features_interior,
        features_exterior,
        features_location,

        geo_lat,
        geo_lng,
        address_formatted,
        nearby_places
      )
      VALUES (
        ${userId},
        ${title},
        ${property_status},
        ${address_line || null},
        ${city || null},
        ${postal_code || null},
        ${country || null},
        ${price || null},
        ${currency || null},
        ${size_sqm || null},
        ${rooms || null},
        ${description || null},
        ${ownerClientId},

        ${housing_type || null},
        ${housing_shape || null},
        ${bedrooms ?? null},
        ${living_rooms ?? null},
        ${bathrooms ?? null},
        ${gross_area_sqm ?? null},
        ${net_area_sqm ?? null},
        ${total_floors ?? null},
        ${floor_number ?? null},
        ${building_age ?? null},
        ${heating_type || null},
        ${elevator ?? null},
        ${parking_type || null},
        ${title_deed_status || null},
        ${furnished_status || null},
        ${mortgage_eligible ?? null},
        ${construction_type || null},
        ${usage_status || null},
        ${facade || null},
        ${deposit ?? null},
        ${dues ?? null},
        ${toJsonbParam(safeFeaturesInterior)}::jsonb,
        ${toJsonbParam(safeFeaturesExterior)}::jsonb,
        ${toJsonbParam(safeFeaturesLocation)}::jsonb,

        ${geoLat},
        ${geoLng},
        ${formattedAddress},
        ${toJsonbParam(nearbyPlaces) || "{}"}::jsonb
      )
      RETURNING *
    `;

    const property = result[0];

    if (photos && photos.length > 0) {
      for (let i = 0; i < photos.length; i++) {
        await sql`
          INSERT INTO property_photos (property_id, storage_path, sort_order)
          VALUES (${property.id}, ${photos[i]}, ${i})
        `;
      }
    }

    return Response.json(property, { status: 201 });
  } catch (error) {
    console.error("POST /api/properties error:", error);
    return Response.json(
      {
        error: "Internal Server Error",
        message: error instanceof Error ? error.message : String(error),
      },
      { status: 500 },
    );
  }
}
