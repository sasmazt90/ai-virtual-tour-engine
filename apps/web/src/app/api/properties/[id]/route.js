import sql from "@/app/api/utils/sql";
import { auth } from "@/auth";
import { getDbUserIdFromSession } from "@/app/api/utils/dbUser";
import {
  buildNearbyPlaces,
  geocodeAddress,
} from "@/app/api/utils/googlePlaces";

const JSONB_FIELDS = new Set([
  "features_interior",
  "features_exterior",
  "features_location",
  "nearby_places",
]);

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

function parseJsonbMaybe(raw) {
  if (!raw) return null;
  if (typeof raw === "object") return raw;
  if (typeof raw === "string") {
    try {
      const parsed = JSON.parse(raw);
      return parsed && typeof parsed === "object" ? parsed : null;
    } catch {
      return null;
    }
  }
  return null;
}

function hasAnyNearbyData(nearbyObj) {
  const n = nearbyObj && typeof nearbyObj === "object" ? nearbyObj : null;
  if (!n) return false;

  const groups = [n.health, n.shopping, n.education, n.transport];
  for (const g of groups) {
    if (Array.isArray(g) && g.length > 0) return true;
  }

  return false;
}

export async function GET(request, { params }) {
  try {
    const session = await auth();
    if (!session || !session.user?.id) {
      return Response.json({ error: "Unauthorized" }, { status: 401 });
    }

    const userId = await getDbUserIdFromSession(session);
    if (!userId) {
      return Response.json({ error: "Unauthorized" }, { status: 401 });
    }

    const { id } = params;

    const properties = await sql`
      SELECT 
        p.*,
        c.full_name as owner_name,
        c.email as owner_email,
        c.phone as owner_phone
      FROM properties p
      LEFT JOIN clients c ON p.owner_client_id = c.id
      WHERE p.id = ${id} AND p.user_id = ${userId}
      LIMIT 1
    `;

    if (properties.length === 0) {
      return Response.json({ error: "Property not found" }, { status: 404 });
    }

    // Backfill nearby_places for older properties:
    // If the address was saved before we shipped nearby places, nearby_places may be null.
    // We compute best-effort during fetch (once) and store it.
    let property = properties[0];

    try {
      const addressParts = [
        property.address_line,
        property.city,
        property.postal_code,
        property.country,
      ]
        .map((x) => (x ? String(x).trim() : ""))
        .filter(Boolean);

      const hasAddress = addressParts.length > 0;
      const nearbyParsed = parseJsonbMaybe(property.nearby_places);
      const alreadyHasNearby = hasAnyNearbyData(nearbyParsed);

      const lat = Number(property.geo_lat);
      const lng = Number(property.geo_lng);
      const hasGeo = Number.isFinite(lat) && Number.isFinite(lng);

      if (hasAddress && !alreadyHasNearby) {
        let nextLat = hasGeo ? lat : null;
        let nextLng = hasGeo ? lng : null;
        let nextFormatted = property.address_formatted || null;

        if (!hasGeo) {
          const geo = await geocodeAddress({
            addressLine: property.address_line,
            city: property.city,
            postalCode: property.postal_code,
            country: property.country,
          });

          if (geo?.ok) {
            nextLat = geo.lat;
            nextLng = geo.lng;
            nextFormatted = geo.formattedAddress || null;
          }
        }

        if (
          Number.isFinite(Number(nextLat)) &&
          Number.isFinite(Number(nextLng))
        ) {
          const nearby = await buildNearbyPlaces({
            lat: Number(nextLat),
            lng: Number(nextLng),
          });

          if (nearby && hasAnyNearbyData(nearby)) {
            const updated = await sql(
              "UPDATE properties SET geo_lat = $1, geo_lng = $2, address_formatted = $3, nearby_places = $4::jsonb WHERE id = $5 AND user_id = $6 RETURNING *",
              [
                nextLat,
                nextLng,
                nextFormatted,
                toJsonbParam(nearby),
                id,
                userId,
              ],
            );

            if (updated && updated.length > 0) {
              property = updated[0];
            } else {
              // If update didn't return (unexpected), still enrich the response.
              property = {
                ...property,
                geo_lat: nextLat,
                geo_lng: nextLng,
                address_formatted: nextFormatted,
                nearby_places: nearby,
              };
            }
          }
        }
      }
    } catch (e) {
      // Best-effort only. Never break property fetch.
      console.warn("GET /api/properties/[id]: nearby backfill failed", e);
    }

    const photos = await sql`
      SELECT * FROM property_photos 
      WHERE property_id = ${id}
      ORDER BY sort_order
    `;

    const interestedClients = await sql`
      SELECT c.* 
      FROM clients c
      JOIN property_interested_clients pic ON c.id = pic.client_id
      WHERE pic.property_id = ${id}
      ORDER BY c.full_name
    `;

    const stagings = await sql`
      SELECT 
        s.*,
        json_agg(
          json_build_object('id', si.id, 'storage_path', si.storage_path)
        ) FILTER (WHERE si.id IS NOT NULL) as images
      FROM stagings s
      LEFT JOIN staging_images si ON s.id = si.staging_id
      WHERE s.property_id = ${id}
      GROUP BY s.id
      ORDER BY s.created_at DESC
    `;

    const tours = await sql`
      SELECT * FROM virtual_tours 
      WHERE property_id = ${id}
      ORDER BY created_at DESC
    `;

    const contracts = await sql`
      SELECT 
        co.*,
        c.full_name as client_name
      FROM contracts co
      LEFT JOIN clients c ON co.client_id = c.id
      WHERE co.property_id = ${id}
      ORDER BY co.created_at DESC
    `;

    return Response.json({
      ...property,
      photos,
      interested_clients: interestedClients,
      stagings,
      tours,
      contracts,
    });
  } catch (error) {
    console.error("GET /api/properties/[id] error:", error);
    return Response.json({ error: "Internal Server Error" }, { status: 500 });
  }
}

export async function PUT(request, { params }) {
  try {
    const session = await auth();
    if (!session || !session.user?.id) {
      return Response.json({ error: "Unauthorized" }, { status: 401 });
    }

    const userId = await getDbUserIdFromSession(session);
    if (!userId) {
      return Response.json({ error: "Unauthorized" }, { status: 401 });
    }

    const { id } = params;
    const body = await request.json();

    const properties = await sql`
      SELECT * FROM properties WHERE id = ${id} AND user_id = ${userId}
    `;

    if (properties.length === 0) {
      return Response.json({ error: "Property not found" }, { status: 404 });
    }

    const setClauses = [];
    const values = [];
    let paramIndex = 1;

    const updatableFields = [
      "title",
      "property_status",
      "address_line",
      "city",
      "postal_code",
      "country",
      "price",
      "currency",
      "size_sqm",
      "rooms",
      "description",
      "owner_client_id",
      "housing_type",
      "housing_shape",
      "bedrooms",
      "living_rooms",
      "bathrooms",
      "gross_area_sqm",
      "net_area_sqm",
      "total_floors",
      "floor_number",
      "building_age",
      "heating_type",
      "elevator",
      "parking_type",
      "title_deed_status",
      "furnished_status",
      "mortgage_eligible",
      "construction_type",
      "usage_status",
      "facade",
      "deposit",
      "dues",
      "features_interior",
      "features_exterior",
      "features_location",
    ];

    updatableFields.forEach((field) => {
      if (body[field] !== undefined) {
        if (JSONB_FIELDS.has(field)) {
          setClauses.push(`${field} = $${paramIndex}::jsonb`);
          values.push(toJsonbParam(body[field]));
        } else {
          setClauses.push(`${field} = $${paramIndex}`);
          values.push(body[field]);
        }
        paramIndex++;
      }
    });

    const addressChanged =
      body.address_line !== undefined ||
      body.city !== undefined ||
      body.postal_code !== undefined ||
      body.country !== undefined;

    if (addressChanged) {
      const nextAddressLine =
        body.address_line !== undefined
          ? body.address_line
          : properties[0].address_line;
      const nextCity = body.city !== undefined ? body.city : properties[0].city;
      const nextPostal =
        body.postal_code !== undefined
          ? body.postal_code
          : properties[0].postal_code;
      const nextCountry =
        body.country !== undefined ? body.country : properties[0].country;

      const geo = await geocodeAddress({
        addressLine: nextAddressLine,
        city: nextCity,
        postalCode: nextPostal,
        country: nextCountry,
      });

      if (geo?.ok) {
        const nearby = await buildNearbyPlaces({ lat: geo.lat, lng: geo.lng });

        setClauses.push(`geo_lat = $${paramIndex}`);
        values.push(geo.lat);
        paramIndex++;

        setClauses.push(`geo_lng = $${paramIndex}`);
        values.push(geo.lng);
        paramIndex++;

        setClauses.push(`address_formatted = $${paramIndex}`);
        values.push(geo.formattedAddress || null);
        paramIndex++;

        setClauses.push(`nearby_places = $${paramIndex}::jsonb`);
        values.push(toJsonbParam(nearby || null));
        paramIndex++;
      }
    }

    if (setClauses.length === 0) {
      return Response.json({ error: "No fields to update" }, { status: 400 });
    }

    const queryStr = `
      UPDATE properties 
      SET ${setClauses.join(", ")} 
      WHERE id = $${paramIndex} AND user_id = $${paramIndex + 1}
      RETURNING *
    `;
    values.push(id, userId);

    const result = await sql(queryStr, values);
    return Response.json(result[0]);
  } catch (error) {
    console.error("PUT /api/properties/[id] error:", error);
    return Response.json({ error: "Internal Server Error" }, { status: 500 });
  }
}

export async function DELETE(request, { params }) {
  try {
    const session = await auth();
    if (!session || !session.user?.id) {
      return Response.json({ error: "Unauthorized" }, { status: 401 });
    }

    const userId = await getDbUserIdFromSession(session);
    if (!userId) {
      return Response.json({ error: "Unauthorized" }, { status: 401 });
    }

    const { id } = params;

    const result = await sql`
      DELETE FROM properties 
      WHERE id = ${id} AND user_id = ${userId}
      RETURNING id
    `;

    if (result.length === 0) {
      return Response.json({ error: "Property not found" }, { status: 404 });
    }

    return Response.json({ success: true, id: result[0].id });
  } catch (error) {
    console.error("DELETE /api/properties/[id] error:", error);
    return Response.json({ error: "Internal Server Error" }, { status: 500 });
  }
}
