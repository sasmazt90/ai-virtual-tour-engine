import sql from "@/app/api/utils/sql";
import { auth } from "@/auth";
import { getDbUserIdFromSession } from "@/app/api/utils/dbUser";
import {
  collectStorageUrlsFromValue,
  deleteSupabaseStorageObjects,
} from "@/app/api/utils/storageCleanup";

function normalizeStagingType(raw) {
  const s = typeof raw === "string" ? raw.trim() : "";
  if (!s) return "";
  const normalized = s.toLowerCase().replace(/\s+/g, "_").replace(/-+/g, "_");

  // Keep this aligned with the DB enum values
  const allowed = new Set([
    "default",
    "vacant",
    "minimalist",
    "luxury",
    "scandinavian",
    "classic",
    "modern",
    "custom",
  ]);

  return allowed.has(normalized) ? normalized : "";
}

export async function POST(request) {
  try {
    const session = await auth();
    if (!session || !session.user?.id) {
      return Response.json(
        { error: "Please sign in to save a virtual tour." },
        { status: 401 },
      );
    }

    const userId = await getDbUserIdFromSession(session);
    if (!userId) {
      return Response.json(
        { error: "Please sign in to save a virtual tour." },
        { status: 401 },
      );
    }

    const body = await request.json().catch(() => ({}));

    const propertyIdRaw = body?.propertyId;
    const dataUrlRaw = body?.data_url;
    const sourceTypeRaw = body?.sourceType;
    const stagingTypeRaw = body?.stagingType;

    const propertyId =
      typeof propertyIdRaw === "string" ? propertyIdRaw.trim() : "";
    const dataUrl = typeof dataUrlRaw === "string" ? dataUrlRaw.trim() : "";

    const sourceType =
      sourceTypeRaw === "staging" || sourceTypeRaw === "original"
        ? sourceTypeRaw
        : "original";

    // Safeguard 1: Reject missing/empty data_url (preview must be generated first)
    if (!dataUrl) {
      return Response.json(
        {
          error:
            "Tour data is missing. Please generate the virtual tour again and then try saving.",
        },
        { status: 400 },
      );
    }

    // Safeguard 2: Ensure we have a valid propertyId
    if (!propertyId) {
      return Response.json(
        {
          error:
            "We couldn't find the property for this tour. Please go back to the property and try again.",
        },
        { status: 400 },
      );
    }

    // Safeguard 3: Verify the property belongs to the current user
    const props = await sql(
      "SELECT id FROM properties WHERE id = $1 AND user_id = $2 LIMIT 1",
      [propertyId, userId],
    );

    if (props.length === 0) {
      return Response.json(
        {
          error:
            "That property isn't available in your account. Please go back and try again.",
        },
        { status: 403 },
      );
    }

    // Slot key for staging-based tours is (property_id, source_type='staging', staging_type)
    // Slot key for original is (property_id, source_type='original', staging_type NULL)

    let baseStagingId = null;
    let stagingTypeDb = null;

    if (sourceType === "staging") {
      const stagingType = normalizeStagingType(stagingTypeRaw);
      if (!stagingType) {
        return Response.json(
          {
            error:
              "Please choose a staging type for this virtual tour and try again.",
          },
          { status: 400 },
        );
      }

      // Ensure staging exists for this property + type (single slot)
      const stagingRows = await sql(
        "SELECT id FROM stagings WHERE property_id = $1 AND staging_type = $2 LIMIT 1",
        [propertyId, stagingType],
      );

      if (stagingRows.length === 0) {
        return Response.json(
          {
            error:
              "That staging isn't available for this property. Please go back and try again.",
          },
          { status: 403 },
        );
      }

      baseStagingId = stagingRows[0].id;
      stagingTypeDb = stagingType;
    }

    const tourPayload = {
      data_url: dataUrl,
      sourceType,
      stagingType: stagingTypeDb,
    };

    // Overwrite behavior (never create a 2nd tour for the same source slot)
    const existing = await sql(
      sourceType === "original"
        ? "SELECT id, tour_payload FROM virtual_tours WHERE property_id = $1 AND source_type = 'original' LIMIT 1"
        : "SELECT id, tour_payload FROM virtual_tours WHERE property_id = $1 AND source_type = 'staging' AND staging_type = $2 LIMIT 1",
      sourceType === "original" ? [propertyId] : [propertyId, stagingTypeDb],
    );

    if (existing.length > 0) {
      const id = existing[0].id;

      const updated = await sql(
        "UPDATE virtual_tours SET base_staging_id = $1, source_type = $2, staging_type = $3, tour_type = $4, tour_payload = $5 WHERE id = $6 RETURNING id, created_at",
        [baseStagingId, sourceType, stagingTypeDb, "panorama", tourPayload, id],
      );
      await deleteSupabaseStorageObjects(
        collectStorageUrlsFromValue(existing[0].tour_payload),
      );

      return Response.json({
        id: updated?.[0]?.id,
        created_at: updated?.[0]?.created_at,
        overwritten: true,
      });
    }

    const rows = await sql(
      "INSERT INTO virtual_tours (property_id, base_staging_id, source_type, staging_type, tour_type, tour_payload) VALUES ($1, $2, $3, $4, $5, $6) RETURNING id, created_at",
      [
        propertyId,
        baseStagingId,
        sourceType,
        stagingTypeDb,
        "panorama",
        tourPayload,
      ],
    );

    return Response.json({
      id: rows?.[0]?.id,
      created_at: rows?.[0]?.created_at,
      overwritten: false,
    });
  } catch (error) {
    console.error("POST /api/virtual-tours error:", error);
    return Response.json(
      { error: "We couldn't save the virtual tour. Please try again." },
      { status: 500 },
    );
  }
}
