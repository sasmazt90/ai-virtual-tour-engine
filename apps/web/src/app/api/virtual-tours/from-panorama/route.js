import sql from "@/app/api/utils/sql";
import { auth } from "@/auth";
import { getDbUserIdFromSession } from "@/app/api/utils/dbUser";

function buildSinglePointPanoramaPayload(panoramaUrl) {
  return {
    type: "virtual_tour",
    initialPointId: "P1",
    points: [
      {
        pointId: "P1",
        label: "Panorama",
        panoramaUrl,
        initialYaw: 0,
        hotspots: [],
      },
    ],
  };
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

    const propertyId = body?.propertyId;
    const panoramaUrl = body?.panoramaUrl;

    if (!propertyId || typeof propertyId !== "string") {
      return Response.json(
        { error: "propertyId is required" },
        { status: 400 },
      );
    }

    if (!panoramaUrl || typeof panoramaUrl !== "string") {
      return Response.json(
        { error: "panoramaUrl is required" },
        { status: 400 },
      );
    }

    const props = await sql(
      "SELECT id FROM properties WHERE id = $1 AND user_id = $2 LIMIT 1",
      [propertyId, userId],
    );

    if (props.length === 0) {
      return Response.json({ error: "Property not found" }, { status: 404 });
    }

    const tourPayload = buildSinglePointPanoramaPayload(panoramaUrl);

    const rows = await sql(
      "INSERT INTO virtual_tours (property_id, base_staging_id, source_type, tour_type, tour_payload) VALUES ($1, $2, $3, $4, $5::jsonb) RETURNING id",
      [propertyId, null, "original", "panorama", JSON.stringify(tourPayload)],
    );

    const tourId = rows?.[0]?.id;

    return Response.json({ tourId, tour_type: "panorama" });
  } catch (error) {
    console.error("POST /api/virtual-tours/from-panorama error:", error);
    return Response.json({ error: "Internal Server Error" }, { status: 500 });
  }
}
