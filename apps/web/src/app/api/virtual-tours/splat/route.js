import sql from "@/app/api/utils/sql";
import { auth } from "@/auth";
import { getDbUserIdFromSession } from "@/app/api/utils/dbUser";

const SUPPORTED_SPLAT_EXTENSIONS = new Set(["ply", "splat", "ksplat"]);

function getExtension(value) {
  try {
    const url = new URL(String(value || ""));
    const cleanPath = url.pathname || "";
    return cleanPath.split(".").pop()?.toLowerCase().trim() || "";
  } catch {
    return String(value || "").split(".").pop()?.toLowerCase().trim() || "";
  }
}

function normalizeNumberArray(raw, fallback) {
  if (!Array.isArray(raw)) return fallback;
  const next = raw.map((v) => Number(v));
  if (next.some((v) => !Number.isFinite(v))) return fallback;
  return next;
}

export async function POST(request) {
  try {
    const session = await auth();
    if (!session || !session.user?.id) {
      return Response.json(
        { error: "Please sign in to save a 3D tour." },
        { status: 401 },
      );
    }

    const userId = await getDbUserIdFromSession(session);
    if (!userId) {
      return Response.json(
        { error: "Please sign in to save a 3D tour." },
        { status: 401 },
      );
    }

    const body = await request.json().catch(() => ({}));
    const propertyId =
      typeof body?.propertyId === "string" ? body.propertyId.trim() : "";
    const fileUrl = typeof body?.fileUrl === "string" ? body.fileUrl.trim() : "";
    const originalName =
      typeof body?.originalName === "string" ? body.originalName.trim() : "";
    const formatRaw =
      typeof body?.format === "string" ? body.format.trim().toLowerCase() : "";
    const format = formatRaw || getExtension(originalName) || getExtension(fileUrl);

    if (!propertyId) {
      return Response.json({ error: "Missing property ID." }, { status: 400 });
    }

    if (!fileUrl || !/^https?:\/\//i.test(fileUrl)) {
      return Response.json(
        { error: "Please upload a valid 3D scan file first." },
        { status: 400 },
      );
    }

    if (!SUPPORTED_SPLAT_EXTENSIONS.has(format)) {
      return Response.json(
        { error: "Supported 3D scan formats are .ply, .splat and .ksplat." },
        { status: 400 },
      );
    }

    const props = await sql(
      "SELECT id FROM properties WHERE id = $1 AND user_id = $2 LIMIT 1",
      [propertyId, userId],
    );

    if (props.length === 0) {
      return Response.json(
        { error: "Property not found in your account." },
        { status: 404 },
      );
    }

    const camera = body?.camera && typeof body.camera === "object" ? body.camera : {};
    const tourPayload = {
      type: "splat3d",
      fileUrl,
      format,
      originalName: originalName || null,
      sourceType: "original",
      camera: {
        up: normalizeNumberArray(camera.up, [0, -1, -0.6]).slice(0, 3),
        position: normalizeNumberArray(camera.position, [-1, -4, 6]).slice(0, 3),
        lookAt: normalizeNumberArray(camera.lookAt, [0, 0, 0]).slice(0, 3),
      },
    };

    const existing = await sql(
      "SELECT id FROM virtual_tours WHERE property_id = $1 AND source_type = 'original' LIMIT 1",
      [propertyId],
    );

    if (existing.length > 0) {
      const rows = await sql(
        "UPDATE virtual_tours SET base_staging_id = NULL, source_type = 'original', staging_type = NULL, tour_type = 'splat3d', tour_payload = $1::jsonb WHERE id = $2 RETURNING id, created_at",
        [JSON.stringify(tourPayload), existing[0].id],
      );
      return Response.json({
        id: rows?.[0]?.id,
        created_at: rows?.[0]?.created_at,
        overwritten: true,
      });
    }

    const rows = await sql(
      "INSERT INTO virtual_tours (property_id, base_staging_id, source_type, staging_type, tour_type, tour_payload) VALUES ($1, NULL, 'original', NULL, 'splat3d', $2::jsonb) RETURNING id, created_at",
      [propertyId, JSON.stringify(tourPayload)],
    );

    return Response.json({
      id: rows?.[0]?.id,
      created_at: rows?.[0]?.created_at,
      overwritten: false,
    });
  } catch (error) {
    console.error("POST /api/virtual-tours/splat error:", error);
    return Response.json(
      { error: "We couldn't save the 3D virtual tour. Please try again." },
      { status: 500 },
    );
  }
}
