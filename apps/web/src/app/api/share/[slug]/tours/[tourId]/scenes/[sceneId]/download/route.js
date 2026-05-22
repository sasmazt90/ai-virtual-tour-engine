import sql from "@/app/api/utils/sql";

function safeFilenamePart(str) {
  return String(str || "")
    .replaceAll(/[^a-zA-Z0-9._-]+/g, "-")
    .slice(0, 80);
}

function safeArray(v) {
  return Array.isArray(v) ? v : [];
}

function uniqStrings(arr) {
  const out = [];
  const seen = new Set();
  for (const v of safeArray(arr)) {
    const s = v === null || v === undefined ? "" : String(v).trim();
    if (!s || seen.has(s)) continue;
    seen.add(s);
    out.push(s);
  }
  return out;
}

function normalizeStagingType(raw) {
  const s = typeof raw === "string" ? raw.trim() : "";
  if (!s) return "";
  const normalized = s.toLowerCase().replace(/\s+/g, "_").replace(/-+/g, "_");
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

function normalizeTourSlot(raw) {
  const sourceType =
    raw?.sourceType === "staging" || raw?.sourceType === "original"
      ? raw.sourceType
      : null;
  if (!sourceType) return null;
  if (sourceType === "original") {
    return { sourceType: "original", stagingType: null };
  }
  const stagingType = normalizeStagingType(raw?.stagingType);
  if (!stagingType) return null;
  return { sourceType: "staging", stagingType };
}

function tourSlotKey(slot) {
  if (slot?.sourceType === "original") return "original";
  if (slot?.sourceType === "staging" && slot?.stagingType) {
    return `staging:${slot.stagingType}`;
  }
  return "";
}

function tourRowKey(tour) {
  if (tour?.source_type === "original") return "original";
  if (tour?.source_type === "staging" && tour?.staging_type) {
    return `staging:${String(tour.staging_type).trim()}`;
  }
  return "";
}

function linkAllowsTour(link, tour) {
  if (String(link.include_tour_id || "") === String(tour.id)) return true;

  const columnIds = uniqStrings(link.include_virtual_tour_ids);
  if (columnIds.includes(String(tour.id))) return true;

  const meta = link.meta && typeof link.meta === "object" ? link.meta : {};
  const metaIds = uniqStrings(meta?.include_tour_ids);
  if (metaIds.includes(String(tour.id))) return true;

  const slotKeys = new Set(
    safeArray(meta?.include_tour_slots)
      .map((slot) => normalizeTourSlot(slot))
      .filter(Boolean)
      .map((slot) => tourSlotKey(slot))
      .filter(Boolean),
  );

  return slotKeys.has(tourRowKey(tour));
}

export async function GET(request, { params }) {
  try {
    const slug = params?.slug;
    const tourId = params?.tourId;
    const sceneId = params?.sceneId;

    if (!slug || !tourId || !sceneId) {
      return Response.json(
        { error: "This file cannot be accessed at the moment." },
        { status: 400 },
      );
    }

    const links = await sql(
      `
      SELECT *
      FROM share_links
      WHERE slug = $1
      LIMIT 1
      `,
      [slug],
    );

    if (links.length === 0) {
      return Response.json(
        { error: "This file is no longer available." },
        { status: 404 },
      );
    }

    const link = links[0];

    if (link.expires_at) {
      const expires = new Date(link.expires_at);
      if (expires.getTime() < Date.now()) {
        return Response.json(
          { error: "This file is no longer available." },
          { status: 410 },
        );
      }
    }

    const rows = await sql(
      `
      SELECT id, tour_type, tour_payload, source_type, staging_type
      FROM virtual_tours
      WHERE id = $1 AND property_id = $2
      LIMIT 1
      `,
      [tourId, link.property_id],
    );

    if (rows.length === 0) {
      return Response.json(
        { error: "This file is no longer available." },
        { status: 404 },
      );
    }

    const tour = rows[0];
    if (!linkAllowsTour(link, tour)) {
      return Response.json(
        { error: "This file is no longer available." },
        { status: 404 },
      );
    }

    const payload = tour.tour_payload || {};
    const scenes = Array.isArray(payload?.scenes) ? payload.scenes : [];

    const scene = scenes.find((s) => String(s?.sceneId) === String(sceneId));
    const url = scene?.imageUrl;

    if (!url) {
      return Response.json(
        { error: "This file is no longer available." },
        { status: 404 },
      );
    }

    const upstream = await fetch(url);
    if (!upstream.ok) {
      const text = await upstream.text().catch(() => "");
      console.error("Share tour scene fetch failed:", upstream.status, text);
      return Response.json(
        { error: "This file cannot be accessed at the moment." },
        { status: 500 },
      );
    }

    const buffer = await upstream.arrayBuffer();
    const contentType =
      upstream.headers.get("content-type") || "application/octet-stream";

    const filename = `tour-${safeFilenamePart(tourId)}-${safeFilenamePart(sceneId)}.bin`;

    return new Response(buffer, {
      status: 200,
      headers: {
        "Content-Type": contentType,
        "Content-Disposition": `inline; filename="${filename}"`,
        "Cache-Control": "private, max-age=60",
      },
    });
  } catch (error) {
    console.error(
      "GET /api/share/[slug]/tours/[tourId]/scenes/[sceneId]/download error:",
      error,
    );
    return Response.json(
      { error: "This file cannot be accessed at the moment." },
      { status: 500 },
    );
  }
}
