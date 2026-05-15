import sql from "@/app/api/utils/sql";

function safeFilenamePart(str) {
  return String(str || "")
    .replaceAll(/[^a-zA-Z0-9._-]+/g, "-")
    .slice(0, 80);
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

    if (String(link.include_tour_id || "") !== String(tourId)) {
      return Response.json(
        { error: "This file is no longer available." },
        { status: 404 },
      );
    }

    const rows = await sql(
      `
      SELECT id, tour_type, tour_payload
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
