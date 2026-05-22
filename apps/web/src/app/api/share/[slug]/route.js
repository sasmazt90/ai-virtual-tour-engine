import sql from "@/app/api/utils/sql";

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
  if (sourceType === "original")
    return { sourceType: "original", stagingType: null };

  const st = normalizeStagingType(raw?.stagingType);
  if (!st) return null;
  return { sourceType: "staging", stagingType: st };
}

function tourSlotKey(slot) {
  if (slot?.sourceType === "original") return "original";
  if (slot?.sourceType === "staging" && slot?.stagingType) {
    return `staging:${slot.stagingType}`;
  }
  return "";
}

export async function GET(request, { params }) {
  try {
    const slug = params.slug;

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
      return Response.json({ error: "Not found" }, { status: 404 });
    }

    const link = links[0];

    const customerRows = await sql(
      "SELECT id, full_name FROM clients WHERE id = $1 LIMIT 1",
      [link.client_id],
    );
    const customer = customerRows[0] || null;

    // Optional agent/company info for header/footer on the public page
    const agentRows = await sql(
      `
      SELECT
        p.id,
        p.full_name as agent_name,
        p.company as company_name,
        au.email as agent_email,
        COALESCE(p.company_logo_url, au.image) as company_logo_url
      FROM profiles p
      LEFT JOIN auth_users au ON au.id = p.id
      WHERE p.id = $1
      LIMIT 1
      `,
      [link.user_id],
    );
    const agent = agentRows[0] || null;

    if (link.expires_at) {
      const expires = new Date(link.expires_at);
      if (expires.getTime() < Date.now()) {
        return Response.json({ error: "Link expired" }, { status: 410 });
      }
    }

    // STEP 11C: lightweight access signal (only on share page load, not per asset).
    // Best-effort: do not fail the share response if logging fails.
    try {
      const userAgent = request.headers.get("user-agent") || "";
      const forwarded = request.headers.get("x-forwarded-for") || "";
      const ip = forwarded ? forwarded.split(",")[0].trim() : null;

      const entry = {
        timestamp: new Date().toISOString(),
        userAgent: String(userAgent).slice(0, 300),
        ip: ip ? String(ip).slice(0, 80) : null,
      };

      const meta = link.meta && typeof link.meta === "object" ? link.meta : {};
      const access = Array.isArray(meta.access) ? meta.access : [];

      // REQUIRED FIX: dedupe rapid refreshes
      // If last entry is from same userAgent and within 60 seconds, do not append.
      const last = access.length ? access[access.length - 1] : null;
      const lastUa = last?.userAgent ? String(last.userAgent) : "";
      const lastTs = last?.timestamp ? new Date(last.timestamp) : null;
      const lastTsMs =
        lastTs && !Number.isNaN(lastTs.getTime()) ? lastTs.getTime() : null;
      const nowMs = Date.now();

      const isDuplicate =
        lastUa &&
        lastUa === entry.userAgent &&
        typeof lastTsMs === "number" &&
        nowMs - lastTsMs >= 0 &&
        nowMs - lastTsMs < 60 * 1000;

      const nextAccess = isDuplicate
        ? access.slice(-50)
        : [...access, entry].slice(-50);

      // Only write if there is a change
      const shouldWrite = !isDuplicate;
      if (shouldWrite) {
        const nextMeta = { ...meta, access: nextAccess };
        await sql("UPDATE share_links SET meta = $1 WHERE id = $2", [
          nextMeta,
          link.id,
        ]);
      }
    } catch (e) {
      console.error("Share access log failed:", e);
    }

    // Availability summary (Step 16): helps the public share page show a neutral
    // "some items unavailable" message without breaking the page.
    const availability = {
      propertyMissing: false,
      missingStagingsCount: 0,
      missingContractsCount: 0,
      missingToursCount: 0,
    };

    const props = await sql(
      `
      SELECT p.*, c.full_name as owner_name, c.email as owner_email, c.phone as owner_phone
      FROM properties p
      LEFT JOIN clients c ON p.owner_client_id = c.id
      WHERE p.id = $1
      LIMIT 1
      `,
      [link.property_id],
    );

    if (props.length === 0) {
      availability.propertyMissing = true;
      return Response.json({
        shareLink: {
          slug: link.slug,
          created_at: link.created_at,
          expires_at: link.expires_at,
        },
        customer,
        agent,
        property: null,
        availability,
      });
    }

    const property = props[0];

    // PHOTOS (no raw storage_path exposed)
    const photos = await sql(
      "SELECT id, sort_order FROM property_photos WHERE property_id = $1 ORDER BY sort_order ASC",
      [link.property_id],
    );

    const photoDownloadUrls = photos.map(
      (p) => `/api/share/${encodeURIComponent(slug)}/photos/${p.id}/download`,
    );

    // Only include selected stagings
    const includeStagingIdsRaw = Array.isArray(link.include_staging_ids)
      ? link.include_staging_ids
      : [];

    const includeStagingIds = uniqStrings(includeStagingIdsRaw);

    const stagings = includeStagingIds.length
      ? await sql(
          `
          SELECT s.*,
            json_agg(json_build_object('id', si.id))
            FILTER (WHERE si.id IS NOT NULL) AS images
          FROM stagings s
          LEFT JOIN staging_images si ON s.id = si.staging_id
          WHERE s.id = ANY($1::uuid[])
          GROUP BY s.id
          ORDER BY s.created_at DESC
          `,
          [includeStagingIds],
        )
      : [];

    availability.missingStagingsCount = Math.max(
      0,
      includeStagingIds.length - stagings.length,
    );

    const safeStagings = stagings.map((s) => {
      const imgs = Array.isArray(s.images) ? s.images : [];
      const safeImages = imgs
        .filter((img) => img && img.id)
        .map((img) => ({
          id: img.id,
          download_url: `/api/share/${encodeURIComponent(slug)}/stagings/${s.id}/images/${img.id}/download`,
        }));

      const urlByImageId = new Map(
        safeImages.map((img) => [String(img.id), String(img.download_url)]),
      );

      // NEW: staged_items for variant toggles (day/night × lights on/off)
      // We never expose raw storage_path publicly.
      const stagedRaw = Array.isArray(s?.meta?.staged) ? s.meta.staged : [];
      const stagedItems = stagedRaw
        .map((it, idx) => {
          const variantsRaw =
            it?.variants && typeof it.variants === "object" ? it.variants : {};

          const variants = {};
          for (const [k, v] of Object.entries(variantsRaw)) {
            const imageId =
              v && typeof v === "object" && v.imageId
                ? String(v.imageId)
                : v && typeof v === "object" && v.image_id
                  ? String(v.image_id)
                  : typeof v === "string"
                    ? v
                    : null;

            if (!imageId) continue;
            const url = urlByImageId.get(imageId);
            if (!url) continue;
            variants[k] = url;
          }

          const keys = Object.keys(variants);
          if (keys.length === 0) return null;

          const photoId = it?.photoId ? String(it.photoId) : null;

          return {
            key: `${s.id}:${photoId || idx}`,
            photo_id: photoId,
            variants,
          };
        })
        .filter(Boolean);

      return {
        ...s,
        images: safeImages,
        staged_items: stagedItems,
      };
    });

    // Virtual Tours (selection can be by slots OR by ids for backward compatibility)
    const meta = link.meta && typeof link.meta === "object" ? link.meta : {};

    const includeTourSlotsRaw = safeArray(meta?.include_tour_slots);
    const includeTourSlots = Array.from(
      new Map(
        includeTourSlotsRaw
          .map((s) => normalizeTourSlot(s))
          .filter(Boolean)
          .map((s) => [tourSlotKey(s), s]),
      ).values(),
    );

    let tours = [];

    if (includeTourSlots.length > 0) {
      const tourRows = await sql(
        "SELECT * FROM virtual_tours WHERE property_id = $1 ORDER BY created_at DESC",
        [link.property_id],
      );

      const byKey = new Map();
      for (const t of tourRows) {
        const st =
          typeof t?.staging_type === "string"
            ? String(t.staging_type).trim()
            : "";
        const key =
          t?.source_type === "staging"
            ? st
              ? `staging:${st}`
              : ""
            : t?.source_type === "original"
              ? "original"
              : "";

        if (!key) continue;
        if (!byKey.has(key)) {
          byKey.set(key, t);
        }
      }

      // Preserve requested ordering
      tours = includeTourSlots
        .map((slot) => byKey.get(tourSlotKey(slot)))
        .filter(Boolean);

      availability.missingToursCount = Math.max(
        0,
        includeTourSlots.length - tours.length,
      );
    } else {
      const includeTourIdsFromMeta = uniqStrings(meta?.include_tour_ids);
      const includeTourIdsFromColumn = uniqStrings(
        link.include_virtual_tour_ids,
      );
      const includeTourIds = includeTourIdsFromMeta.length
        ? includeTourIdsFromMeta
        : includeTourIdsFromColumn.length
          ? includeTourIdsFromColumn
        : link.include_tour_id
          ? [String(link.include_tour_id)]
          : [];

      tours = includeTourIds.length
        ? await sql(
            "SELECT * FROM virtual_tours WHERE property_id = $1 AND id = ANY($2::uuid[]) ORDER BY created_at DESC",
            [link.property_id, includeTourIds],
          )
        : [];

      availability.missingToursCount = Math.max(
        0,
        includeTourIds.length - tours.length,
      );
    }

    const safeTours = tours.map((tour) => {
      const payload = tour.tour_payload || {};
      const scenes = Array.isArray(payload?.scenes) ? payload.scenes : [];

      const safeScenes = scenes.map((s) => {
        const sid = s?.sceneId;
        const safeImageUrl = sid
          ? `/api/share/${encodeURIComponent(slug)}/tours/${tour.id}/scenes/${encodeURIComponent(String(sid))}/download`
          : null;

        return {
          ...s,
          imageUrl: safeImageUrl,
        };
      });

      const safeFrames = safeScenes
        .map((s) => s?.imageUrl)
        .filter((u) => typeof u === "string" && u.length > 0);

      const nextPayload = {
        ...payload,
        scenes: safeScenes,
        frames: safeFrames.length > 1 ? safeFrames : payload?.frames,
        steps:
          typeof payload?.steps === "number" && payload.steps >= 8
            ? payload.steps
            : safeFrames.length || payload?.steps,
        initialIndex:
          typeof payload?.initialIndex === "number" ? payload.initialIndex : 0,
      };

      return {
        id: tour.id,
        property_id: tour.property_id,
        tour_type: tour.tour_type,
        created_at: tour.created_at,
        source_type: tour.source_type || null,
        staging_type: tour.staging_type || null,
        tour_payload: nextPayload,
      };
    });

    const includeContractIdsRaw = Array.isArray(link.include_contract_ids)
      ? link.include_contract_ids
      : [];

    const includeContractIds = uniqStrings(includeContractIdsRaw);

    const contracts = includeContractIds.length
      ? await sql(
          `
          SELECT co.*, c.full_name as client_name
          FROM contracts co
          LEFT JOIN clients c ON co.client_id = c.id
          WHERE co.id = ANY($1::uuid[])
          ORDER BY co.created_at DESC
          `,
          [includeContractIds],
        )
      : [];

    availability.missingContractsCount = Math.max(
      0,
      includeContractIds.length - contracts.length,
    );

    // NEW: never expose raw contract PDF storage paths publicly.
    // Instead, provide a safe download endpoint scoped to this share link.
    const safeContracts = contracts.map((c) => {
      const systemPdf = c?.filled_fields?._system?.pdf || null;
      const sysStorage = systemPdf?.storagePath || null;
      const hasPdf = !!(c.storage_path_pdf || c.pdf_url || sysStorage);
      const { storage_path_pdf, ...rest } = c;
      return {
        ...rest,
        has_pdf: hasPdf,
        pdf_download_url: hasPdf
          ? `/api/share/${encodeURIComponent(slug)}/contracts/${c.id}/download`
          : null,
      };
    });

    return Response.json({
      shareLink: {
        slug: link.slug,
        created_at: link.created_at,
        expires_at: link.expires_at,
      },
      customer,
      agent,
      availability,
      property: {
        ...property,
        photo_download_urls: photoDownloadUrls,
        stagings: safeStagings,
        tours: safeTours,
        contracts: safeContracts,
      },
    });
  } catch (error) {
    console.error("GET /api/share/[slug] error:", error);
    return Response.json({ error: "Internal Server Error" }, { status: 500 });
  }
}
