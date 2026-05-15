import sql from "@/app/api/utils/sql";
import { auth } from "@/auth";
import {
  calculateVirtualTourCreditCost,
  AI_FAKE360_CREDIT_COST,
} from "@/app/api/utils/pricing";
import { getDbUserIdFromSession } from "@/app/api/utils/dbUser";

function makeFake360Payload(imageUrls) {
  // A Fake-360 tour is a *single camera position* 360 "spin".
  // We store `frames` as the primary representation for smooth rotation.
  // We also keep `scenes` as a backward-compatible list so public share links
  // can proxy images via the existing per-scene download endpoint.
  const frames = imageUrls.filter(Boolean);
  const scenes = frames.map((url, idx) => {
    return {
      sceneId: `F${idx + 1}`,
      imageUrl: url,
      initialYaw: 0,
      hotspots: [],
    };
  });

  return {
    type: "fake360",
    frames,
    steps: frames.length,
    initialIndex: 0,
    scenes,
  };
}

function directionToHotspotXY(direction) {
  // consistent, “floor arrow” placements
  if (direction === "left") return { x: 0.2, y: 0.66 };
  if (direction === "right") return { x: 0.8, y: 0.66 };
  if (direction === "back") return { x: 0.5, y: 0.78 };
  // forward default
  return { x: 0.5, y: 0.66 };
}

function normalizeDirection(dir) {
  const d = String(dir || "")
    .toLowerCase()
    .trim();
  if (d === "left" || d === "right" || d === "forward" || d === "back") {
    return d;
  }
  return "forward";
}

function makeNode360Payload({ clusters, edges }) {
  const safeClusters = Array.isArray(clusters) ? clusters : [];
  if (safeClusters.length === 0) {
    return null;
  }

  // Build points (one point per “room/area” cluster)
  const points = safeClusters.map((c, idx) => {
    const pointId = String(c.clusterId || c.id || `P${idx + 1}`);
    const label = String(c.label || c.name || `Area ${idx + 1}`);
    const framesIn = Array.isArray(c.frames) ? c.frames : [];

    // frames can be urls OR {url,yawDeg}
    const frames = framesIn
      .map((f) => {
        if (typeof f === "string") {
          return { url: f, angleDeg: 0 };
        }
        if (f && typeof f === "object") {
          const url = f.url || f.imageUrl || f.src;
          const yawDeg = Number(f.yawDeg ?? f.angleDeg ?? f.angle ?? 0) || 0;
          if (!url) return null;
          return { url, angleDeg: yawDeg };
        }
        return null;
      })
      .filter(Boolean);

    return {
      pointId,
      label,
      clusterId: pointId,
      steps: 36,
      frames,
      initialIndex: 0,
      hotspots: [],
    };
  });

  const byId = new Map(points.map((p) => [p.pointId, p]));

  const safeEdges = Array.isArray(edges) ? edges : [];

  // If no edges, connect linearly to at least allow navigation.
  const effectiveEdges =
    safeEdges.length > 0
      ? safeEdges
      : points.slice(0, -1).map((p, i) => {
          return {
            from: p.pointId,
            to: points[i + 1].pointId,
            direction: "forward",
          };
        });

  for (const e of effectiveEdges) {
    const from = String(e.from || e.fromId || e.source || "");
    const to = String(e.to || e.toId || e.target || "");
    if (!from || !to) continue;
    const fromPoint = byId.get(from);
    if (!fromPoint) continue;

    const direction = normalizeDirection(e.direction);
    const pos = directionToHotspotXY(direction);

    fromPoint.hotspots.push({
      ...pos,
      toPointId: to,
      direction,
      label: String(e.label || "Move"),
    });
  }

  return {
    type: "node360",
    initialPointId: points[0]?.pointId || null,
    points,
  };
}

function toAbsolutePublicUrl(inputUrl) {
  const url = typeof inputUrl === "string" ? inputUrl.trim() : "";
  if (!url) return "";
  if (url.startsWith("http://") || url.startsWith("https://")) return url;
  if (url.startsWith("/")) {
    const base = String(process.env.APP_URL || "").replace(/\/$/, "");
    if (!base) return "";
    return `${base}${url}`;
  }
  return "";
}

// NEW: normalize URLs so OpenAI-returned absolute URLs still match DB-stored relative paths.
function buildAllowedImageUrlSet(imageUrls) {
  const set = new Set();
  const list = Array.isArray(imageUrls) ? imageUrls : [];

  for (const raw of list) {
    const s = typeof raw === "string" ? raw.trim() : "";
    if (!s) continue;
    set.add(s);

    const abs = toAbsolutePublicUrl(s);
    if (abs) set.add(abs);
  }

  return set;
}

async function buildNode360PayloadWithOpenAI(imageUrls) {
  if (!process.env.OPEN_AI_API_KEY) {
    return null;
  }

  const urls = (imageUrls || [])
    .filter((u) => typeof u === "string" && u.trim().length > 0)
    .map((u) => toAbsolutePublicUrl(u))
    .filter(Boolean)
    .slice(0, 18);

  if (urls.length < 2) return null;

  // NEW: allow both the original stored URLs and the absolute URLs we actually sent to OpenAI.
  const allowed = buildAllowedImageUrlSet(imageUrls);
  for (const u of urls) {
    allowed.add(u);
  }

  const controller = new AbortController();
  const t = setTimeout(() => controller.abort(), 45_000);

  const prompt = `You are building a premium real-estate 360 virtual tour from user-uploaded interior photos.

GOAL:
- Group photos into room/area clusters (open plan should stay one cluster).
- Each cluster becomes ONE node (point).
- Estimate an approximate yaw angle (0..360) for each photo within its cluster so the viewer can rotate smoothly.
- Build realistic adjacency edges between clusters. Directions must be one of: forward, back, left, right.

STRICT RULES:
- Do NOT invent new rooms.
- Do NOT fabricate furniture/objects.
- You are only outputting STRUCTURE (JSON). No image generation.

OUTPUT JSON ONLY with this shape:
{
  "clusters": [
    {
      "clusterId": "P1",
      "label": "Living room",
      "frames": [
        {"url": "...", "yawDeg": 0},
        {"url": "...", "yawDeg": 45}
      ]
    }
  ],
  "edges": [
    {"from": "P1", "to": "P2", "direction": "forward"}
  ]
}

Make sure every frame url is exactly one of the provided image URLs.`;

  try {
    const res = await fetch("https://api.openai.com/v1/chat/completions", {
      method: "POST",
      signal: controller.signal,
      headers: {
        Authorization: `Bearer ${process.env.OPEN_AI_API_KEY}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: "gpt-4o-mini",
        temperature: 0.2,
        response_format: { type: "json_object" },
        messages: [
          {
            role: "system",
            content:
              "You are a computer vision assistant for real-estate interior photos. Output JSON only.",
          },
          {
            role: "user",
            content: [
              { type: "text", text: prompt },
              ...urls.map((u) => ({
                type: "image_url",
                image_url: { url: u },
              })),
            ],
          },
        ],
      }),
    });

    if (!res.ok) {
      const txt = await res.text().catch(() => "");
      throw new Error(
        `OpenAI vision request failed: ${res.status} ${res.statusText} ${txt}`,
      );
    }

    const data = await res.json();
    const content = data?.choices?.[0]?.message?.content;
    if (!content) return null;

    const parsed = JSON.parse(content);
    const payload = makeNode360Payload({
      clusters: parsed?.clusters,
      edges: parsed?.edges,
    });

    const ptsIn = Array.isArray(payload?.points) ? payload.points : [];
    if (ptsIn.length === 0) return null;

    // NEW: strict post-filter validation.
    // We had a bug where we validated BEFORE filtering, which could leave 0 frames
    // and cause “No tour image” in the viewer.
    const cleanedPoints = ptsIn
      .map((p) => {
        const framesIn = Array.isArray(p?.frames) ? p.frames : [];
        const cleanedFrames = framesIn
          .map((f) => {
            if (!f) return null;

            // expected shape: { url, angleDeg }
            if (typeof f === "string") {
              const s = f.trim();
              if (!s) return null;
              const abs = toAbsolutePublicUrl(s) || s;
              if (!allowed.has(s) && !allowed.has(abs)) return null;
              return { url: abs, angleDeg: 0 };
            }

            if (typeof f === "object") {
              const rawUrl = String(f.url || f.imageUrl || f.src || "").trim();
              if (!rawUrl) return null;
              const abs = toAbsolutePublicUrl(rawUrl) || rawUrl;
              if (!allowed.has(rawUrl) && !allowed.has(abs)) return null;

              const angleDeg =
                Number(f.angleDeg ?? f.yawDeg ?? f.angle ?? 0) || 0;

              return { url: abs, angleDeg };
            }

            return null;
          })
          .filter(Boolean);

        return {
          ...p,
          frames: cleanedFrames,
        };
      })
      .filter((p) => Array.isArray(p?.frames) && p.frames.length > 0);

    if (cleanedPoints.length === 0) {
      return null;
    }

    // Ensure initialPointId points to a point with frames.
    const wantedInitial = String(payload?.initialPointId || "");
    const initialOk = cleanedPoints.some(
      (p) => String(p.pointId) === wantedInitial,
    );

    return {
      ...payload,
      points: cleanedPoints,
      initialPointId: initialOk
        ? payload.initialPointId
        : cleanedPoints[0].pointId,
    };
  } catch (e) {
    console.error("buildNode360PayloadWithOpenAI error", e);
    return null;
  } finally {
    clearTimeout(t);
  }
}

async function refundCreditsIfNeeded({ userId, credits, jobId }) {
  const existingRefund = await sql(
    "SELECT id FROM credit_transactions WHERE user_id = $1 AND transaction_type = 'refund' AND meta->>'jobId' = $2 LIMIT 1",
    [userId, String(jobId)],
  );
  if (existingRefund.length > 0) return;

  // IMPORTANT: Anything sql.transaction() does NOT accept an async callback.
  // Use a single atomic query instead.
  await sql(
    `
      WITH ensured_wallet AS (
        INSERT INTO credits_wallet (user_id, balance_credits)
        VALUES ($1, 0)
        ON CONFLICT (user_id) DO NOTHING
      ),
      updated_wallet AS (
        UPDATE credits_wallet
        SET balance_credits = balance_credits + $2
        WHERE user_id = $1
        RETURNING balance_credits
      ),
      inserted_tx AS (
        INSERT INTO credit_transactions (user_id, transaction_type, credits_delta, meta)
        VALUES ($1, 'refund', $2, $3::jsonb)
        RETURNING id
      )
      SELECT 1 as ok;
    `,
    [userId, credits, JSON.stringify({ jobId })],
  );
}

async function heartbeat({ jobId, progress }) {
  if (!jobId) return;
  if (typeof progress === "number") {
    await sql(
      "UPDATE ai_jobs SET progress = $1, last_heartbeat_at = NOW() WHERE id = $2",
      [progress, jobId],
    );
    return;
  }
  await sql("UPDATE ai_jobs SET last_heartbeat_at = NOW() WHERE id = $1", [
    jobId,
  ]);
}

export async function processVirtualTourJob({ jobId }) {
  try {
    const jobs = await sql(
      "SELECT id, user_id, property_id, credits_reserved, request_payload FROM ai_jobs WHERE id = $1 LIMIT 1",
      [jobId],
    );

    if (jobs.length === 0) return;

    const job = jobs[0];
    const propertyId = job.property_id;
    const requestPayload = job.request_payload || {};

    await sql(
      "UPDATE ai_jobs SET job_status = 'running', started_at = COALESCE(started_at, NOW()), last_heartbeat_at = NOW(), progress = 10, error_message = NULL WHERE id = $1",
      [jobId],
    );

    const baseView = requestPayload.baseView || {
      type: "default",
      stagingId: null,
    };

    let imageUrls = [];
    let sourceType = "original";
    let stagingType = null;
    let baseStagingId = null;

    if (baseView.type === "staging" && baseView.stagingId) {
      baseStagingId = baseView.stagingId;
      sourceType = "staging";

      const stagingRows = await sql(
        "SELECT id, staging_type FROM stagings WHERE id = $1 AND property_id = $2 LIMIT 1",
        [baseStagingId, propertyId],
      );

      if (stagingRows.length === 0) {
        throw new Error(
          "That staging is not available for this property. Please refresh and try again.",
        );
      }

      stagingType = stagingRows[0].staging_type;

      const images = await sql(
        "SELECT storage_path FROM staging_images WHERE staging_id = $1 ORDER BY created_at ASC",
        [baseStagingId],
      );
      imageUrls = images.map((r) => r.storage_path).filter(Boolean);
    } else {
      const photos = await sql(
        "SELECT storage_path FROM property_photos WHERE property_id = $1 ORDER BY sort_order ASC",
        [propertyId],
      );
      imageUrls = photos.map((r) => r.storage_path).filter(Boolean);
    }

    imageUrls = imageUrls.slice(0, 36);

    if (imageUrls.length < 2) {
      throw new Error(
        "Not enough images to create a 360 tour. Add at least 2 photos (or staging images).",
      );
    }

    await heartbeat({ jobId, progress: 50 });

    const nodePayload = await buildNode360PayloadWithOpenAI(imageUrls);
    const tourPayload = nodePayload || makeFake360Payload(imageUrls);

    const tourType = nodePayload ? "panorama" : "fake360";

    // Slot logic: only ONE tour per source (original) and per staging type.
    // Overwrite existing in-place rather than inserting duplicates.
    const existing = await sql(
      sourceType === "original"
        ? "SELECT id FROM virtual_tours WHERE property_id = $1 AND source_type = 'original' LIMIT 1"
        : "SELECT id FROM virtual_tours WHERE property_id = $1 AND source_type = 'staging' AND staging_type = $2 LIMIT 1",
      sourceType === "original" ? [propertyId] : [propertyId, stagingType],
    );

    let tourId = null;

    if (existing.length > 0) {
      const updated = await sql(
        "UPDATE virtual_tours SET base_staging_id = $1, source_type = $2, staging_type = $3, tour_type = $4, tour_payload = $5::jsonb WHERE id = $6 RETURNING id",
        [
          baseStagingId,
          sourceType,
          stagingType,
          tourType,
          JSON.stringify(tourPayload),
          existing[0].id,
        ],
      );
      tourId = updated?.[0]?.id;
    } else {
      const tourRows = await sql(
        "INSERT INTO virtual_tours (property_id, base_staging_id, source_type, staging_type, tour_type, tour_payload) VALUES ($1, $2, $3, $4, $5, $6::jsonb) RETURNING id",
        [
          propertyId,
          baseStagingId,
          sourceType,
          stagingType,
          tourType,
          JSON.stringify(tourPayload),
        ],
      );
      tourId = tourRows[0].id;
    }

    await sql(
      "UPDATE ai_jobs SET job_status = 'succeeded', progress = 100, last_heartbeat_at = NOW(), result_payload = $1::jsonb WHERE id = $2",
      [JSON.stringify({ tourId }), jobId],
    );
  } catch (err) {
    console.error("processVirtualTourJob error", err);

    try {
      const rows = await sql(
        "SELECT user_id, credits_reserved FROM ai_jobs WHERE id = $1 LIMIT 1",
        [jobId],
      );
      if (rows.length > 0) {
        await refundCreditsIfNeeded({
          userId: rows[0].user_id,
          credits: Number(rows[0].credits_reserved || 0),
          jobId,
        });
      }
    } catch (refundErr) {
      console.error("refundCredits error", refundErr);
    }

    await sql(
      "UPDATE ai_jobs SET job_status = 'failed', last_heartbeat_at = NOW(), error_message = $1 WHERE id = $2",
      [err?.message || "Job failed", jobId],
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

    const propertyId = body?.propertyId;
    const baseView = body?.baseView || { type: "default", stagingId: null };

    if (!propertyId) {
      return Response.json(
        { error: "propertyId is required" },
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

    // Determine whether this tour is based on a custom-furniture staging (preferred items or custom assets).
    let sourceType = "original";
    let stagingHasCustomFurniture = false;

    if (baseView?.type === "staging" && baseView?.stagingId) {
      sourceType = "staging";

      const stRows = await sql(
        "SELECT meta FROM stagings WHERE id = $1 AND property_id = $2 LIMIT 1",
        [baseView.stagingId, propertyId],
      );

      const meta = stRows?.[0]?.meta || {};
      const preferredImgs = Array.isArray(meta?.preferredItemImages)
        ? meta.preferredItemImages
        : [];
      const customAssetIdsUsed = Array.isArray(meta?.customAssetIdsUsed)
        ? meta.customAssetIdsUsed
        : [];

      stagingHasCustomFurniture =
        preferredImgs.length > 0 || customAssetIdsUsed.length > 0;
    }

    const creditsReserved = calculateVirtualTourCreditCost({
      sourceType,
      stagingHasCustomFurniture,
    });

    const requestPayload = {
      baseView,
      tourType: "node360",
      creditCost: creditsReserved,
      baseCost: AI_FAKE360_CREDIT_COST,
      sourceType,
      stagingHasCustomFurniture,
    };

    const spendMeta = {
      kind: "reserve",
      jobType: "virtual_tour",
      creditCost: creditsReserved,
      baseCost: AI_FAKE360_CREDIT_COST,
      sourceType,
      stagingHasCustomFurniture,
    };

    const createdRows = await sql(
      `
        WITH existing_job AS (
          SELECT id, job_status
          FROM ai_jobs
          WHERE user_id = $1
            AND property_id = $2
            AND job_type = 'virtual_tour'
            AND job_status IN ('queued','running')
            AND request_payload->>'tourType' = 'node360'
            AND request_payload->'baseView' = $3::jsonb
          ORDER BY created_at DESC
          LIMIT 1
        ),
        ensured_wallet AS (
          INSERT INTO credits_wallet (user_id, balance_credits)
          VALUES ($1, 0)
          ON CONFLICT (user_id) DO NOTHING
        ),
        deducted AS (
          UPDATE credits_wallet
          SET balance_credits = balance_credits - $4
          WHERE user_id = $1
            AND balance_credits >= $4
            AND NOT EXISTS (SELECT 1 FROM existing_job)
          RETURNING balance_credits
        ),
        created_job AS (
          INSERT INTO ai_jobs (
            user_id,
            property_id,
            job_type,
            job_status,
            progress,
            credits_reserved,
            request_payload,
            started_at,
            last_heartbeat_at
          )
          SELECT $1, $2, 'virtual_tour', 'queued', 0, $4, $5::jsonb, NULL, NOW()
          WHERE NOT EXISTS (SELECT 1 FROM existing_job)
            AND EXISTS (SELECT 1 FROM deducted)
          RETURNING id
        ),
        spend_tx AS (
          INSERT INTO credit_transactions (user_id, transaction_type, credits_delta, meta)
          SELECT
            $1,
            'spend',
            -$4,
            jsonb_set($6::jsonb, '{jobId}', to_jsonb((SELECT id FROM created_job)))
          WHERE EXISTS (SELECT 1 FROM created_job)
          RETURNING id
        )
        SELECT
          COALESCE((SELECT id FROM existing_job), (SELECT id FROM created_job)) AS job_id,
          COALESCE((SELECT job_status FROM existing_job), 'queued') AS job_status,
          CASE
            WHEN EXISTS (SELECT 1 FROM existing_job) THEN 'deduped'
            WHEN EXISTS (SELECT 1 FROM created_job) THEN 'created'
            ELSE 'insufficient'
          END AS outcome;
      `,
      [
        userId,
        propertyId,
        JSON.stringify(baseView),
        creditsReserved,
        JSON.stringify(requestPayload),
        JSON.stringify(spendMeta),
      ],
    );

    const created = createdRows?.[0] || null;

    if (!created || !created.job_id) {
      throw new Error("Could not create job");
    }

    if (created.outcome === "insufficient") {
      return Response.json({ error: "Insufficient credits" }, { status: 402 });
    }

    const jobId = created.job_id;

    if (created.outcome === "created") {
      const start = () => {
        processVirtualTourJob({ jobId }).catch((e) => console.error(e));
      };

      if (typeof queueMicrotask === "function") {
        queueMicrotask(start);
      } else {
        setTimeout(start, 0);
      }
    }

    return Response.json({
      jobId,
      status: created.outcome === "deduped" ? "running" : "queued",
      creditsReserved,
      creditCost: creditsReserved,
      deduped: created.outcome === "deduped",
    });
  } catch (error) {
    console.error("POST /api/ai/virtual-tour/create error:", error);
    return Response.json(
      { error: error?.message || "Internal Server Error" },
      { status: 500 },
    );
  }
}
