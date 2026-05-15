import sql from "@/app/api/utils/sql";
import { auth } from "@/auth";
import {
  calculateStagingCreditCost,
  AI_STAGING_CREDIT_COST,
  AI_STAGING_MAX_PHOTOS_PER_JOB,
} from "@/app/api/utils/pricing";
import { getDbUserIdFromSession } from "@/app/api/utils/dbUser";
import {
  isLikelyAllowedPreferredItemImage,
  safeTrimString,
  normalizeStagingType,
} from "./utils/helpers";
import { processStagingJob } from "./services/stagingProcessor";

export { processStagingJob };

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
    const stagingType = normalizeStagingType(body?.stagingType);

    const propertyPhotoIds = Array.isArray(body?.propertyPhotoIds)
      ? body.propertyPhotoIds
      : [];

    const customAssetIds = Array.isArray(body?.customAssetIds)
      ? body.customAssetIds
      : [];

    // NEW: default true (if user sends nothing, we keep consistency on)
    const useCrossPhotoConsistency = body?.useCrossPhotoConsistency !== false;

    const preferredItemImages = Array.isArray(body?.preferredItemImages)
      ? body.preferredItemImages
      : [];

    const preferredItemHintsRaw = Array.isArray(body?.preferredItemHints)
      ? body.preferredItemHints
      : [];

    const preferredItemsText = safeTrimString(body?.preferredItemsText);

    const preferredItemImagesClean = preferredItemImages
      .map((x) => {
        const url = x && typeof x.url === "string" ? x.url.trim() : "";
        const mimeType =
          x && typeof x.mimeType === "string" ? x.mimeType.trim() : "";
        if (!url) return null;
        const candidate = { url, mimeType };
        if (!isLikelyAllowedPreferredItemImage(candidate)) return null;
        return candidate;
      })
      .filter(Boolean)
      .slice(0, 8);

    if (!propertyId) {
      return Response.json(
        { error: "propertyId is required" },
        { status: 400 },
      );
    }

    if (!stagingType) {
      return Response.json(
        { error: "stagingType is required" },
        { status: 400 },
      );
    }

    if (propertyPhotoIds.length === 0) {
      return Response.json(
        { error: "Please select at least 1 photo to stage." },
        { status: 400 },
      );
    }

    if (propertyPhotoIds.length > AI_STAGING_MAX_PHOTOS_PER_JOB) {
      return Response.json(
        {
          error: `Please select up to ${AI_STAGING_MAX_PHOTOS_PER_JOB} photos per staging batch.`,
        },
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

    const creditsReserved = calculateStagingCreditCost({
      hasPreferredItems: preferredItemImagesClean.length > 0,
      hasCustomAssets: customAssetIds.length > 0,
      photoCount: propertyPhotoIds.length,
    });

    const requestPayload = {
      stagingType,
      propertyPhotoIds,
      customAssetIds,
      useCrossPhotoConsistency,
      preferredItemImages: preferredItemImagesClean,
      preferredItemHints: preferredItemHintsRaw,
      preferredItemsText: preferredItemsText || null,
    };

    const requestPayloadJson = JSON.stringify(requestPayload);

    const spendMeta = {
      kind: "reserve",
      jobType: "staging",
      creditCost: creditsReserved,
      baseCost: AI_STAGING_CREDIT_COST,
      photoCount: propertyPhotoIds.length,
      hasCustomFurniture:
        preferredItemImagesClean.length > 0 || customAssetIds.length > 0,
    };

    const spendMetaJson = JSON.stringify(spendMeta);

    // IMPORTANT: keep SQL parameters aligned with placeholders.
    // The previous version passed an extra argument (stagingType) that wasn't used
    // in the SQL statement, which caused Postgres error:
    // "could not determine data type of parameter $3".
    const createdRows = await sql(
      `
        WITH existing_job AS (
          SELECT id, job_status
          FROM ai_jobs
          WHERE user_id = $1
            AND property_id = $2
            AND job_type = 'staging'
            AND job_status IN ('queued','running')
            AND request_payload = $4::jsonb
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
          SET balance_credits = balance_credits - $3
          WHERE user_id = $1
            AND balance_credits >= $3
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
          SELECT $1, $2, 'staging', 'queued', 0, $3, $4::jsonb, NULL, NOW()
          WHERE NOT EXISTS (SELECT 1 FROM existing_job)
            AND EXISTS (SELECT 1 FROM deducted)
          RETURNING id
        ),
        spend_tx AS (
          INSERT INTO credit_transactions (user_id, transaction_type, credits_delta, meta)
          SELECT
            $1,
            'spend',
            -$3,
            jsonb_set($5::jsonb, '{jobId}', to_jsonb((SELECT id FROM created_job)))
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
      [userId, propertyId, creditsReserved, requestPayloadJson, spendMetaJson],
    );

    const created = createdRows?.[0] || null;

    if (!created || !created.job_id) {
      throw new Error("Could not create job");
    }

    if (created.outcome === "insufficient") {
      return Response.json({ error: "Insufficient credits" }, { status: 402 });
    }

    const jobId = created.job_id;

    // Kick off background processing for newly created jobs (best-effort)
    if (created.outcome === "created") {
      const start = () => {
        processStagingJob({ jobId }).catch((e) => console.error(e));
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
    console.error("POST /api/ai/staging/create error:", error);
    return Response.json(
      { error: error?.message || "Internal Server Error" },
      { status: 500 },
    );
  }
}
