import sql from "@/app/api/utils/sql";
import { deleteSupabaseStorageObjects } from "@/app/api/utils/storageCleanup";

export async function heartbeat({ jobId, progress }) {
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

export async function refundCreditsIfNeeded({ userId, credits, jobId }) {
  const existingRefund = await sql(
    "SELECT id FROM credit_transactions WHERE user_id = $1 AND transaction_type = 'refund' AND meta->>'jobId' = $2 LIMIT 1",
    [userId, String(jobId)],
  );
  if (existingRefund.length > 0) return;

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

export async function getJobData(jobId) {
  const jobs = await sql(
    "SELECT id, user_id, property_id, credits_reserved, request_payload FROM ai_jobs WHERE id = $1 LIMIT 1",
    [jobId],
  );

  if (jobs.length === 0) return null;
  return jobs[0];
}

export async function updateJobStatus({
  jobId,
  status,
  progress,
  errorMessage,
}) {
  const updates = [];
  const values = [];
  let paramIndex = 1;

  if (status) {
    updates.push(`job_status = $${paramIndex++}`);
    values.push(status);
  }

  if (typeof progress === "number") {
    updates.push(`progress = $${paramIndex++}`);
    values.push(progress);
  }

  if (errorMessage !== undefined) {
    updates.push(`error_message = $${paramIndex++}`);
    values.push(errorMessage);
  }

  updates.push(`last_heartbeat_at = NOW()`);

  if (status === "running") {
    updates.push(`started_at = COALESCE(started_at, NOW())`);
  }

  values.push(jobId);

  await sql(
    `UPDATE ai_jobs SET ${updates.join(", ")} WHERE id = $${paramIndex}`,
    values,
  );
}

export async function getPropertyPhotos({ propertyPhotoIds, propertyId }) {
  if (propertyPhotoIds.length === 0) return [];

  return await sql(
    "SELECT id, storage_path FROM property_photos WHERE id = ANY($1::uuid[]) AND property_id = $2 ORDER BY array_position($1::uuid[], id)",
    [propertyPhotoIds, propertyId],
  );
}

export async function getCustomAssets({ customAssetIds, propertyId }) {
  if (customAssetIds.length === 0) return [];

  return await sql(
    "SELECT id, storage_path, label FROM custom_assets WHERE id = ANY($1::uuid[]) AND property_id = $2",
    [customAssetIds, propertyId],
  );
}

export async function findOrCreateStaging({ propertyId, stagingType }) {
  const existing = await sql(
    "SELECT id, version FROM stagings WHERE property_id = $1 AND staging_type = $2 LIMIT 1",
    [propertyId, stagingType],
  );

  if (existing.length > 0) {
    return {
      stagingId: existing[0].id,
      version: Number(existing[0].version || 1),
      isNew: false,
    };
  }

  return { stagingId: null, version: 0, isNew: true };
}

export async function updateStaging({ stagingId, meta, version }) {
  await sql(
    "UPDATE stagings SET prompt_version = $1, meta = $2::jsonb, version = $3 WHERE id = $4",
    ["v4", JSON.stringify(meta), version, stagingId],
  );
}

export async function createStaging({
  propertyId,
  stagingType,
  meta,
  version,
}) {
  const rows = await sql(
    "INSERT INTO stagings (property_id, staging_type, prompt_version, meta, version) VALUES ($1, $2, $3, $4::jsonb, $5) RETURNING id, version",
    [propertyId, stagingType, "v4", JSON.stringify(meta), version],
  );

  return {
    stagingId: rows[0].id,
    version: Number(rows[0].version || 1),
  };
}

export async function deleteStagingImages(stagingId) {
  const rows = await sql(
    "SELECT storage_path FROM staging_images WHERE staging_id = $1",
    [stagingId],
  );
  await sql("DELETE FROM staging_images WHERE staging_id = $1", [stagingId]);
  await deleteSupabaseStorageObjects(rows.map((row) => row.storage_path));
}

export async function insertStagingImage({ stagingId, storagePath }) {
  const rows = await sql(
    "INSERT INTO staging_images (staging_id, storage_path) VALUES ($1, $2) RETURNING id",
    [stagingId, storagePath],
  );
  return rows?.[0]?.id;
}

export async function updateJobResult({ jobId, resultPayload }) {
  await sql(
    "UPDATE ai_jobs SET job_status = 'succeeded', progress = 100, last_heartbeat_at = NOW(), result_payload = $1::jsonb WHERE id = $2",
    [JSON.stringify(resultPayload), jobId],
  );
}

export async function markJobFailed({ jobId, errorMessage, resultPayload }) {
  // Keep existing behavior, but allow attaching debug payload (e.g. VACANT QA details)
  // without flipping the job back to succeeded.
  if (resultPayload && typeof resultPayload === "object") {
    await sql(
      "UPDATE ai_jobs SET job_status = 'failed', last_heartbeat_at = NOW(), error_message = $1, result_payload = COALESCE(result_payload, '{}'::jsonb) || $3::jsonb WHERE id = $2",
      [errorMessage || "Job failed", jobId, JSON.stringify(resultPayload)],
    );
    return;
  }

  await sql(
    "UPDATE ai_jobs SET job_status = 'failed', last_heartbeat_at = NOW(), error_message = $1 WHERE id = $2",
    [errorMessage || "Job failed", jobId],
  );
}
