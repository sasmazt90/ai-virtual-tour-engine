// Central pricing constants (single source of truth)
// Keep this file free of Node-only code so it can be safely imported by web pages too.

export const AI_STAGING_CREDIT_COST = 20;
export const AI_STAGING_CUSTOM_CREDIT_COST = 30;

// How many photos a user can stage in a single job.
// Charging is per-photo, so larger batches can be done by running multiple jobs.
export const AI_STAGING_MAX_PHOTOS_PER_JOB = 12;

export function calculateStagingCreditCost({
  hasPreferredItems,
  hasCustomAssets,
  photoCount,
}) {
  const preferred = !!hasPreferredItems;
  const custom = !!hasCustomAssets;

  const perPhoto =
    preferred || custom
      ? AI_STAGING_CUSTOM_CREDIT_COST
      : AI_STAGING_CREDIT_COST;

  const qty = Math.max(1, Number(photoCount || 1) || 1);
  return perPhoto * qty;
}
