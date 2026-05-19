// Central pricing constants (single source of truth)
// Keep this file free of Node-only code so it can be safely imported by web pages too.

export const AI_STAGING_CREDIT_COST = 35;
export const AI_STAGING_CUSTOM_CREDIT_COST = 50;
export const AI_STAGING_FURNITURE_REFERENCE_CREDIT_COST = 15;

// How many photos a user can stage in a single job.
// Charging is per-photo, so larger batches can be done by running multiple jobs.
export const AI_STAGING_MAX_PHOTOS_PER_JOB = 12;

export const AI_VIDEO_3D_MAX_BYTES = 750 * 1024 * 1024;
export const AI_VIDEO_3D_MAX_FILES = 8;

export const AI_VIDEO_3D_CREDIT_TIERS = [
  { maxBytes: 250 * 1024 * 1024, credits: 300, label: "Up to 250 MB" },
  { maxBytes: 500 * 1024 * 1024, credits: 500, label: "250-500 MB" },
  { maxBytes: AI_VIDEO_3D_MAX_BYTES, credits: 750, label: "500-750 MB" },
];

export const CREDIT_ADMIN_EMAILS = [
  "sasmazt90@gmail.com",
  "tolgar@sasmaz.digital",
];

export function isCreditAdminEmail(email) {
  const normalized = String(email || "").trim().toLowerCase();
  return CREDIT_ADMIN_EMAILS.includes(normalized);
}

export function calculateStagingCreditCost({
  hasPreferredItems,
  hasCustomAssets,
  preferredItemCount,
  customAssetCount,
  photoCount,
}) {
  const qty = Math.max(1, Number(photoCount || 1) || 1);
  const preferredRefs = Math.max(0, Number(preferredItemCount || 0) || 0);
  const customRefs = Math.max(0, Number(customAssetCount || 0) || 0);
  const fallbackRefs = hasPreferredItems || hasCustomAssets ? 1 : 0;
  const referenceCount = Math.max(preferredRefs + customRefs, fallbackRefs);

  return (
    AI_STAGING_CREDIT_COST * qty +
    AI_STAGING_FURNITURE_REFERENCE_CREDIT_COST * referenceCount
  );
}

export function calculateVideo3DTourCreditCost(fileSizeBytes) {
  const size = Math.max(0, Number(fileSizeBytes || 0) || 0);
  const tier =
    AI_VIDEO_3D_CREDIT_TIERS.find((candidate) => size <= candidate.maxBytes) ||
    AI_VIDEO_3D_CREDIT_TIERS[AI_VIDEO_3D_CREDIT_TIERS.length - 1];

  return tier.credits;
}

export function getVideo3DTourCreditTier(fileSizeBytes) {
  const size = Math.max(0, Number(fileSizeBytes || 0) || 0);
  return (
    AI_VIDEO_3D_CREDIT_TIERS.find((candidate) => size <= candidate.maxBytes) ||
    AI_VIDEO_3D_CREDIT_TIERS[AI_VIDEO_3D_CREDIT_TIERS.length - 1]
  );
}
