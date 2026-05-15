// Centralized config for share link behavior.
// Keep this as the single source of truth for default expiry used by:
// - Property "Share Property" modal
// - /links page
// - Backend /api/share-links

export const SHARE_LINK_DEFAULT_EXPIRY_DAYS = 14;
export const SHARE_LINK_MAX_EXPIRY_DAYS = 365;

export function normalizeExpiryDays(value) {
  const raw =
    value === undefined || value === null
      ? SHARE_LINK_DEFAULT_EXPIRY_DAYS
      : value;
  const n = Number(raw);
  if (!Number.isFinite(n) || n <= 0) {
    return SHARE_LINK_DEFAULT_EXPIRY_DAYS;
  }
  return Math.min(Math.floor(n), SHARE_LINK_MAX_EXPIRY_DAYS);
}
