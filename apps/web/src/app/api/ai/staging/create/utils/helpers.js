export function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

export function safeJsonParse(text) {
  try {
    return JSON.parse(text);
  } catch {
    return null;
  }
}

export function safeTrimString(v) {
  return typeof v === "string" ? v.trim() : "";
}

export function normalizeStagingType(value) {
  if (typeof value !== "string") return "default";
  const v = value.trim().toLowerCase();
  if (!v) return "default";
  return v;
}

export function stagingTypeDisplayName(stagingType) {
  const st = normalizeStagingType(stagingType);
  if (st === "vacant") return "Vacant";
  if (st === "minimalist") return "Minimalist";
  if (st === "luxury") return "Luxury";
  if (st === "scandinavian") return "Scandinavian";
  if (st === "classic") return "Classic";
  if (st === "modern") return "Modern";
  if (st === "default") return "Default";
  return st;
}

export function isLikelyAllowedPreferredItemImage({ url, mimeType }) {
  const mt = typeof mimeType === "string" ? mimeType.toLowerCase().trim() : "";
  if (mt) {
    return (
      mt === "image/jpeg" ||
      mt === "image/jpg" ||
      mt === "image/png" ||
      mt === "image/webp"
    );
  }

  const u = typeof url === "string" ? url.toLowerCase() : "";
  return (
    u.endsWith(".jpg") ||
    u.endsWith(".jpeg") ||
    u.endsWith(".png") ||
    u.endsWith(".webp")
  );
}

// NEW: Uploadcare URLs often look like https://ucarecdn.com/<uuid>/-/format/auto/
// and won't match a file extension. For preferred-item images we normalize them to
// a concrete format (jpg/png) so downstream OpenAI calls are predictable.
export function getUploadcareBase(url) {
  if (typeof url !== "string") return null;
  const match = url.match(
    /^https?:\/\/ucarecdn\.com\/([0-9a-f-]{36})(?:\/|$)/i,
  );
  if (!match) return null;
  return `https://ucarecdn.com/${match[1]}/`;
}

export function normalizeUploadcareFormatUrl(url, format = "jpg") {
  const input = typeof url === "string" ? url.trim() : "";
  if (!input) return "";

  const base = getUploadcareBase(input);
  if (!base) return input;

  const fmt = String(format || "jpg").toLowerCase();
  const safeFmt =
    fmt === "png" || fmt === "webp" || fmt === "jpg" ? fmt : "jpg";

  // Preserve the original URL if it already requests a concrete supported format.
  const lower = input.toLowerCase();
  if (
    lower.includes("-/format/jpg") ||
    lower.includes("-/format/png") ||
    lower.includes("-/format/webp")
  ) {
    return input;
  }

  // If it contains format/auto (or no format at all), force a deterministic format.
  return `${base}-/format/${safeFmt}/`;
}

export function toHintObject(raw, index) {
  const label = safeTrimString(raw?.label);
  const type = safeTrimString(raw?.type) || "other";
  const notes = safeTrimString(raw?.notes);
  if (!label && !notes) return null;
  return { index, type, label: label || `Item ${index + 1}`, notes };
}

export function variantKey({ isNight, isLightOn }) {
  return `${isNight ? "night" : "day"}_light_${isLightOn ? "on" : "off"}`;
}

export const STAGING_VARIANTS = [
  { isNight: true, isLightOn: true },
  { isNight: true, isLightOn: false },
  { isNight: false, isLightOn: true },
  { isNight: false, isLightOn: false },
];
