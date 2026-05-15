import {
  isLikelyAllowedPreferredItemImage,
  safeTrimString,
  toHintObject,
  normalizeStagingType,
  stagingTypeDisplayName,
  normalizeUploadcareFormatUrl,
} from "../utils/helpers";

export function validateAndNormalizeRequest(requestPayload) {
  const propertyPhotoIds = Array.isArray(requestPayload.propertyPhotoIds)
    ? requestPayload.propertyPhotoIds
    : [];
  const customAssetIds = Array.isArray(requestPayload.customAssetIds)
    ? requestPayload.customAssetIds
    : [];

  // Determine staging mode EARLY so we never mix rules.
  const stagingType = normalizeStagingType(
    requestPayload.stagingType || "luxury",
  );
  const stagingName = stagingTypeDisplayName(stagingType);
  const isVacant = stagingType === "vacant";

  // NEW: keep a single source of truth for variant count (VACANT=2, others=4)
  const variantsPerPhoto = isVacant ? 2 : 4;

  const useCrossPhotoConsistency = isVacant
    ? true // VACANT: mandatory
    : requestPayload.useCrossPhotoConsistency !== false;

  const preferredItemImagesRaw = Array.isArray(
    requestPayload.preferredItemImages,
  )
    ? requestPayload.preferredItemImages
    : [];

  const preferredItemImagesClean = preferredItemImagesRaw
    .map((x) => {
      const rawUrl = x && typeof x.url === "string" ? x.url.trim() : "";
      const rawMimeType =
        x && typeof x.mimeType === "string" ? x.mimeType.trim() : "";
      if (!rawUrl) return null;

      // If the user pasted an Uploadcare transform URL like .../-/format/auto/
      // it may serve AVIF and get silently filtered out. Normalize to JPG.
      const url = normalizeUploadcareFormatUrl(rawUrl, "jpg");
      const mimeType = rawMimeType
        ? rawMimeType
        : url !== rawUrl
          ? "image/jpeg"
          : "";

      const candidate = { url, mimeType };
      if (!isLikelyAllowedPreferredItemImage(candidate)) return null;
      return candidate;
    })
    .filter(Boolean)
    .slice(0, 8);

  const preferredItemUrls = preferredItemImagesClean
    .map((x) => x.url)
    .filter((u) => typeof u === "string" && u.trim().length > 0);

  const preferredItemHintsRaw = Array.isArray(requestPayload.preferredItemHints)
    ? requestPayload.preferredItemHints
    : [];

  const preferredItemHints = preferredItemHintsRaw
    .map((h, idx) => toHintObject(h, idx))
    .filter(Boolean);

  const preferredItemsText = safeTrimString(requestPayload.preferredItemsText);

  return {
    propertyPhotoIds,
    customAssetIds,
    stagingType,
    stagingName,
    isVacant,
    variantsPerPhoto,
    useCrossPhotoConsistency,
    preferredItemImagesClean,
    preferredItemUrls,
    preferredItemHints,
    preferredItemsText,
  };
}
