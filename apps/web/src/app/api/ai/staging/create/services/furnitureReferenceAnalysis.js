import { heartbeat } from "../utils/database";
import { analyzeAllPreferredItems } from "./preferredItemsAnalyzer";
import { normalizeUploadcareFormatUrl } from "../utils/helpers";

export async function analyzeFurnitureReferences({
  openAiKey,
  isVacant,
  preferredItemUrls,
  customAssetUrls,
  jobId,
}) {
  // Treat selected custom assets as STRICT furniture references (same idea as preferred items)
  // BUT: VACANT mode forbids adding any objects, so we ignore all furniture refs in VACANT.
  const furnitureReferenceUrls = !isVacant
    ? [...preferredItemUrls, ...customAssetUrls]
        .filter((u) => typeof u === "string" && u.trim().length > 0)
        // Normalize Uploadcare format/auto URLs to a deterministic JPG for OpenAI vision.
        .map((u) => normalizeUploadcareFormatUrl(u, "jpg"))
        .filter((u) => typeof u === "string" && u.trim().length > 0)
        .slice(0, 8)
    : [];

  // Analyze furniture references (for prompt + QA verification)
  const preferredItemsRecognition = !isVacant
    ? await analyzeAllPreferredItems({
        openAiKey,
        preferredItemUrls: furnitureReferenceUrls,
        onHeartbeat: async (i) => {
          const total = Math.max(1, furnitureReferenceUrls.length);
          const t = ((i + 1) / total) * 1;
          const progress = Math.min(24, Math.max(10, Math.round(10 + t * 14)));
          await heartbeat({ jobId, progress });
        },
      })
    : null;

  return { furnitureReferenceUrls, preferredItemsRecognition };
}
