import { heartbeat } from "../utils/database";
import { analyzeRoomPhoto } from "./visionAnalyzer";

export async function analyzeAllPhotos({ openAiKey, photos, jobId }) {
  const perPhotoAnalyses = [];

  // Progress mapping:
  // - We come into this step around ~25%.
  // - Move progress steadily up to ~35% while we run vision analysis.
  const progressStart = 26;
  const progressEnd = 35;
  const total = Math.max(1, Array.isArray(photos) ? photos.length : 1);
  const span = progressEnd - progressStart;

  for (let idx = 0; idx < photos.length; idx++) {
    const photo = photos[idx];
    const photoId = photo?.id;
    const photoUrl = photo?.storage_path;
    if (!photoId || !photoUrl) continue;

    // Validate URL before sending to vision API
    const urlStr = String(photoUrl).trim();
    if (!urlStr.startsWith("http://") && !urlStr.startsWith("https://")) {
      console.error(
        `[analyzeAllPhotos] Skipping photo ${photoId}: storage_path is not a valid URL: ${urlStr.slice(0, 120)}`,
      );
      continue;
    }

    // IMPORTANT: For a single photo, the old logic would heartbeat at 25% and then
    // do a long OpenAI call, making the UI look "stuck". We bump progress before
    // and after each analysis so the UI keeps moving.
    const before = Math.min(
      progressEnd,
      Math.max(progressStart, Math.round(progressStart + (idx / total) * span)),
    );
    const after = Math.min(
      progressEnd,
      Math.max(before, Math.round(progressStart + ((idx + 1) / total) * span)),
    );

    await heartbeat({ jobId, progress: before });

    const analysis = await analyzeRoomPhoto({ openAiKey, photoUrl });
    perPhotoAnalyses.push({ photoId, photoUrl, analysis });

    await heartbeat({ jobId, progress: after });
  }

  const analysisByPhotoId = new Map(
    perPhotoAnalyses.map((x) => [x.photoId, x.analysis]),
  );

  return { perPhotoAnalyses, analysisByPhotoId };
}
