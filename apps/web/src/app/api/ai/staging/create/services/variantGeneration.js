import { heartbeat } from "../utils/database";
import { variantKey } from "../utils/helpers";
import { generateAndVerifyVacantVariant } from "./vacantGeneration";
import { generateFurnishedVariants } from "./furnishedGeneration";

export async function generateVariantsForPhoto({
  photo,
  idx,
  totalPhotos,
  isVacant,
  analysisByPhotoId,
  vacantCrossPhotoPlan,
  crossPhotoPlanByPhotoId,
  openAiKey,
  stagingName,
  vacantRules,
  preferredItemsRule,
  preferredItemsBlock,
  customAssetNotes,
  furnitureReferenceUrls,
  preferredItemsRecognition,
  stagingType,
  jobId,
  vacantQaResults,
}) {
  const photoId = photo?.id;
  const photoUrl = photo?.storage_path;
  if (!photoId || !photoUrl) return null;

  const base = 35;
  const span = 50;
  const step = span / Math.max(1, totalPhotos);

  const photoStartProgress = Math.min(85, Math.round(base + idx * step));
  const photoEndProgress = Math.min(85, Math.round(base + (idx + 1) * step));
  const p1 = photoStartProgress;
  const p2 = Math.max(p1, Math.round(p1 + (photoEndProgress - p1) * 0.25));
  const p3 = Math.max(p2, Math.round(p1 + (photoEndProgress - p1) * 0.5));
  const p4 = Math.max(p3, Math.round(p1 + (photoEndProgress - p1) * 0.75));

  await heartbeat({ jobId, progress: p1 });

  const analysis = analysisByPhotoId.get(photoId) || null;

  const crossPhotoPlan = isVacant
    ? vacantCrossPhotoPlan
    : crossPhotoPlanByPhotoId[photoId] || null;

  const variants = {};
  const flatImageUrls = [];

  if (isVacant) {
    // VACANT MODE (EXTREMELY STRICT):
    // - Only two outputs: Day and Night.
    // - HARD RESET: each output is generated directly from the ORIGINAL photo.
    // - Flash/torch does NOT change the image; no indoor/artificial lighting.

    const dayOff = { isNight: false, isLightOn: false };
    const nightOff = { isNight: true, isLightOn: false };

    const dayRes = await generateAndVerifyVacantVariant({
      openAiKey,
      photoId,
      photoUrl,
      analysis,
      crossPhotoPlan,
      stagingName,
      vacantRules,
      heartbeatAt: async () => heartbeat({ jobId, progress: p2 }),
      variantLabel: "day",
      vacantQaResults,
    });

    const dayVacantPath = dayRes.url;

    await heartbeat({ jobId, progress: p2 });

    const nightRes = await generateAndVerifyVacantVariant({
      openAiKey,
      photoId,
      photoUrl,
      analysis,
      crossPhotoPlan,
      stagingName,
      vacantRules,
      heartbeatAt: async () => heartbeat({ jobId, progress: p3 }),
      variantLabel: "night",
      vacantQaResults,
    });

    const nightVacantPath = nightRes.url;

    await heartbeat({ jobId, progress: p3 });

    // IMPORTANT: VACANT has ONLY 2 slots (Day/Night). No duplicated light-on/off variants.
    variants[variantKey(dayOff)] = { storage_path: dayVacantPath };
    variants[variantKey(nightOff)] = { storage_path: nightVacantPath };

    flatImageUrls.push(dayVacantPath, nightVacantPath);

    await heartbeat({ jobId, progress: Math.max(p3, photoEndProgress) });
  } else {
    // Furnished/default behavior (existing strategy):
    const result = await generateFurnishedVariants({
      openAiKey,
      photoId,
      photoUrl,
      analysis,
      stagingName,
      vacantRules,
      preferredItemsRule,
      preferredItemsBlock,
      customAssetNotes,
      furnitureReferenceUrls,
      preferredItemsRecognition,
      crossPhotoPlan,
      stagingType,
      jobId,
      p2,
      p3,
      p4,
    });

    Object.assign(variants, result.variants);
    flatImageUrls.push(...result.flatImageUrls);

    await heartbeat({ jobId, progress: Math.max(p4, photoEndProgress) });
  }

  return { photoId, photoUrl, variants, flatImageUrls };
}
