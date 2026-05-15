import {
  getJobData,
  updateJobStatus,
  getPropertyPhotos,
  getCustomAssets,
  heartbeat,
} from "../utils/database";
import { validateAndNormalizeRequest } from "./requestValidation";
import { analyzeFurnitureReferences } from "./furnitureReferenceAnalysis";
import { buildPromptComponents } from "./promptBuilder";
import { analyzeAllPhotos } from "./photoAnalysis";
import {
  buildVacantCrossPhotoPlan,
  buildFurnishedCrossPhotoPlans,
} from "./crossPhotoPlanning";
import { generateVariantsForPhoto } from "./variantGeneration";
import { buildStagingMetadata } from "./metadataBuilder";
import { persistStagingResults } from "./stagingPersistence";
import { handleJobError } from "./errorHandling";

export async function processStagingJob({ jobId }) {
  // NEW: allow us to attach VACANT QA debug info to the job even if we fail mid-run.
  const debugPayload = {
    stagingType: null,
    vacantQaResults: null,
  };

  try {
    const job = await getJobData(jobId);
    if (!job) return;

    const userId = job.user_id;
    const propertyId = job.property_id;
    const creditsReserved = job.credits_reserved;
    const requestPayload = job.request_payload || {};

    await updateJobStatus({
      jobId,
      status: "running",
      progress: 5,
      errorMessage: null,
    });

    const {
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
    } = validateAndNormalizeRequest(requestPayload);

    debugPayload.stagingType = stagingType;

    const photos = await getPropertyPhotos({ propertyPhotoIds, propertyId });
    const assets = await getCustomAssets({ customAssetIds, propertyId });

    const photoUrls = photos
      .map((p) => p.storage_path)
      .filter((u) => typeof u === "string" && u.trim().length > 0);

    if (photoUrls.length === 0) {
      throw new Error(
        "No property photos found. Add photos before generating staging.",
      );
    }

    // Validate all photo URLs are full public URLs (not relative paths)
    for (const p of photos) {
      const url = String(p?.storage_path || "").trim();
      if (url && !url.startsWith("http://") && !url.startsWith("https://")) {
        throw new Error(
          `Property photo ${p.id} has an invalid storage_path (not a full URL): ${url.slice(0, 100)}. ` +
            "Please re-upload the photo.",
        );
      }
    }

    const openAiKey = process.env.OPEN_AI_API_KEY;
    if (!openAiKey) {
      throw new Error(
        "OPEN_AI_API_KEY is not set. Add it in Secrets to enable staging.",
      );
    }

    await heartbeat({ jobId, progress: 10 });

    const customAssetUrls = !isVacant
      ? assets.map((a) => a.storage_path).filter(Boolean)
      : [];

    const { furnitureReferenceUrls, preferredItemsRecognition } =
      await analyzeFurnitureReferences({
        openAiKey,
        isVacant,
        preferredItemUrls,
        customAssetUrls,
        jobId,
      });

    await heartbeat({ jobId, progress: 25 });

    const {
      vacantRules,
      preferredItemsRule,
      preferredItemsBlock,
      customAssetNotes,
    } = buildPromptComponents({
      isVacant,
      furnitureReferenceUrls,
      preferredItemsRecognition,
      assets,
      preferredItemHints,
      preferredItemsText,
    });

    // --- pre-analyze all photos once ---
    const { perPhotoAnalyses, analysisByPhotoId } = await analyzeAllPhotos({
      openAiKey,
      photos,
      jobId,
    });

    // VACANT: build one shared plan across all selected photos to keep removals consistent.
    const vacantCrossPhotoPlan = isVacant
      ? buildVacantCrossPhotoPlan(perPhotoAnalyses, useCrossPhotoConsistency)
      : null;

    // Furnished modes: group photos by room and build a furniture-set plan per room group.
    const { roomGroups, crossPhotoPlanByPhotoId } = !isVacant
      ? await buildFurnishedCrossPhotoPlans({
          openAiKey,
          perPhotoAnalyses,
          useCrossPhotoConsistency,
          stagingName,
          preferredItemsRecognition,
        })
      : { roomGroups: [], crossPhotoPlanByPhotoId: {} };

    // --- generation loop ---
    const flatImageUrls = [];
    const stagedItems = [];

    // NEW: collect per-image VACANT QA results so failures are debuggable
    const vacantQaResults = [];
    debugPayload.vacantQaResults = vacantQaResults;

    for (let idx = 0; idx < photos.length; idx++) {
      const photo = photos[idx];

      const result = await generateVariantsForPhoto({
        photo,
        idx,
        totalPhotos: photos.length,
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
      });

      if (result) {
        stagedItems.push({
          photoId: result.photoId,
          photoUrl: result.photoUrl,
          variants: result.variants,
        });
        flatImageUrls.push(...result.flatImageUrls);
      }
    }

    await heartbeat({ jobId, progress: 90 });

    const meta = buildStagingMetadata({
      stagedItems,
      perPhotoAnalyses,
      isVacant,
      preferredItemsRecognition,
      preferredItemImagesClean,
      preferredItemHints,
      preferredItemsText,
      customAssetIds,
      propertyPhotoIds,
      variantsPerPhoto,
      flatImageUrls,
      creditsReserved,
      stagingType,
      vacantQaResults,
    });

    await persistStagingResults({
      propertyId,
      stagingType,
      stagedItems,
      meta,
      jobId,
    });
  } catch (err) {
    await handleJobError({ jobId, error: err, debugPayload });
  }
}
