import { heartbeat } from "../utils/database";
import { buildLightingVariantText } from "../utils/prompts";
import { generateStagingVariant } from "./imageGenerator";
import { variantKey } from "../utils/helpers";

export async function generateFurnishedVariants({
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
}) {
  const variants = {};
  const flatImageUrls = [];

  const dayOff = { isNight: false, isLightOn: false };
  const dayOn = { isNight: false, isLightOn: true };
  const nightOn = { isNight: true, isLightOn: true };
  const nightOff = { isNight: true, isLightOn: false };

  // Helper to validate generated URLs
  const validateUrl = (url, label) => {
    if (!url || typeof url !== "string" || !url.trim()) {
      throw new Error(
        `Staging generation returned empty URL for ${label}. Cannot proceed with lighting variants.`,
      );
    }
    return url;
  };

  // 1) Create ONE base staged image from the original room photo.
  const baseStoragePath = validateUrl(
    await generateStagingVariant({
      openAiKey,
      analysis,
      stagingName,
      vacantRules,
      preferredItemsRule,
      lightingVariantText: buildLightingVariantText({
        ...dayOff,
        editMode: "stage",
      }),
      preferredItemsBlock,
      customAssetNotes,
      preferredItemUrls: furnitureReferenceUrls,
      preferredItemsRecognition,
      sourceImageUrls: [photoUrl],
      editMode: "stage",
      crossPhotoPlan,
    }),
    "base image",
  );

  await heartbeat({ jobId, progress: p2 });

  variants[variantKey(dayOff)] = { storage_path: baseStoragePath };
  flatImageUrls.push(baseStoragePath);

  const lightingVacantRules = stagingType === "vacant" ? vacantRules : "";

  // 2) Create lighting variants by editing that base image ONLY.
  const dayOnStoragePath = validateUrl(
    await generateStagingVariant({
      openAiKey,
      analysis,
      stagingName,
      vacantRules: lightingVacantRules,
      preferredItemsRule: "",
      lightingVariantText: buildLightingVariantText(dayOn),
      preferredItemsBlock: "",
      customAssetNotes: "",
      preferredItemUrls: [],
      preferredItemsRecognition: null,
      sourceImageUrls: [baseStoragePath],
      editMode: "lighting_only",
    }),
    "day-on lighting variant",
  );

  await heartbeat({ jobId, progress: p3 });

  variants[variantKey(dayOn)] = { storage_path: dayOnStoragePath };
  flatImageUrls.push(dayOnStoragePath);

  const nightOnStoragePath = validateUrl(
    await generateStagingVariant({
      openAiKey,
      analysis,
      stagingName,
      vacantRules: lightingVacantRules,
      preferredItemsRule: "",
      lightingVariantText: buildLightingVariantText(nightOn),
      preferredItemsBlock: "",
      customAssetNotes: "",
      preferredItemUrls: [],
      preferredItemsRecognition: null,
      sourceImageUrls: [baseStoragePath],
      editMode: "lighting_only",
    }),
    "night-on lighting variant",
  );

  await heartbeat({ jobId, progress: p4 });

  variants[variantKey(nightOn)] = { storage_path: nightOnStoragePath };
  flatImageUrls.push(nightOnStoragePath);

  const nightOffStoragePath = validateUrl(
    await generateStagingVariant({
      openAiKey,
      analysis,
      stagingName,
      vacantRules: lightingVacantRules,
      preferredItemsRule: "",
      lightingVariantText: buildLightingVariantText(nightOff),
      preferredItemsBlock: "",
      customAssetNotes: "",
      preferredItemUrls: [],
      preferredItemsRecognition: null,
      sourceImageUrls: [nightOnStoragePath],
      editMode: "lighting_only",
    }),
    "night-off lighting variant",
  );

  variants[variantKey(nightOff)] = { storage_path: nightOffStoragePath };
  flatImageUrls.push(nightOffStoragePath);

  return { variants, flatImageUrls };
}
