import { upload } from "@/app/api/utils/upload";
import { editImageWithOpenAI, generateImageWithOpenAI } from "../utils/openai";
import { buildImageGenerationPrompt } from "../utils/prompts";
import { verifyPreferredItemsInResult } from "./preferredItemsVerifier";

function buildPreferredItemsRetryHint(preferredItemsRecognition) {
  if (
    !Array.isArray(preferredItemsRecognition) ||
    !preferredItemsRecognition.length
  ) {
    return "";
  }

  const lines = preferredItemsRecognition.slice(0, 8).map((it, idx) => {
    const label = it?.label ? String(it.label) : `Item ${idx + 1}`;
    const type = it?.type ? String(it.type) : "item";
    const colors = Array.isArray(it?.colors) ? it.colors.slice(0, 4) : [];
    const mats = Array.isArray(it?.materials) ? it.materials.slice(0, 4) : [];

    const bits = [];
    bits.push(`${label} (${type})`);
    if (colors.length) bits.push(`colors: ${colors.join(", ")}`);
    if (mats.length) bits.push(`materials: ${mats.join(", ")}`);

    return `- ${bits.join(" — ")}`;
  });

  return (
    "\nRETRY OVERRIDE (VERY IMPORTANT):\n" +
    "- The result FAILED because the referenced furniture was missing or unrealistic.\n" +
    "- You MUST include EACH referenced item clearly in the output.\n" +
    "- Match the item style/colors as closely as possible.\n" +
    "- Place items in physically correct locations (on floor, correct scale, not floating).\n" +
    "MUST INCLUDE ITEMS:\n" +
    lines.join("\n") +
    "\n"
  );
}

export async function generateStagingVariant({
  openAiKey,
  analysis,
  stagingName,
  vacantRules,
  preferredItemsRule,
  lightingVariantText,
  preferredItemsBlock,
  customAssetNotes,
  preferredItemUrls,
  preferredItemsRecognition,
  sourceImageUrls,
  editMode,
  // NEW: enforce same furniture set across multiple photos in one batch
  crossPhotoPlan,
  // NEW: allow a caller (e.g. VACANT) to bypass the big prompt builder
  overridePrompt,
}) {
  const basePrompt =
    typeof overridePrompt === "string" && overridePrompt.trim().length > 0
      ? overridePrompt
      : buildImageGenerationPrompt({
          stagingName,
          vacantRules,
          preferredItemsRule,
          lightingVariantText,
          analysis,
          preferredItemsBlock,
          customAssetNotes,
          editMode,
          crossPhotoPlan,
        });

  const shouldVerify =
    preferredItemUrls.length > 0 && preferredItemsRecognition;

  let lastVerification = null;
  let lastUploadedUrl = null;

  // NEW: if preferred-item verification fails, retry once with a stronger override.
  const maxAttempts = shouldVerify ? 2 : 1;
  for (let attempt = 0; attempt < maxAttempts; attempt++) {
    const retryHint =
      attempt > 0
        ? buildPreferredItemsRetryHint(preferredItemsRecognition)
        : "";
    const imagePrompt = `${basePrompt}${retryHint}`;

    // Prefer EDITS with input images (room photo).
    let generated = null;

    if (Array.isArray(sourceImageUrls) && sourceImageUrls.length > 0) {
      const editImageUrls = [
        ...sourceImageUrls,
        ...(Array.isArray(preferredItemUrls) ? preferredItemUrls : []),
      ].filter(Boolean);

      generated = await editImageWithOpenAI({
        openAiKey,
        prompt: imagePrompt,
        imageUrls: editImageUrls,
        retries: 2,
      });
    } else {
      generated = await generateImageWithOpenAI({
        openAiKey,
        prompt: imagePrompt,
        retries: 3,
      });
    }

    const uploaded =
      generated?.kind === "b64_json"
        ? await upload({
            base64: `data:image/png;base64,${generated.b64_json}`,
          })
        : await upload({ url: generated?.url });

    if (uploaded.error) {
      throw new Error(uploaded.error);
    }

    if (
      !uploaded.url ||
      typeof uploaded.url !== "string" ||
      !uploaded.url.trim()
    ) {
      throw new Error(
        "Upload succeeded but returned no URL. " +
          `Generated kind=${generated?.kind}, modelUsed=${generated?.modelUsed || "unknown"}`,
      );
    }

    lastUploadedUrl = uploaded.url;

    if (!shouldVerify) {
      return uploaded.url;
    }

    const check = await verifyPreferredItemsInResult({
      openAiKey,
      generatedImageUrl: uploaded.url,
      preferredItems: preferredItemsRecognition,
      preferredItemUrls,
    });

    lastVerification = check;

    if (check?.overallPass) {
      return uploaded.url;
    }
  }

  throw new Error(
    "The staging did not include the selected furniture items in a realistic way. " +
      "Tips: use close-up item photos (item fills the frame), plain background, good lighting, and avoid clutter/people/hands. " +
      "Also try adding a short note like 'place this sofa as the main seating'. " +
      (lastVerification?.missing?.length
        ? ` Missing: ${lastVerification.missing.join(", ")}.`
        : ""),
  );
}
