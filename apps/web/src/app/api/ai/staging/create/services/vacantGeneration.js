// NOTE: This file does not update job progress directly; the caller provides heartbeatAt().
import { buildVacantLightingVariantText } from "../utils/prompts";
import { generateStagingVariant } from "./imageGenerator";
import { verifyVacantStaging } from "./vacantVerifier";

export function buildVacantTargetedCorrection(check) {
  const lines = [];
  const forbidden = Array.isArray(check?.forbidden_objects_detected)
    ? check.forbidden_objects_detected
    : [];
  const newArch = Array.isArray(check?.new_architecture_detected)
    ? check.new_architecture_detected
    : [];

  const addObj = (name, msg) => {
    if (forbidden.includes(name)) {
      lines.push(msg);
    }
  };

  // Forbidden objects (targeted)
  addObj(
    "wardrobe",
    "- REMOVE WARDROBE / CLOSET COMPLETELY. Repainting/recoloring/blending is FORBIDDEN. Leave EMPTY wall+floor with continuous materials.",
  );
  addObj(
    "closet",
    "- REMOVE WARDROBE / CLOSET COMPLETELY. Repainting/recoloring/blending is FORBIDDEN. Leave EMPTY wall+floor with continuous materials.",
  );
  addObj(
    "bed",
    "- REMOVE BED COMPLETELY. Do NOT leave silhouette/shadow/outline. Leave EMPTY wall+floor with continuous materials.",
  );
  addObj(
    "mirror",
    "- REMOVE MIRROR COMPLETELY. Do NOT leave reflection-like panel. Leave EMPTY wall+floor with continuous materials.",
  );
  addObj(
    "shelves",
    "- REMOVE SHELVES COMPLETELY. No outlines, no ghosting, no flat panels.",
  );
  addObj(
    "decor",
    "- REMOVE DECOR / ACCESSORIES COMPLETELY. Do NOT replace with any object or any bright patch.",
  );
  addObj(
    "accessories",
    "- REMOVE DECOR / ACCESSORIES COMPLETELY. Do NOT replace with any object or any bright patch.",
  );
  addObj(
    "furniture",
    "- REMOVE ALL FURNITURE COMPLETELY. Do NOT repaint to wall color. Do NOT leave silhouettes or flat panels.",
  );

  // Repaint / silhouette detection
  if (check?.silhouette_or_repaint_detected === true) {
    lines.push(
      "- CRITICAL: Repainting/neutralizing is NOT removal. Do NOT leave a flat rectangular panel or ghost outline. The object must be PHYSICALLY GONE with continuous wall+floor textures.",
    );
    lines.push(
      "- Mirror/wardrobe removal MUST NOT leave patched wall rectangles, brightness anomalies, or texture discontinuities. The wall must look continuous and unaltered.",
    );
  }

  // Architecture violations
  const hasNewWindow =
    newArch.includes("new_window") ||
    newArch.includes("new_opening") ||
    newArch.includes("moved_window") ||
    newArch.includes("resized_window");

  if (hasNewWindow) {
    lines.push(
      "- CRITICAL WINDOW HARD LOCK: DO NOT ADD WINDOWS/OPENINGS and DO NOT IMPLY WINDOWS. This includes: bright rectangles, cut-outs, depth illusions, window frames, glass openings, exterior light rectangles not in the original.",
    );
    lines.push(
      "- Any wall that had NO window in the original MUST remain a solid wall with continuous paint texture and NO brightness break/anomaly.",
    );
    lines.push(
      "- Do NOT use brightness patches, perspective tricks, or wall cut-outs to fill removed-object areas. Empty means EMPTY wall+floor.",
    );
    lines.push(
      "- FORBIDDEN: Creating a bright rectangular area, light patch, or window-like opening on a wall that was originally solid (even if it makes the room look nicer).",
    );
  }

  const hasRadiatorViolation =
    newArch.includes("new_radiator") ||
    newArch.includes("moved_radiator") ||
    newArch.includes("added_radiator") ||
    newArch.includes("heater");

  if (hasRadiatorViolation) {
    lines.push(
      "- CRITICAL RADIATOR HARD LOCK: DO NOT ADD OR MOVE any radiators/heaters/panels/baseboard heaters/convectors anywhere.",
    );
    lines.push(
      "- If the original wall had NO radiator, the output MUST have NONE on that wall (and do NOT add any radiator-like substitute).",
    );
    lines.push(
      "- Do NOT add any long horizontal white/metal unit, vent, grill, panel, or heater-looking object along the base of the wall. Keep only the original skirting/baseboard trim.",
    );
  }

  if (newArch.includes("new_door")) {
    lines.push("- DO NOT ADD DOORS. Preserve doors EXACTLY.");
  }
  if (newArch.includes("changed_wall")) {
    lines.push("- DO NOT CHANGE WALL GEOMETRY/POSITION/ANGLES.");
  }
  if (newArch.includes("changed_ceiling")) {
    lines.push("- DO NOT CHANGE CEILING SHAPE/TEXTURE/HEIGHT.");
  }
  if (newArch.includes("altered_fixture")) {
    lines.push(
      "- DO NOT ALTER FIXTURES: outlets/switches/curtain rails/ceiling lamp mounting points must remain EXACTLY.",
    );
  }

  // Lighting violations
  if (check?.indoor_light_detected_in_vacant === true) {
    lines.push(
      "- CRITICAL LIGHTING: VACANT HAS ZERO INDOOR LIGHT. Ceiling lamp may be visible but MUST NOT emit light: no glow/halo/hotspot on ceiling/walls/floor. Even subtle glow is forbidden.",
    );
    lines.push(
      "- Day brightness MUST come ONLY from the original windows and overall camera exposure (uniform), NEVER from indoor light sources.",
    );
  }

  // If the QA returned generic violations but we didn't match keys, still add a safe baseline.
  if (lines.length === 0) {
    lines.push(
      "- FOLLOW VACANT STRICTLY: remove bed/wardrobe/mirror/shelves/decor fully; no repaint; no new objects; preserve architecture; no indoor lighting; no window-like bright rectangles; no radiators/heaters/panels.",
    );
  }

  return (
    "\nVACANT QA CORRECTION (TARGETED — FIX THESE EXACT VIOLATIONS):\n" +
    lines.join("\n") +
    "\n"
  );
}

// NEW: VACANT uses a smaller, more controlled prompt than the general builder.
// The general prompt includes analysis JSON and other context that can increase hallucinations
// (e.g. inventing windows/radiators to "explain" emptiness).
function buildVacantOverridePrompt({
  variantLabel,
  lightingVariantText,
  targetedCorrection,
  extraHardLock,
  existingFurnitureList,
  // NEW: if we are correcting an already-vacant generated image, we must not
  // re-run removal logic and risk reintroducing objects or inventing architecture.
  correctionOnly,
}) {
  const time = variantLabel === "night" ? "NIGHT" : "DAY";

  const correctionHeader = correctionOnly
    ? "\nCORRECTION-ONLY MODE (VERY IMPORTANT):\n" +
      "- The input image is ALREADY an attempted VACANT result.\n" +
      "- Do NOT add, remove, or reintroduce ANY objects. Keep the room EMPTY.\n" +
      "- Do NOT repaint/patch walls in a way that creates rectangles or panel artifacts.\n" +
      "- ONLY fix the specific QA violations listed below while preserving framing and architecture.\n"
    : "";

  // Build a specific removal list from vision analysis
  const mandatoryRemoval = [
    "bed",
    "wardrobe",
    "closet",
    "mirror",
    "shelves",
    "decor",
    "accessories",
  ];
  const detected = Array.isArray(existingFurnitureList)
    ? existingFurnitureList
    : [];
  const allRemoval = Array.from(new Set([...mandatoryRemoval, ...detected]))
    .map((x) => String(x || "").trim())
    .filter(Boolean)
    .slice(0, 40);
  const removalListText = allRemoval.join(", ");

  return (
    "You are editing a REAL ESTATE PHOTO in VACANT mode.\n" +
    "HARD RESET: Use ONLY the provided input photo as ground truth. Ignore all prior generations.\n" +
    "\n" +
    "DO NOT CHANGE CAMERA/FRAMING: do not crop, do not extend canvas, do not add borders, do not change perspective.\n" +
    correctionHeader +
    "\n" +
    "VACANT GOAL: The room must be COMPLETELY EMPTY.\n" +
    `- REMOVE COMPLETELY (physical removal, not repaint): ${removalListText}, and ALL other movable furniture/objects.\n` +
    "- Every single piece of furniture, decoration, and movable object must be removed — no exceptions.\n" +
    "- Removed areas must remain EMPTY floor + EMPTY wall with continuous realistic textures (no flat panels, no patches, no silhouettes).\n" +
    "- If unsure whether a large cabinet is built-in or movable: treat it as movable and REMOVE it.\n" +
    "\n" +
    "ARCHITECTURE HARD LOCK:\n" +
    "- Preserve EXACT wall/ceiling geometry and ALL fixtures (outlets, switches, curtain rails, trims, ceiling lamp mounting point).\n" +
    "- Preserve EXACT windows and doors.\n" +
    "- NEVER add or imply a new window/opening (no bright rectangles, no cut-outs, no depth illusions, no exterior openings).\n" +
    "- NEVER add/move radiators/heaters/panels/baseboard heaters/convectors.\n" +
    "\n" +
    "LIGHTING HARD LOCK (VACANT):\n" +
    "- There is ZERO indoor lighting. Ceiling lamp fixture may be visible but MUST NOT emit light (no glow/halo/hotspot).\n" +
    "- Flash/torch toggles must have NO effect.\n" +
    `\nOUTPUT: ${time}.\n` +
    lightingVariantText +
    (targetedCorrection || "") +
    (extraHardLock || "") +
    "\nReturn a photorealistic, unedited-looking real estate photo. No text/watermarks."
  );
}

export function formatVacantQaSummary(check) {
  const forbidden = Array.isArray(check?.forbidden_objects_detected)
    ? check.forbidden_objects_detected
    : [];
  const newArch = Array.isArray(check?.new_architecture_detected)
    ? check.new_architecture_detected
    : [];

  const parts = [];
  if (forbidden.length) {
    parts.push(`forbidden_objects_detected=[${forbidden.join(", ")}]`);
  }
  if (check?.silhouette_or_repaint_detected === true) {
    parts.push("silhouette_or_repaint_detected=true");
  }
  if (newArch.length) {
    parts.push(`new_architecture_detected=[${newArch.join(", ")}]`);
  }
  if (check?.indoor_light_detected_in_vacant === true) {
    parts.push("indoor_light_detected_in_vacant=true");
  }
  if (check?.multi_angle_inconsistency === true) {
    parts.push("multi_angle_inconsistency=true");
  }

  const violations = Array.isArray(check?.violations) ? check.violations : [];
  if (violations.length) {
    parts.push(`violations=[${violations.join(" | ")}]`);
  }

  return parts.join("\n");
}

function shouldSwitchToCorrectionOnlyBase(lastCheck) {
  if (!lastCheck || typeof lastCheck !== "object") return false;

  const forbidden = Array.isArray(lastCheck?.forbidden_objects_detected)
    ? lastCheck.forbidden_objects_detected
    : [];

  // If we still see forbidden objects, do NOT switch bases — we must go back to the original.
  if (forbidden.length > 0) return false;

  // If we see silhouette/repaint, going correction-only tends to amplify artifacts.
  if (lastCheck?.silhouette_or_repaint_detected === true) return false;

  // If indoor light is detected, we also prefer regenerating from scratch
  // because correction-only mode cannot reliably remove glow artifacts.
  if (lastCheck?.indoor_light_detected_in_vacant === true) return false;

  const newArch = Array.isArray(lastCheck?.new_architecture_detected)
    ? lastCheck.new_architecture_detected
    : [];

  // Only switch to correction-only if the ONLY remaining failures are architecture-related
  // (e.g. new window, moved radiator). All objects must already be removed.
  if (newArch.length === 0) return false;

  const allowedCorrectionOnlyFlags = new Set([
    "new_window",
    "new_opening",
    "moved_window",
    "resized_window",
    "new_radiator",
    "moved_radiator",
    "added_radiator",
    "heater",
  ]);

  return newArch.every((x) => allowedCorrectionOnlyFlags.has(x));
}

export async function generateAndVerifyVacantVariant({
  openAiKey,
  photoId,
  photoUrl,
  analysis,
  crossPhotoPlan,
  stagingName,
  vacantRules,
  heartbeatAt,
  variantLabel,
  vacantQaResults,
}) {
  let lastCheck = null;
  let lastGeneratedUrl = null;

  // Extract existing furniture list from analysis for more targeted removal
  const existingFurnitureList =
    analysis && Array.isArray(analysis.existingFurniture)
      ? analysis.existingFurniture.filter(Boolean).slice(0, 25)
      : [];

  // Try up to 3 times:
  // - attempt 1: generate from ORIGINAL photo
  // - attempt 2: targeted correction (may use previous generated as base if it is already empty)
  // - attempt 3: targeted correction + extra hard lock
  for (let attempt = 0; attempt < 3; attempt++) {
    const targetedCorrection =
      attempt === 0 ? "" : buildVacantTargetedCorrection(lastCheck);

    const extraHardLock =
      attempt < 2
        ? ""
        : "\nFINAL HARD LOCK (DO NOT IGNORE):\n" +
          "- You are NOT allowed to invent any architectural elements to make the room look nicer or to cover empty space.\n" +
          "- Absolutely NO new windows, NO window-like bright rectangles, NO wall cut-outs, NO depth illusions.\n" +
          "- Absolutely NO radiators/heaters/panels/baseboard heaters/convectors (and do not move existing ones).\n" +
          "- Absolutely NO indoor light glow (ceiling lamp must not emit).\n" +
          "- Every piece of furniture must be GONE — not repainted, not blended, not silhouetted.\n";

    const lightingVariantText = buildVacantLightingVariantText({
      isNight: variantLabel === "night",
    });

    // NEW: choose the safest base image.
    // - Always start from the ORIGINAL photo.
    // - If the previous attempt already removed all forbidden objects, but failed on
    //   window/radiator/light artifacts, do a correction-only edit on the previous image.
    const correctionOnlyBase =
      attempt > 0 && shouldSwitchToCorrectionOnlyBase(lastCheck);

    const baseImageUrl =
      attempt === 0
        ? photoUrl
        : correctionOnlyBase
          ? lastGeneratedUrl
          : photoUrl;

    const overridePrompt = buildVacantOverridePrompt({
      variantLabel,
      lightingVariantText,
      existingFurnitureList,
      targetedCorrection:
        attempt === 0
          ? ""
          : "\nHARD RESET: Ignore any previous generations. Use ONLY the ORIGINAL photo as ground truth.\n" +
            targetedCorrection,
      extraHardLock,
      correctionOnly: correctionOnlyBase,
    });

    const url = await generateStagingVariant({
      openAiKey,
      // NOTE: analysis is intentionally not used in VACANT overridePrompt
      analysis,
      stagingName,
      vacantRules,
      preferredItemsRule: "",
      lightingVariantText: "",
      preferredItemsBlock: "",
      customAssetNotes: "",
      preferredItemUrls: [],
      preferredItemsRecognition: null,
      // HARD RESET: default to ORIGINAL photo, but allow correction-only edits
      // on the previous generated image when it's already empty.
      sourceImageUrls: baseImageUrl ? [baseImageUrl] : [photoUrl],
      editMode: "stage",
      crossPhotoPlan,
      overridePrompt,
    });

    if (!url || typeof url !== "string" || !url.trim()) {
      throw new Error(
        `VACANT staging generation returned empty URL for photoId=${photoId} variant=${variantLabel} attempt=${attempt + 1}`,
      );
    }

    lastGeneratedUrl = url;

    lastCheck = await verifyVacantStaging({
      openAiKey,
      originalPhotoUrl: photoUrl,
      generatedImageUrl: url,
    });

    const passed = lastCheck?.overallPass === true;

    // Record per-image QA immediately (even for failures)
    vacantQaResults.push({
      photoId,
      variant: variantLabel,
      attempt: attempt + 1,
      pass: passed,
      originalPhotoUrl: photoUrl,
      generatedImageUrl: url,
      verifier: {
        overallPass: lastCheck?.overallPass === true,
        violations: Array.isArray(lastCheck?.violations)
          ? lastCheck.violations
          : [],
        forbidden_objects_detected: Array.isArray(
          lastCheck?.forbidden_objects_detected,
        )
          ? lastCheck.forbidden_objects_detected
          : [],
        silhouette_or_repaint_detected:
          lastCheck?.silhouette_or_repaint_detected === true,
        new_architecture_detected: Array.isArray(
          lastCheck?.new_architecture_detected,
        )
          ? lastCheck.new_architecture_detected
          : [],
        indoor_light_detected_in_vacant:
          lastCheck?.indoor_light_detected_in_vacant === true,
        multi_angle_inconsistency:
          lastCheck?.multi_angle_inconsistency === true,
        notes: Array.isArray(lastCheck?.notes) ? lastCheck.notes : [],
        qa_raw_text:
          typeof lastCheck?.qa_raw_text === "string"
            ? lastCheck.qa_raw_text
            : "",
        qa_parsed_json: lastCheck?.qa_parsed_json || null,
      },
    });

    if (passed) {
      return { url, qa: lastCheck };
    }

    if (typeof heartbeatAt === "function") {
      await heartbeatAt();
    }
  }

  const summary = formatVacantQaSummary(lastCheck);
  const raw =
    typeof lastCheck?.qa_raw_text === "string"
      ? lastCheck.qa_raw_text.slice(0, 1200)
      : "";

  const parsedSnippet = (() => {
    try {
      if (!lastCheck?.qa_parsed_json) return "";
      return JSON.stringify(lastCheck.qa_parsed_json).slice(0, 1200);
    } catch {
      return "";
    }
  })();

  throw new Error(
    `VACANT staging QA failed for photoId=${photoId} variant=${variantLabel}.\n` +
      (summary ? summary + "\n" : "") +
      (parsedSnippet ? `qa_parsed_json=${parsedSnippet}\n` : "") +
      (raw ? `qa_raw_text=${raw}` : ""),
  );
}
