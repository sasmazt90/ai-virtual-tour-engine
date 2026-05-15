import { openAiChatJson } from "../utils/openai";

function buildVacantVerificationPrompt() {
  const system =
    "You are a strict QA inspector for VACANT real-estate staging outputs. " +
    "You must follow VACANT rules exactly. " +
    "You MUST respond with VALID JSON ONLY (no markdown, no code fences, no extra text).";

  const user =
    "You will be given TWO images: (1) the ORIGINAL room photo and (2) a GENERATED VACANT result.\n\n" +
    "GROUND TRUTH: The ORIGINAL photo is the only reference.\n" +
    "Previous generations are invalid and must be ignored.\n\n" +
    "VACANT rules (EXTREMELY STRICT):\n" +
    "- REMOVE movable objects completely: bed, wardrobe/closet, mirror, shelves, decor, accessories, ANY furniture.\n" +
    "- IMPORTANT: Repainting/recoloring/blending/fading/neutralizing an object is NOT removal. Leaving a flat panel is NOT removal.\n" +
    "- Removed areas must look EMPTY with continuous realistic wall + floor surfaces (no silhouettes/ghosts/outlines/shadows).\n\n" +
    "ARCHITECTURE LOCK (ABSOLUTE):\n" +
    "- WINDOW HARD LOCK: Do NOT add windows/openings AND do NOT imply them.\n" +
    "  Any of these count as NEW WINDOW and must be flagged: bright rectangular exterior opening, window-like light rectangle, wall cut-out, depth illusion suggesting an opening, added frame/glass, brightness anomaly/patch on a previously solid wall.\n" +
    "  Any wall that had NO window in the ORIGINAL must remain a solid wall (continuous paint texture, no brightness break, no light rectangle).\n" +
    "  CRITICAL: If the ORIGINAL wall was solid and the GENERATED wall has a bright rectangular area (even without a visible frame), flag it as new_window.\n" +
    "- Do NOT add doors.\n" +
    "- Do NOT add radiators/heaters/panels. Do NOT move existing radiators.\n" +
    "- Do NOT change wall/ceiling geometry.\n" +
    "- Outlets/switches/curtain rails/ceiling lamp mounting points MUST remain exactly.\n\n" +
    "LIGHTING (VACANT ONLY):\n" +
    "- There is NO indoor lighting. Ceiling lamp fixture may be visible but must NOT emit light.\n" +
    "  Any glow/halo/hotspot under the ceiling lamp (even subtle) counts as indoor light and must be flagged.\n" +
    "- Allowed: Day=natural sunlight from existing windows. Night=moonlight + subtle exterior ambient light only.\n\n" +
    "Return STRICT JSON ONLY using EXACTLY this schema (no extra keys):\n" +
    "{\n" +
    '  \"overallPass\": boolean,\n' +
    '  \"violations\": string[],\n' +
    '  \"forbidden_objects_detected\": string[],\n' +
    '  \"silhouette_or_repaint_detected\": boolean,\n' +
    '  \"new_architecture_detected\": string[],\n' +
    '  \"indoor_light_detected_in_vacant\": boolean,\n' +
    '  \"multi_angle_inconsistency\": boolean,\n' +
    '  \"notes\": string[]\n' +
    "}\n\n" +
    "Notes:\n" +
    "- forbidden_objects_detected: include any of [bed, wardrobe, mirror, shelves, decor, accessories, furniture].\n" +
    "- new_architecture_detected: include any of [new_window, new_opening, moved_window, resized_window, new_radiator, moved_radiator, new_door, changed_wall, changed_ceiling, altered_fixture].\n" +
    "- If unsure, set overallPass=false and explain in violations.\n";

  return { system, user };
}

function coerceArrayStrings(v) {
  if (!Array.isArray(v)) return [];
  return v
    .map((x) => String(x || "").trim())
    .filter((x) => x.length > 0)
    .slice(0, 30);
}

function toBoolean(v) {
  return v === true;
}

function normalizeVerifierResult(parsed, raw) {
  const base = {
    overallPass: false,
    violations: [],
    forbidden_objects_detected: [],
    silhouette_or_repaint_detected: false,
    new_architecture_detected: [],
    indoor_light_detected_in_vacant: false,
    multi_angle_inconsistency: false,
    notes: [],
    // debugging aids (kept as nested keys to avoid schema drift in the QA JSON)
    qa_raw_text: raw ? String(raw) : "",
    qa_parsed_json: parsed && typeof parsed === "object" ? parsed : null,
  };

  if (!parsed || typeof parsed !== "object") {
    return {
      ...base,
      violations: ["qa_json_parse_failed"],
      notes: [raw ? String(raw).slice(0, 1200) : ""].filter(Boolean),
    };
  }

  // Accept exact schema keys, but also map older keys if they appear.
  const violations = coerceArrayStrings(
    parsed.violations || parsed.Violations || parsed.errors,
  );

  const forbidden = coerceArrayStrings(
    parsed.forbidden_objects_detected || parsed.forbiddenObjectsPresent,
  );

  const repaint =
    toBoolean(parsed.silhouette_or_repaint_detected) ||
    toBoolean(parsed.suspectedRepaintNotRemoval);

  const newArch = coerceArrayStrings(
    parsed.new_architecture_detected || parsed.addedObjects,
  );

  // Map older booleans into structured architecture flags when present.
  const mappedNewArch = [...newArch];
  if (toBoolean(parsed.addedWindowsOrOpenings)) {
    mappedNewArch.push("new_window");
  }
  if (toBoolean(parsed.addedRadiatorOrHeater)) {
    mappedNewArch.push("new_radiator");
  }

  const indoorLight =
    toBoolean(parsed.indoor_light_detected_in_vacant) ||
    toBoolean(parsed.indoorLightDetected) ||
    toBoolean(parsed.indoorLightPresent);

  const multiAngle =
    toBoolean(parsed.multi_angle_inconsistency) ||
    toBoolean(parsed.multiAngleInconsistency);

  const notes = coerceArrayStrings(parsed.notes);

  const overallPass = toBoolean(parsed.overallPass);

  return {
    ...base,
    overallPass,
    violations,
    forbidden_objects_detected: forbidden,
    silhouette_or_repaint_detected: repaint,
    new_architecture_detected: Array.from(new Set(mappedNewArch)).slice(0, 30),
    indoor_light_detected_in_vacant: indoorLight,
    multi_angle_inconsistency: multiAngle,
    notes,
  };
}

async function runVacantQaOnce({
  openAiKey,
  originalPhotoUrl,
  generatedImageUrl,
}) {
  const { system, user } = buildVacantVerificationPrompt();

  return await openAiChatJson({
    openAiKey,
    model: "gpt-4o-mini",
    messages: [
      { role: "system", content: system },
      {
        role: "user",
        content: [
          { type: "text", text: user },
          { type: "text", text: "ORIGINAL:" },
          // IMPORTANT: pass through raw value; openAiChatJson will normalize into image_url:{url:"..."}
          { type: "image_url", image_url: originalPhotoUrl },
          { type: "text", text: "GENERATED:" },
          { type: "image_url", image_url: generatedImageUrl },
        ],
      },
    ],
    retries: 2,
  });
}

export async function verifyVacantStaging({
  openAiKey,
  originalPhotoUrl,
  generatedImageUrl,
}) {
  // 1) Run QA once.
  const first = await runVacantQaOnce({
    openAiKey,
    originalPhotoUrl,
    generatedImageUrl,
  });

  const normalizedFirst = normalizeVerifierResult(first.parsed, first.raw);
  if (normalizedFirst.qa_parsed_json) {
    return normalizedFirst;
  }

  // 2) If JSON parse failed, immediately re-ask ONCE for strict JSON only.
  const { system } = buildVacantVerificationPrompt();
  const repairUser =
    "Your previous response was NOT valid JSON. " +
    "Return VALID JSON ONLY using EXACTLY the schema provided. No extra keys. No markdown.";

  const second = await openAiChatJson({
    openAiKey,
    model: "gpt-4o-mini",
    messages: [
      { role: "system", content: system },
      {
        role: "user",
        content: [
          { type: "text", text: repairUser },
          { type: "text", text: "ORIGINAL:" },
          { type: "image_url", image_url: originalPhotoUrl },
          { type: "text", text: "GENERATED:" },
          { type: "image_url", image_url: generatedImageUrl },
        ],
      },
    ],
    retries: 1,
  });

  return normalizeVerifierResult(second.parsed, second.raw);
}
