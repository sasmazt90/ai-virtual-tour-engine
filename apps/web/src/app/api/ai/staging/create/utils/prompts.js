export function buildVacantRulesText() {
  return (
    "\nSTAGING MODE: VACANT\n" +
    "CRITICAL RULES — NO EXCEPTIONS:\n" +
    "1. VACANT STAGING MEANS:\n" +
    // Strengthen mandatory removals (explicit + redundant on purpose)
    "   - REMOVE ALL MOVABLE OBJECTS COMPLETELY (beds, wardrobes, TVs, shelves, mirrors, decorations, lamps, accessories, rugs, chairs, tables, plants, clutter).\n" +
    "   - Mandatory removals (always remove if present): bed, wardrobe, mirror, shelves, decor, accessories.\n" +
    // ... replace the overly-broad 'if unsure remove it' line with a safer constraint ...
    "   - If unsure whether a LARGE CABINET is built-in or movable: treat it as a movable wardrobe/closet and REMOVE it.\n" +
    "   - IMPORTANT: NEVER remove or alter architectural fixtures: windows, doors, radiators, outlets, switches, curtain rails, ceiling lamp mounting points, trims.\n" +
    "   - OBJECT REMOVAL MEANS PHYSICAL REMOVAL: do NOT repaint, recolor, blend, fade, neutralize, or paint objects wall-color.\n" +
    "   - Do NOT replace removed objects with a flat panel/surface. The area must remain EMPTY wall + EMPTY floor, with realistic continuous materials.\n" +
    "   - DO NOT REPLACE removed objects with anything.\n" +
    "   - DO NOT ADD ANY NEW OBJECTS.\n" +
    "   - DO NOT ADD DOORS.\n" +
    "   - DO NOT ADD WINDOWS.\n" +
    "   - DO NOT ADD FURNITURE.\n" +
    "   - DO NOT ADD ARCHITECTURAL ELEMENTS.\n" +
    "   - DO NOT add radiators/heaters/panels/fixtures that were not present in the original.\n" +
    // NEW: explicitly block the most common "empty-space cover" cheat: baseboard heaters / radiators.
    "   - CRITICAL: If the original photo has NO radiator/heater on a wall, the output MUST also have NONE. Never invent a baseboard heater, convector, panel, or long horizontal unit to fill empty space.\n" +
    "   - The wall area where bed/wardrobe were removed must remain a plain continuous wall with nothing mounted or attached (no panels).\n" +
    "2. ARCHITECTURE MUST BE 100% PRESERVED:\n" +
    "   - Wall positions must remain identical.\n" +
    "   - Ceiling shape/structure/height must remain identical.\n" +
    "   - Door locations must remain identical.\n" +
    "   - Window size/position/count must remain identical (NEVER add a new window).\n" +
    "   - Radiators (if present), sockets, switches, trims, curtain rails, window frames, lamp mounting points must remain identical.\n" +
    // NEW: extremely strict window hard-lock language to align with QA expectations
    "   - WINDOW HARD LOCK: A window-like bright rectangle, cut-out, depth illusion, or exterior opening that was not present in the original is FORBIDDEN and counts as NEW WINDOW.\n" +
    "   - Any wall that had NO window originally MUST remain a solid wall with continuous paint texture and no brightness break.\n" +
    "3. LIGHTING RULES FOR VACANT MODE:\n" +
    "   - ALL INDOOR LIGHT SOURCES ARE CONSIDERED REMOVED / UNAVAILABLE.\n" +
    "   - Ceiling lamps must NOT emit light (no glow/halo/hotspot on ceiling/walls/floor).\n" +
    "   - Flash / indoor light / torch toggles MUST NOT change the image (Flash ON must equal Flash OFF).\n" +
    "4. ALLOWED OUTPUT VARIATIONS (VACANT ONLY):\n" +
    "   - Daylight version (natural sunlight from existing windows only; NO indoor lighting).\n" +
    "   - Night version (moonlight + subtle exterior ambient light only; NO indoor lighting; NO added windows; NO bright rectangles that resemble windows).\n" +
    "   - NO other lighting variants are allowed.\n" +
    "5. MULTI-ANGLE CONSISTENCY IS MANDATORY:\n" +
    "   - All generated views must represent the SAME ROOM.\n" +
    "   - If an object is removed, it must be removed in ALL angles.\n" +
    "   - No angle-specific changes are allowed.\n" +
    "6. STRICTLY FORBIDDEN IN VACANT MODE:\n" +
    "   - Adding doors, windows, walls or ceiling changes.\n" +
    "   - Keeping TVs, wardrobes, beds, mirrors, shelves or any furniture.\n" +
    "   - Adding radiators/heaters/panels or moving existing radiators.\n" +
    "   - Redesigning the room or changing proportions.\n" +
    "VACANT STAGING IS NOT REDESIGN. It is ONLY: Object removal + daylight/night simulation + full architectural preservation.\n"
  );
}

// VACANT mode lighting: ONLY daylight vs night ambience.
// NOTE: In VACANT mode, indoor lights/torch/flash are considered unavailable.
export function buildVacantLightingVariantText({ isNight }) {
  const time = isNight ? "NIGHT" : "DAY";

  return (
    "\nVACANT LIGHTING (CRITICAL — NO EXCEPTIONS):\n" +
    `- Output type: ${time}.\n` +
    "- HARD RESET: treat ONLY the provided original room photo as the source of truth. Ignore any prior generations.\n" +
    // NEW: window hard lock (QA-aligned)
    "- WINDOW HARD LOCK: Preserve EXACT window count/size/position/shape.\n" +
    "- DO NOT add windows or imply windows: no bright rectangles, no cut-outs, no depth illusion in any wall.\n" +
    "- Any wall with NO original window MUST stay a solid wall with continuous paint texture and no brightness break/anomaly.\n" +
    // Lighting hard lock (QA-aligned)
    "- There is NO indoor/artificial lighting available in VACANT mode.\n" +
    "- Ceiling lamp fixture MAY be visible but MUST NOT emit light (no glow/halo/hotspot under the lamp).\n" +
    "- Flash/torch/indoor light toggles MUST NOT change the image (Flash ON must equal Flash OFF).\n" +
    // Brightness guidance so the model doesn't create fake window rectangles
    "- If the room would become too dark: increase overall camera exposure uniformly. Do NOT create localized bright patches.\n" +
    // NEW: explicitly block radiator/baseboard heater invention (common VACANT failure mode)
    "- RADIATOR HARD LOCK: Do NOT add radiators/heaters/panels/baseboard heaters/convectors anywhere. If the original photo has none, the output must have none.\n" +
    "- Do NOT move or change any existing radiators.\n" +
    "- Do NOT add/remove/move any objects EXCEPT: you MUST REMOVE all movable objects (bed/wardrobe/mirror/shelves/decor/accessories) completely.\n" +
    "- Removal must be physical removal (not repainting/recoloring/blending). Removed areas must remain EMPTY wall + EMPTY floor.\n" +
    // NEW: block using "architecture" or "light" as a replacement for removed objects
    "- CRITICAL: Do NOT replace removed furniture with openings, niches, panels, or any wall-mounted unit. The wall where items were removed must remain a plain continuous wall.\n" +
    (isNight
      ? "- NIGHT: subtle moonlight + soft exterior ambient light only (cool, indirect). No interior lighting. No fake bright rectangles.\n"
      : "- DAY: natural daylight/sunlight coming from the existing windows only. No interior lighting. No fake bright rectangles.\n")
  );
}

export function buildLightingVariantText({
  isNight,
  isLightOn,
  editMode = "lighting_only",
}) {
  const time = isNight ? "NIGHT" : "DAY";
  const light = isLightOn ? "LIGHTS ON" : "LIGHTS OFF";
  const isLightingOnly = editMode === "lighting_only";

  const base =
    `\nLIGHTING TARGET (VERY IMPORTANT):\n` +
    `- Time of day: ${time}.\n` +
    `- Artificial lights: ${light}.\n` +
    // IMPORTANT: never invent fixtures to justify lighting.
    "- Do NOT add or invent new lamps or light fixtures.\n" +
    // IMPORTANT: This text is used for BOTH stage + lighting-only.
    // Lighting-only restrictions are enforced elsewhere in the prompt.
    "- If you are in LIGHTING_ONLY edit mode: ONLY change lighting/exposure/color temperature; keep ALL objects exactly the same.\n" +
    "- If you are in STAGE mode: treat this as the desired lighting mood for the staged photo (while following the staging-style rules).\n" +
    "\nOBJECT FREEZE LOCK (CRITICAL — READ BEFORE GENERATING):\n" +
    "- The input image is the GROUND TRUTH for all objects, furniture, decor, and surfaces.\n" +
    "- You MUST produce an output that is PIXEL-IDENTICAL to the input for all non-lighting aspects.\n" +
    "- EVERY piece of furniture, every object, every item of decor in the input MUST appear in the output at the EXACT same position, size, shape, color, and material.\n" +
    "- Do NOT add ANY new furniture, objects, decor, rugs, pillows, lamps, plants, or items of any kind.\n" +
    "- Do NOT remove ANY existing furniture, objects, decor, rugs, pillows, lamps, plants, or items.\n" +
    "- Do NOT move, resize, recolor, reshape, or alter ANY existing object.\n" +
    "- Do NOT change wall colors, floor materials, ceiling textures, or architectural elements.\n" +
    "- The ONLY change allowed is: lighting direction, intensity, color temperature, shadows, and reflections caused by the time-of-day and light-source state.\n";

  const stagingChangeTarget =
    "\nSTAGING CHANGE TARGET (CRITICAL):\n" +
    "- This is the INITIAL staging generation, not a lighting-only edit.\n" +
    "- Do NOT keep the room mostly unchanged. Apply a visible, coherent staging transformation in the selected style.\n" +
    "- Replace or restyle movable furniture/decor as needed to match the staging style.\n" +
    "- Preserve only architecture and permanent fixtures: walls, ceiling, floor shape/material, windows, doors, beams, columns, switches, outlets, radiators, plumbing, built-ins, camera angle, and framing.\n" +
    "- Never resize, move, add, or remove windows/doors/openings. Window dimensions and positions are hard-locked.\n";

  const effectiveBase = isLightingOnly
    ? base
    : base.includes("\nOBJECT FREEZE LOCK")
      ? `${base.slice(0, base.indexOf("\nOBJECT FREEZE LOCK"))}${stagingChangeTarget}`
      : `${base}${stagingChangeTarget}`;

  const reminder = isLightingOnly
    ? "- REMINDER: Keep ALL furniture and objects EXACTLY as they are in the input. Only lighting changes.\n"
    : "- REMINDER: Preserve architecture exactly while applying the staging style to movable furniture/decor.\n";

  if (isNight) {
    if (isLightOn) {
      return (
        effectiveBase +
        "- It must feel like nighttime (dark exterior through windows).\n" +
        "- Turn ON ONLY existing visible light sources (e.g. a ceiling light already in the room).\n" +
        "- Do NOT add new lamps or new ceiling fixtures.\n" +
        "- If a TV/screen is visible, it may emit a subtle glow BUT no readable content/text.\n" +
        reminder
      );
    }
    return (
      effectiveBase +
      "- It must feel like nighttime (dark exterior through windows).\n" +
      "- Turn OFF all artificial lights (no lamps, no ceiling lights).\n" +
      "- Keep the room visible with realistic ambient exposure (camera long exposure feel), not pitch black.\n" +
      reminder
    );
  }

  // DAY
  if (isLightOn) {
    return (
      effectiveBase +
      "- It must feel like daytime with natural sunlight.\n" +
      "- Turn ON ONLY existing visible light sources (e.g. ceiling light already in the room) in addition to daylight.\n" +
      "- Do NOT add new lamps or new ceiling fixtures.\n" +
      "- If a TV/screen is visible, it may emit a subtle glow BUT no readable content/text.\n" +
      reminder
    );
  }

  return (
    effectiveBase +
    "- It must feel like daytime with natural sunlight.\n" +
    "- Turn OFF all artificial lights (no lamps, no ceiling lights).\n" +
    reminder
  );
}

export function buildVisionAnalysisPrompt() {
  const system =
    "You are a meticulous interior space analyst. You must describe the room geometry and immutable elements so that an image generation system can preserve them.";

  const user =
    'Analyze the provided SINGLE room photo for real-estate staging.\nReturn STRICT JSON only, no prose.\nGoals:\n- Identify room type, layout, approximate geometry, and camera viewpoint.\n- Identify immutable elements that must remain unchanged: plumbing, columns, beams, radiators, windows, doors, electrical outlets, switches, built-in cabinets.\n- Describe their relative positions using simple relational language.\n- Identify existing furniture and materials (floor, wall colors, ceiling, trim).\n- Note light sources and shadows.\n\nOutput JSON schema: {\n  "roomType": "...",\n  "materials": {\n    "floor": "...",\n    "walls": "...",\n    "ceiling": "...",\n    "trim": "..."\n  },\n  "cameraView": {\n    "description": "standing position + facing direction",\n    "notes": "key elements visible"\n  },\n  "immutableElements": [{\n    "type": "outlet|switch|radiator|column|beam|plumbing|window|door|built-in",\n    "location": "relative description",\n    "constraints": "must not move or disappear"\n  }],\n  "existingFurniture": ["..."],\n  "lighting": {\n    "natural": "description",\n    "artificial": "description"\n  },\n  "stagingRisks": ["what could go wrong if geometry changes"]\n}\nReturn STRICT JSON only.';

  return { system, user };
}

export function buildPreferredItemAnalysisPrompt() {
  const system =
    "You are a strict vision analyst for interior design. You must identify the primary item in the image and output STRICT JSON only.";

  const user =
    'Analyze this single preferred-item reference image. Return STRICT JSON only, no prose.\n\nYou MUST identify what the item is (examples: pillow, sofa, chair, lamp, rug, table, TV, artwork).\nAlso provide strict semantic placement rules so the item is placed in a physically correct location in a staged real-estate room.\nInfer the item as a complete 3D object: visible sides, likely hidden/back surfaces, scale cues, floor/wall contact points, shadows, and any cropped/missing parts that must be completed naturally.\n\nOutput JSON schema:\n{\n  "index": number,\n  "type": "pillow|sofa|chair|lamp|rug|table|tv|artwork|bed|shelf|decor|other",\n  "label": "short human name",\n  "styleHints": ["..."],\n  "colors": ["..."],\n  "materials": ["..."],\n  "shape": "...",\n  "keyFeatures": ["..."],\n  "threeDimensionalPlacement": {\n    "visibleSurfaces": ["..."],\n    "inferredHiddenSurfaces": ["..."],\n    "scaleCues": ["..."],\n    "contactPoints": ["..."],\n    "shadowNeeds": ["..."]\n  },\n  "semanticPlacement": {\n    "allowed": ["..."],\n    "forbidden": ["..."],\n    "notes": ["..."],\n    "mustNot": ["..."]\n  }\n}\n\nMandatory placement guidance examples:\n- Pillows: on sofas/beds/chairs only; never on the floor\n- Lamps: on side tables or floor-standing positions; never floating\n- Rugs: on the floor under seating areas\n- Wall art: mounted on walls at realistic height\n- TVs: on TV units or wall-mounted; never on the floor or floating\n\nReturn STRICT JSON only.';

  return { system, user };
}

export function buildPreferredItemVerificationPrompt() {
  const system =
    "You are a strict QA inspector for real-estate staging images. You must validate preferred items are present and placed realistically. Return STRICT JSON only.";

  const user =
    'You will be given (1) a generated staging image and (2) the preferred item reference images + extracted item types.\n\nYou MUST check:\n- Each preferred item type is clearly visible in the generated image (or explain why it is physically impossible).\n- Items are placed in semantically and physically correct locations.\n- Items are NOT floating, on the floor incorrectly, or decorative-only in unrealistic positions.\n- Items are physically integrated: correct perspective, scale, contact shadows, occlusion, and no flat sticker/collage effect.\n- If a reference item was cropped or single-view, the generated image completed hidden surfaces plausibly.\n\nIf any item is missing, misidentified, placed unrealistically, or pasted as a flat cutout: overallPass MUST be false.\n\nReturn STRICT JSON schema:\n{\n  "overallPass": boolean,\n  "missing": ["..."] ,\n  "misplaced": ["..."],\n  "notes": ["..."]\n}\n\nReturn STRICT JSON only.';

  return { system, user };
}

function buildPhotorealismRules() {
  return (
    "\nPHOTOREALISM (VERY IMPORTANT):\n" +
    "- The result must look like an unedited real estate photo (NOT CGI, NOT illustration).\n" +
    "- No dreamy filters, no heavy HDR, no cartoon look, no painterly style.\n" +
    "- Keep realistic lens distortion and perspective.\n" +
    "- Keep realistic materials and textures.\n"
  );
}

export function buildImageGenerationPrompt({
  stagingName,
  vacantRules,
  preferredItemsRule,
  lightingVariantText,
  analysis,
  preferredItemsBlock,
  customAssetNotes,
  editMode,
  // NEW: cross-photo consistency plan (same room, multiple angles)
  crossPhotoPlan,
}) {
  const mode = editMode === "lighting_only" ? "LIGHTING_ONLY" : "STAGE";
  const isVacantPrompt = String(vacantRules || "").includes(
    "STAGING MODE: VACANT",
  );

  // NEW: Put the most failure-prone VACANT locks at the TOP of the prompt.
  // This helps prevent the model from "covering emptiness" by inventing radiators or window-like openings.
  const vacantHardLockSummary =
    mode === "STAGE" && isVacantPrompt
      ? "\nVACANT HARD LOCK SUMMARY (READ FIRST):\n" +
        "- DO NOT add or imply any new windows/openings (no bright rectangles, no cut-outs, no depth illusions).\n" +
        "- DO NOT add radiators/heaters/panels/baseboard heaters/convectors anywhere (and do not move existing radiators).\n" +
        "- NO indoor lighting/glow (ceiling lamp must not emit light).\n" +
        "- Remove movable objects completely; removed areas must be EMPTY wall + EMPTY floor with continuous materials.\n"
      : "";

  const lightingOnlyText =
    mode === "LIGHTING_ONLY"
      ? isVacantPrompt
        ? "\nEDIT MODE: LIGHTING_ONLY (VERY IMPORTANT — ZERO TOLERANCE):\n" +
          "- You are given an ALREADY VACANT photo as input.\n" +
          "- You MUST NOT change the room layout, architecture, windows/doors, or emptiness.\n" +
          "- ONLY change time-of-day ambience as requested below (DAY vs NIGHT).\n" +
          "- There is NO indoor/artificial lighting in VACANT mode. Flash/torch MUST NOT change the image.\n" +
          "- NEVER add windows, doors, or any new openings.\n" +
          "- Do not add, remove, or move ANY objects. The room MUST remain EMPTY.\n" +
          "- FORBIDDEN: adding furniture, decor, rugs, plants, lamps, or any items whatsoever.\n"
        : "\nEDIT MODE: LIGHTING_ONLY (VERY IMPORTANT — ZERO TOLERANCE):\n" +
          "- You are given an ALREADY STAGED photo as input.\n" +
          "- This is a LIGHTING-ONLY edit. You are ONLY allowed to change lighting conditions.\n" +
          "- ABSOLUTE OBJECT FREEZE: Every single piece of furniture, every object, every item of decor MUST remain EXACTLY as it is in the input image.\n" +
          "- Do NOT add ANY new furniture, objects, decor, rugs, pillows, lamps, plants, accessories, or items of any kind.\n" +
          "- Do NOT remove ANY existing furniture, objects, decor, or items from the scene.\n" +
          "- Do NOT move, resize, recolor, reshape, replace, or alter ANY existing object in any way.\n" +
          "- Do NOT change the room layout, wall colors, floor materials, or architectural elements.\n" +
          "- The ONLY changes allowed are: exterior daylight level (day vs night through windows), artificial light source state (on vs off for EXISTING fixtures only), resulting shadows, reflections, and color temperature.\n" +
          "- If the input room is empty (vacant), the output MUST remain empty.\n" +
          "- If the input room has furniture, the output MUST have the EXACT SAME furniture in the EXACT SAME positions.\n" +
          "- VIOLATION CHECK: If your output contains ANY object not present in the input, or is missing ANY object from the input, the result is INVALID.\n"
      : "\nEDIT MODE (VERY IMPORTANT):\n" +
        "- You are given the ORIGINAL room photo as input.\n" +
        "- Your job is to do REAL ESTATE STAGING according to the selected STAGING STYLE.\n" +
        "- Preserve architecture 100%: do NOT change walls/windows/doors/flooring/ceiling and do NOT change camera angle.\n";

  // IMPORTANT:
  // - LIGHTING_ONLY: no object changes.
  // - VACANT + STAGE: REMOVE movable objects, but NEVER add objects.
  // - Other STAGE modes: furniture/decor changes may be allowed.
  const consistencyRules =
    mode === "LIGHTING_ONLY"
      ? "\nCONSISTENCY (VERY IMPORTANT — LIGHTING ONLY MODE):\n" +
        "- ABSOLUTE FREEZE on all objects, furniture, decor, and room contents.\n" +
        "- Do NOT add, remove, move, resize, recolor, or modify ANY objects.\n" +
        "- Do NOT add random furniture, decor, rugs, pillows, plants, or accessories.\n" +
        "- Do NOT add doors/windows/walls or any architectural elements.\n" +
        "- Do NOT change wall paint, floor material, or ceiling texture.\n" +
        "- The input image defines EVERYTHING about the room except lighting. Respect it completely.\n" +
        "- ONLY lighting/shadows/reflections/color-temperature may change.\n"
      : isVacantPrompt
        ? "\nCONSISTENCY (VERY IMPORTANT):\n" +
          "- You MUST REMOVE movable objects to make the room empty (VACANT).\n" +
          "- You MUST NOT add any new objects, furniture, decor, fixtures, doors, or windows.\n" +
          "- Preserve architecture 100%: no changes to walls/doors/windows/ceiling/floor geometry.\n"
        : "\nCONSISTENCY (VERY IMPORTANT):\n" +
          "- You MAY add/replace/reposition furniture and decor ONLY (if allowed by the selected STAGING STYLE).\n" +
          "- Do NOT add doors/windows/walls/ceilings or any architectural elements.\n" +
          "- Do NOT add people, pets, reflections, text, logos, or unrealistic objects.\n";

  const planMode = crossPhotoPlan?.planMode;

  const crossPhotoText =
    mode === "STAGE" && crossPhotoPlan && typeof crossPhotoPlan === "object"
      ? planMode === "vacant"
        ? "\nCROSS-PHOTO CONSISTENCY (VACANT — CRITICAL):\n" +
          "- These images are different angles of the SAME room.\n" +
          "- If an object is removed, it MUST be removed in ALL angles.\n" +
          "- Do NOT make angle-specific changes.\n" +
          "- Follow the VACANT plan below strictly.\n" +
          `Plan JSON:\n${JSON.stringify(crossPhotoPlan)}\n`
        : "\nCROSS-PHOTO CONSISTENCY (CRITICAL):\n" +
          "- This image is part of a ROOM GROUP (same room, different angles) in the current batch.\n" +
          "- You MUST use ONE consistent furniture/decor set across all angles in this ROOM GROUP.\n" +
          "- Do NOT create a different furniture set per angle.\n" +
          "- If an item is out of view in one angle, do NOT replace it with different items.\n" +
          "- Use the plan below as the allowed furniture set, colors, and rules.\n" +
          `Plan JSON:\n${JSON.stringify(crossPhotoPlan)}\n`
      : "";

  const existingFurnitureList =
    analysis && Array.isArray(analysis.existingFurniture)
      ? analysis.existingFurniture.filter(Boolean).slice(0, 25)
      : [];

  // VACANT: always include a mandatory removal checklist even if vision missed it.
  const mandatoryVacantRemoval = isVacantPrompt
    ? ["bed", "wardrobe", "mirror", "shelves", "decor", "accessories"]
    : [];

  const removalList = isVacantPrompt
    ? Array.from(new Set([...mandatoryVacantRemoval, ...existingFurnitureList]))
        .map((x) => String(x || "").trim())
        .filter(Boolean)
        .slice(0, 60)
    : existingFurnitureList;

  const vacantRemovalListText =
    isVacantPrompt && removalList.length > 0
      ? "\nVACANT REMOVAL LIST (MANDATORY):\n" +
        `- Remove these visible/movable items completely: ${removalList.join(", ")}.\n` +
        "- The floor area where items were removed must remain EMPTY.\n"
      : "";

  // Avoid confusing the model with furniture reference instructions in VACANT.
  const customAssetsText = isVacantPrompt
    ? ""
    : `\nIf custom assets are provided:\n- treat them as STRICT visual references\n- include them clearly and realistically\n- match scale and perspective to the room\n`;
  const customAssetIntegrationText = isVacantPrompt
    ? ""
    : "\nCUSTOM FURNITURE INTEGRATION (CRITICAL):\n" +
      "- Reconstruct each custom furniture reference as a complete 3D object, even if the upload shows only one side.\n" +
      "- Complete missing/cropped sides in the same material, color, and design language.\n" +
      "- Place items only where they would physically stand or be supported.\n" +
      "- Add realistic contact shadows, floor contact, wall contact, occlusion, and perspective.\n" +
      "- Never paste the item as a flat cutout; it must look naturally photographed in the room.\n";

  return `Create a photorealistic staged version of the room based on the provided input image(s).\n\nYou MUST preserve:\n- all windows/doors openings and positions\n- all outlets/switches/radiators/columns/beams/plumbing points and their relative positions\n- wall/floor/ceiling geometry (do not warp the room)\n- camera viewpoint and framing\n\n${vacantHardLockSummary}\n${buildPhotorealismRules()}\n${lightingOnlyText}\n${consistencyRules}\n${crossPhotoText}\n\nSTAGING STYLE: ${stagingName}\n${vacantRules}${vacantRemovalListText}${preferredItemsRule}${lightingVariantText}\n\n${customAssetsText}${customAssetIntegrationText}\nOutput requirements:\n- ultra realistic real estate photography look\n- no text, no watermarks, no logos\n\nAnalysis plan JSON (helpful guidance, do not contradict the input image):\n${JSON.stringify(analysis)}\n${preferredItemsBlock}\n(Optional custom assets):\n${customAssetNotes}`;
}
