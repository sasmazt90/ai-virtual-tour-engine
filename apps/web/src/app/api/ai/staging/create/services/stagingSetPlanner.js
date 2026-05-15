import { openAiChatJson } from "../utils/openai";

function pickEvenlySpacedIndices(count, max) {
  const c = Math.max(0, Number(count || 0) || 0);
  const m = Math.max(1, Number(max || 1) || 1);
  if (c <= m) {
    return Array.from({ length: c }, (_, i) => i);
  }

  const out = [];
  for (let i = 0; i < m; i++) {
    const idx = Math.round((i * (c - 1)) / (m - 1));
    out.push(idx);
  }

  // de-dupe (rounding can create duplicates for small ranges)
  return Array.from(new Set(out)).slice(0, m);
}

function selectRepresentativeUrls(urls, maxImages) {
  const list = Array.isArray(urls) ? urls.filter(Boolean) : [];
  const indices = pickEvenlySpacedIndices(list.length, maxImages);
  return indices.map((i) => list[i]).filter(Boolean);
}

function buildStagingSetPlanPrompt({
  stagingName,
  hasFurnitureRefs,
  roomContext,
}) {
  const system =
    "You are a meticulous real-estate virtual staging designer. You must produce a single consistent furniture set that can be reused across multiple photos that are DIFFERENT ANGLES of the SAME room. Output STRICT JSON only.";

  const user =
    "We have multiple photos that belong to the SAME room, taken from different angles.\n" +
    "Your job: propose ONE consistent staging furniture/decor set that will be reused across all angles in this room group.\n\n" +
    `Staging style: ${stagingName}.\n` +
    (hasFurnitureRefs
      ? "Some custom furniture reference items may be required. If so, include them in the plan and do not conflict with them.\n"
      : "") +
    (roomContext
      ? `\nRoom context (per-photo analyses, may be incomplete):\n${roomContext}\n`
      : "") +
    "Rules (VERY IMPORTANT):\n" +
    "- Keep room architecture/camera angles intact; do not remodel.\n" +
    "- Use the SAME furniture types and overall look across all angles.\n" +
    "- Do NOT invent different sets per angle.\n" +
    "- If an item is out of view in one angle, do NOT replace it with a different item; just keep the set consistent.\n" +
    "- Keep colors/materials consistent across all images.\n\n" +
    "Return STRICT JSON only, no prose.\n\n" +
    "JSON schema:\n" +
    "{\n" +
    '  "theme": "...",\n' +
    '  "colorPalette": ["..."],\n' +
    '  "mustUseItems": [\n' +
    "    {\n" +
    '      "type": "sofa|chair|coffee_table|side_table|rug|tv_unit|tv|bed|nightstand|dining_table|dining_chair|lamp|plant|artwork|decor|storage|other",\n' +
    '      "label": "short name",\n' +
    '      "style": "...",\n' +
    '      "colors": ["..."],\n' +
    '      "materials": ["..."],\n' +
    '      "placement": "where it should go relative to room"\n' +
    "    }\n" +
    "  ],\n" +
    '  "forbiddenItems": ["..."],\n' +
    '  "consistencyRules": ["..."]\n' +
    "}\n";

  return { system, user };
}

function buildRoomGroupingPrompt({ summaries }) {
  const system =
    "You are a strict real-estate photo organizer. You group photos by ROOM. Photos in the same group are different angles of the SAME room. Output STRICT JSON only.";

  const user =
    "We have multiple interior photos from the same property. Some may be different rooms.\n" +
    "Group them by room so that each group can be staged with a consistent furniture set.\n\n" +
    "Rules (CRITICAL):\n" +
    "- Each photo index MUST appear in exactly one group.\n" +
    "- If you are unsure whether two photos are the same room: put them in separate groups.\n" +
    "- Do not try to force everything into one group.\n\n" +
    "Photo summaries (JSON-ish text):\n" +
    `${summaries}\n\n` +
    "Return STRICT JSON only, no prose.\n\n" +
    "JSON schema:\n" +
    "{\n" +
    '  "groups": [\n' +
    "    {\n" +
    '      "groupId": "room_1",\n' +
    '      "label": "living room|bedroom|kids room|kitchen|bathroom|hallway|office|other",\n' +
    '      "photoIndices": [0,1]\n' +
    "    }\n" +
    "  ],\n" +
    '  "notes": ["..."]\n' +
    "}\n";

  return { system, user };
}

export async function buildPhotoRoomGroupsFromAnalyses({
  openAiKey,
  photoAnalyses,
}) {
  const items = Array.isArray(photoAnalyses) ? photoAnalyses : [];
  if (items.length < 2) {
    return {
      groups: [{ groupId: "room_1", label: "other", photoIndices: [0] }],
      notes: [],
    };
  }

  const summaries = items
    .map((it, idx) => {
      const a =
        it?.analysis && typeof it.analysis === "object" ? it.analysis : {};
      const roomType = a?.roomType ? String(a.roomType) : "";
      const floor = a?.materials?.floor ? String(a.materials.floor) : "";
      const walls = a?.materials?.walls ? String(a.materials.walls) : "";
      const cam = a?.cameraView?.notes ? String(a.cameraView.notes) : "";
      const furniture = Array.isArray(a?.existingFurniture)
        ? a.existingFurniture.filter(Boolean).slice(0, 12)
        : [];

      return {
        index: idx,
        roomType,
        floor,
        walls,
        cameraNotes: cam,
        existingFurniture: furniture,
      };
    })
    .map((x) => JSON.stringify(x))
    .join("\n");

  const { system, user } = buildRoomGroupingPrompt({ summaries });

  const { parsed } = await openAiChatJson({
    openAiKey,
    model: "gpt-4o-mini",
    messages: [
      { role: "system", content: system },
      { role: "user", content: user },
    ],
    retries: 2,
  });

  if (parsed && typeof parsed === "object" && Array.isArray(parsed.groups)) {
    return parsed;
  }

  // fallback: each photo is its own group
  return {
    groups: items.map((_, idx) => ({
      groupId: `room_${idx + 1}`,
      label: "other",
      photoIndices: [idx],
    })),
    notes: ["Fallback grouping used."],
  };
}

export async function buildCrossPhotoStagingPlan({
  openAiKey,
  stagingName,
  photoUrls,
  preferredItemsRecognition,
  photoAnalyses,
  maxImagesForPlan = 8,
}) {
  const urls = Array.isArray(photoUrls) ? photoUrls.filter(Boolean) : [];
  if (urls.length < 2) {
    return null;
  }

  // Keep token usage bounded: pick representative angles across the batch.
  const imageUrls = selectRepresentativeUrls(urls, maxImagesForPlan);

  const hasFurnitureRefs =
    Array.isArray(preferredItemsRecognition) &&
    preferredItemsRecognition.length > 0;

  const roomContext = Array.isArray(photoAnalyses)
    ? photoAnalyses
        .map((x, idx) => {
          const a =
            x?.analysis && typeof x.analysis === "object" ? x.analysis : {};
          const roomType = a?.roomType ? String(a.roomType) : "";
          const floor = a?.materials?.floor ? String(a.materials.floor) : "";
          const walls = a?.materials?.walls ? String(a.materials.walls) : "";
          const notes = a?.cameraView?.notes ? String(a.cameraView.notes) : "";
          return `#${idx + 1} roomType=${roomType} floor=${floor} walls=${walls} notes=${notes}`;
        })
        .slice(0, 12)
        .join("\n")
    : "";

  const { system, user } = buildStagingSetPlanPrompt({
    stagingName,
    hasFurnitureRefs,
    roomContext,
  });

  const preferredItemsText = hasFurnitureRefs
    ? `Preferred furniture reference recognition JSON:\n${JSON.stringify(preferredItemsRecognition)}`
    : null;

  const content = [{ type: "text", text: user }];
  if (preferredItemsText) {
    content.push({ type: "text", text: preferredItemsText });
  }
  for (const u of imageUrls) {
    // IMPORTANT: pass raw value; openAiChatJson normalizes image blocks
    content.push({ type: "image_url", image_url: u });
  }

  const { parsed } = await openAiChatJson({
    openAiKey,
    model: "gpt-4o-mini",
    messages: [
      { role: "system", content: system },
      {
        role: "user",
        content,
      },
    ],
    retries: 2,
  });

  if (parsed && typeof parsed === "object") {
    return parsed;
  }

  return null;
}
