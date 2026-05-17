import { buildVacantRulesText } from "../utils/prompts";

export function buildPromptComponents({
  isVacant,
  furnitureReferenceUrls,
  preferredItemsRecognition,
  assets,
  preferredItemHints,
  preferredItemsText,
}) {
  const vacantRules = isVacant ? buildVacantRulesText() : "";

  const preferredItemsSummary = Array.isArray(preferredItemsRecognition)
    ? preferredItemsRecognition
        .map((it, idx) => {
          const type = it?.type ? String(it.type) : "item";
          const label = it?.label ? String(it.label) : `Item ${idx + 1}`;
          const colors = Array.isArray(it?.colors) ? it.colors.slice(0, 4) : [];
          const mats = Array.isArray(it?.materials)
            ? it.materials.slice(0, 4)
            : [];
          const bits = [];
          bits.push(`${label} (${type})`);
          if (colors.length) bits.push(`colors: ${colors.join(", ")}`);
          if (mats.length) bits.push(`materials: ${mats.join(", ")}`);
          return `- ${bits.join(" — ")}`;
        })
        .slice(0, 8)
        .join("\n")
    : "";

  const userHintsText =
    Array.isArray(preferredItemHints) && preferredItemHints.length
      ? "\nUSER NOTES FOR FURNITURE (IMPORTANT):\n" +
        preferredItemHints
          .slice(0, 8)
          .map((h) => {
            const label = h?.label ? String(h.label) : "Item";
            const notes = h?.notes ? String(h.notes) : "";
            return `- ${label}: ${notes}`;
          })
          .join("\n") +
        "\n"
      : "";

  const preferredItemsFreeText = preferredItemsText
    ? `\nUSER FURNITURE REQUEST (IMPORTANT):\n${preferredItemsText}\n`
    : "";

  const preferredItemsRule =
    !isVacant && furnitureReferenceUrls.length
      ? "\nFURNITURE REFERENCES (IMPORTANT — NO EXCEPTIONS):\n" +
        "- You MUST include EACH referenced furniture item clearly in the final staged room photo.\n" +
        "- Do NOT swap it for a different style/color; match the reference as closely as possible.\n" +
        "- Treat each reference as a real 3D object: infer the hidden/back/side surfaces when only one view is provided.\n" +
        "- The item MUST be placed physically correctly (correct scale, on the floor or supported by furniture, never floating).\n" +
        "- Match room perspective, contact shadows, occlusion, reflections, and floor/wall contact points.\n" +
        "- The item must look naturally photographed in the room, not pasted on top as a flat cutout or collage element.\n" +
        "- If the item is partially cropped in the reference, complete the missing side in the same style/material.\n" +
        "- Place the item only where it belongs semantically: sofas/chairs on floors, art on walls, pillows on seating/beds, rugs flat on floors, lamps on tables/floors.\n" +
        "- You MUST preserve the original room architecture and camera angle.\n" +
        "- You MUST only change furniture/decor; do not remodel the room.\n" +
        (preferredItemsSummary
          ? `\nMUST INCLUDE LIST:\n${preferredItemsSummary}\n`
          : "") +
        userHintsText +
        preferredItemsFreeText
      : "";

  const preferredItemsBlock =
    !isVacant && preferredItemsRecognition
      ? `\nFurniture reference recognition JSON (follow this):\n${JSON.stringify(preferredItemsRecognition)}\n`
      : "";

  const customAssetNotes = !isVacant
    ? assets
        .map((a) => `${a.label || "asset"}: ${a.storage_path}`)
        .slice(0, 8)
        .join("\n")
    : "";

  return {
    vacantRules,
    preferredItemsRule,
    preferredItemsBlock,
    customAssetNotes,
  };
}
