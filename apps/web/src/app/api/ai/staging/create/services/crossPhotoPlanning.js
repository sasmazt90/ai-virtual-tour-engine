import {
  buildCrossPhotoStagingPlan,
  buildPhotoRoomGroupsFromAnalyses,
} from "./stagingSetPlanner";

export function buildVacantCrossPhotoPlan(
  perPhotoAnalyses,
  useCrossPhotoConsistency,
) {
  if (!useCrossPhotoConsistency || perPhotoAnalyses.length <= 1) {
    return null;
  }

  const allFurniture = [];
  for (const it of perPhotoAnalyses) {
    const a = it?.analysis;
    const list = Array.isArray(a?.existingFurniture) ? a.existingFurniture : [];
    for (const x of list) {
      const s = String(x || "").trim();
      if (s) allFurniture.push(s);
    }
  }

  return {
    planMode: "vacant",
    removeAllMovableObjects: true,
    mandatoryRemovalList: Array.from(new Set(allFurniture)).slice(0, 80),
    consistencyRules: [
      "Same room, multiple angles: removed items must be removed in ALL angles.",
      "Do not add any objects.",
      "Preserve architecture 100% (doors/windows/walls/ceiling/outlets/switches/radiators).",
    ],
  };
}

export async function buildFurnishedCrossPhotoPlans({
  openAiKey,
  perPhotoAnalyses,
  useCrossPhotoConsistency,
  stagingName,
  preferredItemsRecognition,
}) {
  const roomGroups = [];
  const crossPhotoPlanByPhotoId = {};

  if (!useCrossPhotoConsistency || perPhotoAnalyses.length <= 1) {
    const photoIds = perPhotoAnalyses.map((x) => x.photoId).filter(Boolean);
    roomGroups.push({ groupId: "all", label: "other", photoIds });
    return { roomGroups, crossPhotoPlanByPhotoId };
  }

  const grouping = await buildPhotoRoomGroupsFromAnalyses({
    openAiKey,
    photoAnalyses: perPhotoAnalyses,
  });

  const rawGroups = Array.isArray(grouping?.groups) ? grouping.groups : [];

  const used = new Set();
  for (const g of rawGroups) {
    const indices = Array.isArray(g?.photoIndices) ? g.photoIndices : [];

    const photoIds = indices
      .map((i) => perPhotoAnalyses[i])
      .filter(Boolean)
      .map((x) => x.photoId)
      .filter(Boolean);

    const deduped = photoIds.filter((id) => {
      if (used.has(id)) return false;
      used.add(id);
      return true;
    });

    if (deduped.length > 0) {
      roomGroups.push({
        groupId: String(g?.groupId || `room_${roomGroups.length + 1}`),
        label: String(g?.label || "other"),
        photoIds: deduped,
      });
    }
  }

  for (const it of perPhotoAnalyses) {
    if (!it?.photoId) continue;
    if (used.has(it.photoId)) continue;
    roomGroups.push({
      groupId: `room_${roomGroups.length + 1}`,
      label: "other",
      photoIds: [it.photoId],
    });
  }

  for (const g of roomGroups) {
    const ids = Array.isArray(g?.photoIds) ? g.photoIds : [];
    if (ids.length < 2) continue;

    const groupItems = perPhotoAnalyses.filter((x) => ids.includes(x.photoId));
    const groupUrls = groupItems.map((x) => x.photoUrl).filter(Boolean);

    const plan = await buildCrossPhotoStagingPlan({
      openAiKey,
      stagingName,
      photoUrls: groupUrls,
      preferredItemsRecognition,
      photoAnalyses: groupItems,
      maxImagesForPlan: 8,
    });

    if (plan && typeof plan === "object") {
      const wrapped = { planMode: "furniture_set", ...plan };
      for (const pid of ids) {
        crossPhotoPlanByPhotoId[pid] = wrapped;
      }
    }
  }

  return { roomGroups, crossPhotoPlanByPhotoId };
}
