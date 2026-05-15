import {
  findOrCreateStaging,
  updateStaging,
  createStaging,
  deleteStagingImages,
  insertStagingImage,
  updateJobResult,
} from "../utils/database";

export async function persistStagingResults({
  propertyId,
  stagingType,
  stagedItems,
  meta,
  jobId,
}) {
  // Slot logic: ONE staging row per (property_id, staging_type)
  const { stagingId, version, isNew } = await findOrCreateStaging({
    propertyId,
    stagingType,
  });

  let finalStagingId = stagingId;
  let nextVersion = version + 1;

  if (!isNew) {
    await updateStaging({
      stagingId: finalStagingId,
      meta,
      version: nextVersion,
    });
    await deleteStagingImages(finalStagingId);
  } else {
    const result = await createStaging({
      propertyId,
      stagingType,
      meta,
      version: nextVersion,
    });
    finalStagingId = result.stagingId;
    nextVersion = result.version;
  }

  // Insert variant images and capture their IDs
  const stagedWithIds = [];
  for (const item of stagedItems) {
    const variantsWithIds = {};
    const vEntries =
      item?.variants && typeof item.variants === "object" ? item.variants : {};

    for (const [k, v] of Object.entries(vEntries)) {
      const storagePath = v?.storage_path;
      if (!storagePath) continue;

      const imageId = await insertStagingImage({
        stagingId: finalStagingId,
        storagePath,
      });
      if (!imageId) continue;

      variantsWithIds[k] = {
        imageId,
        storage_path: storagePath,
      };
    }

    stagedWithIds.push({
      photoId: item.photoId,
      photoUrl: item.photoUrl,
      variants: variantsWithIds,
    });
  }

  // Update meta with IDs
  const metaWithIds = {
    ...meta,
    staged: stagedWithIds,
  };

  await updateStaging({
    stagingId: finalStagingId,
    meta: metaWithIds,
    version: nextVersion,
  });

  const allVariantStoragePaths = [];
  for (const item of stagedWithIds) {
    const v = item?.variants || {};
    for (const vv of Object.values(v)) {
      if (vv?.storage_path) allVariantStoragePaths.push(vv.storage_path);
    }
  }

  await updateJobResult({
    jobId,
    resultPayload: {
      stagingId: finalStagingId,
      version: nextVersion,
      staged: stagedWithIds,
      variantsPerPhoto: meta.variantsPerPhoto,
      photoCountSelected: meta.photoCountSelected,
      photoCountStaged: stagedWithIds.length,
      totalVariantImages: allVariantStoragePaths.length,
    },
  });

  return { finalStagingId, nextVersion, stagedWithIds };
}
