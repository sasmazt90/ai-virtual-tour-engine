export function buildStagingMetadata({
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
}) {
  return {
    staged: stagedItems,
    perPhotoAnalyses,
    preferredItems: isVacant ? null : preferredItemsRecognition,
    preferredItemImages: isVacant ? [] : preferredItemImagesClean,
    preferredItemHints: isVacant ? [] : preferredItemHints,
    preferredItemsText: isVacant ? null : preferredItemsText || null,
    // VACANT: furniture refs are forbidden/ignored, so reflect that in meta
    customAssetIdsUsed: isVacant ? [] : customAssetIds,
    hasCustomFurniture: isVacant
      ? false
      : (preferredItemImagesClean?.length || 0) > 0 ||
        (customAssetIds?.length || 0) > 0,
    photoCountSelected: propertyPhotoIds.length,
    photoCountStaged: stagedItems.length,
    variantsPerPhoto,
    totalVariantImages: flatImageUrls.length,
    creditsReserved,
    // helpful debug flag
    stagingMode: stagingType,
    // NEW: attach QA results for VACANT runs to aid debugging and UI visibility
    vacantQaResults: isVacant ? vacantQaResults : undefined,
  };
}
