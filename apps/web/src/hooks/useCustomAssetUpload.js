import { useCallback } from "react";
import useUpload from "@/utils/useUpload";

export function useCustomAssetUpload(
  propertyId,
  customAssetFiles,
  setCustomAssetFiles,
  setAiError,
  refetchCustomAssets,
) {
  const [upload, { loading: uploading }] = useUpload();

  const onPickCustomAssets = useCallback(
    (e) => {
      const files = Array.from(e.target.files || []);
      setCustomAssetFiles(files);
    },
    [setCustomAssetFiles],
  );

  const uploadCustomAssets = useCallback(async () => {
    setAiError(null);
    if (customAssetFiles.length === 0) return;

    for (const file of customAssetFiles) {
      const { url, error } = await upload({ file });
      if (error) {
        throw new Error(error);
      }

      const res = await fetch(`/api/properties/${propertyId}/custom-assets`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ storage_path: url, label: file.name || null }),
      });

      if (!res.ok) {
        throw new Error(
          `When posting /api/properties/${propertyId}/custom-assets, the response was [${res.status}] ${res.statusText}`,
        );
      }
    }

    setCustomAssetFiles([]);
    await refetchCustomAssets();
  }, [
    customAssetFiles,
    propertyId,
    refetchCustomAssets,
    upload,
    setCustomAssetFiles,
    setAiError,
  ]);

  const onUploadCustomAssetsClick = useCallback(async () => {
    try {
      await uploadCustomAssets();
    } catch (err) {
      console.error(err);
      setAiError(err?.message || "Could not upload custom assets");
    }
  }, [uploadCustomAssets, setAiError]);

  return {
    uploading,
    onPickCustomAssets,
    onUploadCustomAssetsClick,
  };
}
