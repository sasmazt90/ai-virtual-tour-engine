import { useCallback, useState } from "react";
import { pollJob } from "@/utils/e2eTestHelpers";

export function useE2ETests({
  selectedPropertyId,
  firstPhotoId,
  customAssetIds,
  hasCustomAssets,
  furnitureFile,
  upload,
  queryClient,
}) {
  const [e2eRunning, setE2eRunning] = useState(false);
  const [e2eError, setE2eError] = useState(null);
  const [results, setResults] = useState([]);

  const pushResult = useCallback((next) => {
    setResults((prev) => {
      const out = Array.isArray(prev) ? [...prev] : [];
      out.push(next);
      return out;
    });
  }, []);

  const updateLastResult = useCallback((patch) => {
    setResults((prev) => {
      const out = Array.isArray(prev) ? [...prev] : [];
      if (out.length === 0) return out;
      const last = out[out.length - 1];
      out[out.length - 1] = { ...last, ...patch };
      return out;
    });
  }, []);

  const runStaging = useCallback(
    async ({ label, stagingType, withCustomFurniture }) => {
      const neededCustom = withCustomFurniture === true;

      let ensuredCustomAssetIds = [];

      if (neededCustom) {
        if (hasCustomAssets) {
          ensuredCustomAssetIds = [customAssetIds[0]];
        } else if (furnitureFile) {
          const up = await upload({ file: furnitureFile });
          if (up?.error) {
            throw new Error(up.error || "Could not upload custom furniture");
          }

          const resSave = await fetch(
            `/api/properties/${encodeURIComponent(selectedPropertyId)}/custom-assets`,
            {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({
                storage_path: up.url,
                label: furnitureFile?.name || "custom furniture",
              }),
            },
          );

          if (!resSave.ok) {
            const body = await resSave.json().catch(() => ({}));
            throw new Error(body?.error || "Could not save custom furniture");
          }

          const saved = await resSave.json().catch(() => null);
          const newId = saved?.id || null;
          if (newId) {
            ensuredCustomAssetIds = [newId];
          }

          await queryClient.invalidateQueries({
            queryKey: ["custom-assets", "e2e", selectedPropertyId],
          });
        } else {
          throw new Error(
            "Modern + Custom Furniture test needs at least 1 custom asset. Upload one furniture image in this page first.",
          );
        }
      }

      pushResult({
        test: label,
        status: "queued",
        jobId: null,
        details: "Starting…",
      });

      const res = await fetch("/api/ai/staging/create", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          propertyId: selectedPropertyId,
          stagingType,
          propertyPhotoIds: [firstPhotoId],
          customAssetIds: ensuredCustomAssetIds,
          preferredItemImages: [],
          preferredItemHints: [],
          preferredItemsText: "",
        }),
      });

      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body?.error || `Could not start staging (${label})`);
      }

      const json = await res.json();
      const jobId = json?.jobId;
      if (!jobId) {
        throw new Error("Staging did not return a jobId");
      }

      updateLastResult({ status: "running", jobId, details: "Running…" });

      const done = await pollJob({ jobId, timeoutMs: 6 * 60 * 1000 });

      if (done?.status !== "succeeded") {
        const msg = done?.error || "Job failed";
        updateLastResult({ status: "fail", details: msg });
        return;
      }

      updateLastResult({ status: "pass", details: "Succeeded" });
    },
    [
      customAssetIds,
      firstPhotoId,
      furnitureFile,
      hasCustomAssets,
      pushResult,
      queryClient,
      selectedPropertyId,
      updateLastResult,
      upload,
    ],
  );

  const runTour = useCallback(async () => {
    pushResult({
      test: "Test 4: Virtual Tour = Default",
      status: "queued",
      jobId: null,
      details: "Starting…",
    });

    const res = await fetch("/api/ai/virtual-tour/create", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        propertyId: selectedPropertyId,
        baseView: { type: "default", stagingId: null },
      }),
    });

    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error(body?.error || "Could not start virtual tour");
    }

    const json = await res.json();
    const jobId = json?.jobId;
    if (!jobId) {
      throw new Error("Virtual tour did not return a jobId");
    }

    updateLastResult({ status: "running", jobId, details: "Running…" });

    const done = await pollJob({ jobId, timeoutMs: 6 * 60 * 1000 });

    if (done?.status !== "succeeded") {
      const msg = done?.error || "Job failed";
      updateLastResult({ status: "fail", details: msg });
      return;
    }

    updateLastResult({ status: "pass", details: "Succeeded" });
  }, [pushResult, selectedPropertyId, updateLastResult]);

  const runE2E = useCallback(async () => {
    if (!selectedPropertyId) {
      setE2eError("Pick a property first.");
      return;
    }

    if (!firstPhotoId) {
      setE2eError("This property has no photos. Add photos and try again.");
      return;
    }

    setE2eRunning(true);
    setE2eError(null);
    setResults([]);

    try {
      await runStaging({
        label: "Test 1: Staging = Vacant",
        stagingType: "vacant",
        withCustomFurniture: false,
      });

      await runStaging({
        label: "Test 2: Staging = Luxury",
        stagingType: "luxury",
        withCustomFurniture: false,
      });

      await runStaging({
        label: "Test 3: Staging = Modern + Custom Furniture",
        stagingType: "modern",
        withCustomFurniture: true,
      });

      await runTour();

      await queryClient.invalidateQueries({ queryKey: ["property"] });
      await queryClient.invalidateQueries({ queryKey: ["properties"] });
      await queryClient.invalidateQueries({ queryKey: ["properties", "e2e"] });
      await queryClient.invalidateQueries({
        queryKey: ["custom-assets", "e2e", selectedPropertyId],
      });
    } catch (err) {
      console.error(err);
      const msg = err instanceof Error ? err.message : "E2E tests failed";
      setE2eError(msg);
    } finally {
      setE2eRunning(false);
    }
  }, [firstPhotoId, queryClient, runStaging, runTour, selectedPropertyId]);

  return {
    e2eRunning,
    e2eError,
    results,
    runE2E,
  };
}
