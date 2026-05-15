import { useCallback, useState } from "react";
import { pollJob } from "@/utils/e2eTestHelpers";

export function useVacantQuickTest({
  selectedPropertyId,
  firstPhotoId,
  queryClient,
}) {
  const [vacantQuick, setVacantQuick] = useState({
    status: "idle",
    jobId: null,
    dayUrl: null,
    nightUrl: null,
    error: null,
  });

  const runVacantQuickTest = useCallback(async () => {
    if (!selectedPropertyId) {
      setVacantQuick({
        status: "fail",
        jobId: null,
        dayUrl: null,
        nightUrl: null,
        error: "Pick a property first.",
      });
      return;
    }

    if (!firstPhotoId) {
      setVacantQuick({
        status: "fail",
        jobId: null,
        dayUrl: null,
        nightUrl: null,
        error: "This property has no photos. Add photos and try again.",
      });
      return;
    }

    setVacantQuick({
      status: "queued",
      jobId: null,
      dayUrl: null,
      nightUrl: null,
      error: null,
    });

    try {
      const res = await fetch("/api/ai/staging/create", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          propertyId: selectedPropertyId,
          stagingType: "vacant",
          propertyPhotoIds: [firstPhotoId],
          customAssetIds: [],
          preferredItemImages: [],
          preferredItemHints: [],
          preferredItemsText: "",
          useCrossPhotoConsistency: false,
        }),
      });

      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body?.error || "Could not start VACANT test");
      }

      const json = await res.json();
      const jobId = json?.jobId;
      if (!jobId) {
        throw new Error("VACANT test did not return a jobId");
      }

      setVacantQuick({
        status: "running",
        jobId,
        dayUrl: null,
        nightUrl: null,
        error: null,
      });

      const done = await pollJob({ jobId, timeoutMs: 6 * 60 * 1000 });

      if (done?.status !== "succeeded") {
        const msg = done?.error || "Job failed";
        setVacantQuick({
          status: "fail",
          jobId,
          dayUrl: null,
          nightUrl: null,
          error: msg,
        });
        return;
      }

      const staged = Array.isArray(done?.result?.staged)
        ? done.result.staged
        : [];
      const first = staged[0] || null;
      const variants =
        first?.variants && typeof first.variants === "object"
          ? first.variants
          : {};

      const dayUrl = variants?.day_light_off?.storage_path || null;
      const nightUrl = variants?.night_light_off?.storage_path || null;

      setVacantQuick({
        status: "pass",
        jobId,
        dayUrl,
        nightUrl,
        error: null,
      });

      await queryClient.invalidateQueries({ queryKey: ["property"] });
      await queryClient.invalidateQueries({ queryKey: ["properties"] });
      await queryClient.invalidateQueries({ queryKey: ["properties", "e2e"] });
    } catch (err) {
      console.error(err);
      const msg = err instanceof Error ? err.message : "VACANT test failed";
      setVacantQuick({
        status: "fail",
        jobId: null,
        dayUrl: null,
        nightUrl: null,
        error: msg,
      });
    }
  }, [firstPhotoId, queryClient, selectedPropertyId]);

  return {
    vacantQuick,
    runVacantQuickTest,
  };
}
