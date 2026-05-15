import { useCallback, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import useUpload from "@/utils/useUpload";

export function useAIStudio(userId, propertyId, property) {
  const queryClient = useQueryClient();
  const [aiStudioOpen, setAiStudioOpen] = useState(false);
  const [stagingType, setStagingType] = useState("luxury");
  const [customAssetFiles, setCustomAssetFiles] = useState([]);
  const [preferredItemFiles, setPreferredItemFiles] = useState([]);
  const [aiError, setAiError] = useState(null);
  const [stagingJobId, setStagingJobId] = useState(null);
  const [tourBase, setTourBase] = useState("default");
  const [tourJobId, setTourJobId] = useState(null);

  // NEW (policy/UI2): remember the exact backend-reported credit cost for the latest generation.
  const [lastStagingCreditCost, setLastStagingCreditCost] = useState(null);
  const [lastTourCreditCost, setLastTourCreditCost] = useState(null);

  const [upload, { loading: preferredItemsUploading }] = useUpload();

  const onPickPreferredItemFiles = useCallback(
    (filesOrEvent) => {
      const files = (() => {
        if (Array.isArray(filesOrEvent)) return filesOrEvent;
        const list = filesOrEvent?.target?.files;
        return list ? Array.from(list) : [];
      })();

      const allowedTypes = new Set(["image/jpeg", "image/png", "image/webp"]);

      const accepted = [];
      for (const f of files) {
        if (!f) continue;
        // Some browsers may not set type reliably; still prefer type when available.
        if (f.type && !allowedTypes.has(f.type)) {
          continue;
        }
        accepted.push(f);
      }

      if (accepted.length !== files.length) {
        setAiError(
          "Some files were skipped. Preferred items must be JPG, PNG, or WEBP.",
        );
      }

      setPreferredItemFiles((prev) => [...prev, ...accepted]);

      // allow re-selecting same file
      if (filesOrEvent?.target) {
        filesOrEvent.target.value = "";
      }
    },
    [setAiError],
  );

  const onRemovePreferredItemFile = useCallback((index) => {
    setPreferredItemFiles((prev) => prev.filter((_, i) => i !== index));
  }, []);

  const { data: customAssets = [], refetch: refetchCustomAssets } = useQuery({
    queryKey: ["custom-assets", userId, propertyId],
    queryFn: async () => {
      const res = await fetch(`/api/properties/${propertyId}/custom-assets`);
      if (!res.ok) {
        // user-facing copy should stay calm/non-technical
        throw new Error("Could not load custom assets.");
      }
      return res.json();
    },
    enabled: !!userId && !!propertyId && aiStudioOpen,
  });

  const { data: jobData } = useQuery({
    queryKey: ["ai-job", userId, stagingJobId],
    queryFn: async () => {
      if (!stagingJobId) return null;
      const res = await fetch(`/api/ai/jobs/${stagingJobId}`);
      if (!res.ok) {
        throw new Error("Could not load job status.");
      }
      return res.json();
    },
    enabled: !!userId && !!stagingJobId,
    refetchInterval: (query) => {
      const status = query?.state?.data?.status;
      if (status === "queued" || status === "running") {
        return 2000;
      }
      return false;
    },
  });

  const { data: tourJobData } = useQuery({
    queryKey: ["ai-tour-job", userId, tourJobId],
    queryFn: async () => {
      if (!tourJobId) return null;
      const res = await fetch(`/api/ai/jobs/${tourJobId}`);
      if (!res.ok) {
        throw new Error("Could not load job status.");
      }
      return res.json();
    },
    enabled: !!userId && !!tourJobId,
    refetchInterval: (query) => {
      const status = query?.state?.data?.status;
      if (status === "queued" || status === "running") {
        return 2000;
      }
      return false;
    },
  });

  const jobStatus = jobData?.status || null;
  const jobProgress = jobData?.progress ?? 0;
  const jobError = jobData?.error || null;

  const tourJobStatus = tourJobData?.status || null;
  const tourJobProgress = tourJobData?.progress ?? 0;
  const tourJobError = tourJobData?.error || null;

  const retryJobMutation = useMutation({
    mutationFn: async (id) => {
      const res = await fetch(`/api/ai/jobs/${id}/retry`, { method: "POST" });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));

        if (res.status === 402) {
          throw new Error(
            body?.error || "Not enough credits to retry this job.",
          );
        }

        if (res.status === 409) {
          throw new Error(
            body?.error ||
              "A retry is already in progress. Please wait a moment.",
          );
        }

        if (res.status === 401) {
          throw new Error("Please sign in again to retry this job.");
        }

        throw new Error(
          body?.error || "Could not retry the job. Please try again.",
        );
      }
      return res.json();
    },
    onError: (err) => {
      console.error(err);
      setAiError(err?.message || "Could not retry the job.");
    },
    onSuccess: async () => {
      await queryClient.invalidateQueries({ queryKey: ["credits", userId] });
    },
  });

  const createStagingMutation = useMutation({
    mutationFn: async ({ stagingType: chosenType }) => {
      const photoIds = (property?.photos || []).map((p) => p.id);
      const customAssetIds = (customAssets || []).map((a) => a.id);

      let preferredItemImages = [];
      if (preferredItemFiles.length > 0) {
        const uploaded = [];
        for (const file of preferredItemFiles) {
          const out = await upload({ file });
          if (out?.error) {
            throw new Error(
              out.error || "Could not upload preferred item images.",
            );
          }
          uploaded.push({ url: out.url, mimeType: out.mimeType });
        }
        preferredItemImages = uploaded;
      }

      const res = await fetch("/api/ai/staging/create", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          propertyId,
          stagingType: chosenType,
          propertyPhotoIds: photoIds,
          customAssetIds,
          preferredItemImages,
        }),
      });

      if (!res.ok) {
        const body = await res.json().catch(() => ({}));

        if (res.status === 402) {
          const message =
            body?.error || "Not enough credits to start this job.";
          throw new Error(message);
        }

        if (res.status === 401) {
          throw new Error("Please sign in again to start this job.");
        }

        throw new Error(
          body?.error || "Could not start staging. Please try again.",
        );
      }

      return res.json();
    },
    onSuccess: (data) => {
      setAiError(null);
      setStagingJobId(data.jobId);
      setLastStagingCreditCost(
        Number(data?.creditCost ?? data?.creditsReserved ?? 0) || null,
      );
      queryClient.invalidateQueries({ queryKey: ["credits", userId] });
    },
    onError: (err) => {
      console.error(err);
      setAiError(err?.message || "Could not start staging.");
    },
  });

  const createTourMutation = useMutation({
    mutationFn: async () => {
      const baseView = (() => {
        if (tourBase && tourBase.startsWith("staging:")) {
          const stagingId = tourBase.replace("staging:", "");
          return { type: "staging", stagingId };
        }
        return { type: "default", stagingId: null };
      })();

      const res = await fetch("/api/ai/virtual-tour/create", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          propertyId,
          baseView,
        }),
      });

      if (!res.ok) {
        const body = await res.json().catch(() => ({}));

        if (res.status === 402) {
          const message =
            body?.error || "Not enough credits to start this job.";
          throw new Error(message);
        }

        if (res.status === 401) {
          throw new Error("Please sign in again to start this job.");
        }

        throw new Error(
          body?.error || "Could not start the virtual tour. Please try again.",
        );
      }

      return res.json();
    },
    onSuccess: (data) => {
      setAiError(null);
      setTourJobId(data.jobId);
      setLastTourCreditCost(
        Number(data?.creditCost ?? data?.creditsReserved ?? 0) || null,
      );
      queryClient.invalidateQueries({ queryKey: ["credits", userId] });
    },
    onError: (err) => {
      console.error(err);
      setAiError(err?.message || "Could not start the virtual tour.");
    },
  });

  const onOpenAiStudio = useCallback(() => {
    setAiError(null);
    setStagingJobId(null);
    setTourJobId(null);
    setLastStagingCreditCost(null);
    setLastTourCreditCost(null);
    setAiStudioOpen(true);
  }, []);

  const onCloseAiStudio = useCallback(() => {
    setAiStudioOpen(false);
  }, []);

  const stagingJobDone = jobStatus === "succeeded" || jobStatus === "failed";
  const tourJobDone =
    tourJobStatus === "succeeded" || tourJobStatus === "failed";

  return {
    aiStudioOpen,
    setAiStudioOpen,
    stagingType,
    setStagingType,
    customAssetFiles,
    setCustomAssetFiles,
    preferredItemFiles,
    setPreferredItemFiles,
    preferredItemsUploading,
    onPickPreferredItemFiles,
    onRemovePreferredItemFile,
    aiError,
    setAiError,
    stagingJobId,
    setStagingJobId,
    tourBase,
    setTourBase,
    tourJobId,
    setTourJobId,
    customAssets,
    refetchCustomAssets,
    jobData,
    jobStatus,
    jobProgress,
    jobError,
    tourJobData,
    tourJobStatus,
    tourJobProgress,
    tourJobError,
    retryJobMutation,
    createStagingMutation,
    createTourMutation,
    onOpenAiStudio,
    onCloseAiStudio,
    stagingJobDone,
    tourJobDone,
    lastStagingCreditCost,
    lastTourCreditCost,
  };
}
