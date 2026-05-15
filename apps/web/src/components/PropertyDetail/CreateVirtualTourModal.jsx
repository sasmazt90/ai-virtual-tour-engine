import { useCallback, useMemo, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { ModalShell } from "./ModalShell";
import { titleCase } from "@/utils/formatters";

export function CreateVirtualTourModal({
  open,
  onClose,
  propertyId,
  userId,
  stagings,
  tours,
}) {
  const queryClient = useQueryClient();
  const [status, setStatus] = useState("idle"); // idle | loading | error | queued
  const [error, setError] = useState(null);
  const [jobId, setJobId] = useState(null);

  const stagingTypeOptions = useMemo(() => {
    const list = Array.isArray(stagings) ? stagings : [];

    // One option per staging TYPE (property-level slots)
    const typeSet = new Set();
    const orderedTypes = [];

    for (const s of list) {
      const raw =
        typeof s?.staging_type === "string" ? s.staging_type.trim() : "";
      if (!raw) continue;
      if (typeSet.has(raw)) continue;
      typeSet.add(raw);
      orderedTypes.push(raw);
    }

    return orderedTypes.map((t) => {
      return {
        value: `stagingType:${t}`,
        sourceType: "staging",
        stagingType: t,
        label: titleCase(t),
      };
    });
  }, [stagings]);

  const [sourceValue, setSourceValue] = useState("original");

  const selectedSource = useMemo(() => {
    if (sourceValue === "original") {
      return { sourceType: "original", stagingType: null };
    }

    if (
      typeof sourceValue === "string" &&
      sourceValue.startsWith("stagingType:")
    ) {
      const stagingType = sourceValue.replace("stagingType:", "").trim();
      if (stagingType) {
        return { sourceType: "staging", stagingType };
      }
    }

    return { sourceType: "original", stagingType: null };
  }, [sourceValue]);

  const returnUrl = useMemo(() => {
    return propertyId ? `/properties/${propertyId}` : "/properties";
  }, [propertyId]);

  const disableActions = status === "loading";

  const safeOnClose = useCallback(() => {
    if (disableActions) return;
    onClose();
  }, [disableActions, onClose]);

  const tourKeySet = useMemo(() => {
    const set = new Set();
    const list = Array.isArray(tours) ? tours : [];

    for (const t of list) {
      const sourceType =
        t?.source_type === "staging" || t?.source_type === "original"
          ? t.source_type
          : null;

      if (sourceType === "original") {
        set.add("original");
      } else if (sourceType === "staging") {
        const st =
          typeof t?.staging_type === "string" ? t.staging_type.trim() : "";
        if (st) {
          set.add(`staging:${st}`);
        }
      }
    }

    return set;
  }, [tours]);

  const replaceWarning = useMemo(() => {
    if (selectedSource.sourceType === "original") {
      if (tourKeySet.has("original")) {
        return "This will replace the existing Original virtual tour.";
      }
      return null;
    }

    if (selectedSource.sourceType === "staging" && selectedSource.stagingType) {
      if (tourKeySet.has(`staging:${selectedSource.stagingType}`)) {
        return `This will replace the existing ${titleCase(
          selectedSource.stagingType,
        )} virtual tour.`;
      }
    }

    return null;
  }, [selectedSource.sourceType, selectedSource.stagingType, tourKeySet]);

  const selectedStagingId = useMemo(() => {
    if (
      selectedSource.sourceType !== "staging" ||
      !selectedSource.stagingType
    ) {
      return null;
    }

    const list = Array.isArray(stagings) ? stagings : [];
    const want = String(selectedSource.stagingType || "")
      .toLowerCase()
      .trim();

    const found = list.find((s) => {
      const have = String(s?.staging_type || "")
        .toLowerCase()
        .trim();
      return have && have === want;
    });

    return found?.id || null;
  }, [selectedSource.sourceType, selectedSource.stagingType, stagings]);

  const { data: jobData } = useQuery({
    queryKey: ["aiJob", jobId],
    queryFn: async () => {
      const res = await fetch(`/api/ai/jobs/${jobId}`);
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body?.error || "Could not fetch job");
      }
      return res.json();
    },
    enabled: !!jobId,
    refetchInterval: (q) => {
      const st = q?.state?.data?.status;
      if (st === "succeeded" || st === "failed") {
        return false;
      }
      return 2000;
    },
  });

  const onGenerate = useCallback(async () => {
    if (!propertyId) {
      setError("Missing property ID");
      setStatus("error");
      return;
    }

    if (selectedSource.sourceType === "staging" && !selectedStagingId) {
      setError("Please choose a staging source.");
      setStatus("error");
      return;
    }

    setError(null);
    setStatus("loading");
    setJobId(null);

    try {
      const baseView =
        selectedSource.sourceType === "staging"
          ? { type: "staging", stagingId: selectedStagingId }
          : { type: "default", stagingId: null };

      const res = await fetch("/api/ai/virtual-tour/create", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ propertyId, baseView }),
      });

      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(
          body?.error ||
            `When posting /api/ai/virtual-tour/create, the response was [${res.status}] ${res.statusText}`,
        );
      }

      const json = await res.json();
      const nextJobId =
        (typeof json?.jobId === "string" && json.jobId.trim()) || null;

      if (!nextJobId) {
        throw new Error("Server did not return a jobId");
      }

      setJobId(nextJobId);
      setStatus("queued");

      // ensure UI updates once job completes
      queryClient.invalidateQueries({ queryKey: ["aiJobs"] });
    } catch (err) {
      console.error(err);
      const msg = err instanceof Error ? err.message : null;
      const text =
        msg && msg.trim()
          ? msg
          : "We couldn’t generate a virtual tour. Please try again.";
      setError(text);
      setStatus("error");
    }
  }, [propertyId, queryClient, selectedSource.sourceType, selectedStagingId]);

  const effectiveProgress = useMemo(() => {
    const p = Number(jobData?.progress || 0);
    return Number.isFinite(p) ? p : 0;
  }, [jobData?.progress]);

  const effectiveJobStatus = jobData?.status || null;

  const done =
    effectiveJobStatus === "succeeded" || effectiveJobStatus === "failed";

  const onDoneRefresh = useCallback(async () => {
    // FIX: refresh property assets so the new tour appears in the list
    if (userId && propertyId) {
      await queryClient.invalidateQueries({
        queryKey: ["property", userId, propertyId],
      });
    } else {
      await queryClient.invalidateQueries({ queryKey: ["property"] });
    }

    onClose();
  }, [onClose, propertyId, queryClient, userId]);

  if (!open) {
    return null;
  }

  const primaryLabel =
    status === "error" ? "Try again" : "Generate Virtual Tour";

  return (
    <ModalShell title="Create Virtual Tour" onClose={safeOnClose}>
      <div className="space-y-4">
        <div>
          <label className="block text-sm font-medium text-gray-900 dark:text-gray-100 font-jetbrains-mono">
            Source
          </label>
          <div className="mt-1">
            <select
              value={sourceValue}
              onChange={(e) => setSourceValue(e.target.value)}
              disabled={disableActions}
              className="w-full rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-[#1E1E1E] text-gray-900 dark:text-gray-100 px-3 py-2 text-sm font-jetbrains-mono"
            >
              <option value="original">Original photos</option>
              {stagingTypeOptions.map((opt) => (
                <option key={opt.value} value={opt.value}>
                  {opt.label}
                </option>
              ))}
            </select>
          </div>
        </div>

        {replaceWarning ? (
          <div className="text-sm text-amber-700 dark:text-amber-300 font-jetbrains-mono">
            {replaceWarning}
          </div>
        ) : null}

        <div className="text-sm text-gray-700 dark:text-gray-200 font-jetbrains-mono">
          A virtual tour will be generated using the photos already added to
          this property.
        </div>

        {status === "loading" ? (
          <div className="text-sm text-gray-700 dark:text-gray-200 font-jetbrains-mono">
            Starting job…
          </div>
        ) : null}

        {jobId ? (
          <div className="rounded-lg border border-gray-200 dark:border-gray-700 p-3">
            <div className="text-sm text-gray-800 dark:text-gray-100 font-jetbrains-mono">
              Job: {jobData?.status || "queued"} • {effectiveProgress}%
            </div>
            {jobData?.error ? (
              <div className="mt-2 text-sm text-red-600 dark:text-red-400 font-jetbrains-mono">
                {jobData.error}
              </div>
            ) : null}

            {done ? (
              <div className="mt-3 flex items-center justify-end gap-2">
                <button
                  type="button"
                  onClick={onDoneRefresh}
                  className="inline-flex items-center px-3 py-2 rounded-lg bg-gray-900 dark:bg-gray-100 text-white dark:text-gray-900 text-sm font-medium hover:bg-gray-800 dark:hover:bg-gray-200 font-jetbrains-mono"
                >
                  Done
                </button>
              </div>
            ) : (
              <div className="mt-2 text-xs text-gray-500 dark:text-gray-500 font-jetbrains-mono">
                This can take a minute. You can close this window; the job will
                keep running.
              </div>
            )}
          </div>
        ) : null}

        {error ? (
          <div className="text-sm text-red-600 dark:text-red-400 font-jetbrains-mono">
            {error}
          </div>
        ) : null}

        <div className="flex items-center justify-end gap-2">
          <button
            type="button"
            onClick={safeOnClose}
            disabled={disableActions}
            className="inline-flex items-center px-3 py-2 rounded-lg border border-gray-200 dark:border-gray-700 text-gray-700 dark:text-gray-200 text-sm font-medium hover:bg-gray-50 dark:hover:bg-gray-800 disabled:opacity-50 font-jetbrains-mono"
          >
            Cancel
          </button>
          <button
            type="button"
            onClick={onGenerate}
            disabled={disableActions}
            className="inline-flex items-center px-3 py-2 rounded-lg bg-gray-900 dark:bg-gray-100 text-white dark:text-gray-900 text-sm font-medium hover:bg-gray-800 dark:hover:bg-gray-200 disabled:opacity-50 font-jetbrains-mono"
          >
            {primaryLabel}
          </button>
        </div>
      </div>
    </ModalShell>
  );
}
