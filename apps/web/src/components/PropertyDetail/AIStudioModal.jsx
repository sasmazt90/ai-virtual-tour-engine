import { Loader2, Sparkles } from "lucide-react";
import { useEffect, useMemo, useRef } from "react";
import { ModalShell } from "./ModalShell";
import { useAIBusy } from "@/hooks/useAIBusy";
import { StatusBanner } from "@/components/StatusBanner";

const STAGING_TYPES = [
  "classic",
  "default",
  "luxury",
  "minimalist",
  "modern",
  "scandinavian",
  "vacant",
];

// REQUIRED FIX: keep this note stable and rendered in exactly one place per modal open
const BUSY_QUEUE_NOTE_TEXT =
  "Jobs are processed in order. Your job may start shortly.";

export function AIStudioModal({
  aiStudioOpen,
  onCloseAiStudio,
  userId,
  creditsBalance,
  estimatedStagingCredits,
  canRunStaging,
  stagingType,
  setStagingType,
  preferredItemFiles,
  preferredItemsUploading,
  onPickPreferredItemFiles,
  onRemovePreferredItemFile,
  onPickCustomAssets,
  uploading,
  customAssetFiles,
  onUploadCustomAssetsClick,
  customAssets,
  jobStatus,
  jobProgress,
  jobError,
  stagingJobId,
  retryJobMutation,
  setAiError,
  setStagingJobId,
  createStagingMutation,
  stagingJobDone,
  onRefreshAfterJob,
  property,
  formatStagingLabel,
  aiError,
  lastStagingCreditCost,
}) {
  const { data: busyData } = useAIBusy(userId, {
    enabled: !!userId && !!aiStudioOpen,
    refetchInterval: 10000,
  });

  const showBusy = busyData?.busy === true;
  const busyHint = showBusy
    ? "You already have jobs running; new jobs may queue."
    : null;
  const isJobActive = jobStatus === "queued" || jobStatus === "running";
  const autoRefreshedJobRef = useRef(null);

  // STEP 15C: additional subtle note when busy
  // REQUIRED FIX: memoize the rendered note so it won't duplicate across re-renders/refetches
  const busyQueueNoteNode = useMemo(() => {
    if (!showBusy) return null;
    return (
      <div className="mt-2 text-xs text-gray-500 dark:text-gray-500 font-jetbrains-mono">
        {BUSY_QUEUE_NOTE_TEXT}
      </div>
    );
  }, [showBusy]);

  const stagingReplaceWarning = useMemo(() => {
    const list = Array.isArray(property?.stagings) ? property.stagings : [];
    const exists = list.some((s) => s?.staging_type === stagingType);
    if (!exists) return null;
    return `This will replace the existing ${stagingType} staging.`;
  }, [property?.stagings, stagingType]);

  useEffect(() => {
    if (!aiStudioOpen) return;
    if (!stagingJobId) return;
    if (jobStatus !== "succeeded") return;
    if (autoRefreshedJobRef.current === stagingJobId) return;

    autoRefreshedJobRef.current = stagingJobId;
    onRefreshAfterJob?.();
  }, [aiStudioOpen, jobStatus, onRefreshAfterJob, stagingJobId]);

  if (!aiStudioOpen) return null;

  return (
    <ModalShell title="AI Studio" onClose={onCloseAiStudio}>
      <div className="space-y-10">
        <div className="rounded-lg border border-gray-200 dark:border-gray-700 p-4 bg-gray-50 dark:bg-gray-800">
          <div className="flex items-center justify-between gap-4">
            <div>
              <div className="flex items-center gap-2">
                <div className="text-sm font-semibold text-gray-900 dark:text-gray-100 font-jetbrains-mono">
                  Credits
                </div>
                {showBusy ? (
                  <span className="inline-flex items-center justify-center px-2 py-1 rounded-full text-xs leading-none whitespace-nowrap bg-amber-50 dark:bg-amber-900/20 text-amber-800 dark:text-amber-200 font-jetbrains-mono">
                    AI processing in progress
                  </span>
                ) : null}
              </div>
              <div className="mt-1 text-sm text-gray-600 dark:text-gray-300 font-jetbrains-mono">
                Balance: {creditsBalance.toLocaleString()} - Total cost:{" "}
                {Number(estimatedStagingCredits || 0).toLocaleString()}
              </div>
              {busyQueueNoteNode}
            </div>
            <a
              href="/credits"
              className="text-sm text-[var(--brandDark)] dark:text-[var(--brand)] hover:underline font-jetbrains-mono"
            >
              Buy credits
            </a>
          </div>
          {!canRunStaging ? (
            <div className="mt-3 text-sm text-red-600 dark:text-red-400 font-jetbrains-mono">
              Not enough credits for staging.
            </div>
          ) : null}
        </div>

        {/* STAGING */}
        <div className="space-y-6">
          <h4 className="text-sm font-semibold text-gray-900 dark:text-gray-100 font-jetbrains-mono">
            Staging
          </h4>

          {/* Furniture references */}
          <div className="rounded-lg border border-gray-200 dark:border-gray-700 p-4 bg-white dark:bg-gray-900">
            <div className="text-sm font-semibold text-gray-900 dark:text-gray-100 font-jetbrains-mono">
              Furniture references (optional)
            </div>
            <div className="mt-1 text-xs text-gray-500 dark:text-gray-400 font-jetbrains-mono">
              Add item photos only when you want them included in the room.
            </div>

            <div className="mt-3">
              <input
                type="file"
                accept="image/jpeg,image/png,image/webp,.jpg,.jpeg,.png,.webp"
                multiple
                onChange={onPickPreferredItemFiles}
                className="block w-full text-sm text-gray-600 dark:text-gray-300 font-jetbrains-mono"
              />
              {preferredItemsUploading ? (
                <div className="mt-2 text-xs text-gray-500 dark:text-gray-500 font-jetbrains-mono">
                  Uploading preferred items...
                </div>
              ) : null}

              {preferredItemFiles && preferredItemFiles.length > 0 ? (
                <div className="mt-3 grid grid-cols-3 gap-2">
                  {preferredItemFiles.map((f, idx) => {
                    const previewUrl =
                      typeof window !== "undefined" && f
                        ? URL.createObjectURL(f)
                        : null;

                    return (
                      <div
                        key={`${f?.name || "file"}-${idx}`}
                        className="relative rounded-lg overflow-hidden border border-gray-200 dark:border-gray-700"
                      >
                        {previewUrl ? (
                          <img
                            src={previewUrl}
                            alt={f?.name || "Preferred item"}
                            className="w-full h-20 object-cover"
                            onLoad={() => {
                              try {
                                URL.revokeObjectURL(previewUrl);
                              } catch {
                                // ignore
                              }
                            }}
                          />
                        ) : (
                          <div className="w-full h-20 flex items-center justify-center text-xs text-gray-500 font-jetbrains-mono">
                            Image
                          </div>
                        )}

                        <button
                          type="button"
                          onClick={() => onRemovePreferredItemFile(idx)}
                          className="absolute top-1 right-1 inline-flex items-center justify-center h-6 w-6 rounded-full bg-white/90 dark:bg-gray-800/90 border border-gray-200 dark:border-gray-700 text-xs text-gray-800 dark:text-gray-100"
                          aria-label="Remove"
                        >
                          X
                        </button>
                      </div>
                    );
                  })}
                </div>
              ) : (
                <div className="mt-3 text-xs text-gray-500 dark:text-gray-500 font-jetbrains-mono">
                  No preferred items yet.
                </div>
              )}
            </div>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
            <div className="space-y-2">
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 font-jetbrains-mono">
                Staging Type
              </label>
              <select
                value={stagingType}
                onChange={(e) => setStagingType(e.target.value)}
                className="w-full px-4 py-3 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-600 rounded-lg text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-[var(--brand)] font-jetbrains-mono"
              >
                {STAGING_TYPES.map((t) => (
                  <option key={t} value={t}>
                    {t}
                  </option>
                ))}
              </select>

              {stagingReplaceWarning ? (
                <div className="text-xs text-amber-700 dark:text-amber-300 font-jetbrains-mono">
                  {stagingReplaceWarning}
                </div>
              ) : null}
            </div>

            <div className="space-y-2">
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 font-jetbrains-mono">
                Furniture references
              </label>
              <input
                type="file"
                accept="image/*"
                multiple
                onChange={onPickCustomAssets}
                className="block w-full text-sm text-gray-600 dark:text-gray-300 font-jetbrains-mono"
              />
              <button
                type="button"
                disabled={uploading || customAssetFiles.length === 0}
                onClick={onUploadCustomAssetsClick}
                className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-gray-900 dark:bg-gray-100 text-white dark:text-gray-900 text-sm font-medium hover:bg-gray-800 dark:hover:bg-gray-200 disabled:opacity-50 font-jetbrains-mono"
              >
                {uploading ? (
                  <Loader2 size={16} className="animate-spin" />
                ) : null}
                Upload items
              </button>
              {customAssets.length > 0 ? (
                <div className="mt-3 grid grid-cols-3 gap-2">
                  {customAssets.map((a) => (
                    <div
                      key={a.id}
                      className="rounded-lg overflow-hidden border border-gray-200 dark:border-gray-700"
                    >
                      <img
                        src={a.storage_path}
                        alt={a.label || "Asset"}
                        className="w-full h-20 object-cover"
                      />
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-xs text-gray-500 dark:text-gray-500 font-jetbrains-mono">
                  No furniture references yet.
                </div>
              )}
            </div>
          </div>

          <div className="flex flex-col sm:flex-row gap-3 items-start sm:items-center justify-between">
            <div className="text-sm text-gray-600 dark:text-gray-300 font-jetbrains-mono">
              {jobStatus
                ? `${jobStatus
                    .replaceAll("_", " ")
                    .replace(/^\w/, (c) => c.toUpperCase())} - ${Number(
                    jobProgress || 0,
                  )}%`
                : "Ready when you are."}
              {jobError
                ? " - We could not create this staging. Try again with a clearer photo or fewer furniture references."
                : ""}
            </div>
            <div className="flex items-center gap-2">
              {jobStatus === "failed" && stagingJobId ? (
                <button
                  type="button"
                  disabled={retryJobMutation.isPending}
                  onClick={async () => {
                    setAiError(null);
                    const data =
                      await retryJobMutation.mutateAsync(stagingJobId);
                    setStagingJobId(data.jobId);
                  }}
                  className="inline-flex items-center justify-center gap-2 px-4 py-3 bg-gray-900 dark:bg-gray-100 text-white dark:text-gray-900 rounded-lg font-medium transition-colors disabled:opacity-50 font-jetbrains-mono"
                >
                  Retry
                </button>
              ) : null}

              <div className="flex flex-col items-end">
                <button
                  type="button"
                  disabled={
                    createStagingMutation.isPending ||
                    retryJobMutation.isPending ||
                    isJobActive ||
                    (jobStatus !== "succeeded" && !canRunStaging)
                  }
                  onClick={async () => {
                    if (jobStatus === "succeeded") {
                      await onRefreshAfterJob?.();
                      onCloseAiStudio?.();
                      return;
                    }
                    createStagingMutation.mutate({ stagingType });
                  }}
                  className="inline-flex items-center justify-center gap-2 px-6 py-3 bg-[var(--brand)] hover:bg-[var(--brandHover)] text-white rounded-lg font-medium transition-colors disabled:opacity-50 font-jetbrains-mono"
                >
                  {createStagingMutation.isPending || isJobActive ? (
                    <Loader2 size={18} className="animate-spin" />
                  ) : (
                    <Sparkles size={18} />
                  )}
                  {isJobActive
                    ? "Generating..."
                    : jobStatus === "succeeded"
                      ? "View in Assets"
                      : "Generate Staging"}
                </button>
              </div>
            </div>
          </div>

          {/* UX policy confirmation:
              There is no additional cost for preview, saving, or discarding results.
              Credits are consumed only when you click Generate. */}
          <div className="text-xs text-gray-500 dark:text-gray-500 font-jetbrains-mono">
            Credits are consumed when you click Generate. Saving or discarding
            results is free.
          </div>

          {lastStagingCreditCost ? (
            <div className="text-xs text-gray-600 dark:text-gray-400 font-jetbrains-mono">
              Credits have been consumed for this generation (
              {Number(lastStagingCreditCost).toLocaleString()}).
            </div>
          ) : null}

          {busyHint ? (
            <div className="text-xs text-gray-500 dark:text-gray-500 font-jetbrains-mono">
              {busyHint}
            </div>
          ) : null}

          {jobStatus === "succeeded" ? (
            <div className="rounded-lg border border-emerald-200 bg-emerald-50 p-3 text-sm text-emerald-800 dark:border-emerald-900/50 dark:bg-emerald-900/20 dark:text-emerald-200 font-jetbrains-mono">
              Staging completed. Generated images have been added to Assets.
            </div>
          ) : null}
        </div>
        {aiError ? (
          <StatusBanner variant="error">{aiError}</StatusBanner>
        ) : null}

        <p className="text-xs text-gray-500 dark:text-gray-500 font-jetbrains-mono">
          Jobs run asynchronously via the backend.
        </p>
      </div>
    </ModalShell>
  );
}
