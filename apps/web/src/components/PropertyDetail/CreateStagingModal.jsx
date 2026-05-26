import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  ChevronLeft,
  ChevronRight,
  Loader2,
  Sparkles,
} from "lucide-react";
import useUpload from "@/utils/useUpload";
import { ModalShell } from "./ModalShell";
import { StatusBanner } from "@/components/StatusBanner";
import {
  AI_STAGING_MAX_PHOTOS_PER_JOB,
  AI_STAGING_FURNITURE_REFERENCE_CREDIT_COST,
  calculateStagingCreditCost,
} from "@/app/api/utils/pricing";

const STAGING_TYPES = [
  "classic",
  "default",
  "luxury",
  "minimalist",
  "modern",
  "scandinavian",
  "vacant",
];

const PHOTO_PAGE_SIZE = 18; // 6 per row x 3 rows on desktop

export default function CreateStagingModal({
  open,
  onClose,
  userId,
  propertyId,
  property,
  creditsBalance,
  onRefreshAfterJob,
}) {
  const queryClient = useQueryClient();
  const [aiError, setAiError] = useState(null);
  const [stagingType, setStagingType] = useState("luxury");

  // NEW: when staging multiple photos, keep the same furniture set across angles
  const [keepConsistentAcrossPhotos, setKeepConsistentAcrossPhotos] =
    useState(true);

  // Simplified UX: only one saved furniture library (custom_assets), selectable per run
  const [furnitureFiles, setFurnitureFiles] = useState([]);
  const [selectedFurnitureAssetIds, setSelectedFurnitureAssetIds] = useState(
    [],
  );

  // NEW: auto-clean furniture reference images (background removal + centering)
  const [autoCleanFurniture, setAutoCleanFurniture] = useState(true);

  // Let user choose which photos to stage (per-photo pricing)
  const photosList = useMemo(() => {
    const list = Array.isArray(property?.photos) ? property.photos : [];
    return list
      .map((p) => ({ id: p?.id, url: p?.storage_path }))
      .filter((p) => !!p.id && !!p.url);
  }, [property?.photos]);

  // Pagination for photo picker
  const [photoPage, setPhotoPage] = useState(0);

  useEffect(() => {
    if (!open) return;
    setPhotoPage(0);
  }, [open, stagingType]);

  const photoPageCount = useMemo(() => {
    const total = photosList.length;
    return Math.max(1, Math.ceil(total / PHOTO_PAGE_SIZE));
  }, [photosList.length]);

  const pagedPhotos = useMemo(() => {
    const start = photoPage * PHOTO_PAGE_SIZE;
    const end = start + PHOTO_PAGE_SIZE;
    return photosList.slice(start, end);
  }, [photoPage, photosList]);

  const [selectedPhotoIds, setSelectedPhotoIds] = useState([]);

  useEffect(() => {
    if (!open) return;

    // Default selection: first photo only (avoids surprising credit spend)
    setSelectedPhotoIds((prev) => {
      if (Array.isArray(prev) && prev.length > 0) return prev;
      const first = photosList.slice(0, 1).map((p) => p.id);
      return first;
    });
  }, [open, photosList]);

  const selectedCount = selectedPhotoIds.length;
  const maxSelectable = AI_STAGING_MAX_PHOTOS_PER_JOB;

  // NEW: VACANT produces ONLY 2 variants (Day/Night). Other staging types produce 4 (Day/Night × Lights On/Off).
  const variantsPerPhoto = stagingType === "vacant" ? 2 : 4;
  const estimatedOutputImages = selectedCount * variantsPerPhoto;

  const toggleSelectedPhoto = useCallback(
    (photoId) => {
      setSelectedPhotoIds((prev) => {
        const exists = prev.includes(photoId);
        if (exists) {
          return prev.filter((id) => id !== photoId);
        }
        if (prev.length >= maxSelectable) {
          return prev;
        }
        return [...prev, photoId];
      });
    },
    [maxSelectable],
  );

  const [stagingJobId, setStagingJobId] = useState(null);
  const [lastCreditCost, setLastCreditCost] = useState(null);
  const autoRefreshedJobRef = useRef(null);

  const [upload, { loading: uploadingFurniture }] = useUpload();

  const { data: customAssets = [], refetch: refetchCustomAssets } = useQuery({
    queryKey: ["custom-assets", userId, propertyId],
    queryFn: async () => {
      const res = await fetch(`/api/properties/${propertyId}/custom-assets`);
      if (!res.ok) {
        throw new Error("Could not load custom assets.");
      }
      return res.json();
    },
    enabled: !!open && !!userId && !!propertyId,
  });

  const propertyPhotoIds = useMemo(() => {
    return selectedPhotoIds;
  }, [selectedPhotoIds]);

  // VACANT ignores furniture references — disable furniture library UI and clear selection
  const furnitureDisabled = stagingType === "vacant";

  // IMPORTANT: custom furniture pricing only applies if user selects furniture for this run
  const hasPreferredItems = false;
  const hasCustomAssets =
    !furnitureDisabled && selectedFurnitureAssetIds.length > 0;

  const perPhotoCredits = useMemo(() => {
    return calculateStagingCreditCost({
      hasPreferredItems,
      hasCustomAssets,
      customAssetCount: 0,
      photoCount: 1,
    });
  }, []);

  const selectedFurnitureCreditCost = useMemo(() => {
    return furnitureDisabled
      ? 0
      : selectedFurnitureAssetIds.length *
          AI_STAGING_FURNITURE_REFERENCE_CREDIT_COST;
  }, [furnitureDisabled, selectedFurnitureAssetIds.length]);

  const estimatedCredits = useMemo(() => {
    return calculateStagingCreditCost({
      hasPreferredItems,
      hasCustomAssets,
      customAssetCount: furnitureDisabled ? 0 : selectedFurnitureAssetIds.length,
      photoCount: selectedCount || 1,
    });
  }, [
    furnitureDisabled,
    hasCustomAssets,
    selectedCount,
    selectedFurnitureAssetIds.length,
  ]);

  const canRun =
    selectedCount > 0 &&
    Number(creditsBalance || 0) >= Number(estimatedCredits || 0);

  useEffect(() => {
    if (!open) return;
    if (!furnitureDisabled) return;

    // Prevent wasted uploads/credit calculations: VACANT cannot use furniture refs.
    setFurnitureFiles([]);
    setSelectedFurnitureAssetIds([]);
  }, [furnitureDisabled, open]);

  const { data: jobData } = useQuery({
    queryKey: ["ai-job", userId, stagingJobId],
    queryFn: async () => {
      if (!stagingJobId) return null;
      const res = await fetch(`/api/ai/jobs/${stagingJobId}`);
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body?.error || "Could not load job status.");
      }
      return res.json();
    },
    enabled: !!open && !!userId && !!stagingJobId,
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
  const jobResult = jobData?.result || null;
  const isJobActive = jobStatus === "queued" || jobStatus === "running";

  const stagedSummary = useMemo(() => {
    if (!jobResult || typeof jobResult !== "object") return null;

    const selected = Number(jobResult.photoCountSelected);
    const staged = Number(jobResult.photoCountStaged);
    const total = Number(jobResult.totalVariantImages);

    const parts = [];
    if (Number.isFinite(selected)) parts.push(`Selected: ${selected}`);
    if (Number.isFinite(staged)) parts.push(`Processed: ${staged}`);
    if (Number.isFinite(total)) parts.push(`Output images: ${total}`);

    return parts.length > 0 ? parts.join(" - ") : null;
  }, [jobResult]);

  const jobStatusLabel =
    jobStatus === "queued"
      ? "Queued"
      : jobStatus === "running"
        ? "Creating staging"
        : jobStatus === "succeeded"
          ? "Completed"
          : jobStatus === "failed"
            ? "Could not complete"
            : "Ready";

  const jobLine = jobStatus
    ? `${jobStatusLabel} - ${Number(jobProgress || 0)}%`
    : "";

  const retryJobMutation = useMutation({
    mutationFn: async (id) => {
      const res = await fetch(`/api/ai/jobs/${id}/retry`, { method: "POST" });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        if (res.status === 402) {
          throw new Error(body?.error || "Not enough credits to retry.");
        }
        if (res.status === 409) {
          throw new Error(body?.error || "A retry is already in progress.");
        }
        if (res.status === 401) {
          throw new Error("Please sign in again.");
        }
        throw new Error(body?.error || "Could not retry. Please try again.");
      }
      return res.json();
    },
    onSuccess: async () => {
      await queryClient.invalidateQueries({ queryKey: ["credits", userId] });
    },
    onError: (e) => {
      console.error(e);
      setAiError(e?.message || "Could not retry.");
    },
  });

  const createStagingMutation = useMutation({
    mutationFn: async () => {
      if (!propertyId) {
        throw new Error("Missing property ID");
      }

      if (propertyPhotoIds.length === 0) {
        throw new Error("Please select at least 1 photo.");
      }

      const customAssetIds = furnitureDisabled ? [] : selectedFurnitureAssetIds;

      // IMPORTANT: send ONLY the selected photo IDs
      const selectedIds = Array.isArray(propertyPhotoIds)
        ? propertyPhotoIds.filter(Boolean)
        : [];

      const res = await fetch("/api/ai/staging/create", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          propertyId,
          stagingType,
          propertyPhotoIds: selectedIds,
          customAssetIds,
          // NEW
          useCrossPhotoConsistency: keepConsistentAcrossPhotos,
          // keep legacy fields empty (backend is already built to accept them)
          preferredItemImages: [],
          preferredItemHints: [],
          preferredItemsText: "",
        }),
      });

      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        if (res.status === 402) {
          throw new Error(body?.error || "Not enough credits.");
        }
        if (res.status === 401) {
          throw new Error("Please sign in again.");
        }
        throw new Error(body?.error || "Could not start staging.");
      }

      return res.json();
    },
    onSuccess: (data) => {
      setAiError(null);
      setStagingJobId(data.jobId);
      autoRefreshedJobRef.current = null;
      setLastCreditCost(
        Number(data?.creditCost ?? data?.creditsReserved ?? 0) || null,
      );
      queryClient.invalidateQueries({ queryKey: ["credits", userId] });
    },
    onError: (e) => {
      console.error(e);
      setAiError(e?.message || "Could not start staging.");
    },
  });

  const onPickFurniture = useCallback(
    (e) => {
      if (furnitureDisabled) {
        return;
      }
      const files = Array.from(e.target.files || []);
      setFurnitureFiles(files);
    },
    [furnitureDisabled],
  );

  const toggleSelectedFurniture = useCallback(
    (assetId) => {
      if (furnitureDisabled) {
        return;
      }
      setSelectedFurnitureAssetIds((prev) => {
        const exists = prev.includes(assetId);
        if (exists) return prev.filter((x) => x !== assetId);
        return [...prev, assetId];
      });
    },
    [furnitureDisabled],
  );

  const uploadFurnitureMutation = useMutation({
    mutationFn: async () => {
      setAiError(null);
      if (!propertyId) {
        throw new Error("Missing property ID");
      }
      if (furnitureFiles.length === 0) return;

      const createdIds = [];

      for (const file of furnitureFiles) {
        const { url, error } = await upload({ file });
        if (error) {
          throw new Error(error);
        }

        // NEW: preprocess furniture images to reduce staging verification failures
        // (clean background + centered item)
        let finalUrl = url;
        if (autoCleanFurniture) {
          try {
            const prepRes = await fetch("/api/ai/furniture/preprocess", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ imageUrl: url }),
            });

            if (!prepRes.ok) {
              const body = await prepRes.json().catch(() => ({}));
              // Don't block upload if cleanup fails; just fall back to original.
              console.error("Furniture preprocess failed:", body);
            } else {
              const body = await prepRes.json().catch(() => ({}));
              if (body?.url) {
                finalUrl = body.url;
              }
            }
          } catch (e) {
            console.error("Furniture preprocess error:", e);
          }
        }

        const res = await fetch(`/api/properties/${propertyId}/custom-assets`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            storage_path: finalUrl,
            label: file?.name || null,
          }),
        });

        if (!res.ok) {
          const body = await res.json().catch(() => ({}));
          throw new Error(body?.error || "Could not save furniture item.");
        }

        const created = await res.json().catch(() => null);
        const createdId = created?.id;
        if (createdId) createdIds.push(createdId);
      }

      setFurnitureFiles([]);
      await refetchCustomAssets();

      if (createdIds.length > 0) {
        setSelectedFurnitureAssetIds((prev) => {
          const next = new Set(prev);
          for (const id of createdIds) next.add(id);
          return Array.from(next);
        });
      }
    },
    onError: (e) => {
      console.error(e);
      setAiError(e?.message || "Could not upload furniture.");
    },
  });

  const resetModalState = useCallback(() => {
    setAiError(null);
    setFurnitureFiles([]);
    setSelectedFurnitureAssetIds([]);
    setStagingJobId(null);
    setLastCreditCost(null);
    setStagingType("luxury");
    setSelectedPhotoIds([]);
    setAutoCleanFurniture(true);
    setKeepConsistentAcrossPhotos(true);
  }, []);

  const safeOnClose = useCallback(() => {
    if (createStagingMutation.isPending) return;
    resetModalState();
    onClose();
  }, [createStagingMutation.isPending, onClose, resetModalState]);

  const onClickRefresh = useCallback(async () => {
    try {
      await onRefreshAfterJob?.();
      await queryClient.invalidateQueries({
        queryKey: ["property", userId, propertyId],
      });
    } catch (e) {
      console.error(e);
    }
  }, [onRefreshAfterJob, propertyId, queryClient, userId]);

  useEffect(() => {
    if (!open) return;
    if (!stagingJobId) return;
    if (jobStatus !== "succeeded") return;
    if (autoRefreshedJobRef.current === stagingJobId) return;

    autoRefreshedJobRef.current = stagingJobId;
    onClickRefresh();
  }, [jobStatus, onClickRefresh, open, stagingJobId]);

  if (!open) return null;

  const disableActions =
    createStagingMutation.isPending ||
    isJobActive ||
    uploadFurnitureMutation.isPending ||
    uploadingFurniture;

  const creditLine = `${Number(creditsBalance || 0).toLocaleString()} balance - ${selectedCount || 0} photos x ${Number(perPhotoCredits || 0).toLocaleString()} - ${furnitureDisabled ? 0 : selectedFurnitureAssetIds.length} furniture refs x ${AI_STAGING_FURNITURE_REFERENCE_CREDIT_COST} - ${Number(estimatedCredits || 0).toLocaleString()} total`;

  const lastCreditCostLine = lastCreditCost
    ? `Last generation cost: ${Number(lastCreditCost).toLocaleString()}.`
    : null;

  const pageLabelLeft = Math.min(photoPage + 1, photoPageCount);

  return (
    <ModalShell title="Add Staging" onClose={safeOnClose}>
      <div className="space-y-6">
        <div className="rounded-lg border border-gray-200 dark:border-gray-700 p-4 bg-gray-50 dark:bg-gray-800">
          <div className="flex items-center justify-between gap-4">
            <div>
              <div className="text-sm font-semibold text-gray-900 dark:text-gray-100 font-jetbrains-mono">
                Credits
              </div>
              <div className="mt-1 text-sm text-gray-600 dark:text-gray-300 font-jetbrains-mono">
                {creditLine}
              </div>
              {selectedFurnitureCreditCost > 0 ? (
                <div className="mt-1 text-xs text-gray-500 dark:text-gray-400 font-jetbrains-mono">
                  Furniture references add{" "}
                  {selectedFurnitureCreditCost.toLocaleString()} credits to
                  this run.
                </div>
              ) : null}
              {!canRun ? (
                <div className="mt-2 text-xs text-red-600 dark:text-red-400 font-jetbrains-mono">
                  Not enough credits.
                </div>
              ) : null}
            </div>

            <a
              href="/credits"
              className="text-sm text-[var(--brandDark)] dark:text-[var(--brand)] hover:underline font-jetbrains-mono"
            >
              Buy credits
            </a>
          </div>
        </div>

        <div className="rounded-lg border border-gray-200 dark:border-gray-700 p-4 bg-white dark:bg-gray-900">
          <div className="text-sm font-semibold text-gray-900 dark:text-gray-100 font-jetbrains-mono">
            Select photos
          </div>
          <div className="mt-1 text-xs text-gray-500 dark:text-gray-400 font-jetbrains-mono">
            {stagingType === "vacant"
              ? "Empty-room mode creates day and night versions."
              : "Each selected photo creates day, night, lights-on, and lights-off versions."}
          </div>
          {selectedCount > 0 ? (
            <div className="mt-1 text-xs text-gray-500 dark:text-gray-400 font-jetbrains-mono">
              Expected output: {estimatedOutputImages} staged images.
            </div>
          ) : null}

          <div className="mt-3 flex items-center justify-between gap-3">
            <div className="text-xs text-gray-600 dark:text-gray-300 font-jetbrains-mono">
              Selected: {selectedCount} / {maxSelectable}
            </div>
            <div className="flex items-center gap-3">
              <button
                type="button"
                disabled={disableActions}
                onClick={() => setSelectedPhotoIds([])}
                className="text-xs text-[var(--brandDark)] dark:text-[var(--brand)] hover:underline font-jetbrains-mono disabled:opacity-50"
              >
                Clear
              </button>
              <button
                type="button"
                disabled={disableActions}
                onClick={() => {
                  const first = photosList.slice(0, 1).map((p) => p.id);
                  setSelectedPhotoIds(first);
                }}
                className="text-xs text-[var(--brandDark)] dark:text-[var(--brand)] hover:underline font-jetbrains-mono disabled:opacity-50"
              >
                Reset to first
              </button>
            </div>
          </div>

          {photosList.length > 0 ? (
            <div className="mt-3">
              <div className="flex items-center justify-between gap-3">
                <div className="text-xs text-gray-500 dark:text-gray-400 font-jetbrains-mono">
                  Page {pageLabelLeft} / {photoPageCount}
                </div>
                <div className="flex items-center gap-2">
                  <button
                    type="button"
                    disabled={disableActions || photoPage <= 0}
                    onClick={() => setPhotoPage((p) => Math.max(0, p - 1))}
                    className="inline-flex items-center gap-1 px-2 py-1 rounded-lg border border-gray-200 dark:border-gray-700 text-xs text-gray-700 dark:text-gray-200 hover:bg-gray-50 dark:hover:bg-gray-800 font-jetbrains-mono disabled:opacity-50"
                  >
                    <ChevronLeft size={14} />
                    Prev
                  </button>
                  <button
                    type="button"
                    disabled={disableActions || photoPage >= photoPageCount - 1}
                    onClick={() =>
                      setPhotoPage((p) => Math.min(photoPageCount - 1, p + 1))
                    }
                    className="inline-flex items-center gap-1 px-2 py-1 rounded-lg border border-gray-200 dark:border-gray-700 text-xs text-gray-700 dark:text-gray-200 hover:bg-gray-50 dark:hover:bg-gray-800 font-jetbrains-mono disabled:opacity-50"
                  >
                    Next
                    <ChevronRight size={14} />
                  </button>
                </div>
              </div>

              <div className="mt-3 grid grid-cols-3 sm:grid-cols-6 gap-2">
                {pagedPhotos.map((p) => {
                  const isSelected = selectedPhotoIds.includes(p.id);
                  const ring = isSelected ? "ring-2 ring-[var(--brand)]" : "";
                  const opacity = isSelected ? "opacity-100" : "opacity-70";
                  return (
                    <button
                      key={p.id}
                      type="button"
                      onClick={() => toggleSelectedPhoto(p.id)}
                      disabled={disableActions}
                      className={`relative rounded-lg overflow-hidden border border-gray-200 dark:border-gray-700 ${ring} ${opacity}`}
                      title={isSelected ? "Selected" : "Click to select"}
                    >
                      <img
                        src={p.url}
                        alt="Photo"
                        className="w-full h-16 object-cover"
                      />
                      <div className="absolute top-1 right-1 h-5 w-5 rounded-full bg-white/90 dark:bg-gray-800/90 border border-gray-200 dark:border-gray-700 text-xs text-gray-900 dark:text-gray-100 flex items-center justify-center">
                        {isSelected ? "✓" : "+"}
                      </div>
                    </button>
                  );
                })}
              </div>
            </div>
          ) : (
            <div className="mt-3 text-xs text-gray-500 dark:text-gray-500 font-jetbrains-mono">
              No photos yet.
            </div>
          )}

          {selectedCount >= maxSelectable ? (
            <div className="mt-2 text-xs text-amber-700 dark:text-amber-400 font-jetbrains-mono">
              Max {maxSelectable} photos selected (run another batch for more).
            </div>
          ) : null}
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
          <div className="space-y-2">
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 font-jetbrains-mono">
              Staging Type
            </label>
            <select
              value={stagingType}
              onChange={(e) => setStagingType(e.target.value)}
              disabled={disableActions}
              className="w-full px-4 py-3 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-600 rounded-lg text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-[var(--brand)] font-jetbrains-mono"
            >
              {STAGING_TYPES.map((t) => (
                <option key={t} value={t}>
                  {t}
                </option>
              ))}
            </select>
          </div>

          <div className="space-y-2">
            <div className="text-sm font-medium text-gray-700 dark:text-gray-300 font-jetbrains-mono">
              Furniture references (optional)
            </div>

            {/* NEW: VACANT disable notice */}
            {furnitureDisabled ? (
              <div className="text-xs text-amber-700 dark:text-amber-400 font-jetbrains-mono">
                Disabled in VACANT mode (VACANT removes objects and cannot use
                furniture references).
              </div>
            ) : (
              <div className="text-xs text-gray-500 dark:text-gray-400 font-jetbrains-mono">
                Upload clear item photos. Use a plain background, good light,
                and make the item fill most of the frame.
              </div>
            )}

            {/* Auto clean toggle stays visible but disabled in VACANT to reduce confusion */}
            <label className="mt-1 flex items-center gap-2 text-xs text-gray-700 dark:text-gray-300 font-jetbrains-mono">
              <input
                type="checkbox"
                checked={autoCleanFurniture}
                onChange={(e) => setAutoCleanFurniture(e.target.checked)}
                disabled={disableActions || furnitureDisabled}
              />
              Clean item photos automatically
            </label>

            <div className="flex items-center gap-2">
              <input
                type="file"
                accept="image/*"
                multiple
                onChange={onPickFurniture}
                disabled={disableActions || furnitureDisabled}
                className="block w-full text-sm text-gray-600 dark:text-gray-300 font-jetbrains-mono"
              />
              <button
                type="button"
                onClick={() => uploadFurnitureMutation.mutate()}
                disabled={
                  disableActions ||
                  furnitureDisabled ||
                  furnitureFiles.length === 0
                }
                className="shrink-0 inline-flex items-center gap-2 px-3 py-2 rounded-lg bg-gray-900 dark:bg-gray-100 text-white dark:text-gray-900 text-sm font-medium hover:bg-gray-800 dark:hover:bg-gray-200 disabled:opacity-50 font-jetbrains-mono"
              >
                {uploadFurnitureMutation.isPending ? (
                  <Loader2 size={16} className="animate-spin" />
                ) : null}
                Add
              </button>
            </div>

            {Array.isArray(customAssets) && customAssets.length > 0 ? (
              <div className="mt-2 grid grid-cols-3 sm:grid-cols-4 gap-2">
                {customAssets.map((a) => {
                  const isSelected = selectedFurnitureAssetIds.includes(a.id);
                  const ring = isSelected ? "ring-2 ring-[var(--brand)]" : "";
                  const cardTitle = furnitureDisabled
                    ? "Disabled in VACANT mode"
                    : isSelected
                      ? "Selected"
                      : "Click to select";

                  return (
                    <button
                      key={a.id}
                      type="button"
                      onClick={() => toggleSelectedFurniture(a.id)}
                      disabled={disableActions || furnitureDisabled}
                      className={`relative rounded-lg overflow-hidden border border-gray-200 dark:border-gray-700 ${ring} ${furnitureDisabled ? "opacity-60" : ""}`}
                      title={cardTitle}
                    >
                      <img
                        src={a.storage_path}
                        alt={a.label || "Furniture"}
                        className="w-full h-20 object-cover"
                      />
                      <div className="absolute top-1 right-1 h-5 w-5 rounded-full bg-white/90 dark:bg-gray-800/90 border border-gray-200 dark:border-gray-700 text-xs text-gray-900 dark:text-gray-100 flex items-center justify-center">
                        {isSelected ? "✓" : "+"}
                      </div>
                    </button>
                  );
                })}
              </div>
            ) : (
              <div className="mt-2 text-xs text-gray-500 dark:text-gray-500 font-jetbrains-mono">
                No furniture items yet.
              </div>
            )}

            <div className="flex items-center justify-between gap-3">
              <div className="text-xs text-gray-500 dark:text-gray-400 font-jetbrains-mono">
                Selected for this run:{" "}
                {furnitureDisabled ? 0 : selectedFurnitureAssetIds.length}
              </div>
              <button
                type="button"
                disabled={
                  disableActions ||
                  furnitureDisabled ||
                  selectedFurnitureAssetIds.length === 0
                }
                onClick={() => setSelectedFurnitureAssetIds([])}
                className="text-xs text-[var(--brandDark)] dark:text-[var(--brand)] hover:underline font-jetbrains-mono disabled:opacity-50"
              >
                Clear
              </button>
            </div>
          </div>
        </div>

        {/* NEW: multi-photo consistency toggle */}
        {selectedCount > 1 && stagingType !== "vacant" ? (
          <div className="rounded-lg border border-gray-200 dark:border-gray-700 p-4 bg-white dark:bg-gray-900">
            <div className="text-sm font-semibold text-gray-900 dark:text-gray-100 font-jetbrains-mono">
              Multi-photo consistency
            </div>
            <div className="mt-1 text-xs text-gray-500 dark:text-gray-400 font-jetbrains-mono">
              If these photos are different angles of the same room, this keeps
              the furniture set consistent. If you picked multiple rooms in one
              batch, we will try to auto-split them into room groups so each
              room gets its own furniture set.
            </div>
            <label className="mt-3 flex items-center gap-2 text-xs text-gray-700 dark:text-gray-300 font-jetbrains-mono">
              <input
                type="checkbox"
                checked={keepConsistentAcrossPhotos}
                onChange={(e) =>
                  setKeepConsistentAcrossPhotos(e.target.checked)
                }
                disabled={disableActions}
              />
              Keep furniture consistent across photos (same-room angles)
            </label>
            <div className="mt-2 text-[11px] text-gray-500 dark:text-gray-400 font-jetbrains-mono">
              Tip: Turn this off if you want each photo treated independently.
            </div>
          </div>
        ) : null}

        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
          <div className="text-sm text-gray-600 dark:text-gray-300 font-jetbrains-mono">
            {jobLine ? <div>{jobLine}</div> : null}
            {jobError ? (
              <div className="mt-1 text-xs text-red-600 dark:text-red-400 font-jetbrains-mono">
                We could not create this staging. Try again with a clearer
                photo or fewer furniture references.
              </div>
            ) : null}
            {stagedSummary ? (
              <div className="mt-1 text-xs text-gray-500 dark:text-gray-400 font-jetbrains-mono">
                {stagedSummary}
              </div>
            ) : null}
          </div>

          <div className="flex items-center gap-2">
            {jobStatus === "failed" && stagingJobId ? (
              <button
                type="button"
                disabled={disableActions || retryJobMutation.isPending}
                onClick={async () => {
                  try {
                    setAiError(null);
                    const data =
                      await retryJobMutation.mutateAsync(stagingJobId);
                    setStagingJobId(data.jobId);
                  } catch (e) {
                    // handled in mutation
                  }
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
                  uploadFurnitureMutation.isPending ||
                  uploadingFurniture ||
                  isJobActive ||
                  (jobStatus !== "succeeded" && !canRun)
                }
                onClick={async () => {
                  if (jobStatus === "succeeded") {
                    await onClickRefresh();
                    resetModalState();
                    onClose();
                    return;
                  }

                  setAiError(null);
                  createStagingMutation.mutate();
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

        <div className="text-xs text-gray-500 dark:text-gray-500 font-jetbrains-mono">
          Credits are used when generation starts.
          {lastCreditCostLine ? <span>{` ${lastCreditCostLine}`}</span> : null}
        </div>

        {jobStatus === "succeeded" ? (
          <div className="rounded-lg border border-emerald-200 bg-emerald-50 p-3 text-sm text-emerald-800 dark:border-emerald-900/50 dark:bg-emerald-900/20 dark:text-emerald-200 font-jetbrains-mono">
            Staging completed. Generated images have been added to Assets.
          </div>
        ) : null}

        {aiError ? (
          <StatusBanner variant="error">{aiError}</StatusBanner>
        ) : null}
      </div>
    </ModalShell>
  );
}
