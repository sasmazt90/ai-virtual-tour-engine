import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { CheckCircle2, Clock3, Loader2, Trash2, Video, XCircle } from "lucide-react";
import { ModalShell } from "./ModalShell";
import {
  AI_VIDEO_3D_CREDIT_TIERS,
  AI_VIDEO_3D_MAX_BYTES,
  AI_VIDEO_3D_MAX_FILES,
  calculateVideo3DTourCreditCost,
} from "@/app/api/utils/pricing";
import { uploadLargeFile } from "@/utils/largeUpload";

const SUPPORTED_EXTENSIONS = new Set(["mp4", "mov", "m4v", "webm"]);
const DEFAULT_TOTAL_MS = 90 * 60 * 1000;
const MAX_VIDEO_BYTES = AI_VIDEO_3D_MAX_BYTES;
const HAS_VIDEO_SIZE_LIMIT = Number.isFinite(MAX_VIDEO_BYTES);
const MIN_VIDEO_WIDTH = 1280;
const MIN_VIDEO_HEIGHT = 720;
const MIN_VIDEO_DURATION_SECONDS = 25;
const MIN_VIDEO_BITRATE_MBPS = 2.5;

function extensionFromName(name) {
  return String(name || "").split(".").pop()?.toLowerCase().trim() || "";
}

function formatDuration(ms) {
  if (!Number.isFinite(ms) || ms <= 0) return "calculating";
  const minutes = Math.max(1, Math.round(ms / 60000));
  if (minutes < 60) return `${minutes} min`;
  const hours = Math.floor(minutes / 60);
  const rest = minutes % 60;
  return rest ? `${hours}h ${rest}m` : `${hours}h`;
}

function formatFileSize(bytes) {
  const n = Number(bytes || 0);
  if (!Number.isFinite(n) || n <= 0) return "0 KB";
  if (n < 1024 * 1024) return `${Math.round(n / 1024).toLocaleString()} KB`;
  if (n >= 1024 * 1024 * 1024) return `${(n / 1024 / 1024 / 1024).toFixed(1)} GB`;
  return `${(n / 1024 / 1024).toFixed(1)} MB`;
}

function formatVideoDuration(seconds) {
  const n = Number(seconds || 0);
  if (!Number.isFinite(n) || n <= 0) return "unknown length";
  const minutes = Math.floor(n / 60);
  const rest = Math.round(n % 60);
  return minutes ? `${minutes}m ${rest}s` : `${rest}s`;
}

function formatBitrate(mbps) {
  const n = Number(mbps || 0);
  if (!Number.isFinite(n) || n <= 0) return "unknown bitrate";
  return `${n.toFixed(1)} Mbps`;
}

function fileListFromInput(fileList) {
  return Array.from(fileList || []);
}

function inspectVideoFile(file) {
  return new Promise((resolve) => {
    if (typeof document === "undefined") {
      resolve({ file, warnings: [] });
      return;
    }

    const url = URL.createObjectURL(file);
    const video = document.createElement("video");
    video.preload = "metadata";
    video.muted = true;

    const finish = (result) => {
      URL.revokeObjectURL(url);
      resolve(result);
    };

    video.onloadedmetadata = () => {
      const width = Number(video.videoWidth || 0);
      const height = Number(video.videoHeight || 0);
      const duration = Number(video.duration || 0);
      const bitrateMbps =
        duration > 0 ? (Number(file.size || 0) * 8) / duration / 1000 / 1000 : 0;
      const warnings = [];

      if (width > 0 && height > 0 && height > width) {
        warnings.push("record in landscape mode");
      }
      if (width < MIN_VIDEO_WIDTH || height < MIN_VIDEO_HEIGHT) {
        warnings.push(`use at least ${MIN_VIDEO_WIDTH}x${MIN_VIDEO_HEIGHT} resolution`);
      }
      if (duration < MIN_VIDEO_DURATION_SECONDS) {
        warnings.push(`record at least ${MIN_VIDEO_DURATION_SECONDS} seconds`);
      }
      if (bitrateMbps < MIN_VIDEO_BITRATE_MBPS) {
        warnings.push(`export at ${MIN_VIDEO_BITRATE_MBPS} Mbps or higher`);
      }

      finish({ file, width, height, duration, bitrateMbps, warnings });
    };

    video.onerror = () => {
      finish({
        file,
        warnings: ["this video could not be inspected before upload"],
      });
    };

    video.src = url;
  });
}

function uploadLargeVideo(file, onProgress) {
  return uploadLargeFile(file, {
    fallbackName: "iphone-video",
    onProgress,
  });
}

function dateMs(value) {
  const ms = value ? new Date(value).getTime() : NaN;
  return Number.isFinite(ms) ? ms : null;
}

function estimateRemainingMs({ status, progress, startedAt, createdAt }) {
  if (status === "succeeded" || progress >= 100) return 0;
  if (status === "failed") return 0;

  const anchor = dateMs(startedAt) || dateMs(createdAt);
  const elapsed = anchor ? Date.now() - anchor : 0;
  const p = Math.max(0, Math.min(99, Number(progress || 0)));

  if (p >= 5 && elapsed > 30000) {
    return Math.max(60000, (elapsed / p) * (100 - p));
  }

  return Math.max(60000, DEFAULT_TOTAL_MS - elapsed);
}

function stageFor({ status, progress, uploadLoading }) {
  if (uploadLoading) return "Uploading video";
  if (!status) return "Starting";
  if (status === "queued") return "Queued";
  if (status === "failed") return "Failed";
  if (status === "succeeded") return "Ready";

  const p = Number(progress || 0);
  if (p < 15) return "Preparing video";
  if (p < 35) return "Reading video";
  if (p < 70) return "Building 3D structure";
  if (p < 88) return "Creating 3D tour";
  if (p < 95) return "Uploading tour";
  return "Finishing";
}

function statusDescription({ status, uploadLoading }) {
  if (uploadLoading) return "Your video is uploading.";
  if (!status) return "Preparing your video.";
  if (status === "queued") return "Your video is waiting to be processed.";
  if (status === "failed") return "The 3D tour could not be created.";
  if (status === "succeeded") return "Your 3D tour is ready.";
  return "Processing is in progress.";
}

function userSafeJobError(rawError) {
  if (!rawError) return "";
  const text = String(rawError);
  if (
    text.includes("not suitable for a sellable 3D tour") ||
    text.includes("not reliable enough for a sellable 3D tour")
  ) {
    return text;
  }
  return "The video could not be converted into a 3D tour. Please try a landscape, well-lit walkthrough video with slow movement and clear overlap between clips.";
}

export function CreateVideo3DTourModal({ open, onClose, propertyId, userId }) {
  const queryClient = useQueryClient();
  const fileInputRef = useRef(null);
  const [files, setFiles] = useState([]);
  const [status, setStatus] = useState("idle");
  const [error, setError] = useState("");
  const [jobId, setJobId] = useState(null);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [videoChecks, setVideoChecks] = useState([]);
  const [checkingVideos, setCheckingVideos] = useState(false);

  const totalFileSize = useMemo(
    () => files.reduce((sum, item) => sum + Number(item?.size || 0), 0),
    [files],
  );
  const estimatedCreditCost = useMemo(
    () => (files.length ? calculateVideo3DTourCreditCost(totalFileSize) : null),
    [files.length, totalFileSize],
  );
  const uploadLoading = status === "uploading";
  const disableActions = status === "uploading" || status === "starting";
  const videoWarnings = useMemo(
    () => videoChecks.filter((item) => item.warnings?.length),
    [videoChecks],
  );
  const hasVideoWarnings = videoWarnings.length > 0;

  useEffect(() => {
    let cancelled = false;
    if (!files.length) {
      setVideoChecks([]);
      setCheckingVideos(false);
      return () => {
        cancelled = true;
      };
    }

    setCheckingVideos(true);
    Promise.all(files.map((file) => inspectVideoFile(file))).then((results) => {
      if (cancelled) return;
      setVideoChecks(results);
      setCheckingVideos(false);
    });

    return () => {
      cancelled = true;
    };
  }, [files]);

  const { data: jobData } = useQuery({
    queryKey: ["video-3d-tour-job", jobId],
    queryFn: async () => {
      const res = await fetch(`/api/ai/jobs/${jobId}`);
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body?.error || "Could not fetch progress.");
      }
      return res.json();
    },
    enabled: !!jobId,
    refetchInterval: (q) => {
      const st = q?.state?.data?.status;
      return st === "succeeded" || st === "failed" ? false : 3000;
    },
  });

  const safeOnClose = useCallback(() => {
    if (disableActions) return;
    onClose();
  }, [disableActions, onClose]);

  const refreshAndClose = useCallback(async () => {
    if (userId && propertyId) {
      await queryClient.invalidateQueries({
        queryKey: ["property", userId, propertyId],
      });
    } else {
      await queryClient.invalidateQueries({ queryKey: ["property"] });
    }
    onClose();
  }, [onClose, propertyId, queryClient, userId]);

  const resetFailedJob = useCallback(() => {
    setJobId(null);
    setStatus("idle");
    setError("");
  }, []);

  const removeFileAt = useCallback(
    (removeIndex) => {
      if (disableActions || jobId) return;
      setFiles((current) => current.filter((_, index) => index !== removeIndex));
      setError("");
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    },
    [disableActions, jobId],
  );

  const onStart = useCallback(async () => {
    if (!propertyId) {
      setStatus("error");
      setError("Missing property ID.");
      return;
    }

    if (files.length === 0) {
      setStatus("error");
      setError("Choose at least one iPhone video first.");
      return;
    }

    if (files.length > AI_VIDEO_3D_MAX_FILES) {
      setStatus("error");
      setError(`Upload ${AI_VIDEO_3D_MAX_FILES} videos or fewer.`);
      return;
    }

    const invalidFile = files.find(
      (candidate) => !SUPPORTED_EXTENSIONS.has(extensionFromName(candidate.name)),
    );
    if (invalidFile) {
      setStatus("error");
      setError("Supported video formats are .mp4, .mov, .m4v and .webm.");
      return;
    }

    if (HAS_VIDEO_SIZE_LIMIT && totalFileSize > MAX_VIDEO_BYTES) {
      setStatus("error");
      setError(
        `Videos are too large (${formatFileSize(totalFileSize)} total). Please upload ${formatFileSize(MAX_VIDEO_BYTES)} total or less.`,
      );
      return;
    }

    if (checkingVideos) {
      setStatus("error");
      setError("Video checks are still running. Please wait a moment.");
      return;
    }

    if (hasVideoWarnings) {
      const first = videoWarnings[0];
      setStatus("error");
      setError(
        `${first.file.name} is not ready for a sellable 3D tour yet: ${first.warnings.join(", ")}.`,
      );
      return;
    }

    setStatus("uploading");
    setUploadProgress(0);
    setError("");
    setJobId(null);

    try {
      const uploadedVideos = [];
      let completedBytes = 0;
      for (const [index, item] of files.entries()) {
        const uploaded = await uploadLargeVideo(item, (fileProgress) => {
          const currentBytes =
            (Number(fileProgress || 0) / 100) * item.size;
          setUploadProgress(
            Math.round(((completedBytes + currentBytes) / totalFileSize) * 100),
          );
        });
        if (uploaded?.error || !uploaded?.url) {
          throw new Error(uploaded?.error || "Video upload failed.");
        }
        completedBytes += item.size;
        uploadedVideos.push({
          videoUrl: uploaded.url,
          originalName: item.name,
          sourceOriginalName: item.name,
          fileSizeBytes: uploaded.sizeBytes || item.size,
          originalFileSizeBytes: item.size,
          compressed: false,
          storageProvider: uploaded.provider || null,
          objectPath: uploaded.objectPath || null,
          bucket: uploaded.bucket || null,
          index,
        });
      }

      setStatus("starting");

      const res = await fetch("/api/ai/video-3d-tour/create", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          propertyId,
          videos: uploadedVideos,
        }),
      });

      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body?.error || "Could not start the 3D tour.");
      }

      const body = await res.json();
      if (!body?.jobId) {
        throw new Error("Could not start the 3D tour.");
      }

      setJobId(body.jobId);
      setStatus("queued");
      if (userId) {
        await queryClient.invalidateQueries({ queryKey: ["credits", userId] });
      }
    } catch (err) {
      console.error(err);
      setStatus("error");
      setError(
        err instanceof Error ? err.message : "Could not start the 3D tour.",
      );
    }
  }, [
    checkingVideos,
    files,
    hasVideoWarnings,
    propertyId,
    queryClient,
    totalFileSize,
    userId,
    videoWarnings,
  ]);

  if (!open) return null;

  const jobStatus = jobData?.status || (jobId ? "queued" : null);
  const rawProgress = Number(jobData?.progress || 0);
  const progress = Number.isFinite(rawProgress) ? rawProgress : 0;
  const done = jobStatus === "succeeded" || jobStatus === "failed";
  const visibleProgress = uploadLoading
    ? Math.max(1, Math.min(99, uploadProgress))
    : status === "starting"
      ? 5
      : Math.max(0, Math.min(100, progress));
  const activeStage = stageFor({ status: jobStatus, progress, uploadLoading });
  const remainingMs = estimateRemainingMs({
    status: jobStatus,
    progress,
    startedAt: jobData?.startedAt,
    createdAt: jobData?.createdAt,
  });
  const elapsedMs =
    dateMs(jobData?.startedAt) || dateMs(jobData?.createdAt)
      ? Date.now() - (dateMs(jobData?.startedAt) || dateMs(jobData?.createdAt))
      : 0;
  const remainingLabel =
    jobStatus === "succeeded"
      ? "done"
      : jobStatus === "failed"
        ? "stopped"
        : formatDuration(remainingMs);
  const queuedForMs =
    jobStatus === "queued" && jobData?.createdAt
      ? Date.now() - (dateMs(jobData.createdAt) || Date.now())
      : 0;

  return (
    <ModalShell title="Create 3D Tour" onClose={safeOnClose}>
      <div className="space-y-4">
        <div className="rounded-lg border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-[#171717] p-4">
          <div className="flex items-start gap-3">
            <Video className="mt-0.5 h-5 w-5 text-amber-500" />
            <div>
              <div className="text-sm font-semibold text-gray-900 dark:text-gray-100 font-jetbrains-mono">
                Create from iPhone video
              </div>
              <p className="mt-1 text-sm text-gray-600 dark:text-gray-300 font-jetbrains-mono">
                Upload one or more high-quality walkthrough videos. We will turn them
                into an interactive 3D tour for this property.
              </p>
              <p className="mt-2 text-xs text-gray-500 dark:text-gray-400 font-jetbrains-mono">
                Credits are charged by total uploaded video size:{" "}
                {AI_VIDEO_3D_CREDIT_TIERS.map(
                  (tier) => `${tier.label}: ${tier.credits}`,
                ).join(" - ")}
                .
              </p>
            </div>
          </div>
        </div>

        <div className="rounded-lg border border-amber-500/30 bg-amber-500/5 p-4">
          <div className="text-sm font-semibold text-gray-900 dark:text-gray-100 font-jetbrains-mono">
            Instructions
          </div>
          <ul className="mt-2 space-y-2 text-sm text-gray-600 dark:text-gray-300 font-jetbrains-mono list-disc pl-5">
            <li>Record in landscape mode at 1080p or 4K.</li>
            <li>Use bright, even lighting and avoid dark corners.</li>
            <li>Walk slowly; keep the phone steady at chest height.</li>
            <li>Move through the room instead of standing still and turning in place.</li>
            <li>Capture each wall or furniture area from more than one angle.</li>
            <li>Record one room or one connected area per video.</li>
            <li>Upload videos in walking order: entrance, hallway, room, next room.</li>
            <li>Leave 5-10 seconds of overlap between connected clips.</li>
            <li>Avoid fast turns, motion blur, mirrors, large windows, blank walls, and people moving through the scene.</li>
            <li>Keep each clip around 45-90 seconds for best results.</li>
            <li>For the highest quality, upload a ready 3D tour file as a .ply, .splat or .ksplat file.</li>
          </ul>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-900 dark:text-gray-100 font-jetbrains-mono">
            iPhone videos
          </label>
          <input
            ref={fileInputRef}
            type="file"
            accept="video/mp4,video/quicktime,video/webm,.mp4,.mov,.m4v,.webm"
            multiple
            disabled={disableActions || !!jobId}
            onChange={(e) => {
              setFiles(fileListFromInput(e.target.files));
              setError("");
            }}
            className="mt-2 block w-full text-sm text-gray-700 dark:text-gray-200 font-jetbrains-mono file:mr-3 file:rounded-md file:border-0 file:bg-amber-500 file:px-3 file:py-2 file:text-sm file:font-medium file:text-white hover:file:bg-amber-600"
          />
          {files.length ? (
            <div className="mt-2 space-y-1 text-xs text-gray-500 dark:text-gray-400 font-jetbrains-mono">
              <div>
                {files.length} video{files.length === 1 ? "" : "s"} -{" "}
                {formatFileSize(totalFileSize)} total
                {HAS_VIDEO_SIZE_LIMIT
                  ? ` - Limit ${formatFileSize(MAX_VIDEO_BYTES)}`
                  : " - No file size limit"}
                {estimatedCreditCost
                  ? ` - ${estimatedCreditCost.toLocaleString()} credits`
                  : ""}
              </div>
              <ol className="space-y-1">
                {files.map((item, index) => (
                  <li
                    key={`${item.name}-${item.size}-${item.lastModified}`}
                    className="grid grid-cols-[1.5rem_minmax(0,1fr)_auto] items-center gap-2"
                  >
                    <span className="text-right tabular-nums">{index + 1}.</span>
                    <span className="min-w-0 truncate">
                      {item.name} - {formatFileSize(item.size)} -{" "}
                      {extensionFromName(item.name) || "unknown"}
                    </span>
                    <button
                      type="button"
                      onClick={() => removeFileAt(index)}
                      disabled={disableActions || !!jobId}
                      title="Remove video"
                      aria-label={`Remove ${item.name}`}
                      className="inline-flex h-7 w-7 items-center justify-center rounded-md text-gray-500 hover:bg-red-500/10 hover:text-red-600 disabled:cursor-not-allowed disabled:opacity-40 dark:text-gray-400 dark:hover:text-red-300"
                    >
                      <Trash2 className="h-3.5 w-3.5" />
                    </button>
                  </li>
                ))}
              </ol>
            </div>
          ) : null}
          {checkingVideos ? (
            <div className="mt-3 flex items-center gap-2 text-xs text-gray-500 dark:text-gray-400 font-jetbrains-mono">
              <Loader2 className="h-3.5 w-3.5 animate-spin" />
              Checking video quality...
            </div>
          ) : null}
          {!checkingVideos && files.length && videoChecks.length ? (
            <div className="mt-3 space-y-2">
              {videoChecks.map((item) => {
                const ready = !item.warnings?.length;
                const fileIndex = files.findIndex(
                  (file) =>
                    file.name === item.file.name &&
                    file.size === item.file.size &&
                    file.lastModified === item.file.lastModified,
                );
                return (
                  <div
                    key={`${item.file.name}-${item.file.size}-${item.file.lastModified}-check`}
                    className={`grid grid-cols-[minmax(0,1fr)_auto] items-start gap-3 rounded-md border px-3 py-2 text-xs font-jetbrains-mono ${
                      ready
                        ? "border-emerald-500/30 bg-emerald-500/5 text-emerald-700 dark:text-emerald-300"
                        : "border-red-500/30 bg-red-500/5 text-red-700 dark:text-red-300"
                    }`}
                  >
                    <div className="min-w-0">
                      <div className="truncate font-semibold">
                        {ready ? "Ready" : "Needs a better recording"}:{" "}
                        {item.file.name}
                      </div>
                      <div className="mt-1 text-gray-500 dark:text-gray-400">
                        {item.width && item.height
                          ? `${item.width}x${item.height}`
                          : "unknown size"}
                        {" - "}
                        {formatVideoDuration(item.duration)}
                        {" - "}
                        {formatBitrate(item.bitrateMbps)}
                      </div>
                      {!ready ? (
                        <div className="mt-1">{item.warnings.join(", ")}.</div>
                      ) : null}
                    </div>
                    <button
                      type="button"
                      onClick={() => removeFileAt(fileIndex)}
                      disabled={fileIndex < 0 || disableActions || !!jobId}
                      title="Remove video"
                      aria-label={`Remove ${item.file.name}`}
                      className="inline-flex h-8 w-8 items-center justify-center rounded-md text-current hover:bg-black/5 disabled:cursor-not-allowed disabled:opacity-40 dark:hover:bg-white/10"
                    >
                      <Trash2 className="h-4 w-4" />
                    </button>
                  </div>
                );
              })}
            </div>
          ) : null}
        </div>

        {jobId || uploadLoading || status === "starting" ? (
          <div className="rounded-lg border border-gray-200 dark:border-gray-700 bg-white/60 dark:bg-black/20 p-4">
            <div className="flex items-start justify-between gap-4">
              <div className="min-w-0">
                <div className="flex items-center gap-2 text-sm font-semibold text-gray-900 dark:text-gray-100 font-jetbrains-mono">
                  {jobStatus === "succeeded" ? (
                    <CheckCircle2 className="h-4 w-4 text-emerald-500" />
                  ) : jobStatus === "failed" ? (
                    <XCircle className="h-4 w-4 text-red-500" />
                  ) : (
                    <Loader2 className="h-4 w-4 animate-spin text-amber-500" />
                  )}
                  <span>{activeStage}</span>
                </div>
                <div className="mt-1 text-xs text-gray-500 dark:text-gray-400 font-jetbrains-mono">
                  {statusDescription({ status: jobStatus, uploadLoading })}
                </div>
              </div>
              <div className="shrink-0 text-right text-sm font-semibold text-gray-900 dark:text-gray-100 font-jetbrains-mono">
                {Math.round(visibleProgress)}%
              </div>
            </div>

            <div className="mt-3 h-2 overflow-hidden rounded-full bg-gray-200 dark:bg-gray-800">
              <div
                className="h-full rounded-full bg-amber-500 transition-all duration-500"
                style={{ width: `${visibleProgress}%` }}
              />
            </div>

            <div className="mt-3 grid gap-2 text-xs text-gray-600 dark:text-gray-300 font-jetbrains-mono sm:grid-cols-2">
              <div className="flex items-center gap-2">
                <Clock3 className="h-3.5 w-3.5 text-amber-500" />
                <span>Remaining: {remainingLabel}</span>
              </div>
              <div>Elapsed: {formatDuration(elapsedMs)}</div>
            </div>

            {jobData?.error ? (
              <div className="mt-2 text-sm text-red-600 dark:text-red-400 font-jetbrains-mono">
                {userSafeJobError(jobData.error)}
              </div>
            ) : null}
            {!done && queuedForMs > 2 * 60 * 1000 ? (
              <div className="mt-2 text-xs text-gray-500 dark:text-gray-500 font-jetbrains-mono">
                This is taking longer than usual to start. Progress will update
                automatically.
              </div>
            ) : null}
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
          {jobStatus === "succeeded" ? (
            <button
              type="button"
              onClick={refreshAndClose}
              className="inline-flex items-center px-3 py-2 rounded-lg bg-gray-900 dark:bg-gray-100 text-white dark:text-gray-900 text-sm font-medium hover:bg-gray-800 dark:hover:bg-gray-200 font-jetbrains-mono"
            >
              Done
            </button>
          ) : jobStatus === "failed" ? (
            <button
              type="button"
              onClick={resetFailedJob}
              className="inline-flex items-center px-3 py-2 rounded-lg bg-gray-900 dark:bg-gray-100 text-white dark:text-gray-900 text-sm font-medium hover:bg-gray-800 dark:hover:bg-gray-200 font-jetbrains-mono"
            >
              Try Again
            </button>
          ) : (
            <button
              type="button"
              onClick={onStart}
              disabled={disableActions || !!jobId || checkingVideos || hasVideoWarnings}
              className="inline-flex items-center px-3 py-2 rounded-lg bg-gray-900 dark:bg-gray-100 text-white dark:text-gray-900 text-sm font-medium hover:bg-gray-800 dark:hover:bg-gray-200 disabled:opacity-50 font-jetbrains-mono"
            >
              Start 3D Tour
            </button>
          )}
        </div>
      </div>
    </ModalShell>
  );
}
