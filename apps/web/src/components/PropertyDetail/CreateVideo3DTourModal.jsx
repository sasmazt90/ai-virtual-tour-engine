import { useCallback, useMemo, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { CheckCircle2, Clock3, Loader2, Video, XCircle } from "lucide-react";
import { ModalShell } from "./ModalShell";

const SUPPORTED_EXTENSIONS = new Set(["mp4", "mov", "m4v"]);
const DEFAULT_TOTAL_MS = 90 * 60 * 1000;
const MAX_VIDEO_BYTES = 750 * 1024 * 1024;

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
  if (!Number.isFinite(n) || n <= 0) return "0 MB";
  if (n >= 1024 * 1024 * 1024) return `${(n / 1024 / 1024 / 1024).toFixed(1)} GB`;
  return `${(n / 1024 / 1024).toFixed(1)} MB`;
}

function uploadLargeVideo(file, onProgress) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/api/upload/large");
    xhr.setRequestHeader("Content-Type", file.type || "application/octet-stream");
    xhr.setRequestHeader("X-Filename", encodeURIComponent(file.name || "iphone-video"));

    xhr.upload.onprogress = (event) => {
      if (!event.lengthComputable) return;
      onProgress(Math.round((event.loaded / event.total) * 100));
    };

    xhr.onload = () => {
      let body = {};
      try {
        body = JSON.parse(xhr.responseText || "{}");
      } catch {
        body = {};
      }

      if (xhr.status >= 200 && xhr.status < 300 && body?.url) {
        resolve(body);
        return;
      }

      reject(
        new Error(
          body?.error ||
            `Video upload failed: [${xhr.status}] ${xhr.statusText}`,
        ),
      );
    };

    xhr.onerror = () => reject(new Error("Video upload failed."));
    xhr.onabort = () => reject(new Error("Video upload was cancelled."));
    xhr.send(file);
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
  if (!status) return "Starting 3D tour job";
  if (status === "queued") return "Queued for GPU worker";
  if (status === "failed") return "Failed";
  if (status === "succeeded") return "Completed";

  const p = Number(progress || 0);
  if (p < 15) return "Downloading video";
  if (p < 35) return "Extracting frames";
  if (p < 70) return "Reconstructing camera poses";
  if (p < 88) return "Training Gaussian Splat";
  if (p < 95) return "Uploading 3D tour";
  return "Saving tour";
}

export function CreateVideo3DTourModal({ open, onClose, propertyId, userId }) {
  const queryClient = useQueryClient();
  const [file, setFile] = useState(null);
  const [status, setStatus] = useState("idle");
  const [error, setError] = useState("");
  const [jobId, setJobId] = useState(null);
  const [uploadProgress, setUploadProgress] = useState(0);

  const format = useMemo(() => extensionFromName(file?.name), [file?.name]);
  const uploadLoading = status === "uploading";
  const disableActions = status === "uploading" || status === "starting";

  const { data: jobData } = useQuery({
    queryKey: ["video-3d-tour-job", jobId],
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

  const onStart = useCallback(async () => {
    if (!propertyId) {
      setStatus("error");
      setError("Missing property ID.");
      return;
    }

    if (!file) {
      setStatus("error");
      setError("Choose an iPhone video first.");
      return;
    }

    const nextFormat = extensionFromName(file.name);
    if (!SUPPORTED_EXTENSIONS.has(nextFormat)) {
      setStatus("error");
      setError("Supported video formats are .mp4, .mov and .m4v.");
      return;
    }

    if (file.size > MAX_VIDEO_BYTES) {
      setStatus("error");
      setError(
        `Video is too large (${formatFileSize(file.size)}). Please upload a file under ${formatFileSize(MAX_VIDEO_BYTES)}.`,
      );
      return;
    }

    setStatus("uploading");
    setUploadProgress(0);
    setError("");
    setJobId(null);

    try {
      const uploaded = await uploadLargeVideo(file, setUploadProgress);
      if (uploaded?.error || !uploaded?.url) {
        throw new Error(uploaded?.error || "Video upload failed.");
      }

      setStatus("starting");

      const res = await fetch("/api/ai/video-3d-tour/create", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          propertyId,
          videoUrl: uploaded.url,
          originalName: file.name,
        }),
      });

      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body?.error || "Could not start 3D tour job.");
      }

      const body = await res.json();
      if (!body?.jobId) {
        throw new Error("Server did not return a job ID.");
      }

      setJobId(body.jobId);
      setStatus("queued");
    } catch (err) {
      console.error(err);
      setStatus("error");
      setError(
        err instanceof Error ? err.message : "Could not start 3D tour job.",
      );
    }
  }, [file, propertyId]);

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
  const lastHeartbeatMs = dateMs(jobData?.lastHeartbeatAt);
  const heartbeatAgeMs = lastHeartbeatMs ? Date.now() - lastHeartbeatMs : null;
  const remainingLabel =
    jobStatus === "succeeded"
      ? "done"
      : jobStatus === "failed"
        ? "stopped"
        : formatDuration(remainingMs);

  return (
    <ModalShell title="Create 3D Tour from iPhone Video" onClose={safeOnClose}>
      <div className="space-y-4">
        <div className="rounded-lg border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-[#171717] p-4">
          <div className="flex items-start gap-3">
            <Video className="mt-0.5 h-5 w-5 text-amber-500" />
            <div>
              <div className="text-sm font-semibold text-gray-900 dark:text-gray-100 font-jetbrains-mono">
                Video to 3D Gaussian Splat
              </div>
              <p className="mt-1 text-sm text-gray-600 dark:text-gray-300 font-jetbrains-mono">
                Upload a 2-3 minute walkthrough video. The worker extracts
                frames, reconstructs camera poses, trains a Gaussian Splat and
                saves the result as the property virtual tour.
              </p>
            </div>
          </div>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-900 dark:text-gray-100 font-jetbrains-mono">
            iPhone video
          </label>
          <input
            type="file"
            accept="video/mp4,video/quicktime,.mp4,.mov,.m4v"
            disabled={disableActions || !!jobId}
            onChange={(e) => setFile(e.target.files?.[0] || null)}
            className="mt-2 block w-full text-sm text-gray-700 dark:text-gray-200 font-jetbrains-mono file:mr-3 file:rounded-md file:border-0 file:bg-amber-500 file:px-3 file:py-2 file:text-sm file:font-medium file:text-white hover:file:bg-amber-600"
          />
          {file ? (
            <div className="mt-2 text-xs text-gray-500 dark:text-gray-400 font-jetbrains-mono">
              {file.name} - {(file.size / 1024 / 1024).toFixed(1)} MB -{" "}
              {format || "unknown"}
              {" "}• Limit {formatFileSize(MAX_VIDEO_BYTES)}
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
                  {jobId ? `Job ${String(jobId).slice(0, 8)}...` : "Preparing job"}
                  {jobStatus ? ` • ${jobStatus}` : null}
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

            <div className="mt-3 grid gap-2 text-xs text-gray-600 dark:text-gray-300 font-jetbrains-mono sm:grid-cols-3">
              <div className="flex items-center gap-2">
                <Clock3 className="h-3.5 w-3.5 text-amber-500" />
                <span>Remaining: {remainingLabel}</span>
              </div>
              <div>Elapsed: {formatDuration(elapsedMs)}</div>
              <div>
                Heartbeat:{" "}
                {heartbeatAgeMs == null
                  ? "waiting"
                  : `${formatDuration(heartbeatAgeMs)} ago`}
              </div>
            </div>

            {jobData?.error ? (
              <div className="mt-2 text-sm text-red-600 dark:text-red-400 font-jetbrains-mono">
                {jobData.error}
              </div>
            ) : null}
            {!done ? (
              <div className="mt-2 text-xs text-gray-500 dark:text-gray-500 font-jetbrains-mono">
                Keep the GPU worker running in RunPod. This panel updates every
                few seconds while the job is moving.
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
              disabled={disableActions || !!jobId}
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
