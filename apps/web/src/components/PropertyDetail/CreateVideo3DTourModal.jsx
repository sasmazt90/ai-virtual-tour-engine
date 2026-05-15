import { useCallback, useMemo, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { Video } from "lucide-react";
import { useUpload } from "@/utils/useUpload";
import { ModalShell } from "./ModalShell";

const SUPPORTED_EXTENSIONS = new Set(["mp4", "mov", "m4v"]);

function extensionFromName(name) {
  return String(name || "").split(".").pop()?.toLowerCase().trim() || "";
}

export function CreateVideo3DTourModal({ open, onClose, propertyId, userId }) {
  const queryClient = useQueryClient();
  const [upload, { loading: uploadLoading }] = useUpload();
  const [file, setFile] = useState(null);
  const [status, setStatus] = useState("idle");
  const [error, setError] = useState("");
  const [jobId, setJobId] = useState(null);

  const format = useMemo(() => extensionFromName(file?.name), [file?.name]);
  const disableActions = uploadLoading || status === "starting";

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
      return st === "succeeded" || st === "failed" ? false : 5000;
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

    setStatus("starting");
    setError("");
    setJobId(null);

    try {
      const uploaded = await upload({ file });
      if (uploaded?.error || !uploaded?.url) {
        throw new Error(uploaded?.error || "Video upload failed.");
      }

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
  }, [file, propertyId, upload]);

  if (!open) return null;

  const jobStatus = jobData?.status || (jobId ? "queued" : null);
  const progress = Number(jobData?.progress || 0);
  const done = jobStatus === "succeeded" || jobStatus === "failed";

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
            </div>
          ) : null}
        </div>

        {jobId ? (
          <div className="rounded-lg border border-gray-200 dark:border-gray-700 p-3">
            <div className="text-sm text-gray-800 dark:text-gray-100 font-jetbrains-mono">
              Job: {jobStatus} - {Number.isFinite(progress) ? progress : 0}%
            </div>
            {jobData?.error ? (
              <div className="mt-2 text-sm text-red-600 dark:text-red-400 font-jetbrains-mono">
                {jobData.error}
              </div>
            ) : null}
            {!done ? (
              <div className="mt-2 text-xs text-gray-500 dark:text-gray-500 font-jetbrains-mono">
                Processing can take a long time depending on video length and
                worker GPU/CPU speed.
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
          {done ? (
            <button
              type="button"
              onClick={refreshAndClose}
              className="inline-flex items-center px-3 py-2 rounded-lg bg-gray-900 dark:bg-gray-100 text-white dark:text-gray-900 text-sm font-medium hover:bg-gray-800 dark:hover:bg-gray-200 font-jetbrains-mono"
            >
              Done
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
