import { useCallback, useMemo, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { Upload } from "lucide-react";
import { useUpload } from "@/utils/useUpload";
import { ModalShell } from "./ModalShell";

const SUPPORTED_EXTENSIONS = new Set(["ply", "splat", "ksplat"]);

function extensionFromName(name) {
  return String(name || "").split(".").pop()?.toLowerCase().trim() || "";
}

export function CreateSplatTourModal({ open, onClose, propertyId, userId }) {
  const queryClient = useQueryClient();
  const [upload, { loading: uploadLoading }] = useUpload();
  const [file, setFile] = useState(null);
  const [status, setStatus] = useState("idle");
  const [error, setError] = useState("");

  const format = useMemo(() => extensionFromName(file?.name), [file?.name]);
  const disableActions = uploadLoading || status === "saving";

  const safeOnClose = useCallback(() => {
    if (disableActions) return;
    onClose();
  }, [disableActions, onClose]);

  const onSave = useCallback(async () => {
    if (!propertyId) {
      setError("Missing property ID.");
      setStatus("error");
      return;
    }

    if (!file) {
      setError("Choose a .ply, .splat or .ksplat file.");
      setStatus("error");
      return;
    }

    const nextFormat = extensionFromName(file.name);
    if (!SUPPORTED_EXTENSIONS.has(nextFormat)) {
      setError("Supported 3D scan formats are .ply, .splat and .ksplat.");
      setStatus("error");
      return;
    }

    setStatus("saving");
    setError("");

    try {
      const uploaded = await upload({ file });
      if (uploaded?.error || !uploaded?.url) {
        throw new Error(uploaded?.error || "Upload failed.");
      }

      const res = await fetch("/api/virtual-tours/splat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          propertyId,
          fileUrl: uploaded.url,
          originalName: file.name,
          format: nextFormat,
        }),
      });

      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body?.error || "Could not save the 3D tour.");
      }

      if (userId && propertyId) {
        await queryClient.invalidateQueries({
          queryKey: ["property", userId, propertyId],
        });
      } else {
        await queryClient.invalidateQueries({ queryKey: ["property"] });
      }

      setFile(null);
      setStatus("idle");
      onClose();
    } catch (err) {
      console.error(err);
      setStatus("error");
      setError(err instanceof Error ? err.message : "Could not save 3D tour.");
    }
  }, [file, onClose, propertyId, queryClient, upload, userId]);

  if (!open) return null;

  return (
    <ModalShell title="Upload 3D Virtual Tour" onClose={safeOnClose}>
      <div className="space-y-4">
        <div className="rounded-lg border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-[#171717] p-4">
          <div className="flex items-start gap-3">
            <Upload className="mt-0.5 h-5 w-5 text-amber-500" />
            <div>
              <div className="text-sm font-semibold text-gray-900 dark:text-gray-100 font-jetbrains-mono">
                Gaussian Splat scan
              </div>
              <p className="mt-1 text-sm text-gray-600 dark:text-gray-300 font-jetbrains-mono">
                Upload a 3D scan exported as .ply, .splat or .ksplat. This will
                replace the existing Original virtual tour for this property.
              </p>
            </div>
          </div>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-900 dark:text-gray-100 font-jetbrains-mono">
            3D scan file
          </label>
          <input
            type="file"
            accept=".ply,.splat,.ksplat"
            disabled={disableActions}
            onChange={(e) => setFile(e.target.files?.[0] || null)}
            className="mt-2 block w-full text-sm text-gray-700 dark:text-gray-200 font-jetbrains-mono file:mr-3 file:rounded-md file:border-0 file:bg-amber-500 file:px-3 file:py-2 file:text-sm file:font-medium file:text-white hover:file:bg-amber-600"
          />
          {file ? (
            <div className="mt-2 text-xs text-gray-500 dark:text-gray-400 font-jetbrains-mono">
              {file.name} • {(file.size / 1024 / 1024).toFixed(1)} MB •{" "}
              {format || "unknown"}
            </div>
          ) : null}
        </div>

        <div className="text-xs text-gray-500 dark:text-gray-400 font-jetbrains-mono">
          Tip: for faster loading on the web, export or convert large scans to
          .ksplat before uploading.
        </div>

        {status === "saving" ? (
          <div className="text-sm text-gray-700 dark:text-gray-200 font-jetbrains-mono">
            Uploading and saving 3D tour...
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
            onClick={onSave}
            disabled={disableActions}
            className="inline-flex items-center px-3 py-2 rounded-lg bg-gray-900 dark:bg-gray-100 text-white dark:text-gray-900 text-sm font-medium hover:bg-gray-800 dark:hover:bg-gray-200 disabled:opacity-50 font-jetbrains-mono"
          >
            Save 3D Tour
          </button>
        </div>
      </div>
    </ModalShell>
  );
}
