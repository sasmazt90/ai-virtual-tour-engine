import { useCallback, useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { FileUp, FileText, Loader2 } from "lucide-react";
import useUpload from "@/utils/useUpload";
import { ModalShell } from "./ModalShell";
import { ClientCombobox } from "@/components/Calendar/ClientCombobox";

const TEMPLATE_OPTIONS = [
  { value: "sale_agreement", label: "Sale Agreement" },
  { value: "rental_agreement", label: "Rental Agreement" },
];

function isPdfMime(mimeType) {
  if (!mimeType) return false;
  const s = String(mimeType).toLowerCase();
  return s === "application/pdf" || s.endsWith("/pdf");
}

export default function AddContractModal({
  open,
  onClose,
  propertyId,
  userId,
  defaultClientId,
}) {
  const queryClient = useQueryClient();

  const [mode, setMode] = useState("upload"); // 'upload' | 'generate'
  const [error, setError] = useState(null);

  const [selectedFile, setSelectedFile] = useState(null);
  const [templateType, setTemplateType] = useState(TEMPLATE_OPTIONS[0]?.value);

  // NEW: which client card should this contract be saved under?
  const [clientId, setClientId] = useState(defaultClientId || "");

  const [upload, { loading: uploadLoading }] = useUpload();

  const { data: clients = [], isLoading: clientsLoading } = useQuery({
    queryKey: ["clients", userId],
    queryFn: async () => {
      const res = await fetch("/api/clients?type=all");
      if (!res.ok) {
        throw new Error("Could not load clients");
      }
      return res.json();
    },
    enabled: !!open && !!userId,
  });

  const resetState = useCallback(() => {
    setError(null);
    setMode("upload");
    setSelectedFile(null);
    setTemplateType(TEMPLATE_OPTIONS[0]?.value);
    setClientId(defaultClientId || "");
  }, [defaultClientId]);

  const closeAndReset = useCallback(() => {
    resetState();
    onClose();
  }, [onClose, resetState]);

  const invalidateProperty = useCallback(async () => {
    if (!userId || !propertyId) return;
    await queryClient.invalidateQueries({
      queryKey: ["property", userId, propertyId],
    });
  }, [propertyId, queryClient, userId]);

  const uploadMutation = useMutation({
    mutationFn: async () => {
      setError(null);
      if (!propertyId) throw new Error("Missing property context.");
      if (!selectedFile) throw new Error("Please select a PDF file.");

      const {
        url,
        mimeType,
        error: upErr,
      } = await upload({ file: selectedFile });
      if (upErr) throw new Error(upErr);
      if (!isPdfMime(mimeType)) {
        throw new Error("Please upload a PDF file.");
      }

      const res = await fetch("/api/contracts/upload", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          propertyId,
          pdfUrl: url,
          fileName: selectedFile?.name || null,
          clientId: clientId || null,
        }),
      });

      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body?.error || "Could not upload contract.");
      }

      return res.json();
    },
    onSuccess: async () => {
      await invalidateProperty();
      closeAndReset();
    },
    onError: (err) => {
      console.error(err);
      setError(err?.message || "Could not upload contract.");
    },
  });

  const generateMutation = useMutation({
    mutationFn: async () => {
      setError(null);
      if (!propertyId) throw new Error("Missing property context.");
      if (!templateType) throw new Error("Please select a template.");

      const res = await fetch("/api/contracts/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          propertyId,
          templateType,
          clientId: clientId || null,
        }),
      });

      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body?.error || "Could not generate contract.");
      }

      return res.json();
    },
    onSuccess: async () => {
      await invalidateProperty();
      closeAndReset();
    },
    onError: (err) => {
      console.error(err);
      setError(err?.message || "Could not generate contract.");
    },
  });

  const isBusy =
    uploadLoading || uploadMutation.isPending || generateMutation.isPending;

  const primaryLabel = useMemo(() => {
    if (mode === "upload") return "Upload PDF";
    return "Generate PDF";
  }, [mode]);

  const onPrimary = useCallback(() => {
    if (mode === "upload") {
      uploadMutation.mutate();
    } else {
      generateMutation.mutate();
    }
  }, [generateMutation, mode, uploadMutation]);

  if (!open) return null;

  return (
    <ModalShell
      title="Add Contract"
      onClose={isBusy ? () => {} : closeAndReset}
    >
      <div className="space-y-4 font-jetbrains-mono">
        {/* NEW: attach to client */}
        <div className="space-y-2">
          <div className="text-sm text-gray-700 dark:text-gray-300">
            Save under client (so it appears on the client card)
          </div>
          <ClientCombobox
            value={clientId}
            onChange={setClientId}
            clients={clients}
            placeholder={
              clientsLoading ? "Loading clients…" : "Select a client (optional)"
            }
          />
          <div className="text-xs text-gray-600 dark:text-gray-400">
            If you pick a client, this contract will show under Directory → that
            client.
          </div>
        </div>

        <div className="flex gap-2">
          <button
            type="button"
            onClick={() => {
              setError(null);
              setMode("upload");
            }}
            disabled={isBusy}
            className={`px-3 py-2 rounded-lg border text-sm ${
              mode === "upload"
                ? "bg-gray-900 text-white border-gray-900"
                : "bg-white dark:bg-[#262626] text-gray-900 dark:text-gray-100 border-gray-200 dark:border-gray-700"
            }`}
          >
            Upload PDF
          </button>

          <button
            type="button"
            onClick={() => {
              setError(null);
              setMode("generate");
            }}
            disabled={isBusy}
            className={`px-3 py-2 rounded-lg border text-sm ${
              mode === "generate"
                ? "bg-gray-900 text-white border-gray-900"
                : "bg-white dark:bg-[#262626] text-gray-900 dark:text-gray-100 border-gray-200 dark:border-gray-700"
            }`}
          >
            Generate from template
          </button>
        </div>

        {mode === "upload" ? (
          <div className="space-y-2">
            <div className="text-sm text-gray-700 dark:text-gray-300">
              Upload PDF
            </div>
            <input
              type="file"
              accept="application/pdf"
              disabled={isBusy}
              onChange={(e) => {
                const file = e.target.files?.[0] || null;
                setSelectedFile(file);
              }}
              className="block w-full text-sm text-gray-700 dark:text-gray-200"
            />
            {selectedFile ? (
              <div className="text-xs text-gray-600 dark:text-gray-400">
                Selected: {selectedFile.name}
              </div>
            ) : (
              <div className="text-xs text-gray-600 dark:text-gray-400">
                PDF only.
              </div>
            )}
          </div>
        ) : (
          <div className="space-y-2">
            <div className="text-sm text-gray-700 dark:text-gray-300">
              Template
            </div>
            <select
              value={templateType || ""}
              disabled={isBusy}
              onChange={(e) => setTemplateType(e.target.value)}
              className="w-full px-4 py-3 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-600 rounded-lg text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-[var(--brand)]"
            >
              {TEMPLATE_OPTIONS.map((opt) => (
                <option key={opt.value} value={opt.value}>
                  {opt.label}
                </option>
              ))}
            </select>
            <div className="text-xs text-gray-600 dark:text-gray-400">
              PDF is generated automatically. No manual edits.
            </div>
          </div>
        )}

        {isBusy ? (
          <div className="text-sm text-gray-600 dark:text-gray-300">
            {mode === "upload" ? "Uploading…" : "Generating…"}
          </div>
        ) : null}

        {error ? (
          <div className="rounded-lg bg-red-50 dark:bg-red-900/30 p-3 text-sm text-red-600 dark:text-red-400">
            {error}
          </div>
        ) : null}

        <div className="flex flex-col sm:flex-row gap-3">
          <button
            type="button"
            onClick={onPrimary}
            disabled={isBusy || !propertyId}
            className="inline-flex items-center justify-center gap-2 w-full px-6 py-3 bg-gray-900 dark:bg-gray-100 text-white dark:text-gray-900 rounded-lg font-medium transition-colors hover:bg-gray-800 dark:hover:bg-gray-200 disabled:opacity-50"
          >
            {isBusy ? (
              <Loader2 size={18} className="animate-spin" />
            ) : mode === "upload" ? (
              <FileUp size={18} />
            ) : (
              <FileText size={18} />
            )}
            {primaryLabel}
          </button>

          <button
            type="button"
            onClick={closeAndReset}
            disabled={isBusy}
            className="inline-flex items-center justify-center gap-2 w-full px-6 py-3 rounded-lg font-medium transition-colors border border-gray-200 dark:border-gray-700 bg-white dark:bg-[#262626] text-gray-900 dark:text-gray-100 hover:bg-gray-50 dark:hover:bg-gray-800 disabled:opacity-50"
          >
            Cancel
          </button>
        </div>
      </div>
    </ModalShell>
  );
}
