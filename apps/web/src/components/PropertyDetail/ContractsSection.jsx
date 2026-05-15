import { useMemo, useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Download, Eye, Trash2, User } from "lucide-react";
import useUser from "@/utils/useUser";
import AddContractModal from "./AddContractModal";

function titleFromKey(key) {
  const raw = typeof key === "string" ? key : "";
  const spaced = raw.replace(/_/g, " ").trim();
  if (!spaced) return "Contract";
  return spaced
    .split(/\s+/)
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(" ");
}

function getContractPdfUrl(contract) {
  const rawPdfState = contract?.filled_fields?._system?.pdf || null;
  const pdfStoragePath = rawPdfState?.storagePath || null;
  return (
    contract?.storage_path_pdf || contract?.pdf_url || pdfStoragePath || null
  );
}

function getContractDisplayName(contract) {
  const meta =
    contract?.metadata && typeof contract.metadata === "object"
      ? contract.metadata
      : null;
  const metaName = meta?.display_name ? String(meta.display_name) : "";
  if (metaName) return metaName;

  const templateType = contract?.template_type
    ? String(contract.template_type)
    : "Contract";
  if (templateType === "uploaded_pdf") return "Uploaded contract";
  return titleFromKey(templateType);
}

function getSourceLabel(contract) {
  const s = contract?.source_type ? String(contract.source_type) : "";
  if (s === "upload") return "Uploaded";
  if (s === "generated") return "Generated";
  // Backward compatibility: older generated contracts may not have source_type
  return contract?.storage_path_pdf ? "Generated" : "Uploaded";
}

export function ContractsSection({ property, propertyId }) {
  const queryClient = useQueryClient();
  const { data: user } = useUser();

  const [addOpen, setAddOpen] = useState(false);

  const contracts = useMemo(() => {
    return Array.isArray(property?.contracts) ? property.contracts : [];
  }, [property?.contracts]);

  const deleteMutation = useMutation({
    mutationFn: async (contractId) => {
      const res = await fetch(`/api/contracts/${contractId}`, {
        method: "DELETE",
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body?.error || "Could not delete contract");
      }
      return res.json();
    },
    onSuccess: async () => {
      if (user?.id && propertyId) {
        await queryClient.invalidateQueries({
          queryKey: ["property", user.id, propertyId],
        });
      }
    },
    onError: (err) => {
      console.error(err);
      // No UI changes requested; keep it quiet besides console.
    },
  });

  const addNewButtonClass =
    "inline-flex items-center gap-2 px-3 py-2 rounded-lg border border-gray-200 dark:border-gray-700 text-gray-700 dark:text-gray-200 text-sm font-medium hover:bg-gray-50 dark:hover:bg-gray-800 font-jetbrains-mono";

  return (
    <div>
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-semibold text-gray-900 dark:text-gray-100 font-jetbrains-mono">
          Contracts
        </h3>
        <button
          type="button"
          onClick={() => setAddOpen(true)}
          className={addNewButtonClass}
        >
          + Add New
        </button>
      </div>

      {contracts.length === 0 ? (
        <div className="text-sm text-gray-600 dark:text-gray-400 font-jetbrains-mono">
          No contracts yet.
        </div>
      ) : (
        <div className="space-y-2">
          {contracts.map((c) => {
            const pdfUrl = getContractPdfUrl(c);
            const name = getContractDisplayName(c);
            const sourceLabel = getSourceLabel(c);
            const hasPdf = !!pdfUrl;

            const clientName = c?.client_name ? String(c.client_name) : "";
            const clientLine = clientName ? `Client: ${clientName}` : null;

            const viewHref = hasPdf
              ? `/api/contracts/${encodeURIComponent(String(c.id))}/download?disposition=inline`
              : undefined;
            const downloadHref = hasPdf
              ? `/api/contracts/${encodeURIComponent(String(c.id))}/download`
              : undefined;

            return (
              <div
                key={c.id}
                className="rounded-lg border border-gray-200 dark:border-gray-700 p-3 text-sm text-gray-700 dark:text-gray-300 font-jetbrains-mono"
              >
                <div className="flex items-center justify-between gap-3">
                  <div className="min-w-0">
                    <div className="truncate text-gray-900 dark:text-gray-100">
                      {name}
                    </div>
                    <div className="mt-1 text-xs text-gray-500 dark:text-gray-400">
                      {sourceLabel}
                      {clientLine ? ` • ${clientLine}` : ""}
                    </div>
                  </div>

                  <div className="flex items-center gap-2">
                    <a
                      href={viewHref}
                      target="_blank"
                      rel="noreferrer"
                      className={`inline-flex items-center gap-2 px-3 py-2 rounded-lg border border-gray-200 dark:border-gray-700 ${
                        hasPdf
                          ? "hover:bg-gray-50 dark:hover:bg-gray-800"
                          : "opacity-50 pointer-events-none"
                      }`}
                    >
                      <Eye size={16} /> View
                    </a>

                    <a
                      href={downloadHref}
                      className={`inline-flex items-center gap-2 px-3 py-2 rounded-lg border border-gray-200 dark:border-gray-700 ${
                        hasPdf
                          ? "hover:bg-gray-50 dark:hover:bg-gray-800"
                          : "opacity-50 pointer-events-none"
                      }`}
                    >
                      <Download size={16} /> Download
                    </a>

                    <button
                      type="button"
                      onClick={() => {
                        const ok =
                          typeof window !== "undefined"
                            ? window.confirm("Delete this contract?")
                            : false;
                        if (!ok) return;
                        deleteMutation.mutate(c.id);
                      }}
                      disabled={deleteMutation.isPending}
                      className="inline-flex items-center gap-2 px-3 py-2 rounded-lg border border-gray-200 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-800 disabled:opacity-50"
                    >
                      <Trash2 size={16} /> Delete
                    </button>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}

      <AddContractModal
        open={addOpen}
        onClose={() => setAddOpen(false)}
        propertyId={propertyId}
        userId={user?.id}
        defaultClientId={property?.owner_client_id || null}
      />
    </div>
  );
}
