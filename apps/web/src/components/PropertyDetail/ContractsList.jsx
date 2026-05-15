export function ContractsList({ contracts }) {
  if (!contracts || contracts.length === 0) {
    return (
      <div className="text-sm text-gray-600 dark:text-gray-400 font-jetbrains-mono">
        No contracts yet.
      </div>
    );
  }

  return (
    <div className="space-y-2">
      {contracts.map((c) => {
        const rawPdfState = c?.filled_fields?._system?.pdf || null;
        const pdfStatus = rawPdfState?.status || null;
        const pdfStoragePath = rawPdfState?.storagePath || null;
        const hasPdf = !!(c.storage_path_pdf || pdfStoragePath);

        const signedStatus = c?.filled_fields?.signed_status || "unsigned";
        const isSigned = signedStatus === "signed";

        const statusBadgeText = isSigned ? "Signed" : "Draft";
        const statusBadgeClass = isSigned
          ? "text-emerald-700 dark:text-emerald-300"
          : "text-amber-700 dark:text-amber-300";

        const pdfBadgeText = hasPdf
          ? "PDF ready"
          : pdfStatus === "disabled"
            ? "PDF disabled"
            : pdfStatus === "failed"
              ? "PDF unavailable"
              : "PDF pending";

        const pdfBadgeClass = hasPdf
          ? "text-emerald-700 dark:text-emerald-300"
          : "text-amber-700 dark:text-amber-300";

        return (
          <a
            key={c.id}
            href={`/contracts/${c.id}`}
            className="block rounded-lg border border-gray-200 dark:border-gray-700 p-3 text-sm text-gray-700 dark:text-gray-300 font-jetbrains-mono hover:bg-gray-50 dark:hover:bg-gray-800"
          >
            <div className="flex items-center justify-between gap-3">
              <div className="min-w-0 truncate">
                {c.template_type} • {c.client_name || "Client"}
              </div>
              <div className="flex items-center gap-3">
                <div className={`text-xs ${statusBadgeClass}`}>
                  {statusBadgeText}
                </div>
                <div className={`text-xs ${pdfBadgeClass}`}>{pdfBadgeText}</div>
              </div>
            </div>
          </a>
        );
      })}
    </div>
  );
}
