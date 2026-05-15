import {
  ArrowLeft,
  FileText,
  Download,
  RefreshCcw,
  Loader2,
} from "lucide-react";

export function ContractHeader({
  propertyLink,
  templateType,
  propertyTitle,
  clientName,
  pdfDownloadUrl,
  onRegeneratePdf,
  isRegenerating,
  pdfActionsDisabled,
  pdfRegenBlocked,
}) {
  const pdfActionLabel = pdfDownloadUrl ? "Regenerate PDF" : "Generate PDF";

  return (
    <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 mb-8">
      <div>
        <a
          href={propertyLink}
          className="inline-flex items-center gap-2 text-sm text-gray-600 dark:text-gray-300 hover:text-gray-900 dark:hover:text-gray-100 font-jetbrains-mono"
        >
          <ArrowLeft size={16} />
          Back
        </a>
        <div className="mt-2 flex items-center gap-3">
          <FileText className="text-[var(--brand)]" />
          <h1 className="text-3xl sm:text-4xl font-bold text-gray-900 dark:text-gray-100 font-jetbrains-mono">
            {templateType}
          </h1>
        </div>
        <p className="mt-2 text-sm text-gray-600 dark:text-gray-300 font-jetbrains-mono">
          {propertyTitle || "Property"} • {clientName || "Client"}
        </p>
      </div>

      <div className="flex flex-col sm:flex-row gap-3">
        {pdfDownloadUrl ? (
          <a
            href={pdfDownloadUrl}
            className="inline-flex items-center justify-center gap-2 px-5 py-3 bg-gray-900 dark:bg-gray-100 text-white dark:text-gray-900 rounded-lg font-medium hover:bg-gray-800 dark:hover:bg-gray-200 transition-colors font-jetbrains-mono"
          >
            <Download size={18} />
            Download PDF
          </a>
        ) : (
          <button
            type="button"
            disabled
            className="inline-flex items-center justify-center gap-2 px-5 py-3 rounded-lg font-medium font-jetbrains-mono bg-gray-200 text-gray-500 dark:bg-gray-800 dark:text-gray-400 cursor-not-allowed"
            title="PDF not available yet"
          >
            <Download size={18} />
            Download PDF
          </button>
        )}

        <button
          type="button"
          onClick={onRegeneratePdf}
          disabled={pdfActionsDisabled}
          className="inline-flex items-center justify-center gap-2 px-5 py-3 rounded-lg font-medium font-jetbrains-mono border border-gray-200 dark:border-gray-700 bg-white dark:bg-[#262626] text-gray-900 dark:text-gray-100 hover:bg-gray-50 dark:hover:bg-gray-800 disabled:opacity-50"
          title={
            pdfRegenBlocked
              ? "PDF generation is blocked after signing"
              : "Generate or regenerate the PDF"
          }
        >
          {isRegenerating ? (
            <Loader2 size={18} className="animate-spin" />
          ) : (
            <RefreshCcw size={18} />
          )}
          {pdfActionLabel}
        </button>
      </div>
    </div>
  );
}
