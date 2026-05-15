import { Download, ExternalLink } from "lucide-react";

export function PDFPreview({ pdfInlineUrl, pdfDownloadUrl, pdfMessage }) {
  const downloadDisabled = !pdfDownloadUrl;
  const openDisabled = !pdfInlineUrl;

  return (
    <div className="bg-white dark:bg-[#262626] rounded-xl shadow-lg dark:shadow-none dark:ring-1 dark:ring-gray-700 p-6">
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 mb-4">
        <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100 font-jetbrains-mono">
          PDF Preview
        </h2>

        <div className="flex items-center gap-2">
          <a
            href={pdfInlineUrl || "#"}
            target={openDisabled ? undefined : "_blank"}
            rel={openDisabled ? undefined : "noreferrer"}
            aria-disabled={openDisabled}
            onClick={(e) => {
              if (openDisabled) {
                e.preventDefault();
              }
            }}
            className={
              openDisabled
                ? "inline-flex items-center justify-center gap-2 px-4 py-2 rounded-lg text-sm font-medium font-jetbrains-mono bg-gray-200 text-gray-500 dark:bg-gray-800 dark:text-gray-400 cursor-not-allowed"
                : "inline-flex items-center justify-center gap-2 px-4 py-2 rounded-lg text-sm font-medium font-jetbrains-mono bg-white dark:bg-[#1E1E1E] text-gray-900 dark:text-gray-100 border border-gray-200 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-[#2A2A2A] transition-colors"
            }
            title={openDisabled ? "PDF not available" : "Open PDF"}
          >
            <ExternalLink size={16} />
            Open
          </a>

          <a
            href={pdfDownloadUrl || "#"}
            aria-disabled={downloadDisabled}
            onClick={(e) => {
              if (downloadDisabled) {
                e.preventDefault();
              }
            }}
            className={
              downloadDisabled
                ? "inline-flex items-center justify-center gap-2 px-4 py-2 rounded-lg text-sm font-medium font-jetbrains-mono bg-gray-200 text-gray-500 dark:bg-gray-800 dark:text-gray-400 cursor-not-allowed"
                : "inline-flex items-center justify-center gap-2 px-4 py-2 rounded-lg text-sm font-medium font-jetbrains-mono bg-gray-900 dark:bg-gray-100 text-white dark:text-gray-900 hover:bg-gray-800 dark:hover:bg-gray-200 transition-colors"
            }
            title={downloadDisabled ? "PDF not available" : "Download PDF"}
          >
            <Download size={16} />
            Download
          </a>
        </div>
      </div>

      {pdfInlineUrl ? (
        <iframe
          src={pdfInlineUrl}
          title="Contract PDF"
          className="w-full rounded-lg border border-gray-200 dark:border-gray-700"
          style={{ height: 720 }}
        />
      ) : (
        <div className="text-sm text-gray-600 dark:text-gray-400 font-jetbrains-mono">
          {pdfMessage}
        </div>
      )}
    </div>
  );
}
