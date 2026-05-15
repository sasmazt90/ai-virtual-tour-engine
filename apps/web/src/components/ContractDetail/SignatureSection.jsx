import { Loader2, Check } from "lucide-react";

export function SignatureSection({
  signatureMethod,
  signedStatus,
  signedAt,
  signedByClientName,
  signedByAgentName,
  onClientNameChange,
  onAgentNameChange,
  onMarkSigned,
  onMarkUnsigned,
  isMarkingSignedPending,
  isMarkingUnsignedPending,
}) {
  const isSigned = signedStatus === "signed";
  const canMarkSigned = !isSigned;

  const signatureStatusLabel = isSigned ? "Signed" : "Unsigned";
  const signatureBadgeClass = isSigned
    ? "text-emerald-700 dark:text-emerald-300"
    : "text-amber-700 dark:text-amber-300";

  return (
    <div className="bg-white dark:bg-[#262626] rounded-xl shadow-lg dark:shadow-none dark:ring-1 dark:ring-gray-700 p-6">
      <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-4 font-jetbrains-mono">
        Signatures
      </h2>

      <div className="text-sm text-gray-700 dark:text-gray-200 font-jetbrains-mono">
        <div className="flex items-center justify-between gap-3">
          <div>
            <div className="text-xs uppercase tracking-wide text-gray-500 dark:text-gray-400">
              Status
            </div>
            <div className={`mt-1 font-medium ${signatureBadgeClass}`}>
              {signatureStatusLabel}
            </div>
          </div>

          <div className="text-right">
            <div className="text-xs uppercase tracking-wide text-gray-500 dark:text-gray-400">
              Method
            </div>
            <div className="mt-1">{signatureMethod}</div>
          </div>
        </div>

        {signedStatus === "signed" && signedAt ? (
          <div className="mt-3">
            <div className="text-xs uppercase tracking-wide text-gray-500 dark:text-gray-400">
              Signed at
            </div>
            <div className="mt-1">{new Date(signedAt).toLocaleString()}</div>
          </div>
        ) : null}

        {canMarkSigned ? (
          <div className="mt-4 space-y-3">
            <div className="space-y-2">
              <label className="block text-xs uppercase tracking-wide text-gray-500 dark:text-gray-400">
                Signed by (client)
              </label>
              <input
                value={signedByClientName}
                onChange={(e) => onClientNameChange(e.target.value)}
                className="w-full px-3 py-2 rounded-lg bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-600 text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-[var(--brand)] font-jetbrains-mono"
              />
            </div>
            <div className="space-y-2">
              <label className="block text-xs uppercase tracking-wide text-gray-500 dark:text-gray-400">
                Signed by (agent)
              </label>
              <input
                value={signedByAgentName}
                onChange={(e) => onAgentNameChange(e.target.value)}
                className="w-full px-3 py-2 rounded-lg bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-600 text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-[var(--brand)] font-jetbrains-mono"
              />
            </div>

            <button
              type="button"
              disabled={isMarkingSignedPending}
              onClick={onMarkSigned}
              className="inline-flex items-center justify-center gap-2 w-full px-4 py-2 rounded-lg bg-gray-900 dark:bg-gray-100 text-white dark:text-gray-900 text-sm font-medium hover:bg-gray-800 dark:hover:bg-gray-200 disabled:opacity-50 font-jetbrains-mono"
            >
              {isMarkingSignedPending ? (
                <Loader2 size={16} className="animate-spin" />
              ) : (
                <Check size={16} />
              )}
              Mark as Signed
            </button>

            <p className="text-xs text-gray-500 dark:text-gray-500">
              This does not apply an e-signature. It only tracks that the
              document was signed off-platform.
            </p>
          </div>
        ) : (
          <div className="mt-4 space-y-3">
            <div className="text-xs text-gray-500 dark:text-gray-500 font-jetbrains-mono">
              This is off-platform tracking only.
            </div>

            <button
              type="button"
              disabled={isMarkingUnsignedPending}
              onClick={onMarkUnsigned}
              className="inline-flex items-center justify-center gap-2 w-full px-4 py-2 rounded-lg border border-amber-300 dark:border-amber-700 bg-amber-50 dark:bg-amber-900/20 text-amber-900 dark:text-amber-200 text-sm font-medium hover:bg-amber-100 dark:hover:bg-amber-900/30 disabled:opacity-50 font-jetbrains-mono"
              title="Correction flow: mark the contract as unsigned"
            >
              {isMarkingUnsignedPending ? (
                <Loader2 size={16} className="animate-spin" />
              ) : (
                <Check size={16} />
              )}
              Mark as Unsigned
            </button>

            <p className="text-xs text-amber-800 dark:text-amber-200 font-jetbrains-mono">
              Warning: Use this only for corrections. This does not delete any
              PDFs.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
