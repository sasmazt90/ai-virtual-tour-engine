import { Copy, Shield, Timer, XCircle, Check } from "lucide-react";
import { copyToClipboard } from "@/utils/shareLinkHelpers";

export function ShareLinkDisplay({
  activeLink,
  activeUrl,
  supportSafeUrl,
  activeExpired,
  lastViewedText,
  disableMutation,
  extendMutation,
}) {
  return (
    <div className="space-y-2">
      <div className="flex items-center gap-3 text-xs font-jetbrains-mono">
        <span
          className={
            activeExpired
              ? "inline-flex items-center gap-1 text-red-600 dark:text-red-400"
              : "inline-flex items-center gap-1 text-green-700 dark:text-green-300"
          }
        >
          {activeExpired ? <XCircle size={14} /> : <Check size={14} />}
          {activeExpired ? "Expired" : "Active"}
        </span>
        <span className="text-gray-600 dark:text-gray-400 inline-flex items-center gap-1">
          <Timer size={14} />
          Expires:{" "}
          {activeLink.expires_at
            ? new Date(activeLink.expires_at).toLocaleString()
            : "No expiry"}
        </span>
        <span className="text-gray-600 dark:text-gray-400">
          Last viewed: {lastViewedText}
        </span>
      </div>

      <div className="flex items-center justify-between gap-3">
        <a
          href={activeUrl}
          target="_blank"
          rel="noreferrer"
          className="text-sm text-[var(--brandDark)] dark:text-[var(--brand)] hover:underline font-jetbrains-mono truncate"
        >
          {activeUrl}
        </a>
        <div className="flex flex-wrap gap-2">
          <button
            type="button"
            onClick={() => copyToClipboard(activeUrl)}
            className="inline-flex items-center gap-2 px-3 py-2 rounded-lg bg-gray-100 dark:bg-gray-800 hover:bg-gray-200 dark:hover:bg-gray-700"
          >
            <Copy size={16} />
            <span className="text-sm font-jetbrains-mono">Copy</span>
          </button>

          {supportSafeUrl ? (
            <button
              type="button"
              onClick={() => copyToClipboard(supportSafeUrl)}
              className="inline-flex items-center gap-2 px-3 py-2 rounded-lg bg-gray-100 dark:bg-gray-800 hover:bg-gray-200 dark:hover:bg-gray-700"
              title="Same URL with ?view=readonly hint"
            >
              <Shield size={16} />
              <span className="text-sm font-jetbrains-mono">Support-safe</span>
            </button>
          ) : null}

          <button
            type="button"
            onClick={() =>
              extendMutation.mutate({
                id: activeLink.id,
                extendDays: 7,
              })
            }
            disabled={extendMutation.isPending}
            className="inline-flex items-center gap-2 px-3 py-2 rounded-lg bg-white dark:bg-[#262626] border border-gray-200 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-800 disabled:opacity-50"
          >
            <Timer size={16} />
            <span className="text-sm font-jetbrains-mono">+7d</span>
          </button>

          <button
            type="button"
            onClick={() =>
              extendMutation.mutate({
                id: activeLink.id,
                extendDays: 30,
              })
            }
            disabled={extendMutation.isPending}
            className="inline-flex items-center gap-2 px-3 py-2 rounded-lg bg-white dark:bg-[#262626] border border-gray-200 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-800 disabled:opacity-50"
          >
            <Timer size={16} />
            <span className="text-sm font-jetbrains-mono">+30d</span>
          </button>

          <button
            type="button"
            onClick={() => disableMutation.mutate({ id: activeLink.id })}
            disabled={disableMutation.isPending}
            className="inline-flex items-center gap-2 px-3 py-2 rounded-lg bg-white dark:bg-[#262626] border border-gray-200 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-800 disabled:opacity-50"
            title="Disable immediately"
          >
            <XCircle size={16} />
            <span className="text-sm font-jetbrains-mono">Disable</span>
          </button>
        </div>
      </div>
    </div>
  );
}
