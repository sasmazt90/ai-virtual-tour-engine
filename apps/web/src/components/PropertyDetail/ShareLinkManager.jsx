import { useState } from "react";
import { useShareLink } from "@/hooks/useShareLink";
import { ShareLinkDisplay } from "./ShareLinkDisplay";

export function ShareLinkManager({
  propertyId,
  ownerClient,
  interestedClients,
  onOpenShareForClient,
}) {
  const [shareClientId, setShareClientId] = useState("");

  const {
    activeLink,
    activeUrl,
    supportSafeUrl,
    activeExpired,
    lastViewedText,
    hasActiveLink,
    activeLinkLoading,
    disableMutation,
    extendMutation,
  } = useShareLink({ propertyId, shareClientId });

  const primaryShareCtaLabel = hasActiveLink
    ? "Replace Share Link"
    : "Create Share Link";

  const showTopShareButton =
    !shareClientId || activeLinkLoading || hasActiveLink;

  const onCreateOrReplace = () => {
    if (!shareClientId) return;
    if (typeof onOpenShareForClient === "function") {
      onOpenShareForClient(shareClientId);
    }
  };

  return (
    <div className="mb-4 rounded-lg border border-gray-200 dark:border-gray-700 p-3">
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
        <div className="min-w-0">
          <div className="text-xs text-gray-500 dark:text-gray-400 font-jetbrains-mono">
            Share link (client-specific)
          </div>
          <div className="mt-1">
            <select
              value={shareClientId}
              onChange={(e) => setShareClientId(e.target.value)}
              className="w-full sm:w-[320px] px-3 py-2 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-600 rounded-lg text-sm text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-[var(--brand)] font-jetbrains-mono"
            >
              <option value="">Select client…</option>
              {ownerClient?.id ? (
                <optgroup label="Owner">
                  <option value={ownerClient.id}>
                    {ownerClient.label} (Owner)
                  </option>
                </optgroup>
              ) : null}
              {interestedClients.length > 0 ? (
                <optgroup label="Interested Clients">
                  {interestedClients.map((c) => (
                    <option key={c.id} value={c.id}>
                      {c.label}
                    </option>
                  ))}
                </optgroup>
              ) : null}
            </select>
          </div>
        </div>

        <div className="flex flex-wrap gap-2">
          {showTopShareButton ? (
            <button
              type="button"
              onClick={onCreateOrReplace}
              disabled={!shareClientId}
              className="inline-flex items-center gap-2 px-3 py-2 rounded-lg bg-gray-900 dark:bg-gray-100 text-white dark:text-gray-900 text-sm font-medium hover:bg-gray-800 dark:hover:bg-gray-200 disabled:opacity-50 font-jetbrains-mono"
              title="Creates a new share link for the selected client (older active links will be disabled)"
            >
              <svg
                xmlns="http://www.w3.org/2000/svg"
                width="16"
                height="16"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71" />
                <path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71" />
              </svg>
              {primaryShareCtaLabel}
            </button>
          ) : null}
        </div>
      </div>

      {!shareClientId ? (
        <div className="mt-2 text-xs text-amber-700 dark:text-amber-300 font-jetbrains-mono">
          Pick exactly one client to manage the share link.
        </div>
      ) : null}

      {shareClientId ? (
        <div className="mt-3">
          {activeLinkLoading ? (
            <div className="text-xs text-gray-600 dark:text-gray-400 font-jetbrains-mono">
              Loading share link…
            </div>
          ) : hasActiveLink ? (
            <ShareLinkDisplay
              activeLink={activeLink}
              activeUrl={activeUrl}
              supportSafeUrl={supportSafeUrl}
              activeExpired={activeExpired}
              lastViewedText={lastViewedText}
              disableMutation={disableMutation}
              extendMutation={extendMutation}
            />
          ) : (
            <div className="flex items-center justify-between gap-3">
              <div className="text-xs text-gray-600 dark:text-gray-400 font-jetbrains-mono">
                No active share link for this client.
              </div>
              <button
                type="button"
                onClick={onCreateOrReplace}
                className="inline-flex items-center gap-2 px-3 py-2 rounded-lg bg-gray-900 dark:bg-gray-100 text-white dark:text-gray-900 text-sm font-medium hover:bg-gray-800 dark:hover:bg-gray-200 font-jetbrains-mono"
                title="Create a new share link for the selected client"
              >
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  width="16"
                  height="16"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                >
                  <path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71" />
                  <path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71" />
                </svg>
                Create Share Link
              </button>
            </div>
          )}
        </div>
      ) : null}
    </div>
  );
}
