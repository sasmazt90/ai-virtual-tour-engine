import { useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import useUser from "@/utils/useUser";
import { Header } from "@/components/Header";
import { ArrowLeft, RefreshCcw } from "lucide-react";
import { StatusBanner } from "@/components/StatusBanner";
import { useAIBusy } from "@/hooks/useAIBusy";

function formatDurationMs(ms) {
  if (!Number.isFinite(ms) || ms < 0) return "—";
  const sec = Math.round(ms / 1000);
  if (sec < 60) return `${sec}s`;
  const min = Math.floor(sec / 60);
  const rem = sec % 60;
  return `${min}m ${rem}s`;
}

export default function AIJobsPage() {
  const queryClient = useQueryClient();
  const { data: user, loading: userLoading } = useUser();
  const [actionError, setActionError] = useState(null);

  const { data: busyData } = useAIBusy(user?.id, {
    enabled: !!user?.id,
    refetchInterval: 10000,
  });

  const {
    data: jobs = [],
    isLoading,
    error,
  } = useQuery({
    queryKey: ["ai-jobs", user?.id],
    queryFn: async () => {
      const res = await fetch("/api/ai/jobs?limit=20");
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body?.error || "Failed to load AI jobs");
      }
      return res.json();
    },
    enabled: !!user?.id,
  });

  const retryMutation = useMutation({
    mutationFn: async (jobId) => {
      const res = await fetch(`/api/ai/jobs/${jobId}/retry`, {
        method: "POST",
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));

        const err = new Error(body?.error || "Could not retry that job.");
        err.status = res.status;
        throw err;
      }
      return res.json();
    },
    onSuccess: async () => {
      setActionError(null);
      await queryClient.invalidateQueries({ queryKey: ["ai-jobs", user?.id] });
      await queryClient.invalidateQueries({ queryKey: ["credits", user?.id] });
    },
    onError: (err) => {
      // Final copy pass: keep this calm and non-technical.
      const status = Number(err?.status || 0);
      const msg = String(err?.message || "");

      const friendly =
        status === 402
          ? "Not enough credits to retry this job."
          : status === 409
            ? "A retry is already in progress. Please wait a moment."
            : status === 401
              ? "Please sign in again and try retrying."
              : msg
                ? msg
                : "Could not retry that job. Please try again.";

      setActionError(friendly);
    },
  });

  const rows = useMemo(() => {
    const arr = Array.isArray(jobs) ? jobs : [];
    return arr.map((j) => {
      const createdAt = j?.created_at ? new Date(j.created_at) : null;
      const startedAt = j?.started_at ? new Date(j.started_at) : null;
      const updatedAt = j?.updated_at ? new Date(j.updated_at) : null;

      const isDone =
        j?.job_status === "succeeded" || j?.job_status === "failed";
      const durationMs =
        isDone &&
        startedAt &&
        updatedAt &&
        !Number.isNaN(startedAt.getTime()) &&
        !Number.isNaN(updatedAt.getTime())
          ? updatedAt.getTime() - startedAt.getTime()
          : null;

      return {
        ...j,
        createdAtText:
          createdAt && !Number.isNaN(createdAt.getTime())
            ? createdAt.toLocaleString()
            : "—",
        durationText: durationMs ? formatDurationMs(durationMs) : "—",
      };
    });
  }, [jobs]);

  if (userLoading) {
    return (
      <div className="min-h-screen bg-gray-50 dark:bg-[#1E1E1E]">
        <Header />
        <div className="pt-16 flex items-center justify-center min-h-screen">
          <p className="text-gray-600 dark:text-gray-400 font-jetbrains-mono">
            Loading…
          </p>
        </div>
      </div>
    );
  }

  if (!user) {
    if (typeof window !== "undefined") {
      window.location.href = "/account/signin";
    }
    return null;
  }

  const errorMessage = error ? "Could not load AI jobs right now." : null;
  const actionErrorMessage = actionError ? String(actionError) : null;

  const showBusy = busyData?.busy === true;
  const busyText = showBusy
    ? `AI processing in progress${
        busyData?.queued || busyData?.running
          ? ` (queued: ${Number(busyData?.queued || 0)}, running: ${Number(
              busyData?.running || 0,
            )})`
          : ""
      }`
    : null;

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-[#1E1E1E]">
      <Header />

      <div className="pt-16">
        <div className="max-w-6xl mx-auto px-4 sm:px-8 py-8 sm:py-12">
          <a
            href="/profile"
            className="inline-flex items-center gap-2 text-sm text-gray-600 dark:text-gray-300 hover:text-gray-900 dark:hover:text-gray-100 font-jetbrains-mono"
          >
            <ArrowLeft size={16} />
            Back to Profile
          </a>

          <div className="mt-6 mb-6">
            <div className="flex flex-col sm:flex-row sm:items-end sm:justify-between gap-3">
              <div>
                <h1 className="text-3xl sm:text-4xl font-bold text-gray-900 dark:text-gray-100 mb-2 font-jetbrains-mono">
                  AI Jobs
                </h1>
                <p className="text-gray-600 dark:text-gray-300 font-jetbrains-mono">
                  Lightweight operations view (last 20).
                </p>
              </div>
              {showBusy ? (
                <div className="inline-flex items-center">
                  <span className="inline-flex items-center justify-center px-2 py-1 rounded-full text-xs leading-none whitespace-nowrap bg-amber-50 dark:bg-amber-900/20 text-amber-800 dark:text-amber-200 font-jetbrains-mono">
                    AI processing in progress
                  </span>
                </div>
              ) : null}
            </div>
          </div>

          {showBusy && busyText ? (
            <div className="mb-4">
              <StatusBanner variant="warning" title="Busy">
                {busyText}
              </StatusBanner>
            </div>
          ) : null}

          {errorMessage ? (
            <div className="mb-4">
              <StatusBanner variant="error">{errorMessage}</StatusBanner>
            </div>
          ) : null}

          {actionErrorMessage ? (
            <div className="mb-4">
              <StatusBanner variant="error">{actionErrorMessage}</StatusBanner>
            </div>
          ) : null}

          <div className="bg-white dark:bg-[#262626] rounded-xl shadow-lg dark:shadow-none dark:ring-1 dark:ring-gray-700 overflow-hidden">
            <div className="flex items-center justify-between px-6 py-4 border-b border-gray-200 dark:border-gray-700">
              <div className="text-sm text-gray-600 dark:text-gray-300 font-jetbrains-mono">
                {isLoading ? "Loading…" : `${rows.length} jobs`}
              </div>
              <button
                type="button"
                onClick={() =>
                  queryClient.invalidateQueries({
                    queryKey: ["ai-jobs", user?.id],
                  })
                }
                className="inline-flex items-center gap-2 px-3 py-2 rounded-lg bg-gray-100 dark:bg-gray-800 hover:bg-gray-200 dark:hover:bg-gray-700"
              >
                <RefreshCcw size={16} />
                <span className="text-sm font-jetbrains-mono">Refresh</span>
              </button>
            </div>

            {rows.length === 0 && !isLoading ? (
              <div className="p-8 text-sm text-gray-600 dark:text-gray-300 font-jetbrains-mono">
                No jobs yet.
              </div>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead className="bg-gray-50 dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700">
                    <tr>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider font-jetbrains-mono">
                        Created
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider font-jetbrains-mono">
                        Type
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider font-jetbrains-mono">
                        Property
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider font-jetbrains-mono">
                        Status
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider font-jetbrains-mono">
                        Duration
                      </th>
                      <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider font-jetbrains-mono">
                        Action
                      </th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
                    {rows.map((j) => {
                      const canRetry = j.job_status === "failed";
                      const propertyText =
                        j.property_title || j.property_id || "—";
                      const propertyHref = j.property_id
                        ? `/properties/${j.property_id}`
                        : null;

                      const statusClass =
                        j.job_status === "succeeded"
                          ? "text-green-700 dark:text-green-300"
                          : j.job_status === "failed"
                            ? "text-red-700 dark:text-red-300"
                            : "text-amber-700 dark:text-amber-300";

                      const isRetryingThis =
                        retryMutation.isPending &&
                        retryMutation.variables === j.id;

                      const handleRetry = () => {
                        if (isRetryingThis) return;
                        setActionError(null);
                        retryMutation.mutate(j.id);
                      };

                      const actionButton = canRetry ? (
                        <button
                          type="button"
                          disabled={isRetryingThis}
                          onClick={handleRetry}
                          className="inline-flex items-center gap-2 px-3 py-2 rounded-lg bg-gray-900 dark:bg-gray-100 text-white dark:text-gray-900 text-sm font-medium hover:bg-gray-800 dark:hover:bg-gray-200 disabled:opacity-50 font-jetbrains-mono"
                        >
                          {isRetryingThis ? "Retrying…" : "Retry"}
                        </button>
                      ) : (
                        <span className="text-xs text-gray-500 dark:text-gray-400 font-jetbrains-mono">
                          —
                        </span>
                      );

                      const rawError = j.error_message
                        ? String(j.error_message)
                        : "";

                      const resultPayload =
                        j.result_payload && typeof j.result_payload === "object"
                          ? j.result_payload
                          : null;

                      const vacantQaResults = Array.isArray(
                        resultPayload?.vacantQaResults,
                      )
                        ? resultPayload.vacantQaResults
                        : [];

                      const isVacantQaError =
                        rawError.includes("VACANT staging QA failed") ||
                        rawError.includes("VACANT QA") ||
                        vacantQaResults.length > 0;

                      const showExpandableError =
                        isVacantQaError ||
                        rawError.includes("\n") ||
                        rawError.length > 120;

                      const shortError =
                        rawError.length > 160
                          ? rawError.slice(0, 160) + "…"
                          : rawError;

                      const qaSummaryLines = vacantQaResults
                        .slice(-12)
                        .map((r) => {
                          const passText = r?.pass ? "PASS" : "FAIL";
                          const variant = r?.variant ? String(r.variant) : "";
                          const attempt = Number(r?.attempt || 0) || 0;
                          const forbidden = Array.isArray(
                            r?.verifier?.forbidden_objects_detected,
                          )
                            ? r.verifier.forbidden_objects_detected
                            : [];
                          const newArch = Array.isArray(
                            r?.verifier?.new_architecture_detected,
                          )
                            ? r.verifier.new_architecture_detected
                            : [];

                          const parts = [];
                          parts.push(
                            `${passText} photo=${String(r?.photoId || "").slice(0, 8)} variant=${variant} attempt=${attempt}`,
                          );

                          if (forbidden.length) {
                            parts.push(`forbidden=[${forbidden.join(", ")}]`);
                          }

                          if (r?.verifier?.silhouette_or_repaint_detected) {
                            parts.push("repaint_or_silhouette=true");
                          }

                          if (newArch.length) {
                            parts.push(`new_arch=[${newArch.join(", ")}]`);
                          }

                          if (r?.verifier?.indoor_light_detected_in_vacant) {
                            parts.push("indoor_light=true");
                          }

                          const violations = Array.isArray(
                            r?.verifier?.violations,
                          )
                            ? r.verifier.violations
                            : [];
                          if (violations.length) {
                            parts.push(`violations=${violations.join(" | ")}`);
                          }

                          return parts.join(" • ");
                        });

                      const errorBlock =
                        !rawError &&
                        vacantQaResults.length ===
                          0 ? null : showExpandableError ? (
                          <details className="mt-1">
                            <summary className="text-xs text-gray-700 dark:text-gray-300 cursor-pointer select-none">
                              View error details
                            </summary>

                            {rawError ? (
                              <pre className="mt-2 whitespace-pre-wrap break-words text-[11px] leading-snug text-gray-700 dark:text-gray-300 bg-gray-50 dark:bg-gray-900/40 border border-gray-200 dark:border-gray-700 rounded-lg p-2 max-w-[520px]">
                                {rawError}
                              </pre>
                            ) : null}

                            {qaSummaryLines.length > 0 ? (
                              <pre className="mt-2 whitespace-pre-wrap break-words text-[11px] leading-snug text-gray-700 dark:text-gray-300 bg-white dark:bg-black/20 border border-gray-200 dark:border-gray-700 rounded-lg p-2 max-w-[520px]">
                                {qaSummaryLines.join("\n")}
                              </pre>
                            ) : null}
                          </details>
                        ) : (
                          <div className="mt-1 text-xs text-gray-600 dark:text-gray-400 max-w-[420px] truncate">
                            {rawError ? shortError : "Job failed"}
                          </div>
                        );

                      return (
                        <tr key={j.id}>
                          <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 dark:text-gray-100 font-jetbrains-mono">
                            {j.createdAtText}
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-700 dark:text-gray-300 font-jetbrains-mono">
                            {j.job_type}
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap text-sm font-jetbrains-mono">
                            {propertyHref ? (
                              <a
                                href={propertyHref}
                                className="text-[var(--brandDark)] dark:text-[var(--brand)] hover:underline"
                              >
                                {propertyText}
                              </a>
                            ) : (
                              <span className="text-gray-700 dark:text-gray-300">
                                {propertyText}
                              </span>
                            )}
                          </td>
                          <td
                            className={`px-6 py-4 whitespace-nowrap text-sm font-jetbrains-mono ${statusClass}`}
                          >
                            {j.job_status}
                            {j.job_status === "running" ||
                            j.job_status === "queued" ? (
                              <span className="text-gray-500 dark:text-gray-400">
                                {" "}
                                • {Number(j.progress || 0)}%
                              </span>
                            ) : null}

                            {j.job_status === "failed" ? errorBlock : null}
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-700 dark:text-gray-300 font-jetbrains-mono">
                            {j.durationText}
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap text-right">
                            {actionButton}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </div>

          <div className="mt-3 text-xs text-gray-500 dark:text-gray-500 font-jetbrains-mono">
            Retry re-checks your current credit balance.
          </div>
        </div>
      </div>
    </div>
  );
}
