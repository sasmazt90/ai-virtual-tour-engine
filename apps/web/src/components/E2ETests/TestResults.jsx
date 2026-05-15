import { StatusPill } from "./StatusPill";

export function TestResults({ results }) {
  return (
    <div className="mt-6 bg-white dark:bg-[#262626] rounded-xl shadow-lg dark:shadow-none dark:ring-1 dark:ring-gray-700 p-5">
      <div className="text-sm font-semibold text-gray-900 dark:text-gray-100 font-jetbrains-mono">
        Results
      </div>

      {results.length === 0 ? (
        <div className="mt-3 text-sm text-gray-600 dark:text-gray-400 font-jetbrains-mono">
          No results yet.
        </div>
      ) : (
        <div className="mt-3 space-y-2">
          {results.map((r, idx) => {
            const details = r?.details ? String(r.details) : "";
            const jobIdText = r?.jobId ? String(r.jobId) : "";

            return (
              <div
                key={`${idx}-${r.test}`}
                className="rounded-lg border border-gray-200 dark:border-gray-700 p-3"
              >
                <div className="flex items-start justify-between gap-3">
                  <div className="min-w-0">
                    <div className="text-sm text-gray-900 dark:text-gray-100 font-jetbrains-mono">
                      {r.test}
                    </div>
                    {jobIdText ? (
                      <div className="mt-1 text-xs text-gray-500 dark:text-gray-500 font-jetbrains-mono">
                        Job: {jobIdText}
                      </div>
                    ) : null}
                    {details ? (
                      <div className="mt-1 text-xs text-gray-600 dark:text-gray-300 font-jetbrains-mono">
                        {details}
                      </div>
                    ) : null}
                  </div>
                  <StatusPill status={r.status} />
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
