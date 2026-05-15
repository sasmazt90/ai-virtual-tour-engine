export function AuditSection({ auditEntries }) {
  const entriesRaw = Array.isArray(auditEntries) ? auditEntries : [];

  // Normalize + sort oldest -> newest (chronological)
  const entries = [...entriesRaw].sort((a, b) => {
    const ta = a?.timestamp ? new Date(a.timestamp).getTime() : 0;
    const tb = b?.timestamp ? new Date(b.timestamp).getTime() : 0;
    if (Number.isNaN(ta) && Number.isNaN(tb)) return 0;
    if (Number.isNaN(ta)) return -1;
    if (Number.isNaN(tb)) return 1;
    return ta - tb;
  });

  return (
    <div className="bg-white dark:bg-[#262626] rounded-xl shadow-lg dark:shadow-none dark:ring-1 dark:ring-gray-700 p-6">
      <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-4 font-jetbrains-mono">
        Audit
      </h2>

      <details className="font-jetbrains-mono">
        <summary className="cursor-pointer text-sm text-gray-700 dark:text-gray-200">
          View audit entries ({entries.length})
        </summary>
        <div className="mt-3 space-y-2">
          {entries.length === 0 ? (
            <div className="text-xs text-gray-500 dark:text-gray-500">
              No audit entries yet.
            </div>
          ) : (
            entries.map((a, idx) => {
              const action = a?.action || "";
              const timestamp = a?.timestamp || "";
              const actor = a?.actor || "";
              const notes = a?.notes || null; // legacy-safe
              const changes =
                a?.changes && typeof a.changes === "object" ? a.changes : null;
              const changeKeys = changes ? Object.keys(changes) : [];

              return (
                <div
                  key={`${timestamp}-${idx}`}
                  className="rounded-lg border border-gray-200 dark:border-gray-700 p-3 text-xs text-gray-700 dark:text-gray-200"
                >
                  <div className="flex items-center justify-between gap-3">
                    <div className="font-semibold">{action}</div>
                    <div className="text-gray-500 dark:text-gray-400">
                      {timestamp ? new Date(timestamp).toLocaleString() : ""}
                    </div>
                  </div>
                  <div className="mt-1 text-gray-500 dark:text-gray-400">
                    Actor: {actor || "—"}
                  </div>

                  {changeKeys.length > 0 ? (
                    <div className="mt-2 space-y-1">
                      {changeKeys.map((k) => {
                        const c = changes[k] || {};
                        const fromVal = c.from;
                        const toVal = c.to;
                        const fromText =
                          fromVal === null || fromVal === undefined
                            ? "—"
                            : String(fromVal);
                        const toText =
                          toVal === null || toVal === undefined
                            ? "—"
                            : String(toVal);
                        return (
                          <div
                            key={k}
                            className="text-gray-600 dark:text-gray-300"
                          >
                            <span className="font-semibold">{k}</span>:{" "}
                            {fromText} → {toText}
                          </div>
                        );
                      })}
                    </div>
                  ) : null}

                  {notes ? (
                    <div className="mt-1 text-gray-600 dark:text-gray-300">
                      Notes: {notes}
                    </div>
                  ) : null}
                </div>
              );
            })
          )}
        </div>
      </details>
    </div>
  );
}
