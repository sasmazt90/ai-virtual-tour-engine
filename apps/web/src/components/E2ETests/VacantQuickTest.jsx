import { StatusBanner } from "@/components/StatusBanner";
import { StatusPill } from "./StatusPill";

export function VacantQuickTest({ vacantQuick, onRun, canRun }) {
  return (
    <div className="mt-5 rounded-lg border border-gray-200 dark:border-gray-700 p-4">
      <div className="flex items-start justify-between gap-3">
        <div>
          <div className="text-sm font-semibold text-gray-900 dark:text-gray-100 font-jetbrains-mono">
            Quick VACANT test (1 photo)
          </div>
          <div className="mt-1 text-xs text-gray-500 dark:text-gray-500 font-jetbrains-mono">
            Runs VACANT on the first photo only and shows the final Day/Night
            image URLs.
          </div>
        </div>
        <StatusPill status={vacantQuick.status} />
      </div>

      <div className="mt-3 flex items-center justify-end gap-2">
        <button
          type="button"
          onClick={onRun}
          disabled={!canRun}
          className="inline-flex items-center px-4 py-2 rounded-lg bg-gray-900 dark:bg-gray-100 text-white dark:text-gray-900 text-sm font-medium hover:bg-gray-800 dark:hover:bg-gray-200 disabled:opacity-50 font-jetbrains-mono"
        >
          Run VACANT (1 photo)
        </button>
      </div>

      {vacantQuick.jobId ? (
        <div className="mt-3 text-xs text-gray-600 dark:text-gray-300 font-jetbrains-mono">
          Job: {vacantQuick.jobId}
        </div>
      ) : null}

      {vacantQuick.error ? (
        <div className="mt-3">
          <StatusBanner variant="error">{vacantQuick.error}</StatusBanner>
        </div>
      ) : null}

      {vacantQuick.status === "pass" ? (
        <div className="mt-4 grid grid-cols-1 sm:grid-cols-2 gap-4">
          <div className="rounded-lg border border-gray-200 dark:border-gray-700 p-3">
            <div className="text-xs text-gray-500 dark:text-gray-500 font-jetbrains-mono">
              Day (day_light_off)
            </div>
            {vacantQuick.dayUrl ? (
              <>
                <a
                  href={vacantQuick.dayUrl}
                  target="_blank"
                  rel="noreferrer"
                  className="mt-1 block text-xs text-blue-700 dark:text-blue-300 break-all font-jetbrains-mono"
                >
                  {vacantQuick.dayUrl}
                </a>
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={vacantQuick.dayUrl}
                  alt="VACANT day"
                  className="mt-2 w-full rounded-md border border-gray-100 dark:border-gray-800"
                />
              </>
            ) : (
              <div className="mt-2 text-xs text-gray-500 dark:text-gray-500 font-jetbrains-mono">
                Missing day URL
              </div>
            )}
          </div>

          <div className="rounded-lg border border-gray-200 dark:border-gray-700 p-3">
            <div className="text-xs text-gray-500 dark:text-gray-500 font-jetbrains-mono">
              Night (night_light_off)
            </div>
            {vacantQuick.nightUrl ? (
              <>
                <a
                  href={vacantQuick.nightUrl}
                  target="_blank"
                  rel="noreferrer"
                  className="mt-1 block text-xs text-blue-700 dark:text-blue-300 break-all font-jetbrains-mono"
                >
                  {vacantQuick.nightUrl}
                </a>
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={vacantQuick.nightUrl}
                  alt="VACANT night"
                  className="mt-2 w-full rounded-md border border-gray-100 dark:border-gray-800"
                />
              </>
            ) : (
              <div className="mt-2 text-xs text-gray-500 dark:text-gray-500 font-jetbrains-mono">
                Missing night URL
              </div>
            )}
          </div>
        </div>
      ) : null}
    </div>
  );
}
