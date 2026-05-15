import { usePanoramaStitching } from "@/hooks/usePanoramaStitching";

export function PanoramaCreator({ propertyId, property }) {
  const {
    panoFiles,
    panoError,
    panoPreviewUrl,
    panoRemoteUrl,
    overlapMode,
    setOverlapMode,
    onPickPanoFiles,
    canGenerate,
    onGeneratePanorama,
    stitchBusy,
  } = usePanoramaStitching({ propertyId, property });

  return (
    <div className="mt-5 rounded-lg border border-dashed border-gray-200 dark:border-gray-700 p-4">
      <div className="flex flex-col sm:flex-row sm:items-start sm:justify-between gap-3">
        <div className="min-w-0">
          <div className="text-sm font-semibold text-gray-900 dark:text-gray-100 font-jetbrains-mono">
            Create panorama tour (beta)
          </div>
          <div className="mt-1 text-xs text-gray-600 dark:text-gray-400 font-jetbrains-mono">
            Pick 2–3 overlapping photos (left → right). We stitch in the browser
            and create a 1-node tour.
          </div>
        </div>

        <div className="shrink-0 flex items-center gap-2">
          <button
            type="button"
            onClick={onGeneratePanorama}
            disabled={!canGenerate || stitchBusy}
            className="inline-flex items-center gap-2 px-3 py-2 rounded-lg bg-gray-900 dark:bg-gray-100 text-white dark:text-gray-900 text-sm font-medium hover:bg-gray-800 dark:hover:bg-gray-200 disabled:opacity-50 font-jetbrains-mono"
          >
            {stitchBusy ? "Working…" : "Generate"}
          </button>
        </div>
      </div>

      <div className="mt-3 grid grid-cols-1 sm:grid-cols-2 gap-3">
        <div>
          <div className="text-xs text-gray-500 dark:text-gray-400 font-jetbrains-mono mb-2">
            Photos
          </div>
          <input
            type="file"
            accept="image/*"
            multiple
            onChange={onPickPanoFiles}
            className="block w-full text-sm text-gray-700 dark:text-gray-300 font-jetbrains-mono"
          />

          {panoFiles.length > 0 ? (
            <div className="mt-3 flex flex-wrap gap-2">
              {panoFiles.map((f) => (
                <span
                  key={f.name + String(f.size)}
                  className="inline-flex items-center px-2 py-1 rounded-full text-[11px] border border-gray-200 dark:border-gray-700 text-gray-700 dark:text-gray-200 font-jetbrains-mono"
                >
                  {f.name}
                </span>
              ))}
            </div>
          ) : null}

          <div className="mt-3">
            <div className="text-xs text-gray-500 dark:text-gray-400 font-jetbrains-mono">
              Overlap
            </div>
            <div className="mt-2 flex flex-col gap-2">
              <label className="inline-flex items-center gap-2 text-xs text-gray-700 dark:text-gray-200 font-jetbrains-mono">
                <input
                  type="radio"
                  name={`overlap_${propertyId}`}
                  checked={overlapMode.mode === "auto"}
                  onChange={() =>
                    setOverlapMode((prev) => ({ ...prev, mode: "auto" }))
                  }
                />
                Auto
              </label>

              <label className="inline-flex items-center gap-2 text-xs text-gray-700 dark:text-gray-200 font-jetbrains-mono">
                <input
                  type="radio"
                  name={`overlap_${propertyId}`}
                  checked={overlapMode.mode === "manual"}
                  onChange={() =>
                    setOverlapMode((prev) => ({
                      ...prev,
                      mode: "manual",
                    }))
                  }
                />
                Manual
              </label>

              {overlapMode.mode === "manual" ? (
                <div className="flex items-center gap-3">
                  <input
                    type="range"
                    min={5}
                    max={80}
                    value={overlapMode.manualPct}
                    onChange={(e) =>
                      setOverlapMode((prev) => ({
                        ...prev,
                        manualPct: Number(e.target.value),
                      }))
                    }
                    className="w-full"
                  />
                  <div className="text-xs text-gray-700 dark:text-gray-200 font-jetbrains-mono w-[56px] text-right">
                    {overlapMode.manualPct}%
                  </div>
                </div>
              ) : null}
            </div>
          </div>

          {panoError ? (
            <div className="mt-3 text-xs text-red-600 dark:text-red-400 font-jetbrains-mono">
              {panoError}
            </div>
          ) : null}

          {panoRemoteUrl ? (
            <div className="mt-3 text-xs text-emerald-700 dark:text-emerald-300 font-jetbrains-mono">
              Panorama saved as a tour.
            </div>
          ) : null}
        </div>

        <div>
          <div className="text-xs text-gray-500 dark:text-gray-400 font-jetbrains-mono mb-2">
            Preview
          </div>

          {panoPreviewUrl ? (
            <div className="rounded-lg overflow-hidden border border-gray-200 dark:border-gray-700 bg-black">
              <img
                src={panoPreviewUrl}
                alt="Panorama preview"
                className="w-full h-40 object-cover"
                draggable={false}
              />
            </div>
          ) : (
            <div className="rounded-lg border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800 h-40 flex items-center justify-center">
              <div className="text-xs text-gray-500 dark:text-gray-400 font-jetbrains-mono">
                Generate to preview
              </div>
            </div>
          )}
        </div>
      </div>

      <div className="mt-3 text-[11px] text-gray-500 dark:text-gray-500 font-jetbrains-mono">
        Tip: keep the phone level and move only by turning in place. If results
        look wrong, switch to Manual overlap.
      </div>
    </div>
  );
}
