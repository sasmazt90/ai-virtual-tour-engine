import { useCallback, useMemo, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import Splat3DViewer from "@/components/Splat3DViewer";
import { CreateSplatTourModal } from "./CreateSplatTourModal";
import { CreateVideo3DTourModal } from "./CreateVideo3DTourModal";
import { ModalShell } from "./ModalShell";
import { titleCase } from "@/utils/formatters";

export function VirtualToursSection({ property, propertyId }) {
  const queryClient = useQueryClient();
  const [splatOpen, setSplatOpen] = useState(false);
  const [video3DOpen, setVideo3DOpen] = useState(false);
  const [viewTourId, setViewTourId] = useState(null);

  const userId = property?.user_id || null;

  const tours = useMemo(() => {
    const list = Array.isArray(property?.tours)
      ? property.tours.filter(
          (tour) =>
            tour?.tour_type !== "fake360" &&
            tour?.tour_payload?.type !== "fake360",
        )
      : [];

    const sorted = list
      .slice()
      .sort(
        (a, b) =>
          new Date(b.created_at).getTime() - new Date(a.created_at).getTime(),
      );

    const byKey = new Map();
    for (const t of sorted) {
      const sourceType =
        t?.source_type === "staging" || t?.source_type === "original"
          ? t.source_type
          : "original";

      const key = (() => {
        if (sourceType === "staging") {
          const st =
            typeof t?.staging_type === "string" ? t.staging_type.trim() : "";
          return st ? `staging:${st}` : null;
        }
        return "original";
      })();

      if (!key) continue;
      if (!byKey.has(key)) {
        byKey.set(key, t);
      }
    }

    return Array.from(byKey.values());
  }, [property?.tours]);

  const displayTitleById = useMemo(() => {
    const map = {};

    for (const t of tours) {
      if (!t?.id) continue;

      if (t?.tour_type === "splat3d") {
        map[t.id] = "3D Virtual Tour - Original Scan";
      } else if (t?.source_type === "staging") {
        const st =
          typeof t?.staging_type === "string" ? t.staging_type.trim() : "";
        map[t.id] = `Virtual Tour - ${titleCase(st || "Staging")}`;
      } else {
        map[t.id] = "Virtual Tour - Original";
      }
    }

    return map;
  }, [tours]);

  const viewingTour = useMemo(() => {
    if (!viewTourId) return null;
    return tours.find((t) => String(t?.id) === String(viewTourId)) || null;
  }, [tours, viewTourId]);

  const onDelete = useCallback(
    async (tourId) => {
      const ok =
        typeof window !== "undefined"
          ? window.confirm("Delete this virtual tour?")
          : false;
      if (!ok) return;

      try {
        const res = await fetch(`/api/virtual-tours/${tourId}`, {
          method: "DELETE",
        });
        if (!res.ok) {
          throw new Error(
            `When deleting /api/virtual-tours/${tourId}, the response was [${res.status}] ${res.statusText}`,
          );
        }

        if (userId && propertyId) {
          await queryClient.invalidateQueries({
            queryKey: ["property", userId, propertyId],
          });
        } else {
          await queryClient.invalidateQueries({ queryKey: ["property"] });
        }
      } catch (e) {
        console.error(e);
        if (typeof window !== "undefined") {
          window.alert("Could not delete this virtual tour.");
        }
      }
    },
    [propertyId, queryClient, userId],
  );

  return (
    <div>
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-semibold text-gray-900 dark:text-gray-100 font-jetbrains-mono">
          Virtual Tour
        </h3>

        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={() => setVideo3DOpen(true)}
            className="inline-flex items-center gap-2 px-3 py-2 rounded-lg bg-amber-500 text-white text-sm font-medium hover:bg-amber-600 font-jetbrains-mono"
          >
            + iPhone Video
          </button>
          <button
            type="button"
            onClick={() => setSplatOpen(true)}
            className="inline-flex items-center gap-2 px-3 py-2 rounded-lg border border-amber-500/70 text-amber-600 dark:text-amber-300 text-sm font-medium hover:bg-amber-50 dark:hover:bg-amber-500/10 font-jetbrains-mono"
          >
            + 3D File
          </button>
        </div>
      </div>

      {tours.length > 0 ? (
        <div className="space-y-2">
          {tours.map((t) => {
            const title = displayTitleById[t.id] || "Virtual Tour";
            const sceneCount = Number(t?.tour_payload?.sceneCount || 0);

            return (
              <div
                key={t.id}
                className="rounded-lg border border-gray-200 dark:border-gray-700 p-3 text-sm text-gray-700 dark:text-gray-300 font-jetbrains-mono"
              >
                <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
                  <div className="min-w-0">
                    <div className="text-sm font-semibold text-gray-900 dark:text-gray-100 font-jetbrains-mono">
                      {title}
                    </div>
                    {sceneCount > 1 ? (
                      <div className="mt-1 text-xs text-gray-500 dark:text-gray-500 font-jetbrains-mono">
                        {sceneCount} areas
                      </div>
                    ) : null}
                  </div>

                  <div className="shrink-0 flex items-center gap-2">
                    <button
                      type="button"
                      onClick={() => setViewTourId(t.id)}
                      className="inline-flex items-center px-3 py-2 rounded-lg bg-gray-900 dark:bg-gray-100 text-white dark:text-gray-900 text-sm font-medium hover:bg-gray-800 dark:hover:bg-gray-200 font-jetbrains-mono"
                    >
                      View
                    </button>

                    <button
                      type="button"
                      onClick={() => onDelete(t.id)}
                      className="inline-flex items-center px-3 py-2 rounded-lg border border-gray-200 dark:border-gray-700 text-gray-700 dark:text-gray-200 text-sm font-medium hover:bg-gray-50 dark:hover:bg-gray-800 font-jetbrains-mono"
                    >
                      Delete
                    </button>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      ) : (
        <div className="text-sm text-gray-600 dark:text-gray-400 font-jetbrains-mono">
          No virtual tours yet.
        </div>
      )}

      <CreateSplatTourModal
        open={splatOpen}
        onClose={() => setSplatOpen(false)}
        propertyId={propertyId}
        userId={userId}
      />

      <CreateVideo3DTourModal
        open={video3DOpen}
        onClose={() => setVideo3DOpen(false)}
        propertyId={propertyId}
        userId={userId}
      />

      {viewTourId && viewingTour?.tour_payload ? (
        <ModalShell
          title={displayTitleById[viewingTour.id] || "Virtual Tour"}
          onClose={() => setViewTourId(null)}
        >
          {viewingTour?.tour_type === "splat3d" ||
          viewingTour?.tour_payload?.type === "splat3d" ? (
            <Splat3DViewer tourPayload={viewingTour.tour_payload} height={560} />
          ) : (
            <div className="rounded-lg border border-gray-200 dark:border-gray-700 bg-gray-950 p-6 text-sm text-gray-200 font-jetbrains-mono">
              This virtual tour format is no longer supported. Create a new 3D
              tour from an iPhone video or upload a ready 3D tour file.
            </div>
          )}
        </ModalShell>
      ) : null}
    </div>
  );
}
