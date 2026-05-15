import { useEffect, useMemo, useState } from "react";
import { ChevronLeft, ChevronRight } from "lucide-react";
import { ModalShell } from "@/components/PropertyDetail/ModalShell";

function clamp(n, min, max) {
  return Math.min(Math.max(n, min), max);
}

/**
 * A simple modal image viewer with prev/next and optional thumbnails.
 *
 * items: [{ key: string, url?: string, thumbnailUrl?: string, alt?: string }]
 * getActiveUrl: (item, index) => string
 * renderOverlay: ({ index }) => ReactNode (rendered on top of the main image)
 * headerActions: ReactNode (rendered in the modal header)
 */
export default function ImageCarouselModal({
  open,
  title,
  onClose,
  items,
  initialIndex,
  showThumbnails,
  getActiveUrl,
  renderOverlay,
  headerActions,
}) {
  const list = Array.isArray(items) ? items : [];

  const [index, setIndex] = useState(0);

  useEffect(() => {
    if (!open) return;
    const next = Number.isFinite(Number(initialIndex))
      ? Number(initialIndex)
      : 0;
    setIndex(clamp(next, 0, Math.max(0, list.length - 1)));
  }, [initialIndex, list.length, open]);

  useEffect(() => {
    if (!open) return;

    const onKeyDown = (e) => {
      if (e.key === "Escape") {
        onClose?.();
      }
      if (e.key === "ArrowLeft") {
        setIndex((i) => (i <= 0 ? Math.max(0, list.length - 1) : i - 1));
      }
      if (e.key === "ArrowRight") {
        setIndex((i) => (i >= list.length - 1 ? 0 : i + 1));
      }
    };

    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [list.length, onClose, open]);

  const safeIndex = useMemo(() => {
    return clamp(index, 0, Math.max(0, list.length - 1));
  }, [index, list.length]);

  const activeItem = list.length > 0 ? list[safeIndex] : null;

  const activeUrl = useMemo(() => {
    if (!activeItem) return "";
    if (typeof getActiveUrl === "function") {
      return String(getActiveUrl(activeItem, safeIndex) || "");
    }
    return typeof activeItem?.url === "string" ? activeItem.url : "";
  }, [activeItem, getActiveUrl, safeIndex]);

  const activeAlt = typeof activeItem?.alt === "string" ? activeItem.alt : "";

  const onPrev = () => {
    setIndex((i) => (i <= 0 ? Math.max(0, list.length - 1) : i - 1));
  };

  const onNext = () => {
    setIndex((i) => (i >= list.length - 1 ? 0 : i + 1));
  };

  if (!open) return null;

  const canNav = list.length > 1;
  const shouldShowThumbnails = showThumbnails !== false && list.length > 1;

  return (
    <ModalShell
      title={title || "Preview"}
      onClose={onClose}
      headerActions={headerActions}
    >
      <div className="space-y-3">
        <div className="relative overflow-hidden rounded-xl border border-gray-200 dark:border-gray-700 bg-black">
          {activeUrl ? (
            <img
              src={activeUrl}
              alt={activeAlt || "Preview"}
              className="w-full h-[55vh] sm:h-[65vh] object-contain bg-black"
            />
          ) : (
            <div className="w-full h-[55vh] sm:h-[65vh] flex items-center justify-center text-sm text-gray-200 font-jetbrains-mono">
              No image.
            </div>
          )}

          {canNav ? (
            <div className="absolute inset-0 z-20 flex items-center justify-between px-3 pointer-events-none">
              <button
                type="button"
                onClick={onPrev}
                className="pointer-events-auto inline-flex items-center justify-center h-10 w-10 rounded-full bg-white/80 hover:bg-white border border-gray-200"
                aria-label="Previous"
                title="Previous"
              >
                <ChevronLeft size={18} />
              </button>
              <button
                type="button"
                onClick={onNext}
                className="pointer-events-auto inline-flex items-center justify-center h-10 w-10 rounded-full bg-white/80 hover:bg-white border border-gray-200"
                aria-label="Next"
                title="Next"
              >
                <ChevronRight size={18} />
              </button>
            </div>
          ) : null}

          {typeof renderOverlay === "function" ? (
            <div className="absolute inset-0 z-10">
              {renderOverlay({ index: safeIndex })}
            </div>
          ) : null}
        </div>

        {shouldShowThumbnails ? (
          <div className="flex gap-2 overflow-x-auto pb-1">
            {list.map((it, i) => {
              const thumbUrl =
                typeof it?.thumbnailUrl === "string" && it.thumbnailUrl
                  ? it.thumbnailUrl
                  : typeof it?.url === "string"
                    ? it.url
                    : "";

              const isActive = i === safeIndex;
              const ring = isActive ? "ring-2 ring-[var(--brand)]" : "";

              return (
                <button
                  key={it?.key || `${i}`}
                  type="button"
                  onClick={() => setIndex(i)}
                  className={`shrink-0 rounded-lg overflow-hidden border border-gray-200 dark:border-gray-700 ${ring}`}
                  aria-label={`Image ${i + 1}`}
                  title={`Image ${i + 1}`}
                >
                  {thumbUrl ? (
                    <img
                      src={thumbUrl}
                      alt={it?.alt || "Thumbnail"}
                      className="h-16 w-24 object-cover"
                      loading="lazy"
                    />
                  ) : (
                    <div className="h-16 w-24 bg-gray-900" />
                  )}
                </button>
              );
            })}
          </div>
        ) : null}

        {list.length > 1 ? (
          <div className="text-xs text-gray-600 dark:text-gray-400 font-jetbrains-mono">
            {safeIndex + 1} / {list.length}
          </div>
        ) : null}
      </div>
    </ModalShell>
  );
}
