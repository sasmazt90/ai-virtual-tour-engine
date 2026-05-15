import { useEffect, useMemo, useRef, useState } from "react";
import { LegacyHotspots } from "./LegacyHotspots";

export function LegacyScenesViewer({
  containerHeight,
  activeScene,
  yaw,
  parallax,
  dragging,
  transitioning,
  goToScene,
  onPointerDown,
  onPointerMove,
  onPointerUp,
  setDragging,
  showHelperHint,
}) {
  const containerRef = useRef(null);
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  const translateX = yaw * 12; // percent
  const parallaxX = parallax.x * 6;
  const parallaxY = parallax.y * 6;

  const baseTransform = `translate(${translateX + parallaxX}%, ${parallaxY}%) scale(1.08)`;
  const opacity = transitioning ? 0.15 : 1;

  const hotspots = Array.isArray(activeScene?.hotspots)
    ? activeScene.hotspots
    : [];

  // Avoid server-rendering <style> because browsers may move it into <head>
  // during HTML parsing, causing hydration mismatch at the document level.
  const kenburnsCss = useMemo(() => {
    return `
      @keyframes kenburns {
        0% {
          transform: ${baseTransform};
        }
        100% {
          transform: ${baseTransform} translate(-1.5%, -1%);
        }
      }
    `;
  }, [baseTransform]);

  if (!activeScene) {
    return (
      <div className="rounded-lg border border-gray-200 dark:border-gray-700 p-6 text-sm text-gray-600 dark:text-gray-400 font-jetbrains-mono">
        No tour data.
      </div>
    );
  }

  return (
    <div
      ref={containerRef}
      onPointerDown={onPointerDown}
      onPointerMove={(e) => onPointerMove(e, containerRef)}
      onPointerUp={(e) => onPointerUp(e, containerRef)}
      onPointerLeave={() => setDragging(false)}
      className="relative w-full overflow-hidden rounded-xl border border-gray-200 dark:border-gray-700 bg-black select-none"
      style={{ height: containerHeight, touchAction: "none" }}
    >
      <img
        src={activeScene.imageUrl}
        alt="Tour scene"
        className="absolute inset-0 w-full h-full object-cover"
        style={{
          transform: baseTransform,
          transition: dragging
            ? "none"
            : "transform 120ms ease, opacity 180ms ease",
          opacity,
          willChange: "transform, opacity",
          filter: "saturate(1.03)",
          animation: dragging
            ? "none"
            : "kenburns 18s ease-in-out infinite alternate",
        }}
        draggable={false}
      />

      {/* One-time helper hint (no arrows/labels) */}
      {showHelperHint ? (
        <div className="absolute top-3 left-1/2 -translate-x-1/2 rounded-full bg-black/60 px-4 py-2 text-xs text-white font-jetbrains-mono">
          Drag to look • Double click a floor marker to move
        </div>
      ) : null}

      {/* Floor markers */}
      <LegacyHotspots hotspots={hotspots} goToScene={goToScene} />

      {mounted ? <style>{kenburnsCss}</style> : null}
    </div>
  );
}
