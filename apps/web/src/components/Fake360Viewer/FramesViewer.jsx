import { useRef, useMemo, useEffect, useState } from "react";
import { PointHotspots } from "./PointHotspots";

export function FramesViewer({
  containerHeight,
  isPointsMode,
  panoramaUrl,
  frames,
  yawDeg,
  pitch,
  zoom,
  isRotating,
  pointTransitioning,
  pointHotspots,
  goToPoint,
  onFramesPointerDown,
  onFramesPointerMove,
  onFramesPointerUp,
  onFramesWheel,
}) {
  const containerRef = useRef(null);

  const safeYaw = Number.isFinite(yawDeg) ? yawDeg : 0;
  const safePitch = Number.isFinite(pitch) ? pitch : 0;
  const safeZoom = Number.isFinite(zoom) ? zoom : 1;

  const safeFrames = useMemo(() => {
    return Array.isArray(frames)
      ? frames.filter((x) => typeof x === "string")
      : [];
  }, [frames]);

  const isPanoMode = typeof panoramaUrl === "string" && !!panoramaUrl;
  const isFrameRotationMode = !isPanoMode && safeFrames.length >= 1;

  // Normalize yaw to [0..360)
  const yaw01 = useMemo(() => {
    const y = ((safeYaw % 360) + 360) % 360;
    return y / 360;
  }, [safeYaw]);

  const pitchRange = 0.28;
  const clampedPitch = Math.max(-pitchRange, Math.min(pitchRange, safePitch));
  const pitchShiftPx = clampedPitch * containerHeight * 0.9;

  // ---------------------------------------------------------------------------
  // MODE A: Panorama texture (repeat-x)
  // ---------------------------------------------------------------------------
  const [panoAspect, setPanoAspect] = useState(2);

  useEffect(() => {
    if (!isPanoMode) return;
    let cancelled = false;

    try {
      const img = new Image();
      img.crossOrigin = "anonymous";
      img.onload = () => {
        if (cancelled) return;
        const w = img.naturalWidth;
        const h = img.naturalHeight;
        const next = h > 0 ? w / h : 2;
        if (Number.isFinite(next) && next > 0.2) {
          setPanoAspect(next);
        } else {
          setPanoAspect(2);
        }
      };
      img.onerror = () => {
        if (cancelled) return;
        setPanoAspect(2);
      };
      img.src = panoramaUrl;
    } catch {
      setPanoAspect(2);
    }

    return () => {
      cancelled = true;
    };
  }, [isPanoMode, panoramaUrl]);

  const panoWidthPx = useMemo(() => {
    if (!isPanoMode) return 0;
    return containerHeight * panoAspect * safeZoom;
  }, [containerHeight, isPanoMode, panoAspect, safeZoom]);

  const backgroundPositionX = useMemo(() => {
    if (!isPanoMode) return 0;
    return -yaw01 * panoWidthPx;
  }, [isPanoMode, panoWidthPx, yaw01]);

  // ---------------------------------------------------------------------------
  // MODE B: Frame-rotation (Polycam-like)
  // ---------------------------------------------------------------------------
  const frameIndex = useMemo(() => {
    if (!isFrameRotationMode) return 0;
    const n = safeFrames.length;
    if (!n) return 0;
    return Math.round(yaw01 * n) % n;
  }, [isFrameRotationMode, safeFrames.length, yaw01]);

  const frameUrl = useMemo(() => {
    if (!isFrameRotationMode) return null;
    return safeFrames[frameIndex] || safeFrames[0] || null;
  }, [frameIndex, isFrameRotationMode, safeFrames]);

  useEffect(() => {
    if (!isFrameRotationMode) return;
    const n = safeFrames.length;
    if (n < 2) return;

    const nextIdx = (frameIndex + 1) % n;
    const prevIdx = (frameIndex - 1 + n) % n;
    const urls = [safeFrames[nextIdx], safeFrames[prevIdx]].filter(Boolean);

    for (const u of urls) {
      try {
        const img = new Image();
        img.decoding = "async";
        img.src = u;
      } catch {
        // ignore
      }
    }
  }, [frameIndex, isFrameRotationMode, safeFrames]);

  const hasAnyImage = isPanoMode || (isFrameRotationMode && !!frameUrl);

  if (!hasAnyImage) {
    return (
      <div className="rounded-lg border border-gray-200 dark:border-gray-700 p-6 text-sm text-gray-600 dark:text-gray-400 font-jetbrains-mono">
        No tour image.
        <div className="mt-2 text-xs text-gray-500 dark:text-gray-500 font-jetbrains-mono">
          This usually means the saved tour payload does not include any image
          URLs (frames or panoramaUrl). Try closing and reopening, or regenerate
          the tour.
        </div>
      </div>
    );
  }

  const hintText = isPointsMode
    ? "Drag to look - scroll or pinch to zoom - double click to move"
    : "Drag to look - scroll or pinch to zoom";

  const panoNode = isPanoMode ? (
    <div
      className="absolute inset-0"
      style={{
        backgroundImage: `url(${panoramaUrl})`,
        backgroundRepeat: "repeat-x",
        backgroundSize: `auto ${Math.round(100 * safeZoom)}%`,
        backgroundPosition: `${backgroundPositionX.toFixed(2)}px ${pitchShiftPx.toFixed(2)}px`,
        transition: isRotating
          ? "none"
          : "background-position 80ms linear, background-size 120ms ease",
        filter: "saturate(1.03)",
        willChange: "background-position, background-size",
      }}
    />
  ) : null;

  const frameNode = !isPanoMode ? (
    <img
      src={frameUrl}
      alt="Virtual tour frame"
      draggable={false}
      className="absolute inset-0 w-full h-full object-cover"
      style={{
        transform: `translate3d(0px, ${pitchShiftPx.toFixed(2)}px, 0px) scale(${safeZoom.toFixed(3)})`,
        transformOrigin: "center center",
        transition: isRotating ? "none" : "transform 120ms ease",
        willChange: "transform",
      }}
    />
  ) : null;

  return (
    <div
      ref={containerRef}
      onPointerDown={onFramesPointerDown}
      onPointerMove={(e) => onFramesPointerMove(e, containerRef)}
      onPointerUp={(e) => onFramesPointerUp(e, containerRef)}
      onWheel={onFramesWheel}
      onContextMenu={(e) => e.preventDefault()}
      className="relative w-full overflow-hidden rounded-xl border border-gray-200 dark:border-gray-700 bg-black select-none"
      style={{ height: containerHeight, touchAction: "none" }}
    >
      {panoNode}
      {frameNode}

      {/* Teleport fade */}
      <div
        className="absolute inset-0 bg-black"
        style={{
          opacity: pointTransitioning ? 0.35 : 0,
          transition: "opacity 220ms ease",
          pointerEvents: "none",
        }}
      />

      {isPointsMode ? (
        <PointHotspots hotspots={pointHotspots} goToPoint={goToPoint} />
      ) : null}

      <div className="absolute bottom-3 left-1/2 -translate-x-1/2 rounded-full bg-black/[0.55] px-4 py-2 text-[11px] text-white/85 font-jetbrains-mono border border-white/10">
        {hintText}
      </div>
    </div>
  );
}
