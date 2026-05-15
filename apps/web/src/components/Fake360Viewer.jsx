import { useState, useEffect, useMemo, useRef } from "react";
import { useFake360Payload } from "@/hooks/useFake360Payload";
import { useFake360Points } from "@/hooks/useFake360Points";
import { useFake360Parallax } from "@/hooks/useFake360Parallax";
import { useFake360LegacyScenes } from "@/hooks/useFake360LegacyScenes";
import { useFake360HelperHint } from "@/hooks/useFake360HelperHint";
import { useFake360FramesInteraction } from "@/hooks/useFake360FramesInteraction";
import { useFake360LegacyInteraction } from "@/hooks/useFake360LegacyInteraction";
import { useFake360Frames } from "@/hooks/useFake360Frames";
import { FramesViewer } from "./Fake360Viewer/FramesViewer";
import { LegacyScenesViewer } from "./Fake360Viewer/LegacyScenesViewer";

function pickPanoramaUrlFromPoint(point) {
  if (!point) return null;

  const direct =
    point.panoramaUrl ||
    point.panorama ||
    point.equirectangularUrl ||
    point.imageUrl ||
    point.url ||
    point.src ||
    point.download_url ||
    point.storage_path ||
    point.storagePath;

  if (typeof direct === "string" && direct) {
    return direct;
  }

  // IMPORTANT: do NOT fall back to frames here.
  // If a point has `frames`, it should be treated as a frame-rotation tour,
  // not as a single panorama texture.
  return null;
}

function pickPanoramaUrlFromPayload(payload) {
  if (!payload) return null;
  const direct =
    payload.panoramaUrl ||
    payload.panorama ||
    payload.imageUrl ||
    payload.url ||
    payload.src ||
    payload.download_url ||
    payload.storage_path ||
    payload.storagePath;
  if (typeof direct === "string" && direct) return direct;

  // IMPORTANT: do NOT fall back to frames here.
  return null;
}

export default function Fake360Viewer({ tourPayload, height }) {
  const effectivePayload = useFake360Payload(tourPayload);

  // Support multi-point tours OR single pano.
  const isPointsMode = Array.isArray(effectivePayload?.points);
  const isFramesMode = !isPointsMode && Array.isArray(effectivePayload?.frames);
  const containerHeight = height ?? 380;

  const { parallax, setParallax } = useFake360Parallax();

  // Camera controls: yaw + pitch + zoom (Street View feel)
  const [yawDeg, setYawDeg] = useState(0);
  const [pitch, setPitch] = useState(0);
  const [zoom, setZoom] = useState(1);

  // Track whether the user is actively rotating (drag or inertia)
  const [isRotating, setIsRotating] = useState(false);

  // Per-node yaw memory (so when you return, you land facing the same direction)
  const pointYawMapRef = useRef(new Map());

  // ---------------------------------------------------------------------------
  // MODE A: points (node-based)
  // ---------------------------------------------------------------------------
  const {
    activePoint,
    pointHotspots,
    pointTransitioning,
    goToPoint: goToPointBase,
  } = useFake360Points(effectivePayload, isPointsMode);

  const { normalizedFrames } = useFake360Frames(
    activePoint,
    effectivePayload,
    isPointsMode,
    isFramesMode,
  );

  // Prefer a true panorama if it exists; otherwise fall back to frame-rotation.
  const directPanoramaUrl = useMemo(() => {
    if (isPointsMode) {
      return pickPanoramaUrlFromPoint(activePoint);
    }
    return pickPanoramaUrlFromPayload(effectivePayload);
  }, [activePoint, effectivePayload, isPointsMode]);

  const hasFrames = !directPanoramaUrl && normalizedFrames.length >= 2;

  const panoramaUrl = hasFrames ? null : directPanoramaUrl;

  // Restore yaw when point changes (or apply its initialYaw)
  useEffect(() => {
    if (!isPointsMode) return;
    const key = activePoint?.pointId || activePoint?.id;
    if (!key) return;

    const saved = pointYawMapRef.current.get(String(key));
    if (typeof saved === "number" && Number.isFinite(saved)) {
      setYawDeg(saved);
      return;
    }

    const initialYaw = Number(activePoint?.initialYaw || 0);
    setYawDeg(Number.isFinite(initialYaw) ? initialYaw : 0);
  }, [
    activePoint?.id,
    activePoint?.initialYaw,
    activePoint?.pointId,
    isPointsMode,
  ]);

  const goToPoint = (nextId) => {
    // While rotating, navigation should be locked.
    if (isRotating) return;

    const key = activePoint?.pointId || activePoint?.id;
    if (key) {
      pointYawMapRef.current.set(String(key), yawDeg);
    }

    goToPointBase(
      nextId,
      () => {},
      0,
      () => {},
    );
  };

  const {
    onFramesPointerDown,
    onFramesPointerMove,
    onFramesPointerUp,
    onFramesWheel,
  } = useFake360FramesInteraction({
    yawDeg,
    setYawDeg,
    setIsRotating: setIsRotating,
    setParallax,
    // camera controls
    pitch,
    setPitch,
    zoom,
    setZoom,
    // navigation
    isPointsMode,
    pointHotspots,
    goToPoint,
    // NEW: tune drag feel when frame count is small
    frameCount: normalizedFrames.length,
    isFrameRotationMode: hasFrames,
    // lock input while transitioning
    pointTransitioning,
  });

  // ---------------------------------------------------------------------------
  // MODE B: legacy scenes graph (kept for compatibility)
  // ---------------------------------------------------------------------------
  const {
    activeScene,
    yaw,
    setYaw,
    dragging,
    setDragging,
    transitioning,
    goToScene,
  } = useFake360LegacyScenes(effectivePayload, isFramesMode, isPointsMode);

  const showHelperHint = useFake360HelperHint();

  const { onPointerDown, onPointerMove, onPointerUp } =
    useFake360LegacyInteraction({
      yaw,
      setYaw,
      setDragging,
      goToScene,
      activeScene,
      setParallax,
    });

  if (isFramesMode || isPointsMode) {
    return (
      <FramesViewer
        containerHeight={containerHeight}
        isPointsMode={isPointsMode}
        panoramaUrl={panoramaUrl}
        frames={normalizedFrames}
        yawDeg={yawDeg}
        pitch={pitch}
        zoom={zoom}
        isRotating={isRotating}
        pointTransitioning={pointTransitioning}
        pointHotspots={pointHotspots}
        goToPoint={goToPoint}
        onFramesPointerDown={onFramesPointerDown}
        onFramesPointerMove={onFramesPointerMove}
        onFramesPointerUp={onFramesPointerUp}
        onFramesWheel={onFramesWheel}
      />
    );
  }

  return (
    <LegacyScenesViewer
      containerHeight={containerHeight}
      activeScene={activeScene}
      yaw={yaw}
      parallax={parallax}
      dragging={dragging}
      transitioning={transitioning}
      goToScene={goToScene}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      setDragging={setDragging}
      showHelperHint={showHelperHint}
    />
  );
}
