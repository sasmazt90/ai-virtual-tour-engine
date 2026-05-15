import { useState, useEffect, useMemo, useCallback } from "react";

export function useFake360Points(effectivePayload, isPointsMode) {
  const points = isPointsMode ? effectivePayload.points : [];

  const initialPointId = useMemo(() => {
    if (!isPointsMode) return null;
    const explicit = effectivePayload?.initialPointId;
    if (explicit) return explicit;
    return points[0]?.pointId || points[0]?.id || null;
  }, [effectivePayload?.initialPointId, isPointsMode, points]);

  const [activePointId, setActivePointId] = useState(initialPointId);
  const [pointTransitioning, setPointTransitioning] = useState(false);

  useEffect(() => {
    if (!isPointsMode) return;
    if (!activePointId && initialPointId) {
      setActivePointId(initialPointId);
    }
  }, [activePointId, initialPointId, isPointsMode]);

  const activePoint = useMemo(() => {
    if (!isPointsMode) return null;
    if (!activePointId) return points[0] || null;
    return (
      points.find((p) => (p.pointId || p.id) === activePointId) ||
      points[0] ||
      null
    );
  }, [activePointId, isPointsMode, points]);

  const pointHotspots = useMemo(() => {
    if (!isPointsMode) return [];
    return Array.isArray(activePoint?.hotspots) ? activePoint.hotspots : [];
  }, [activePoint?.hotspots, isPointsMode]);

  const goToPoint = useCallback(
    (nextId, savePointCursor, frameCursor, stopFrameAnim) => {
      if (!isPointsMode) return;
      if (!nextId) return;
      if (nextId === activePointId) return;

      // Save current cursor for current point.
      savePointCursor(frameCursor);
      stopFrameAnim();

      // Premium “teleport” feel (250–400ms)
      setPointTransitioning(true);
      setTimeout(() => {
        setActivePointId(nextId);
        setPointTransitioning(false);
      }, 320);
    },
    [activePointId, isPointsMode],
  );

  return {
    points,
    activePointId,
    activePoint,
    pointHotspots,
    pointTransitioning,
    goToPoint,
  };
}
