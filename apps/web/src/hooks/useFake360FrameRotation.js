import { useState, useRef, useCallback, useEffect } from "react";

export function useFake360FrameRotation(
  frameCount,
  activePoint,
  effectivePayload,
  isPointsMode,
) {
  const [frameCursor, setFrameCursor] = useState(() => {
    // For points mode, initialize from per-point cursor map if possible.
    if (isPointsMode) {
      const initial = Number(activePoint?.initialIndex || 0);
      return Number.isFinite(initial) ? initial : 0;
    }

    const initial = Number(effectivePayload?.initialIndex || 0);
    return Number.isFinite(initial) ? initial : 0;
  });

  const [frameDragging, setFrameDragging] = useState(false);
  const frameDragRef = useRef({
    startX: 0,
    startCursor: 0,
    lastX: 0,
    lastT: 0,
    velocity: 0,
    moved: false,
    thresholdPassed: false,
    animId: null,
  });

  const pointCursorMapRef = useRef(new Map());

  const normalizeCursor = useCallback(
    (cursor) => {
      if (!frameCount) return 0;
      const n = frameCount;
      let v = cursor;
      v = ((v % n) + n) % n;
      return v;
    },
    [frameCount],
  );

  const getDisplayedFrameIndex = useCallback(
    (cursor) => {
      if (!frameCount) return 0;
      const normalized = normalizeCursor(cursor);
      const base = Math.floor(normalized);
      return ((base % frameCount) + frameCount) % frameCount;
    },
    [frameCount, normalizeCursor],
  );

  const stopFrameAnim = useCallback(() => {
    const id = frameDragRef.current.animId;
    if (id) {
      cancelAnimationFrame(id);
      frameDragRef.current.animId = null;
    }
  }, []);

  const startInertia = useCallback(() => {
    if (!frameCount) return;
    stopFrameAnim();

    const friction = 0.92;
    const minV = 0.0004;

    const tick = () => {
      const v = frameDragRef.current.velocity;
      if (Math.abs(v) < minV) {
        frameDragRef.current.animId = null;
        return;
      }

      frameDragRef.current.velocity = v * friction;
      setFrameCursor((c) => c + frameDragRef.current.velocity);
      frameDragRef.current.animId = requestAnimationFrame(tick);
    };

    frameDragRef.current.animId = requestAnimationFrame(tick);
  }, [frameCount, stopFrameAnim]);

  const savePointCursor = useCallback(
    (cursorToSave) => {
      if (!isPointsMode) return;
      const key = activePoint?.pointId || activePoint?.id;
      if (!key) return;
      pointCursorMapRef.current.set(key, cursorToSave);
    },
    [activePoint?.id, activePoint?.pointId, isPointsMode],
  );

  // When changing active point, restore cursor for that point (or use its initialIndex).
  useEffect(() => {
    if (!isPointsMode) return;
    const key = activePoint?.pointId || activePoint?.id;
    if (!key) return;

    const saved = pointCursorMapRef.current.get(key);
    if (typeof saved === "number" && Number.isFinite(saved)) {
      setFrameCursor(saved);
      return;
    }

    const initial = Number(activePoint?.initialIndex || 0);
    setFrameCursor(Number.isFinite(initial) ? initial : 0);
  }, [
    activePoint?.id,
    activePoint?.initialIndex,
    activePoint?.pointId,
    isPointsMode,
  ]);

  return {
    frameCursor,
    setFrameCursor,
    frameDragging,
    setFrameDragging,
    frameDragRef,
    normalizeCursor,
    getDisplayedFrameIndex,
    stopFrameAnim,
    startInertia,
    savePointCursor,
  };
}
