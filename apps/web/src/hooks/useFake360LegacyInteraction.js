import { useRef, useCallback } from "react";
import { clamp } from "@/utils/fake360Utils";

export function useFake360LegacyInteraction({
  yaw,
  setYaw,
  setDragging,
  goToScene,
  activeScene,
  setParallax,
}) {
  const dragRef = useRef({
    isPointerDown: false,
    startX: 0,
    startYaw: 0,
    startY: 0,
    moved: false,
    thresholdPassed: false,
    downAt: 0,
  });

  const clickRef = useRef({ lastAt: 0, lastX: 0, lastY: 0 });

  const pickNearestHotspot = useCallback(
    (nx, ny) => {
      const hs = Array.isArray(activeScene?.hotspots)
        ? activeScene.hotspots
        : [];
      if (!hs.length) return null;

      let best = null;
      let bestD = Infinity;
      for (const h of hs) {
        const dx = nx - Number(h.x || 0);
        const dy = ny - Number(h.y || 0);
        const d = dx * dx + dy * dy;
        if (d < bestD) {
          bestD = d;
          best = h;
        }
      }

      // Only accept if click is reasonably close to a marker.
      const thresholdSq = 0.035;
      if (bestD > thresholdSq) return null;
      return best;
    },
    [activeScene],
  );

  const onPointerDown = useCallback(
    (e) => {
      dragRef.current.isPointerDown = true;
      dragRef.current.startX = e.clientX;
      dragRef.current.startY = e.clientY;
      dragRef.current.startYaw = yaw;
      dragRef.current.moved = false;
      dragRef.current.thresholdPassed = false;
      dragRef.current.downAt = Date.now();

      // Only set dragging after threshold (Street View feel)
      setDragging(false);

      try {
        e.currentTarget.setPointerCapture(e.pointerId);
      } catch {
        // ignore
      }
    },
    [yaw, setDragging],
  );

  const onPointerMove = useCallback(
    (e, containerRef) => {
      if (!containerRef.current) return;

      // Parallax (mouse move)
      const rect = containerRef.current.getBoundingClientRect();
      const nx = (e.clientX - rect.left) / rect.width - 0.5;
      const ny = (e.clientY - rect.top) / rect.height - 0.5;
      setParallax({ x: nx, y: ny });

      if (!dragRef.current.isPointerDown) return;

      const dx = e.clientX - dragRef.current.startX;
      const dy = e.clientY - dragRef.current.startY;

      if (!dragRef.current.thresholdPassed) {
        const rotateThresholdPx = 10;
        const hasMovedEnough =
          Math.abs(dx) >= rotateThresholdPx ||
          Math.abs(dy) >= rotateThresholdPx;
        if (!hasMovedEnough) {
          return;
        }
        dragRef.current.thresholdPassed = true;
        dragRef.current.moved = true;
        setDragging(true);
      }

      // ROTATE ONLY
      const delta = dx / rect.width;
      const nextYaw = clamp(dragRef.current.startYaw + delta * 2, -1, 1);
      setYaw(nextYaw);
    },
    [setParallax, setDragging, setYaw],
  );

  const onPointerUp = useCallback(
    (e, containerRef) => {
      dragRef.current.isPointerDown = false;

      if (!containerRef.current) {
        setDragging(false);
        return;
      }

      const thresholdPassed = !!dragRef.current.thresholdPassed;
      if (thresholdPassed) {
        setDragging(false);
        return;
      }

      // Click candidate (no rotation): only DOUBLE click/tap near a marker moves.
      setDragging(false);

      const downAt = Number(dragRef.current.downAt || 0);
      const pressMs = downAt ? Date.now() - downAt : 9999;
      const sx = Number(dragRef.current.startX || 0);
      const sy = Number(dragRef.current.startY || 0);
      const dx = e.clientX - sx;
      const dy = e.clientY - sy;
      const movedSq = dx * dx + dy * dy;

      const isShortPress = pressMs > 0 && pressMs < 250;
      const isStill = movedSq < 5 * 5;
      if (!(isShortPress && isStill)) {
        return;
      }

      const now = Date.now();
      const dt = now - clickRef.current.lastAt;
      const ddx = e.clientX - clickRef.current.lastX;
      const ddy = e.clientY - clickRef.current.lastY;
      const distSinceLastSq = ddx * ddx + ddy * ddy;

      const isDouble = dt > 0 && dt < 250 && distSinceLastSq < 5 * 5;

      clickRef.current.lastAt = now;
      clickRef.current.lastX = e.clientX;
      clickRef.current.lastY = e.clientY;

      if (!isDouble) {
        return;
      }

      clickRef.current.lastAt = 0;

      const rect = containerRef.current.getBoundingClientRect();
      const nx = (e.clientX - rect.left) / rect.width;
      const ny = (e.clientY - rect.top) / rect.height;

      const hit = pickNearestHotspot(nx, ny);
      if (hit?.toSceneId) {
        goToScene(hit.toSceneId);
      }
    },
    [goToScene, pickNearestHotspot, setDragging],
  );

  return {
    onPointerDown,
    onPointerMove,
    onPointerUp,
  };
}
