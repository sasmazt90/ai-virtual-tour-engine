import { useRef, useCallback } from "react";

function clamp(n, min, max) {
  return Math.min(max, Math.max(min, n));
}

function distSq(a, b) {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  return dx * dx + dy * dy;
}

export function useFake360FramesInteraction({
  // camera (Street View feel)
  yawDeg,
  setYawDeg,
  setIsRotating,
  pitch,
  setPitch,
  zoom,
  setZoom,

  // navigation
  isPointsMode,
  pointHotspots,
  goToPoint,

  // NEW: used to tune drag sensitivity when frame count is small
  frameCount,
  isFrameRotationMode,

  // UI polish
  setParallax,

  // lock input while transitioning
  pointTransitioning,
}) {
  // RULES:
  // - Drag after threshold = ROTATE ONLY (yaw + pitch). Never navigate.
  // - Single click on empty space = no-op.
  // - Navigation only via: hotspot click OR true double click (manual detection).

  const clickRef = useRef({ lastAt: 0, lastX: 0, lastY: 0 });

  const dragRef = useRef({
    isPointerDown: false,
    startX: 0,
    startY: 0,
    startYaw: 0,
    startPitch: 0,
    lastX: 0,
    lastT: 0,
    velocityYawDegPerTick: 0,
    thresholdPassed: false,
    downAt: 0,
  });

  // Inertia
  const inertiaRef = useRef({ animId: null, active: false });

  // Track multi-touch pointers for pinch zoom
  const pointersRef = useRef(new Map());
  const pinchRef = useRef({ active: false, startDist: 0, startZoom: 1 });

  const stopInertia = useCallback(() => {
    const id = inertiaRef.current.animId;
    if (id) {
      cancelAnimationFrame(id);
      inertiaRef.current.animId = null;
    }
    inertiaRef.current.active = false;
  }, []);

  const startInertia = useCallback(() => {
    stopInertia();

    const friction = 0.92;
    const minV = 0.08; // deg/tick

    inertiaRef.current.active = true;
    setIsRotating(true);

    const tick = () => {
      const v = inertiaRef.current.active
        ? dragRef.current.velocityYawDegPerTick
        : 0;

      if (Math.abs(v) < minV) {
        inertiaRef.current.active = false;
        inertiaRef.current.animId = null;
        setIsRotating(false);
        return;
      }

      dragRef.current.velocityYawDegPerTick = v * friction;
      setYawDeg((prev) => prev + dragRef.current.velocityYawDegPerTick);
      inertiaRef.current.animId = requestAnimationFrame(tick);
    };

    inertiaRef.current.animId = requestAnimationFrame(tick);
  }, [setIsRotating, setYawDeg, stopInertia]);

  const teleportAtEvent = useCallback(
    (e, containerRef) => {
      if (!isPointsMode) return;
      if (pointTransitioning) return;
      if (dragRef.current.thresholdPassed) return;
      if (inertiaRef.current.active) return;

      const hs = Array.isArray(pointHotspots) ? pointHotspots : [];
      if (hs.length === 0) return;

      // User request: double click should move to the NEAREST hotspot
      // (nearest to where you're currently looking), not nearest to the click.
      const yaw01 = (((Number(yawDeg || 0) % 360) + 360) % 360) / 360;

      const mappedForYaw = hs
        .map((h) => {
          const toPointId = h.toPointId || h.toId || h.toSceneId;
          if (!toPointId) return null;
          const hx = Number(h.x);
          if (!Number.isFinite(hx)) return null;
          const d = Math.abs(yaw01 - hx);
          const circular = Math.min(d, 1 - d);
          return { toPointId, d: circular };
        })
        .filter(Boolean);

      if (mappedForYaw.length > 0) {
        mappedForYaw.sort((a, b) => a.d - b.d);
        const best = mappedForYaw[0];
        if (best?.toPointId) {
          goToPoint(best.toPointId);
          return;
        }
      }

      // Fallback: legacy click-position-based selection
      if (!containerRef?.current) return;
      const rect = containerRef.current.getBoundingClientRect();
      const x = (e.clientX - rect.left) / rect.width;
      const y = (e.clientY - rect.top) / rect.height;
      const click = { x, y };

      const mapped = hs
        .map((h) => {
          const toPointId = h.toPointId || h.toId || h.toSceneId;
          if (!toPointId) return null;
          const hx = Number(h.x || 0);
          const hy = Number(h.y || 0);
          return {
            toPointId,
            d: distSq(click, { x: hx, y: hy }),
            dir: h.direction,
          };
        })
        .filter(Boolean);

      mapped.sort((a, b) => a.d - b.d);
      const best = mapped[0];

      if (best && best.toPointId && best.d < 0.08) {
        goToPoint(best.toPointId);
        return;
      }

      const nx = x - 0.5;
      const preferredDir =
        nx < -0.12 ? "left" : nx > 0.12 ? "right" : "forward";
      const byDir = hs.find((h) => String(h.direction) === preferredDir);
      const fallback = byDir || hs[0];
      const toPointId =
        fallback?.toPointId || fallback?.toId || fallback?.toSceneId;
      if (toPointId) {
        goToPoint(toPointId);
      }
    },
    [goToPoint, isPointsMode, pointHotspots, pointTransitioning, yawDeg],
  );

  const onFramesPointerDown = useCallback(
    (e) => {
      if (pointTransitioning) return;

      stopInertia();

      try {
        if (e.pointerType === "touch") {
          e.preventDefault();
        }
      } catch {
        // ignore
      }

      // register pointer
      pointersRef.current.set(e.pointerId, { x: e.clientX, y: e.clientY });

      // If this becomes a 2-finger gesture, start pinch mode
      if (e.pointerType === "touch" && pointersRef.current.size === 2) {
        const pts = Array.from(pointersRef.current.values());
        const dx = pts[0].x - pts[1].x;
        const dy = pts[0].y - pts[1].y;
        pinchRef.current.active = true;
        pinchRef.current.startDist = Math.sqrt(dx * dx + dy * dy);
        pinchRef.current.startZoom = zoom;

        dragRef.current.isPointerDown = false;
        dragRef.current.thresholdPassed = false;
        dragRef.current.downAt = 0;

        setIsRotating(false);
        return;
      }

      pinchRef.current.active = false;

      dragRef.current.isPointerDown = true;
      dragRef.current.startX = e.clientX;
      dragRef.current.startY = e.clientY;
      dragRef.current.startYaw = Number(yawDeg || 0);
      dragRef.current.startPitch = Number(pitch || 0);
      dragRef.current.lastX = e.clientX;
      dragRef.current.lastT =
        typeof performance !== "undefined" ? performance.now() : Date.now();
      dragRef.current.velocityYawDegPerTick = 0;
      dragRef.current.thresholdPassed = false;
      dragRef.current.downAt = Date.now();

      setIsRotating(false); // becomes true only after threshold

      try {
        e.currentTarget.setPointerCapture(e.pointerId);
      } catch {
        // ignore
      }
    },
    [pitch, pointTransitioning, setIsRotating, stopInertia, yawDeg, zoom],
  );

  const onFramesPointerMove = useCallback(
    (e, containerRef) => {
      if (!containerRef.current) return;

      // update pointer
      if (pointersRef.current.has(e.pointerId)) {
        pointersRef.current.set(e.pointerId, { x: e.clientX, y: e.clientY });
      }

      const rect = containerRef.current.getBoundingClientRect();
      const nx = (e.clientX - rect.left) / rect.width - 0.5;
      const ny = (e.clientY - rect.top) / rect.height - 0.5;
      setParallax({ x: nx, y: ny });

      // Pinch zoom (touch)
      if (pinchRef.current.active && pointersRef.current.size === 2) {
        const pts = Array.from(pointersRef.current.values());
        const dx = pts[0].x - pts[1].x;
        const dy = pts[0].y - pts[1].y;
        const d = Math.sqrt(dx * dx + dy * dy);
        const ratio = pinchRef.current.startDist
          ? d / pinchRef.current.startDist
          : 1;
        const nextZoom = clamp(pinchRef.current.startZoom * ratio, 0.9, 1.6);
        setZoom(nextZoom);
        return;
      }

      if (pointTransitioning) return;
      if (!dragRef.current.isPointerDown) return;

      const dxTotal = e.clientX - dragRef.current.startX;
      const dyTotal = e.clientY - dragRef.current.startY;

      // Drag threshold
      if (!dragRef.current.thresholdPassed) {
        const rotateThresholdPx = 10;
        const hasMovedEnough =
          Math.abs(dxTotal) >= rotateThresholdPx ||
          Math.abs(dyTotal) >= rotateThresholdPx;
        if (!hasMovedEnough) {
          return;
        }

        dragRef.current.thresholdPassed = true;
        setIsRotating(true);
      }

      // ROTATE ONLY
      // Slower (more deliberate) drag feel:
      // - In pano mode: one full 360 turn requires a fairly long drag.
      // - In frame-rotation mode: aim for ~50–70px per frame when frame count is small.
      let pixelsPerTurn = Math.max(900, rect.width * 2.6);

      if (
        isFrameRotationMode &&
        Number.isFinite(frameCount) &&
        frameCount > 0
      ) {
        // If there are few frames, make each frame take more drag distance.
        const pxPerFrame = frameCount <= 24 ? 70 : 52;
        const desired = frameCount * pxPerFrame;
        pixelsPerTurn = Math.max(pixelsPerTurn, desired);
      }

      const deltaTurns = dxTotal / pixelsPerTurn;
      const deltaYawDeg = deltaTurns * 360;

      const nextYaw = dragRef.current.startYaw + deltaYawDeg;
      setYawDeg(nextYaw);

      // pitch
      const pitchRange = 0.28;
      const pitchDelta = -(dyTotal / rect.height) * (pitchRange * 2);
      const nextPitch = clamp(
        dragRef.current.startPitch + pitchDelta,
        -pitchRange,
        pitchRange,
      );
      setPitch(nextPitch);

      // velocity for inertia (yaw only)
      const stepDx = e.clientX - dragRef.current.lastX;
      const stepTurns = stepDx / pixelsPerTurn;
      dragRef.current.velocityYawDegPerTick = stepTurns * 360;
      dragRef.current.lastX = e.clientX;
      dragRef.current.lastT =
        typeof performance !== "undefined" ? performance.now() : Date.now();
    },
    [
      frameCount,
      isFrameRotationMode,
      pointTransitioning,
      setParallax,
      setIsRotating,
      setPitch,
      setYawDeg,
      setZoom,
    ],
  );

  const onFramesPointerUp = useCallback(
    (e, containerRef) => {
      // unregister pointer
      pointersRef.current.delete(e.pointerId);
      if (pointersRef.current.size < 2) {
        pinchRef.current.active = false;
      }

      const thresholdPassed = !!dragRef.current.thresholdPassed;
      dragRef.current.isPointerDown = false;

      if (thresholdPassed) {
        // rotate finished -> inertia only
        startInertia();
        return;
      }

      // Not a drag: treat as click candidate.
      setIsRotating(false);
      if (pointTransitioning) return;

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

      // update last click
      clickRef.current.lastAt = now;
      clickRef.current.lastX = e.clientX;
      clickRef.current.lastY = e.clientY;

      if (isDouble) {
        teleportAtEvent(e, containerRef);
        clickRef.current.lastAt = 0;
      }
      // single click: no-op (by design)
    },
    [pointTransitioning, setIsRotating, startInertia, teleportAtEvent],
  );

  const onFramesWheel = useCallback(
    (e) => {
      // scroll / trackpad zoom
      try {
        e.preventDefault();
      } catch {
        // ignore
      }
      const delta = e.deltaY || 0;
      const step = delta > 0 ? -0.06 : 0.06;
      const next = clamp((zoom || 1) + step, 0.9, 1.6);
      setZoom(next);
    },
    [setZoom, zoom],
  );

  return {
    onFramesPointerDown,
    onFramesPointerMove,
    onFramesPointerUp,
    onFramesWheel,
  };
}
