import { useCallback, useEffect, useMemo, useRef, useState } from "react";

export function BeforeAfterSlider({
  beforeSrc,
  afterSrc,
  beforeAlt,
  afterAlt,
}) {
  const containerRef = useRef(null);
  const [ratio, setRatio] = useState(0.55);
  const [dragging, setDragging] = useState(false);

  // Smooth updates even on fast pointer movement
  const ratioRef = useRef(ratio);
  const rafRef = useRef(null);

  const dividerWidthPx = 2;

  const clamp01 = useCallback((n) => {
    // Allow full left/right to match user expectation (0%..100%).
    // We handle divider positioning separately so it stays visible.
    return Math.min(1, Math.max(0, n));
  }, []);

  const commitRatio = useCallback((next) => {
    ratioRef.current = next;

    if (rafRef.current) return;

    // Only use requestAnimationFrame in the browser
    if (typeof window === "undefined") {
      setRatio(next);
      return;
    }

    rafRef.current = window.requestAnimationFrame(() => {
      rafRef.current = null;
      setRatio(ratioRef.current);
    });
  }, []);

  const setFromClientX = useCallback(
    (clientX) => {
      const el = containerRef.current;
      if (!el) return;
      const rect = el.getBoundingClientRect();
      if (rect.width <= 0) return;
      const next = (clientX - rect.left) / rect.width;
      commitRatio(clamp01(next));
    },
    [clamp01, commitRatio],
  );

  const onPointerDown = useCallback(
    (e) => {
      setDragging(true);
      try {
        e.currentTarget.setPointerCapture(e.pointerId);
      } catch {
        // ignore
      }
      setFromClientX(e.clientX);
    },
    [setFromClientX],
  );

  const onPointerUp = useCallback(() => {
    setDragging(false);
  }, []);

  // Track movement even if pointer leaves the slider (fast drags)
  useEffect(() => {
    if (typeof window === "undefined") return;

    if (!dragging) return;

    const onMove = (e) => {
      // PointerEvent has clientX; MouseEvent does too
      if (!e?.clientX && e?.clientX !== 0) return;
      setFromClientX(e.clientX);
    };

    const onUp = () => {
      setDragging(false);
    };

    window.addEventListener("pointermove", onMove, { passive: true });
    window.addEventListener("pointerup", onUp, { passive: true });
    window.addEventListener("pointercancel", onUp, { passive: true });

    return () => {
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
      window.removeEventListener("pointercancel", onUp);
    };
  }, [dragging, setFromClientX]);

  useEffect(() => {
    if (typeof window === "undefined") return;
    return () => {
      if (rafRef.current) {
        window.cancelAnimationFrame(rafRef.current);
        rafRef.current = null;
      }
    };
  }, []);

  const percent = useMemo(() => {
    // Use the raw ratio (not rounded) so the divider can truly reach the edges
    // and move smoothly.
    return `${(ratio * 100).toFixed(4)}%`;
  }, [ratio]);

  const overlayStyle = useMemo(() => {
    return {
      clipPath: `inset(0 ${100 - ratio * 100}% 0 0)`,
      transition: dragging ? "none" : "clip-path 220ms ease",
    };
  }, [dragging, ratio]);

  const handleStyle = useMemo(() => {
    // Keep the divider visible at both extremes.
    const ratioPct = ratio * 100;
    const translateX =
      ratioPct <= 0
        ? 0
        : ratioPct >= 100
          ? -dividerWidthPx
          : -dividerWidthPx / 2;

    return {
      left: percent,
      transform: `translateX(${translateX}px)`,
      transition: dragging ? "none" : "left 220ms ease",
    };
  }, [dragging, percent, ratio]);

  return (
    <div
      ref={containerRef}
      className="relative w-full overflow-hidden rounded-2xl border border-white/10 bg-[#07080A] shadow-[0_30px_90px_rgba(0,0,0,0.55)]"
      style={{ height: 392 }}
    >
      {/* BASE: after image fills the full area */}
      <img
        src={afterSrc}
        alt={afterAlt}
        className="absolute inset-0 w-full h-full object-cover"
        draggable={false}
      />

      {/* OVERLAY: before image clipped to the left portion (so left = before, right = after) */}
      <div className="absolute inset-0" style={overlayStyle}>
        <img
          src={beforeSrc}
          alt={beforeAlt}
          className="absolute inset-0 w-full h-full object-cover"
          draggable={false}
        />
      </div>

      <div className="absolute inset-0 bg-gradient-to-t from-black/60 via-black/20 to-black/0" />

      {/* Interaction layer (press + drag) */}
      <div
        className="absolute inset-0 cursor-ew-resize"
        onPointerDown={onPointerDown}
        onPointerUp={onPointerUp}
        onPointerLeave={onPointerUp}
        style={{ touchAction: "none" }}
        aria-label="Before and after comparison slider"
      />

      {/* Divider (no center knob) */}
      <div
        className="absolute top-0 bottom-0 w-[2px] bg-white/80"
        style={{
          ...handleStyle,
          boxShadow:
            "0 0 0 1px rgba(0,0,0,0.35), 0 0 18px rgba(255,255,255,0.18)",
        }}
      />
    </div>
  );
}
