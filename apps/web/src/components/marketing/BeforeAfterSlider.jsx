import { useCallback, useEffect, useMemo, useRef, useState } from "react";

function clamp01(n) {
  return Math.min(0.94, Math.max(0.06, n));
}

export default function BeforeAfterSlider({
  beforeSrc,
  afterSrc,
  beforeLabel = "Before: Empty room",
  afterLabel = "After: AI staging + furniture placement",
  height = 420,
}) {
  const containerRef = useRef(null);
  const [ratio, setRatio] = useState(0.52);
  const [dragging, setDragging] = useState(false);

  // Additive: show a small, non-blocking hint if assets are missing.
  const [missingAssets, setMissingAssets] = useState(false);

  const setFromClientX = useCallback((clientX) => {
    const el = containerRef.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    if (!rect.width) return;
    const next = (clientX - rect.left) / rect.width;
    setRatio(clamp01(next));
  }, []);

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

  const onPointerMove = useCallback(
    (e) => {
      if (!dragging) return;
      setFromClientX(e.clientX);
    },
    [dragging, setFromClientX],
  );

  const onPointerUp = useCallback(() => {
    setDragging(false);
  }, []);

  // Keyboard support
  useEffect(() => {
    if (typeof window === "undefined") return;
    const onKeyDown = (e) => {
      if (e.key === "ArrowLeft") setRatio((r) => clamp01(r - 0.03));
      if (e.key === "ArrowRight") setRatio((r) => clamp01(r + 0.03));
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, []);

  const percent = useMemo(() => {
    return Math.round(ratio * 100);
  }, [ratio]);

  const overlayStyle = useMemo(() => {
    return {
      clipPath: `inset(0 ${100 - ratio * 100}% 0 0)`,
      transition: dragging ? "none" : "clip-path 170ms ease",
    };
  }, [dragging, ratio]);

  const handleStyle = useMemo(() => {
    return {
      left: `${percent}%`,
      transition: dragging ? "none" : "left 170ms ease",
    };
  }, [dragging, percent]);

  return (
    <div
      ref={containerRef}
      className="relative w-full overflow-hidden rounded-2xl border border-white/10 bg-black shadow-[0_32px_90px_rgba(0,0,0,0.55)]"
      style={{ height }}
    >
      <img
        src={beforeSrc}
        alt={beforeLabel}
        className="absolute inset-0 w-full h-full object-cover"
        draggable={false}
        onError={() => setMissingAssets(true)}
      />

      <div className="absolute inset-0" style={overlayStyle}>
        <img
          src={afterSrc}
          alt={afterLabel}
          className="absolute inset-0 w-full h-full object-cover"
          draggable={false}
          onError={() => setMissingAssets(true)}
        />
      </div>

      <div className="absolute inset-0 bg-gradient-to-t from-black/60 via-black/10 to-black/0" />

      <div className="absolute top-4 left-4 rounded-full border border-white/10 bg-black/50 px-3 py-1 text-xs text-gray-100 font-jetbrains-mono">
        {beforeLabel}
      </div>
      <div className="absolute top-4 right-4 rounded-full border border-white/10 bg-black/50 px-3 py-1 text-xs text-gray-100 font-jetbrains-mono">
        {afterLabel}
      </div>

      <div
        className="absolute inset-0 cursor-ew-resize"
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerLeave={onPointerUp}
        style={{ touchAction: "none" }}
        aria-label="Before and after comparison slider"
      />

      {/* divider */}
      <div
        className="absolute top-0 bottom-0 w-[2px] bg-white/70"
        style={handleStyle}
      />

      {/* knob */}
      <div
        className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2"
        style={handleStyle}
      >
        <div className="relative">
          <div className="w-12 h-12 rounded-full bg-black/[0.55] border border-white/[0.15] backdrop-blur flex items-center justify-center">
            <div className="w-7 h-7 rounded-full bg-white/10 border border-white/20 slider-knob" />
          </div>
          <div className="absolute -bottom-9 left-1/2 -translate-x-1/2 rounded-full bg-black/60 border border-white/10 px-3 py-1 text-[10px] text-gray-100 font-jetbrains-mono">
            Drag
          </div>
        </div>
      </div>

      {missingAssets ? (
        <div className="absolute bottom-4 left-4 right-4 rounded-xl border border-white/10 bg-black/60 px-4 py-3 text-xs text-gray-200 font-jetbrains-mono">
          Missing demo images. Ensure{" "}
          <span className="text-gray-50">Vacant.png</span> and{" "}
          <span className="text-gray-50">existing.png</span> are available at
          the site root.
        </div>
      ) : null}
    </div>
  );
}
