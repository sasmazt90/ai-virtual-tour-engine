import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ArrowUp, ArrowLeft, ArrowRight } from "lucide-react";

function clamp(n, min, max) {
  return Math.min(max, Math.max(min, n));
}

const HELP_KEY = "marketing_fake360_hint_v1";

export default function VirtualTourDemo({ emptySrc, stagedSrc, height = 420 }) {
  // We keep everything in ONE room by reusing the same two images.
  // Different "perspectives" are simulated using different base offsets.
  const scenes = useMemo(() => {
    return [
      {
        id: "A",
        title: "Entry angle",
        img: stagedSrc,
        baseX: -6,
        baseY: -2,
      },
      {
        id: "B",
        title: "Window side",
        img: stagedSrc,
        baseX: 7,
        baseY: -1,
      },
      {
        id: "C",
        title: "Closer view",
        img: stagedSrc,
        baseX: 2,
        baseY: 4,
        scale: 1.12,
      },
      {
        id: "D",
        title: "Empty reference",
        img: emptySrc,
        baseX: 0,
        baseY: 1,
      },
    ];
  }, [emptySrc, stagedSrc]);

  const [activeId, setActiveId] = useState("A");
  const [yaw, setYaw] = useState(0);
  const [dragging, setDragging] = useState(false);
  const [parallax, setParallax] = useState({ x: 0, y: 0 });
  const [showHint, setShowHint] = useState(false);
  const [mounted, setMounted] = useState(false);

  const containerRef = useRef(null);
  const dragRef = useRef({ startX: 0, startYaw: 0 });

  useEffect(() => {
    setMounted(true);
  }, []);

  useEffect(() => {
    if (typeof window === "undefined") return;
    try {
      const already = window.localStorage.getItem(HELP_KEY);
      if (already) return;
      window.localStorage.setItem(HELP_KEY, "1");
      setShowHint(true);
      const t = window.setTimeout(() => setShowHint(false), 2000);
      return () => window.clearTimeout(t);
    } catch {
      setShowHint(true);
      const t = setTimeout(() => setShowHint(false), 2000);
      return () => clearTimeout(t);
    }
  }, []);

  const active = useMemo(() => {
    return scenes.find((s) => s.id === activeId) || scenes[0];
  }, [activeId, scenes]);

  const onPointerDown = useCallback(
    (e) => {
      setDragging(true);
      dragRef.current.startX = e.clientX;
      dragRef.current.startYaw = yaw;
    },
    [yaw],
  );

  const onPointerMove = useCallback(
    (e) => {
      const el = containerRef.current;
      if (!el) return;

      const rect = el.getBoundingClientRect();
      const nx = (e.clientX - rect.left) / rect.width - 0.5;
      const ny = (e.clientY - rect.top) / rect.height - 0.5;
      setParallax({ x: nx, y: ny });

      if (!dragging) return;
      const dx = e.clientX - dragRef.current.startX;
      const delta = dx / rect.width;
      setYaw(clamp(dragRef.current.startYaw + delta * 2, -1, 1));
    },
    [dragging],
  );

  const onPointerUp = useCallback(() => {
    setDragging(false);
  }, []);

  const go = useCallback((id) => {
    setActiveId(id);
    setYaw(0);
  }, []);

  const translateX = useMemo(() => {
    const drift = active.baseX || 0;
    const yawShift = yaw * 10;
    const px = parallax.x * 4;
    return drift + yawShift + px;
  }, [active.baseX, parallax.x, yaw]);

  const translateY = useMemo(() => {
    const drift = active.baseY || 0;
    const py = parallax.y * 4;
    return drift + py;
  }, [active.baseY, parallax.y]);

  const scale = useMemo(() => {
    return active.scale || 1.08;
  }, [active.scale]);

  const transform = useMemo(() => {
    return `translate(${translateX}%, ${translateY}%) scale(${scale})`;
  }, [scale, translateX, translateY]);

  // Google Maps style floor arrows: big, clear, clickable
  const arrows = useMemo(() => {
    // simple graph: A <-> B <-> C and A -> D (empty)
    if (activeId === "A") {
      return [
        { id: "B", label: "Right", Icon: ArrowRight },
        { id: "D", label: "Empty", Icon: ArrowUp },
      ];
    }
    if (activeId === "B") {
      return [
        { id: "A", label: "Left", Icon: ArrowLeft },
        { id: "C", label: "Forward", Icon: ArrowUp },
      ];
    }
    if (activeId === "C") {
      return [{ id: "B", label: "Back", Icon: ArrowLeft }];
    }
    return [{ id: "A", label: "Back", Icon: ArrowLeft }];
  }, [activeId]);

  const driftCss = useMemo(() => {
    return `
      @keyframes tourDrift {
        0% { transform: ${transform}; }
        100% { transform: ${transform} translate(-1.2%, -0.8%); }
      }
    `;
  }, [transform]);

  return (
    <div className="rounded-2xl border border-white/10 bg-white/5 overflow-hidden shadow-[0_30px_90px_rgba(0,0,0,0.55)]">
      <div
        ref={containerRef}
        className="relative w-full overflow-hidden bg-black select-none"
        style={{ height, touchAction: "none" }}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerLeave={onPointerUp}
      >
        <img
          src={active.img}
          alt="Virtual tour demo"
          className="absolute inset-0 w-full h-full object-cover"
          style={{
            transform,
            transition: dragging ? "none" : "transform 140ms ease",
            filter: "saturate(1.02)",
            animation: dragging
              ? "none"
              : "tourDrift 14s ease-in-out infinite alternate",
            willChange: "transform",
          }}
          draggable={false}
        />

        <div className="absolute inset-0 bg-gradient-to-t from-black/60 via-black/10 to-black/0" />

        {showHint ? (
          <div className="absolute top-3 left-1/2 -translate-x-1/2 rounded-full bg-black/60 px-4 py-2 text-xs text-white font-jetbrains-mono">
            Use arrows to explore the space
          </div>
        ) : null}

        <div className="absolute top-4 left-4 rounded-full border border-white/10 bg-black/[0.55] px-3 py-1 text-xs text-gray-100 font-jetbrains-mono">
          Virtual tour demo - {active.title}
        </div>

        {/* Floor arrows */}
        <div className="absolute bottom-5 left-1/2 -translate-x-1/2 flex items-center gap-3">
          {arrows.map((a) => {
            const Icon = a.Icon;
            const cursorClass =
              a.label === "Left"
                ? "cursor-w-resize"
                : a.label === "Right"
                  ? "cursor-e-resize"
                  : "cursor-n-resize";
            return (
              <button
                key={a.id}
                type="button"
                onClick={() => go(a.id)}
                className={`group flex items-center gap-2 rounded-full bg-white/90 hover:bg-white text-gray-900 px-4 py-3 shadow-lg transition-transform hover:-translate-y-0.5 ${cursorClass}`}
                aria-label={`Go ${a.label}`}
              >
                <Icon size={18} />
                <span className="text-sm font-medium font-jetbrains-mono">
                  {a.label}
                </span>
              </button>
            );
          })}
        </div>

        <div className="absolute bottom-3 left-3 rounded-full bg-black/[0.55] px-3 py-1 text-xs text-white font-jetbrains-mono">
          Drag to look • Click arrows
        </div>

        {mounted ? <style>{driftCss}</style> : null}
      </div>

      <div className="px-5 py-4 border-t border-white/10">
        <div className="text-xs text-gray-300 font-jetbrains-mono">
          Not a true 360 pano — an image-based tour that still feels interactive
          for clients.
        </div>
      </div>
    </div>
  );
}
