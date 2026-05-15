import { useRef } from "react";
import { ArrowLeft, ArrowRight, ArrowDown } from "lucide-react";

export function PointHotspots({ hotspots, goToPoint }) {
  const tapMapRef = useRef(new Map());

  return hotspots.map((h, idx) => {
    const dir =
      typeof h.direction === "string" && h.direction.trim().length > 0
        ? h.direction
        : h.x < 0.5
          ? "left"
          : "right";

    // Hide the forward marker entirely.
    // We still allow navigation via double-click (and any other non-visual
    // mechanisms), but we don't draw anything in the middle of the image.
    const isForward = dir === "forward";
    if (isForward) {
      return null;
    }

    const Icon =
      dir === "back" ? ArrowDown : dir === "left" ? ArrowLeft : ArrowRight;

    const rotationDeg = dir === "back" ? 180 : dir === "left" ? -90 : 90;

    // Support common target field names (tours in the app may use toSceneId)
    const toPointId = h.toPointId || h.toId || h.toSceneId;
    const key = `${toPointId || ""}-${idx}`;

    const handleNavigate = (e) => {
      e.preventDefault();
      e.stopPropagation();
      if (!toPointId) return;
      goToPoint(toPointId);
    };

    return (
      <button
        key={key}
        type="button"
        onPointerDown={(e) => e.stopPropagation()}
        onPointerUp={(e) => {
          // Touch: allow single tap on the marker to move (Street View style)
          try {
            if (e.pointerType !== "touch") return;
            if (!toPointId) return;

            tapMapRef.current.set(key, {
              lastAt: Date.now(),
              lastX: e.clientX,
              lastY: e.clientY,
            });

            e.preventDefault();
            e.stopPropagation();
            goToPoint(toPointId);
          } catch {
            // ignore
          }
        }}
        onClick={handleNavigate}
        className="absolute flex items-center justify-center w-10 h-10"
        style={{
          left: `${Number(h.x || 0) * 100}%`,
          top: `${Number(h.y || 0) * 100}%`,
          transform: "translate(-50%, -50%)",
        }}
        aria-label={h.label || "Move"}
        title={h.label || "Move"}
      >
        <span
          className="flex items-center justify-center w-10 h-10 rounded-full border border-white/10 bg-black/35 backdrop-blur-sm shadow-[0_10px_30px_rgba(0,0,0,0.45)] transition-colors hover:bg-black/50"
          style={{
            transform: `rotate(${rotationDeg}deg) translateY(0px)`,
          }}
        >
          <Icon size={18} className="text-white/90" />
        </span>
      </button>
    );
  });
}
