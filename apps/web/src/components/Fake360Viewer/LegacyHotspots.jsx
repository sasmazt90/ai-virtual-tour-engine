import { useRef } from "react";
import { ArrowLeft, ArrowRight, ArrowUp, ArrowDown } from "lucide-react";

export function LegacyHotspots({ hotspots, goToScene }) {
  const tapMapRef = useRef(new Map());

  return hotspots.map((h, idx) => {
    const dir =
      typeof h.direction === "string" && h.direction.trim().length > 0
        ? h.direction
        : h.x < 0.5
          ? "left"
          : "right";

    const Icon =
      dir === "forward"
        ? ArrowUp
        : dir === "back"
          ? ArrowDown
          : dir === "left"
            ? ArrowLeft
            : ArrowRight;

    // Make it feel like a "floor" arrow: smaller, subtle, no label.
    const rotationDeg =
      dir === "forward" ? 0 : dir === "back" ? 180 : dir === "left" ? -90 : 90;

    const targetSceneId = h.toSceneId || h.toId || h.toPointId;
    const key = `${targetSceneId}-${idx}`;

    const handleNavigate = (e) => {
      e.preventDefault();
      e.stopPropagation();
      if (!targetSceneId) return;
      goToScene(targetSceneId);
    };

    return (
      <button
        key={key}
        type="button"
        onPointerDown={(e) => e.stopPropagation()}
        onPointerUp={(e) => {
          // Touch: allow single tap to move
          try {
            if (e.pointerType !== "touch") return;
            if (!targetSceneId) return;

            tapMapRef.current.set(key, {
              lastAt: Date.now(),
              lastX: e.clientX,
              lastY: e.clientY,
            });

            e.preventDefault();
            e.stopPropagation();
            goToScene(targetSceneId);
          } catch {
            // ignore
          }
        }}
        onClick={handleNavigate}
        className="absolute flex items-center justify-center w-10 h-10"
        style={{
          left: `${h.x * 100}%`,
          top: `${h.y * 100}%`,
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
