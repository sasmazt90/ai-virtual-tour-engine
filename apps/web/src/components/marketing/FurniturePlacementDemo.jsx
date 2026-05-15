import { useCallback, useMemo, useState } from "react";
import { Sparkles } from "lucide-react";

export default function FurniturePlacementDemo({
  roomSrc,
  sofaSrc,
  tableSrc,
  pillowSrc,
}) {
  const [showFurnished, setShowFurnished] = useState(true);

  const onToggle = useCallback(() => {
    setShowFurnished((v) => !v);
  }, []);

  const furnishedOpacity = useMemo(() => {
    return showFurnished ? 1 : 0;
  }, [showFurnished]);

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-10 items-center">
      <div>
        <div className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-3 py-1 text-xs text-gray-200 font-jetbrains-mono">
          <Sparkles size={14} className="text-gray-200" />
          AI Furniture Placement
        </div>

        <h3 className="mt-4 text-2xl sm:text-3xl font-bold text-gray-50 font-jetbrains-mono">
          Furniture is added — not just “enhanced”
        </h3>
        <p className="mt-3 text-sm sm:text-base text-gray-300 font-jetbrains-mono max-w-xl">
          360 Estate Suite digitally places sofas, tables, decor, and
          layout-aware accents with correct scale, perspective, and lighting —
          so empty spaces look instantly livable.
        </p>

        <div className="mt-6 flex flex-col sm:flex-row gap-3">
          <button
            type="button"
            onClick={onToggle}
            className="inline-flex items-center justify-center px-4 py-3 rounded-lg border border-white/10 bg-white/5 hover:bg-white/10 text-gray-100 text-sm font-jetbrains-mono transition-colors"
          >
            {showFurnished ? "Show empty" : "Show furnished"}
          </button>
          <a
            href="/signup"
            className="inline-flex items-center justify-center px-4 py-3 rounded-lg bg-[var(--brand)] hover:bg-[var(--brandHover)] text-white text-sm font-jetbrains-mono transition-colors"
          >
            Sign Up
          </a>
        </div>

        <div className="mt-6 text-xs text-gray-400 font-jetbrains-mono">
          Tip: Toggle to see how objects are placed, scaled, and lit.
        </div>
      </div>

      <div className="rounded-2xl border border-white/10 bg-white/5 overflow-hidden shadow-[0_30px_90px_rgba(0,0,0,0.55)]">
        <div className="relative w-full" style={{ aspectRatio: "16 / 10" }}>
          <img
            src={roomSrc}
            alt="Empty room"
            className="absolute inset-0 w-full h-full object-cover"
            draggable={false}
          />

          {/* Furniture overlay (additive demo) */}
          <div
            className="absolute inset-0"
            style={{
              opacity: furnishedOpacity,
              transition: "opacity 260ms ease",
            }}
          >
            <img
              src={sofaSrc}
              alt="AI placed sofa"
              className="absolute"
              style={{
                width: "56%",
                left: "18%",
                bottom: "18%",
                filter: "drop-shadow(0 24px 35px rgba(0,0,0,0.55))",
                transform: "rotate(-1deg)",
              }}
              draggable={false}
            />
            <img
              src={tableSrc}
              alt="AI placed coffee table"
              className="absolute"
              style={{
                width: "24%",
                left: "45%",
                bottom: "14%",
                filter: "drop-shadow(0 20px 30px rgba(0,0,0,0.5))",
              }}
              draggable={false}
            />
            <img
              src={pillowSrc}
              alt="AI placed decor"
              className="absolute"
              style={{
                width: "9%",
                left: "28%",
                bottom: "30%",
                filter: "drop-shadow(0 16px 24px rgba(0,0,0,0.45))",
              }}
              draggable={false}
            />

            <div className="absolute inset-0 bg-gradient-to-t from-black/40 via-black/0 to-black/0" />
          </div>

          <div className="absolute top-4 left-4 rounded-full border border-white/10 bg-black/[0.55] px-3 py-1 text-xs text-gray-100 font-jetbrains-mono">
            {showFurnished ? "Furnished" : "Empty"}
          </div>
        </div>

        <div className="px-5 py-4 border-t border-white/10">
          <div className="text-xs text-gray-300 font-jetbrains-mono">
            Demonstration overlay using real furniture assets — placed with
            correct scale & perspective.
          </div>
        </div>
      </div>
    </div>
  );
}
