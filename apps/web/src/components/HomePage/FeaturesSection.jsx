import { useCallback, useMemo, useRef, useEffect, useState } from "react";
import {
  Building2,
  Calendar,
  FileText,
  Link2,
  Sparkles,
  View,
  ChevronLeft,
  ChevronRight,
} from "lucide-react";
import { AnimatedInView } from "./AnimatedInView";
import { useTheme } from "@/components/ThemeProvider";

export function FeaturesSection() {
  // Replace the overflow scroller with a button-controlled carousel
  // (no scrollbars), and make cards flip to reveal their description.
  const viewportRef = useRef(null);

  const [startIndex, setStartIndex] = useState(0);
  const [cardsPerView, setCardsPerView] = useState(1);
  const [viewportWidth, setViewportWidth] = useState(0);
  const [isMounted, setIsMounted] = useState(false);

  const [hoveredCard, setHoveredCard] = useState(null);
  const [activeCard, setActiveCard] = useState(null);

  // NEW: pointer-drag state for the carousel (no scrollbars)
  const isDraggingRef = useRef(false);
  const [isDragging, setIsDragging] = useState(false);
  const dragStartXRef = useRef(0);
  const dragStartOffsetRef = useRef(0);
  const offsetRef = useRef(0);
  const movedRef = useRef(false);
  const suppressNextCardClickRef = useRef(false);
  const rafRef = useRef(null);
  const [dragOffsetPx, setDragOffsetPx] = useState(0); // 0..maxOffset

  const featureCards = useMemo(() => {
    return [
      {
        title: "Property & Client Management",
        desc: "Keep every listing, owner, lead, and buyer organized — with a workflow built for fast-moving teams.",
        Icon: Building2,
      },
      {
        title: "AI Staging & Decoration",
        desc: "Create premium staged visuals in minutes. Turn ‘hard to imagine’ spaces into ‘must-see’ listings.",
        Icon: Sparkles,
      },
      {
        title: "360 Virtual Tour",
        desc: "Let clients explore like they’re there. A smooth walkthrough experience that keeps attention (and boosts trust).",
        Icon: View,
      },
      {
        title: "Calendar",
        desc: "Stay sharp on visits and meetings. Fewer missed follow-ups, more deals moving forward.",
        Icon: Calendar,
      },
      {
        title: "Contract & PDF Generation",
        desc: "Produce clean, client-ready paperwork fast — with fewer back-and-forth edits and a clearer audit trail.",
        Icon: FileText,
      },
      {
        title: "Secure Client Share",
        desc: "Share a polished package with controlled access. One link that feels enterprise-grade — not hacked together.",
        Icon: Link2,
      },
    ];
  }, []);

  const maxStartIndex = useMemo(() => {
    return Math.max(0, featureCards.length - cardsPerView);
  }, [featureCards.length, cardsPerView]);

  useEffect(() => {
    setIsMounted(true);
  }, []);

  useEffect(() => {
    if (typeof window === "undefined" || !isMounted) {
      return;
    }

    const el = viewportRef.current;
    if (!el) {
      return;
    }

    const update = () => {
      const w = el.getBoundingClientRect().width;
      setViewportWidth(w);

      // Tailwind-ish breakpoints: sm(640), lg(1024)
      const nextCardsPerView = w >= 1024 ? 3 : w >= 640 ? 2 : 1;
      setCardsPerView(nextCardsPerView);
    };

    update();

    let ro;
    if (typeof ResizeObserver !== "undefined") {
      ro = new ResizeObserver(() => update());
      ro.observe(el);
    } else {
      window.addEventListener("resize", update);
    }

    return () => {
      if (ro) {
        ro.disconnect();
      } else {
        window.removeEventListener("resize", update);
      }
    };
  }, [isMounted]);

  useEffect(() => {
    // keep startIndex valid on resize
    setStartIndex((prev) => Math.min(prev, maxStartIndex));
  }, [maxStartIndex]);

  const scrollByCards = useCallback(
    (dir) => {
      setStartIndex((prev) => {
        const next = prev + dir;
        return Math.max(0, Math.min(maxStartIndex, next));
      });
    },
    [maxStartIndex],
  );

  const onCardClick = useCallback((idx) => {
    if (suppressNextCardClickRef.current) {
      suppressNextCardClickRef.current = false;
      return;
    }
    setActiveCard((prev) => (prev === idx ? null : idx));
  }, []);

  const pillClass =
    "inline-flex items-center justify-center px-2 py-1 rounded-full text-xs leading-none whitespace-nowrap font-jetbrains-mono";

  const gapPx = 16;
  const safeWidth = Math.max(0, viewportWidth);
  const computedCardWidth = useMemo(() => {
    if (!safeWidth) {
      return 320;
    }

    const gaps = gapPx * Math.max(0, cardsPerView - 1);
    const w = (safeWidth - gaps) / Math.max(1, cardsPerView);

    // Keep them feeling premium (not tiny) even if viewport is narrow.
    return Math.max(270, Math.floor(w));
  }, [safeWidth, cardsPerView]);

  const stepWidth = computedCardWidth + gapPx;
  const maxOffsetPx = maxStartIndex * stepWidth;

  // NEW: keep pixel offset synced with startIndex (unless dragging)
  useEffect(() => {
    if (isDraggingRef.current) {
      return;
    }
    const next = startIndex * stepWidth;
    offsetRef.current = next;
    setDragOffsetPx(next);
  }, [startIndex, stepWidth]);

  const setOffsetRaf = useCallback((nextOffset) => {
    offsetRef.current = nextOffset;
    if (rafRef.current) {
      cancelAnimationFrame(rafRef.current);
    }
    rafRef.current = requestAnimationFrame(() => {
      setDragOffsetPx(nextOffset);
    });
  }, []);

  const clampOffset = useCallback(
    (v) => {
      return Math.max(0, Math.min(maxOffsetPx, v));
    },
    [maxOffsetPx],
  );

  const onPointerDown = useCallback((e) => {
    // left click / primary touch only
    if (e.button !== undefined && e.button !== 0) {
      return;
    }

    isDraggingRef.current = true;
    setIsDragging(true);
    movedRef.current = false;
    suppressNextCardClickRef.current = false;

    dragStartXRef.current = e.clientX;
    dragStartOffsetRef.current = offsetRef.current;

    // capture moves even if pointer leaves the element
    try {
      e.currentTarget.setPointerCapture?.(e.pointerId);
    } catch {
      // ignore
    }
  }, []);

  const onPointerMove = useCallback(
    (e) => {
      if (!isDraggingRef.current) {
        return;
      }
      const dx = e.clientX - dragStartXRef.current;
      if (Math.abs(dx) > 6) {
        movedRef.current = true;
        suppressNextCardClickRef.current = true;
      }
      const next = clampOffset(dragStartOffsetRef.current - dx);
      setOffsetRaf(next);
    },
    [clampOffset, setOffsetRaf],
  );

  const endDrag = useCallback(() => {
    if (!isDraggingRef.current) {
      return;
    }

    isDraggingRef.current = false;
    setIsDragging(false);

    const currentOffset = offsetRef.current;
    const nextIndex = Math.round(currentOffset / stepWidth);
    const snappedIndex = Math.max(0, Math.min(maxStartIndex, nextIndex));

    setStartIndex(snappedIndex);
    const snappedOffset = snappedIndex * stepWidth;
    offsetRef.current = snappedOffset;
    setDragOffsetPx(snappedOffset);
  }, [stepWidth, maxStartIndex]);

  const onPointerUp = useCallback(() => {
    endDrag();
  }, [endDrag]);

  const onPointerCancel = useCallback(() => {
    endDrag();
  }, [endDrag]);

  const translateX = -dragOffsetPx;

  const { theme } = useTheme();
  const isDark = theme === "dark";

  // Theme-controlled classes (avoid Tailwind `dark:` so OS theme can’t break our light theme)
  const titleClass = isDark
    ? "text-2xl sm:text-3xl font-bold text-gray-50 font-jetbrains-mono"
    : "text-2xl sm:text-3xl font-bold text-gray-950 font-jetbrains-mono";

  const subtitleClass = isDark
    ? "mt-2 text-sm text-gray-300 font-jetbrains-mono max-w-2xl"
    : "mt-2 text-sm text-gray-700 font-jetbrains-mono max-w-2xl";

  // NOTE (UI2): Removing edge-fade overlays because they can create subtle background seams
  // on gradient backgrounds at certain zoom levels.

  const navBtnClass = isDark
    ? "w-9 h-9 rounded-full border border-white/10 bg-white/5 hover:bg-white/10 disabled:opacity-40 disabled:hover:bg-white/5 text-gray-100 transition-colors flex items-center justify-center"
    : "w-9 h-9 rounded-full border border-black/10 bg-black/5 hover:bg-black/10 disabled:opacity-40 disabled:hover:bg-black/5 text-gray-900 transition-colors flex items-center justify-center";

  const cardShellClass = isDark
    ? "relative rounded-2xl border border-white/10 bg-white/5 shadow-[0_18px_60px_rgba(0,0,0,0.35)] overflow-hidden"
    : "relative rounded-2xl border border-black/10 bg-white/80 shadow-[0_18px_60px_rgba(0,0,0,0.10)] overflow-hidden";

  const cardOverlayStyle = isDark
    ? {
        backdropFilter: "blur(16px)",
        WebkitBackdropFilter: "blur(16px)",
      }
    : {
        backdropFilter: "none",
        WebkitBackdropFilter: "none",
      };

  const stepPillClass = isDark
    ? `${pillClass} border border-white/10 bg-black/25 text-gray-200`
    : `${pillClass} border border-black/10 bg-black/5 text-gray-700`;

  const iconBoxClass = isDark
    ? "w-10 h-10 rounded-xl bg-white/5 border border-white/10 flex items-center justify-center"
    : "w-10 h-10 rounded-xl bg-black/5 border border-black/10 flex items-center justify-center";

  const iconClass = isDark ? "text-gray-100" : "text-gray-900";

  const frontTitleClass = isDark
    ? "px-3 text-center text-lg sm:text-xl font-semibold text-gray-50 font-jetbrains-mono tracking-tight leading-snug"
    : "px-3 text-center text-lg sm:text-xl font-semibold text-gray-950 font-jetbrains-mono tracking-tight leading-snug";

  const backDescClass = isDark
    ? "text-center text-sm sm:text-base text-gray-200 font-jetbrains-mono leading-relaxed max-w-[28ch]"
    : "text-center text-sm sm:text-base text-gray-700 font-jetbrains-mono leading-relaxed max-w-[28ch]";

  const frontFrameStyle = {
    boxShadow: isDark
      ? "0 0 0 1px rgba(255,255,255,0.08), 0 0 40px rgba(0,0,0,0.25)"
      : "0 0 0 1px rgba(0,0,0,0.06), 0 0 40px rgba(0,0,0,0.05)",
  };

  const backFrameStyle = {
    boxShadow: isDark
      ? "0 0 0 1px rgba(255,255,255,0.10), 0 0 55px rgba(0,0,0,0.28)"
      : "0 0 0 1px rgba(0,0,0,0.08), 0 0 55px rgba(0,0,0,0.06)",
  };

  return (
    <section className="max-w-7xl mx-auto px-4 sm:px-8 py-14 sm:py-18">
      <div>
        <h2 className={titleClass}>
          Premium features to streamline your workflow
        </h2>
        <p className={subtitleClass}>
          A step-by-step suite that every real estate agent needs.
        </p>
      </div>

      <div className="mt-8 relative">
        {/* Edge fades removed (prevents left-side color line artifact) */}

        {/* Desktop nav buttons */}
        <div className="hidden sm:flex items-center gap-2 absolute -top-12 right-0 z-20">
          <button
            type="button"
            onClick={() => scrollByCards(-1)}
            disabled={startIndex <= 0}
            className={navBtnClass}
            aria-label="Previous"
          >
            <ChevronLeft size={18} />
          </button>
          <button
            type="button"
            onClick={() => scrollByCards(1)}
            disabled={startIndex >= maxStartIndex}
            className={navBtnClass}
            aria-label="Next"
          >
            <ChevronRight size={18} />
          </button>
        </div>

        {/* Viewport is overflow-hidden so no scrollbars are ever shown */}
        <div
          ref={viewportRef}
          className="overflow-hidden"
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={onPointerUp}
          onPointerCancel={onPointerCancel}
          style={{
            cursor: isDragging ? "grabbing" : "grab",
            touchAction: "pan-y",
            userSelect: "none",
          }}
          aria-label="Premium workflows carousel"
        >
          <div
            className="flex gap-4"
            style={{
              transform: `translate3d(${translateX}px, 0, 0)`,
              transition: isDragging
                ? "none"
                : "transform 650ms cubic-bezier(0.22, 1, 0.36, 1)",
              willChange: "transform",
              paddingBottom: 10,
            }}
          >
            {featureCards.map((f, idx) => {
              const Icon = f.Icon;
              const delay = idx * 0.06;
              const isFlipped = hoveredCard === idx || activeCard === idx;

              return (
                <AnimatedInView
                  key={f.title}
                  delay={delay}
                  className="relative"
                  style={{ width: computedCardWidth, flex: "0 0 auto" }}
                >
                  <button
                    type="button"
                    onClick={() => onCardClick(idx)}
                    onMouseEnter={() => setHoveredCard(idx)}
                    onMouseLeave={() => setHoveredCard(null)}
                    className="group w-full text-left"
                    aria-label={`${f.title} details`}
                  >
                    <div
                      className={cardShellClass}
                      style={{
                        height: 240,
                        perspective: "1100px",
                        WebkitPerspective: "1100px",
                      }}
                    >
                      {/* Backdrop blur is premium in dark, but hurts readability in light */}
                      <div
                        className="absolute inset-0 pointer-events-none"
                        style={cardOverlayStyle}
                      />

                      <div
                        className="absolute inset-0"
                        style={{
                          transformStyle: "preserve-3d",
                          WebkitTransformStyle: "preserve-3d",
                          transition:
                            "transform 700ms cubic-bezier(0.22, 1, 0.36, 1)",
                          transform: isFlipped
                            ? "rotateY(180deg)"
                            : "rotateY(0deg)",
                          willChange: "transform",
                        }}
                      >
                        {/* Front */}
                        <div
                          className="absolute inset-0 p-6 flex items-center justify-center"
                          style={{
                            backfaceVisibility: "hidden",
                            WebkitBackfaceVisibility: "hidden",
                            transform: "rotateY(0deg)",
                            WebkitTransform: "rotateY(0deg)",
                            visibility: isFlipped ? "hidden" : "visible",
                          }}
                        >
                          <span
                            className={`${stepPillClass} absolute top-5 left-5`}
                          >
                            Step {idx + 1}
                          </span>

                          <div className={frontTitleClass}>{f.title}</div>

                          <div
                            className="absolute inset-0 rounded-2xl pointer-events-none"
                            style={frontFrameStyle}
                          />
                        </div>

                        {/* Back */}
                        <div
                          className="absolute inset-0"
                          style={{
                            backfaceVisibility: "hidden",
                            WebkitBackfaceVisibility: "hidden",
                            transform: "rotateY(180deg)",
                            WebkitTransform: "rotateY(180deg)",
                            visibility: isFlipped ? "visible" : "hidden",
                          }}
                        >
                          <div className="absolute top-6 left-6 right-6 flex items-start justify-between gap-4">
                            <div className={iconBoxClass}>
                              <Icon size={18} className={iconClass} />
                            </div>
                            <span className={stepPillClass}>
                              Step {idx + 1}
                            </span>
                          </div>

                          <div className="absolute inset-0 flex items-center justify-center px-7">
                            <div className={backDescClass}>{f.desc}</div>
                          </div>

                          <div
                            className="absolute inset-0 rounded-2xl pointer-events-none"
                            style={backFrameStyle}
                          />
                        </div>
                      </div>
                    </div>
                  </button>
                </AnimatedInView>
              );
            })}
          </div>
        </div>

        {/* Mobile nav buttons */}
        <div className="sm:hidden mt-4 flex items-center justify-center gap-2">
          <button
            type="button"
            onClick={() => scrollByCards(-1)}
            disabled={startIndex <= 0}
            className={navBtnClass}
            aria-label="Previous"
          >
            <ChevronLeft size={18} />
          </button>
          <button
            type="button"
            onClick={() => scrollByCards(1)}
            disabled={startIndex >= maxStartIndex}
            className={navBtnClass}
            aria-label="Next"
          >
            <ChevronRight size={18} />
          </button>
        </div>
      </div>
    </section>
  );
}
