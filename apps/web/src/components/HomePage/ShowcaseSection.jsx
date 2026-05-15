import { useEffect, useMemo, useState } from "react";
import { motion } from "motion/react";
import Fake360Viewer from "@/components/Fake360Viewer";
import { AnimatedInView } from "./AnimatedInView";

export function ShowcaseSection() {
  const originalShowcase =
    "https://ucarecdn.com/02486dab-20b1-4020-89d6-df9714cfd241/-/format/auto/";
  const stagedShowcase =
    "https://ucarecdn.com/4e8cb4e0-14d8-4517-a177-34473e6a44c2/-/format/auto/";

  const tourPayload = useMemo(() => {
    const frames = [
      "https://ucarecdn.com/4afa8729-a232-4536-8bc6-bc4684c08927/-/format/auto/",
      "https://ucarecdn.com/7042433d-ad54-4066-b269-a86b6c40c25d/-/format/auto/",
      "https://ucarecdn.com/e13d4cb0-4968-4603-95d9-165d54cdb1d7/-/format/auto/",
      "https://ucarecdn.com/966132bc-9376-4f07-9487-ae81d11efe7e/-/format/auto/",
      "https://ucarecdn.com/a883bb4c-215d-4f7f-9008-ca7bf6763dad/-/format/auto/",
      "https://ucarecdn.com/93ab76b1-ac3b-438e-89c1-2e807292477e/-/format/auto/",
      "https://ucarecdn.com/cf078260-095b-4ced-adfe-8e51267b7cd0/-/format/auto/",
      "https://ucarecdn.com/033ea138-eddb-472b-9d60-1a07eac6557b/-/format/auto/",
      "https://ucarecdn.com/f134b576-0145-4e3f-b3d9-628bde7f05fd/-/format/auto/",
      "https://ucarecdn.com/2c162fc6-3233-4461-8f57-0d73ef733607/-/format/auto/",
      "https://ucarecdn.com/acaf74ca-dcc4-4665-a7d5-d348f69a0a10/-/format/auto/",
      "https://ucarecdn.com/5671bb56-1135-4b37-8dcb-512e327ffc3d/-/format/auto/",
      "https://ucarecdn.com/1a875b5d-790a-4c2f-afa2-586120a277fb/-/format/auto/",
      "https://ucarecdn.com/4bf3a6d3-07f5-49ea-8a05-40adf53637fc/-/format/auto/",
      "https://ucarecdn.com/a2da65b2-40e1-4ee6-95c9-02cb2a06f633/-/format/auto/",
      "https://ucarecdn.com/c1ebf752-0f8d-4b2e-ad10-4828ac631ffd/-/format/auto/",
      "https://ucarecdn.com/84d48b1c-81df-4d42-9fc6-f0f7025d262b/-/format/auto/",
      "https://ucarecdn.com/a860ce7d-6926-4dca-8be5-745c80157ac3/-/format/auto/",
      "https://ucarecdn.com/f10b1e23-b086-473a-b967-d1aa0b384602/-/format/auto/",
      "https://ucarecdn.com/7f899452-c20f-4c1d-bfc9-0f084acb3b10/-/format/auto/",
      "https://ucarecdn.com/d84be1b0-598f-4b38-a3c6-4780cac59dad/-/format/auto/",
    ];

    // NEW: multi-point tour payload.
    // Drag rotates within the current point; clicking a marker moves to another point.
    // NOTE: `steps` tells the viewer how many angles we want around 360°. If some are
    // missing, the viewer fills gaps with the nearest available frame.
    return {
      initialPointId: "P1",
      points: [
        {
          pointId: "P1",
          steps: 36,
          frames,
          initialIndex: 0,
          hotspots: [
            { x: 0.5, y: 0.66, toPointId: "P2", direction: "forward" },
          ],
        },
        {
          pointId: "P2",
          steps: 36,
          frames,
          initialIndex: 6,
          hotspots: [{ x: 0.5, y: 0.78, toPointId: "P1", direction: "back" }],
        },
      ],
    };
  }, []);

  const [showcaseMode, setShowcaseMode] = useState("staged");
  const [isMounted, setIsMounted] = useState(false);

  useEffect(() => {
    setIsMounted(true);
  }, []);

  const showcaseTabs = useMemo(() => {
    return [
      { key: "original", label: "Original photo" },
      { key: "staged", label: "AI staged" },
      { key: "tour", label: "Virtual tour" },
    ];
  }, []);

  const showcaseTitle = useMemo(() => {
    if (showcaseMode === "original") return "Original photo";
    if (showcaseMode === "tour") return "Virtual tour";
    return "AI staged";
  }, [showcaseMode]);

  const showcaseImageSrc = useMemo(() => {
    if (showcaseMode === "original") return originalShowcase;
    return stagedShowcase;
  }, [originalShowcase, stagedShowcase, showcaseMode]);

  const showcaseImageAlt = useMemo(() => {
    if (showcaseMode === "original") return "Original listing photo";
    return "AI staged listing photo";
  }, [showcaseMode]);

  const tabButtonBase =
    "px-3 py-2 rounded-full text-xs sm:text-sm font-jetbrains-mono transition-colors";

  const imageNode = useMemo(() => {
    if (!isMounted) {
      return (
        <img
          key={showcaseMode}
          src={showcaseImageSrc}
          alt={showcaseImageAlt}
          className="absolute inset-0 w-full h-full object-cover"
          draggable={false}
          loading="lazy"
        />
      );
    }

    return (
      <motion.img
        key={showcaseMode}
        src={showcaseImageSrc}
        alt={showcaseImageAlt}
        className="absolute inset-0 w-full h-full object-cover"
        initial={{ opacity: 0, filter: "blur(6px)" }}
        animate={{ opacity: 1, filter: "blur(0px)" }}
        transition={{ duration: 0.55, ease: [0.22, 1, 0.36, 1] }}
        draggable={false}
        loading="lazy"
      />
    );
  }, [isMounted, showcaseMode, showcaseImageAlt, showcaseImageSrc]);

  return (
    <section className="max-w-7xl mx-auto px-4 sm:px-8 py-14 sm:py-18">
      <AnimatedInView>
        <div className="flex flex-col sm:flex-row sm:items-end sm:justify-between gap-6">
          <div>
            <h2 className="text-2xl sm:text-3xl font-bold text-gray-950 dark:text-gray-50 font-jetbrains-mono">
              Unlock the Power of AI for Your Listings
            </h2>
            <p className="mt-2 text-sm text-gray-700 dark:text-gray-300 font-jetbrains-mono max-w-2xl">
              Toggle views to preview the client experience.
            </p>
          </div>

          <div className="inline-flex items-center gap-1 rounded-full border border-black/10 dark:border-white/10 bg-black/5 dark:bg-white/5 p-1">
            {showcaseTabs.map((t) => {
              const active = t.key === showcaseMode;
              const activeClass =
                "bg-black/10 dark:bg-white/10 text-gray-950 dark:text-gray-50 border border-black/10 dark:border-white/10";
              const inactiveClass =
                "text-gray-700 dark:text-gray-300 hover:text-gray-950 dark:hover:text-gray-50";
              const cls = `${tabButtonBase} ${active ? activeClass : inactiveClass}`;

              return (
                <button
                  key={t.key}
                  type="button"
                  className={cls}
                  onClick={() => setShowcaseMode(t.key)}
                  aria-pressed={active}
                >
                  {t.label}
                </button>
              );
            })}
          </div>
        </div>
      </AnimatedInView>

      <AnimatedInView delay={0.08} className="mt-8">
        <div className="rounded-2xl border border-black/10 dark:border-white/10 bg-black/5 dark:bg-white/5 overflow-hidden">
          <div className="p-5">
            {showcaseMode === "tour" ? (
              isMounted ? (
                <Fake360Viewer tourPayload={tourPayload} height={520} />
              ) : (
                <div
                  className="relative overflow-hidden rounded-xl border border-black/10 dark:border-white/10 bg-black flex items-center justify-center"
                  style={{ height: 520 }}
                >
                  <div className="text-gray-300 text-sm font-jetbrains-mono">
                    Loading tour...
                  </div>
                </div>
              )
            ) : (
              <div
                className="relative overflow-hidden rounded-xl border border-black/10 dark:border-white/10 bg-black"
                style={{ height: 520 }}
              >
                {imageNode}
                <div className="absolute inset-0 bg-gradient-to-t from-black/60 via-black/15 to-black/0" />
              </div>
            )}
          </div>
        </div>
      </AnimatedInView>
    </section>
  );
}
