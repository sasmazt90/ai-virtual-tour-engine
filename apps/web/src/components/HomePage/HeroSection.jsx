import { useMemo, useState } from "react";
import { Flashlight, FlashlightOff, Moon, Sun } from "lucide-react";
import { BeforeAfterSlider } from "./BeforeAfterSlider";
import { AnimatedInView } from "./AnimatedInView";

export function HeroSection({ onSignUpClick }) {
  const furnishedNightFlashOn =
    "https://ucarecdn.com/fe9a1c32-58ec-43f9-b132-275d27b9320f/-/format/auto/";
  const furnishedNightFlashOff =
    "https://ucarecdn.com/f8acec5e-3b42-4c95-bcc5-88d96ef58406/-/format/auto/";
  const furnishedDayFlashOn =
    "https://ucarecdn.com/d3c02b60-d7ea-48cc-b684-0a65f6415444/-/format/auto/";
  const furnishedDayFlashOff =
    "https://ucarecdn.com/f2fa0a52-215c-4c78-bc29-03d25bddb7f5/-/format/auto/";

  // NOTE: For the vacant room, both "night + flashlight on" and "night + flashlight off"
  // should use the same supplied night image.
  const vacantNight =
    "https://ucarecdn.com/446f3288-1a85-4c09-86c8-9d08762e96b0/-/format/auto/";
  const vacantDay =
    "https://ucarecdn.com/00d090f1-b934-4063-a1c5-fa2094ac2713/-/format/auto/";

  const [isFlashlightOn, setIsFlashlightOn] = useState(false);
  const [isNight, setIsNight] = useState(false);

  const furnishedSrc = useMemo(() => {
    if (isNight) {
      return isFlashlightOn ? furnishedNightFlashOn : furnishedNightFlashOff;
    }
    return isFlashlightOn ? furnishedDayFlashOn : furnishedDayFlashOff;
  }, [
    isNight,
    isFlashlightOn,
    furnishedNightFlashOn,
    furnishedNightFlashOff,
    furnishedDayFlashOn,
    furnishedDayFlashOff,
  ]);

  const vacantSrc = useMemo(() => {
    return isNight ? vacantNight : vacantDay;
  }, [isNight, vacantNight, vacantDay]);

  const beforeImage = vacantSrc;
  const afterImage = furnishedSrc;

  return (
    <section className="relative border-b border-black/10 dark:border-white/10 overflow-hidden">
      {/*
        IMPORTANT: We already render a global grain overlay in ThemeProvider.
        Duplicating it here makes light mode text look washed out.
      */}

      <div className="max-w-7xl mx-auto px-4 sm:px-8 py-12 sm:py-20">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-10 lg:gap-12 items-center">
          <div>
            <div className="inline-flex items-center gap-2 rounded-full border border-black/10 dark:border-white/10 bg-black/5 dark:bg-white/5 px-3 py-1 text-xs text-gray-700 dark:text-gray-200 font-jetbrains-mono">
              <span className="w-1.5 h-1.5 rounded-full bg-[var(--brand70)]" />
              Built for modern real estate teams
            </div>

            <h1 className="mt-6 text-4xl sm:text-5xl lg:text-6xl font-bold text-gray-950 dark:text-gray-50 font-jetbrains-mono tracking-tight leading-[1.05]">
              <span className="block">The all-in-one</span>
              <span className="block">AI-powered</span>
              <span className="block">estate operating suite</span>
            </h1>

            <p className="mt-4 text-sm sm:text-base text-gray-700 dark:text-gray-300 font-jetbrains-mono max-w-xl">
              From empty rooms to client-ready experiences — in one platform.
            </p>

            <div className="mt-8 flex flex-col sm:flex-row gap-3">
              <a
                href="/account/signup"
                className="inline-flex items-center justify-center px-5 py-3 rounded-lg bg-[var(--brand90)] hover:bg-[var(--brand)] text-white font-medium font-jetbrains-mono transition-colors"
                onClick={onSignUpClick}
              >
                Sign Up
              </a>
              <a
                href="/account/signin"
                className="inline-flex items-center justify-center px-5 py-3 rounded-lg border border-black/10 dark:border-white/10 bg-black/5 dark:bg-white/5 hover:bg-black/10 dark:hover:bg-white/10 text-gray-900 dark:text-gray-100 font-medium font-jetbrains-mono transition-colors"
              >
                Login
              </a>
            </div>

            {/* {{ remove the three feature pills under the buttons per request }} */}
          </div>

          <AnimatedInView delay={0.08}>
            <div className="relative">
              <BeforeAfterSlider
                beforeSrc={beforeImage}
                afterSrc={afterImage}
                beforeAlt="Empty room"
                afterAlt="Original furnished room"
              />

              {/* Top-left: Flashlight toggle */}
              <div className="absolute top-3 left-3 z-20">
                <div className="flex items-center rounded-full border border-black/10 dark:border-white/[0.15] bg-white/70 dark:bg-black/[0.45] backdrop-blur px-1 py-1 shadow-[0_14px_50px_rgba(0,0,0,0.18)] dark:shadow-[0_14px_50px_rgba(0,0,0,0.40)]">
                  <button
                    type="button"
                    onPointerDown={(e) => e.stopPropagation()}
                    onClick={() => setIsFlashlightOn(true)}
                    className={`w-9 h-9 rounded-full flex items-center justify-center transition-colors ${
                      isFlashlightOn
                        ? "bg-black/[0.08] dark:bg-white/[0.18] text-gray-900 dark:text-gray-50"
                        : "text-gray-700 dark:text-gray-300 hover:text-gray-900 dark:hover:text-gray-50"
                    }`}
                    aria-label="Flashlight on"
                    title="Flashlight on"
                  >
                    <Flashlight size={18} />
                  </button>
                  <button
                    type="button"
                    onPointerDown={(e) => e.stopPropagation()}
                    onClick={() => setIsFlashlightOn(false)}
                    className={`w-9 h-9 rounded-full flex items-center justify-center transition-colors ${
                      !isFlashlightOn
                        ? "bg-black/[0.08] dark:bg-white/[0.18] text-gray-900 dark:text-gray-50"
                        : "text-gray-700 dark:text-gray-300 hover:text-gray-900 dark:hover:text-gray-50"
                    }`}
                    aria-label="Flashlight off"
                    title="Flashlight off"
                  >
                    <FlashlightOff size={18} />
                  </button>
                </div>
              </div>

              {/* Top-right: Day/Night toggle */}
              <div className="absolute top-3 right-3 z-20">
                <div className="flex items-center rounded-full border border-black/10 dark:border-white/[0.15] bg-white/70 dark:bg-black/[0.45] backdrop-blur px-1 py-1 shadow-[0_14px_50px_rgba(0,0,0,0.18)] dark:shadow-[0_14px_50px_rgba(0,0,0,0.40)]">
                  <button
                    type="button"
                    onPointerDown={(e) => e.stopPropagation()}
                    onClick={() => setIsNight(false)}
                    className={`w-9 h-9 rounded-full flex items-center justify-center transition-colors ${
                      !isNight
                        ? "bg-black/[0.08] dark:bg-white/[0.18] text-gray-900 dark:text-gray-50"
                        : "text-gray-700 dark:text-gray-300 hover:text-gray-900 dark:hover:text-gray-50"
                    }`}
                    aria-label="Day"
                    title="Day"
                  >
                    <Sun size={18} />
                  </button>
                  <button
                    type="button"
                    onPointerDown={(e) => e.stopPropagation()}
                    onClick={() => setIsNight(true)}
                    className={`w-9 h-9 rounded-full flex items-center justify-center transition-colors ${
                      isNight
                        ? "bg-black/[0.08] dark:bg-white/[0.18] text-gray-900 dark:text-gray-50"
                        : "text-gray-700 dark:text-gray-300 hover:text-gray-900 dark:hover:text-gray-50"
                    }`}
                    aria-label="Night"
                    title="Night"
                  >
                    <Moon size={18} />
                  </button>
                </div>
              </div>
            </div>
          </AnimatedInView>
        </div>
      </div>
    </section>
  );
}
