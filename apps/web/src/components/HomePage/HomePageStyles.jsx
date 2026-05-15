import { useEffect, useState } from "react";

export function HomePageStyles() {
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  // Important: avoid rendering <style> on the server, because some browsers will
  // move body <style> tags into <head> during HTML parsing which can break hydration.
  if (!mounted) {
    return null;
  }

  return (
    <style>{`
      @keyframes subtleGlow {
        0% {
          filter: drop-shadow(0 0 0 rgba(255, 122, 24, 0.0));
          opacity: 0.95;
        }
        50% {
          filter: drop-shadow(0 0 10px rgba(255, 122, 24, 0.14));
          opacity: 1;
        }
        100% {
          filter: drop-shadow(0 0 0 rgba(255, 122, 24, 0.0));
          opacity: 0.95;
        }
      }

      .icon-glow {
        animation: subtleGlow 3.6s ease-in-out infinite;
      }

      @keyframes ctaGradientMove {
        0% {
          transform: translate3d(-10%, -10%, 0);
          opacity: 0.35;
        }
        50% {
          transform: translate3d(10%, 10%, 0);
          opacity: 0.48;
        }
        100% {
          transform: translate3d(-10%, -10%, 0);
          opacity: 0.35;
        }
      }

      .cta-gradient {
        background:
          radial-gradient(800px circle at 18% 30%, rgba(99, 102, 241, 0.22), transparent 55%),
          radial-gradient(900px circle at 86% 40%, rgba(16, 185, 129, 0.14), transparent 60%),
          radial-gradient(700px circle at 60% 115%, rgba(255, 122, 24, 0.12), transparent 60%);
        animation: ctaGradientMove 18s ease-in-out infinite;
      }

      @keyframes grainShift {
        0% {
          background-position: 0% 0%;
          opacity: 0.12;
        }
        50% {
          background-position: 40% 60%;
          opacity: 0.16;
        }
        100% {
          background-position: 0% 0%;
          opacity: 0.12;
        }
      }

      .hero-grain {
        background-image:
          radial-gradient(circle at 25% 20%, rgba(255, 255, 255, 0.06) 0, rgba(255, 255, 255, 0) 50%),
          radial-gradient(circle at 75% 60%, rgba(255, 255, 255, 0.05) 0, rgba(255, 255, 255, 0) 55%),
          repeating-linear-gradient(
            90deg,
            rgba(255, 255, 255, 0.02) 0,
            rgba(255, 255, 255, 0.02) 1px,
            rgba(255, 255, 255, 0) 2px,
            rgba(255, 255, 255, 0) 6px
          );
        mix-blend-mode: overlay;
        animation: grainShift 10s ease-in-out infinite;
      }

      @keyframes knobPulse {
        0% {
          transform: scale(1);
          box-shadow: 0 0 0 rgba(255, 122, 24, 0);
        }
        50% {
          transform: scale(1.06);
          box-shadow: 0 0 18px rgba(255, 122, 24, 0.12);
        }
        100% {
          transform: scale(1);
          box-shadow: 0 0 0 rgba(255, 122, 24, 0);
        }
      }

      .slider-knob {
        animation: knobPulse 3.2s ease-in-out infinite;
      }
    `}</style>
  );
}
