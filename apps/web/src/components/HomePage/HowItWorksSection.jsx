import { useMemo } from "react";
import { AnimatedInView } from "./AnimatedInView";

export function HowItWorksSection() {
  const steps = useMemo(() => {
    return [
      {
        num: "01",
        title: "Add properties & clients",
        desc: "Capture the basics once — reuse everywhere.",
      },
      {
        num: "02",
        title: "Generate staging, tours & contracts",
        desc: "Create on-demand assets with pay-as-you-go credits.",
      },
      {
        num: "03",
        title: "Share securely with clients",
        desc: "Deliver the full experience with controlled access.",
      },
    ];
  }, []);

  return (
    <section className="border-y border-white/10 bg-[#050608]">
      <div className="max-w-7xl mx-auto px-4 sm:px-8 py-14 sm:py-18">
        <h2 className="text-2xl sm:text-3xl font-bold text-gray-50 font-jetbrains-mono">
          How it works
        </h2>

        <div className="mt-8 grid grid-cols-1 md:grid-cols-3 gap-5">
          {steps.map((s, idx) => {
            const delay = idx * 0.09;

            return (
              <AnimatedInView
                key={s.num}
                delay={delay}
                className="relative rounded-2xl border border-white/10 bg-white/5 p-6"
              >
                <div className="text-xs text-gray-300 font-jetbrains-mono">
                  Step {s.num}
                </div>
                <div className="mt-2 text-base font-semibold text-gray-50 font-jetbrains-mono">
                  {s.title}
                </div>
                <div className="mt-2 text-xs text-gray-300 font-jetbrains-mono">
                  {s.desc}
                </div>

                {idx < steps.length - 1 ? (
                  <div className="hidden md:block absolute top-1/2 -right-3 w-6 h-[2px] bg-white/10" />
                ) : null}
              </AnimatedInView>
            );
          })}
        </div>
      </div>
    </section>
  );
}
