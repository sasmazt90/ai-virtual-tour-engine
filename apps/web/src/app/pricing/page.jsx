import { Check } from "lucide-react";
import { useEffect, useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import MarketingHeader from "@/components/MarketingHeader";
import useMarketingAnalytics from "@/hooks/useMarketingAnalytics";

function formatMoney(unitAmount, currency) {
  if (!Number.isFinite(unitAmount) || !currency) {
    return "—";
  }

  try {
    const dollars = unitAmount / 100;
    // Use a fixed locale for SSR/client consistency (avoid hydration mismatches).
    return new Intl.NumberFormat("en-US", {
      style: "currency",
      currency: currency.toUpperCase(),
      maximumFractionDigits: 0,
    }).format(dollars);
  } catch {
    return `$${(unitAmount / 100).toFixed(0)}`;
  }
}

export default function PricingPage() {
  const { trackSignUpClick } = useMarketingAnalytics();

  useEffect(() => {
    if (typeof window !== "undefined") {
      document.title = "Pricing | 360 Estate Suite";
    }
  }, []);

  const onHeroSignUpClick = useMemo(() => {
    return () => trackSignUpClick("pricing_hero");
  }, [trackSignUpClick]);

  const onPlanSignUpClick = useMemo(() => {
    return (planKey) => {
      trackSignUpClick(`pricing_plan_${planKey}`);
    };
  }, [trackSignUpClick]);

  const onBottomSignUpClick = useMemo(() => {
    return () => trackSignUpClick("pricing_bottom_cta");
  }, [trackSignUpClick]);

  const { data, isLoading, error } = useQuery({
    queryKey: ["marketing-pricing"],
    queryFn: async () => {
      const res = await fetch("/api/marketing/pricing");
      if (!res.ok) {
        throw new Error(
          `When fetching /api/marketing/pricing, the response was [${res.status}] ${res.statusText}`,
        );
      }
      return res.json();
    },
    staleTime: 1000 * 60 * 30,
    retry: 1,
  });

  const cards = useMemo(() => {
    const plans = data?.plans || {};

    const starter = plans?.starter || {};
    const pro = plans?.pro || {};
    const agency = plans?.agency || {};

    return [
      {
        key: "starter",
        name: "Starter",
        coins: starter.coins || 100,
        audience: "For individual agents",
        price: starter.price,
        highlight: false,
      },
      {
        key: "pro",
        name: "Pro",
        coins: pro.coins || 250,
        audience: "For growing teams",
        price: pro.price,
        highlight: true,
      },
      {
        key: "agency",
        name: "Agency",
        coins: agency.coins || 500,
        audience: "For agencies & multi-property teams",
        price: agency.price,
        highlight: false,
      },
    ];
  }, [data?.plans]);

  const featureItems = [
    "AI Staging & Virtual Tours",
    "Property & Client Management",
    "Contracts & PDF generation",
    "Secure Share Links",
    "Credit-based usage",
    "Stripe-secured payments",
  ];

  const pricingError = error
    ? "We couldn’t load pricing right now. Please refresh and try again."
    : null;

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-[#1E1E1E]">
      <MarketingHeader />

      <main className="max-w-7xl mx-auto px-4 sm:px-8 py-10 sm:py-14">
        <header className="max-w-3xl">
          <h1 className="text-3xl sm:text-4xl font-bold text-gray-900 dark:text-gray-100 font-jetbrains-mono">
            Simple, transparent pricing for real estate professionals
          </h1>
          <p className="mt-3 text-gray-600 dark:text-gray-300 font-jetbrains-mono">
            Choose the plan that fits your workflow. Upgrade anytime.
          </p>

          <div className="mt-6 flex flex-col sm:flex-row gap-3">
            <a
              href="/signup"
              className="inline-flex items-center justify-center px-5 py-3 rounded-lg bg-[var(--brand)] hover:bg-[var(--brandHover)] text-white font-medium font-jetbrains-mono"
              onClick={onHeroSignUpClick}
            >
              Sign Up
            </a>
            <a
              href="/login"
              className="inline-flex items-center justify-center px-5 py-3 rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-[#121212] text-gray-900 dark:text-gray-100 hover:bg-gray-50 dark:hover:bg-gray-800 font-medium font-jetbrains-mono"
            >
              Login
            </a>
          </div>
          <div className="mt-3 text-xs text-gray-500 dark:text-gray-400 font-jetbrains-mono">
            Cancel anytime · Secure Stripe payments
          </div>

          {pricingError ? (
            <p className="mt-4 text-sm text-gray-600 dark:text-gray-300 font-jetbrains-mono">
              {pricingError}
            </p>
          ) : null}
        </header>

        <section className="mt-10">
          <h2 className="text-xl sm:text-2xl font-bold text-gray-900 dark:text-gray-100 font-jetbrains-mono">
            Plans
          </h2>

          <div className="mt-6 grid grid-cols-1 md:grid-cols-3 gap-6">
            {cards.map((c) => {
              const unitAmount = c.price?.unitAmount;
              const currency = c.price?.currency;
              const interval = c.price?.interval;

              const priceText = formatMoney(unitAmount, currency);
              const intervalText = interval ? `/${interval}` : "/month";

              const cardClass = c.highlight
                ? "bg-white dark:bg-[#262626] rounded-xl shadow-lg dark:shadow-none dark:ring-2 dark:ring-[var(--brand)] p-6"
                : "bg-white dark:bg-[#262626] rounded-xl shadow-lg dark:shadow-none dark:ring-1 dark:ring-gray-700 p-6";

              const onThisPlanSignUp = () => onPlanSignUpClick(c.key);

              return (
                <div key={c.key} className={cardClass}>
                  <div className="flex items-start justify-between gap-4">
                    <div>
                      <div className="text-xl font-semibold text-gray-900 dark:text-gray-100 font-jetbrains-mono">
                        {c.name}
                      </div>
                      <div className="mt-1 text-sm text-gray-600 dark:text-gray-300 font-jetbrains-mono">
                        {c.audience}
                      </div>
                    </div>
                    {c.highlight ? (
                      <div className="px-2 py-1 rounded-full text-xs bg-[var(--brandSoft)] dark:bg-[var(--brandSoftDark)] text-[var(--brandDark)] dark:text-[var(--brand)] font-jetbrains-mono">
                        Most popular
                      </div>
                    ) : null}
                  </div>

                  <div className="mt-6">
                    <div className="text-4xl font-bold text-gray-900 dark:text-gray-100 font-jetbrains-mono">
                      {isLoading ? "…" : priceText}
                      <span className="text-sm font-normal text-gray-600 dark:text-gray-300 ml-2">
                        {intervalText}
                      </span>
                    </div>
                    <div className="mt-2 text-sm text-gray-700 dark:text-gray-200 font-jetbrains-mono">
                      Includes{" "}
                      <span className="font-semibold">{c.coins} coins</span>
                    </div>
                  </div>

                  <a
                    href="/signup"
                    className="mt-6 inline-flex w-full items-center justify-center px-4 py-3 rounded-lg bg-[var(--brand)] hover:bg-[var(--brandHover)] text-white font-medium font-jetbrains-mono"
                    onClick={onThisPlanSignUp}
                  >
                    Sign Up
                  </a>

                  <div className="mt-2 text-xs text-gray-500 dark:text-gray-400 font-jetbrains-mono text-center">
                    Cancel anytime · Secure Stripe payments
                  </div>

                  <div className="mt-6 border-t border-gray-200 dark:border-gray-700 pt-4">
                    <div className="text-xs text-gray-500 dark:text-gray-400 font-jetbrains-mono">
                      Included features
                    </div>
                    <ul className="mt-3 space-y-2">
                      {featureItems.slice(0, 4).map((f) => (
                        <li
                          key={f}
                          className="flex items-start gap-2 text-sm text-gray-700 dark:text-gray-200 font-jetbrains-mono"
                        >
                          <Check
                            size={16}
                            className="mt-0.5 text-emerald-600 dark:text-emerald-300"
                          />
                          <span>{f}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                </div>
              );
            })}
          </div>
        </section>

        <section className="mt-12">
          <h2 className="text-xl sm:text-2xl font-bold text-gray-900 dark:text-gray-100 font-jetbrains-mono">
            What you get
          </h2>
          <div className="mt-4 grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="rounded-xl border border-gray-200 dark:border-gray-700 bg-white dark:bg-[#262626] p-6">
              <div className="text-sm font-semibold text-gray-900 dark:text-gray-100 font-jetbrains-mono">
                Feature list
              </div>
              <ul className="mt-4 space-y-2">
                {featureItems.map((f) => (
                  <li
                    key={f}
                    className="flex items-start gap-2 text-sm text-gray-700 dark:text-gray-200 font-jetbrains-mono"
                  >
                    <Check
                      size={16}
                      className="mt-0.5 text-emerald-600 dark:text-emerald-300"
                    />
                    <span>{f}</span>
                  </li>
                ))}
              </ul>
            </div>

            <div className="rounded-xl border border-gray-200 dark:border-gray-700 bg-white dark:bg-[#262626] p-6">
              <div className="text-sm font-semibold text-gray-900 dark:text-gray-100 font-jetbrains-mono">
                Coins explained
              </div>
              <p className="mt-3 text-sm text-gray-600 dark:text-gray-300 font-jetbrains-mono">
                Coins are your usage balance. You spend coins when you generate
                AI outputs like staging images and virtual tours.
              </p>
              <p className="mt-3 text-sm text-gray-600 dark:text-gray-300 font-jetbrains-mono">
                Coins help keep pricing fair: light users spend less, heavy
                users can buy more as needed.
              </p>
              <p className="mt-3 text-sm text-gray-600 dark:text-gray-300 font-jetbrains-mono">
                <span className="font-semibold">Unused coins carry over</span>{" "}
                while your subscription is active.
              </p>
            </div>
          </div>
        </section>

        <section className="mt-12 rounded-2xl bg-gradient-to-br from-[var(--brand)] to-[var(--brandDark)] p-8 sm:p-10">
          <h2 className="text-2xl sm:text-3xl font-bold text-white font-jetbrains-mono">
            Ready to streamline your workflow?
          </h2>
          <p className="mt-2 text-white/90 font-jetbrains-mono">
            Sign up and start managing properties, clients, and AI outputs in
            one place.
          </p>
          <div className="mt-6 flex flex-col sm:flex-row gap-3">
            <a
              href="/signup"
              className="inline-flex items-center justify-center px-5 py-3 rounded-lg bg-white text-[var(--brandDark)] hover:bg-white/90 font-medium font-jetbrains-mono"
              onClick={onBottomSignUpClick}
            >
              Sign Up
            </a>
            <a
              href="/login"
              className="inline-flex items-center justify-center px-5 py-3 rounded-lg border border-white/40 text-white hover:bg-white/10 font-medium font-jetbrains-mono"
            >
              Login
            </a>
          </div>
        </section>
      </main>
    </div>
  );
}
