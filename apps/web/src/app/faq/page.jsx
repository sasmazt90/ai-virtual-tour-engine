import { ChevronDown } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import MarketingHeader from "@/components/MarketingHeader";
import useMarketingAnalytics from "@/hooks/useMarketingAnalytics";

export default function FAQPage() {
  const { trackSignUpClick } = useMarketingAnalytics();

  useEffect(() => {
    if (typeof window !== "undefined") {
      document.title = "FAQ | 360 Estate Suite";
    }
  }, []);

  const items = useMemo(
    () => [
      {
        q: "What is 360 Estate Suite?",
        a: "360 Estate Suite is an all-in-one AI-powered real estate operating system designed for agents and teams. It combines AI staging, 3D virtual tours, contract creation with PDFs, secure client share links, and client management so you can run more of your listing workflow in one place.",
      },
      {
        q: "Who is 360 Estate Suite for?",
        a: "360 Estate Suite is built for real estate professionals: individual agents, growing teams, and agencies managing multiple listings at once. If you want faster listing preparation, a cleaner client experience, and fewer tools to juggle, it’s a good fit.",
      },
      {
        q: "How does AI staging work?",
        a: "AI staging helps you turn a property photo into a more presentation-ready version. You can create new staging looks without scheduling physical staging or multiple photo shoots. It’s ideal for accelerating time-to-list, improving first impressions, and helping buyers visualize potential.",
      },
      {
        q: "How do 3D virtual tours work?",
        a: "You can create a 3D tour from an iPhone walkthrough video or upload a ready 3D tour file. The result can be saved to the property and shared with clients as part of the listing assets.",
      },
      {
        q: "What are coins and what are they used for?",
        a: "Coins are your usage balance for AI features. You spend coins when you generate AI outputs like staging images and virtual tour scenes. This keeps pricing simple: you only use coins when you actually create AI assets.",
      },
      {
        q: "Do coins expire?",
        a: "Unused coins carry over while your subscription is active. This means you can build a buffer for busy months and use coins when you need them, without feeling pressure to spend immediately.",
      },
      {
        q: "What’s the difference between Starter, Pro, and Agency?",
        a: "The plans are designed around how many coins you want included each month and the scale of your workflow. Starter is for individual agents, Pro is for growing teams, and Agency is for agencies and multi-property teams that need a larger monthly coin balance.",
      },
      {
        q: "Can I upgrade or downgrade my plan?",
        a: "Yes. You can choose a plan that matches your current workflow and change it as your business grows. Many teams start with Starter or Pro and move to Agency once they manage more listings and need more coins available each month.",
      },
      {
        q: "Is payment secure?",
        a: "Yes. Payments are secured by Stripe, a widely used payment platform trusted by many online businesses. 360 Estate Suite does not ask you to email card details or handle payment in an unsafe way.",
      },
      {
        q: "How does secure client sharing work?",
        a: "You can create a share link to deliver listing assets like photos, tours, and documents. Share links are designed to keep client delivery simple while maintaining controlled access compared to sending files in email threads.",
      },
      {
        q: "Can I generate contracts and PDFs?",
        a: "Yes. 360 Estate Suite can generate contracts and produce PDFs so you can quickly prepare documents for clients. This helps reduce manual document assembly and keeps your workflow consistent.",
      },
      {
        q: "Is 360 Estate Suite GDPR-compliant?",
        a: "360 Estate Suite is designed with a privacy-first architecture. You remain in control of what you share with clients through secure links. If you have specific compliance requirements for your region or brokerage, contact us so we can help you align your workflow.",
      },
      {
        q: "Is there a free trial?",
        a: "If a free trial is available, it will be shown on the Pricing page or during Sign Up. Many teams prefer starting with Starter so they can validate the workflow quickly while keeping costs predictable.",
      },
      {
        q: "Can I cancel anytime?",
        a: "Yes. You can cancel when your needs change. Your access and coin carry-over depend on whether your subscription remains active.",
      },
      {
        q: "What happens if I run out of coins?",
        a: "If you run out of coins, you won’t be able to generate new AI assets until you add more. Your existing properties, clients, contracts, and share links remain accessible, and you can continue running the parts of the workflow that don’t require new AI generation.",
      },
      {
        q: "Can I use 360 Estate Suite for rentals and sales?",
        a: "Yes. The system supports real estate workflows for both for-sale and for-rent listings. You can organize properties, manage clients, generate assets, and share deliverables regardless of the listing type.",
      },
      {
        q: "Will my team be able to see the same properties and assets?",
        a: "Teams can coordinate by using a shared workflow and consistent asset generation. If you need a multi-user, agency-level setup, the Agency plan is designed for larger teams managing many listings.",
      },
    ],
    [],
  );

  const faqJsonLd = useMemo(() => {
    return {
      "@context": "https://schema.org",
      "@type": "FAQPage",
      mainEntity: items.map((item) => {
        return {
          "@type": "Question",
          name: item.q,
          acceptedAnswer: {
            "@type": "Answer",
            text: item.a,
          },
        };
      }),
    };
  }, [items]);

  const faqJsonLdText = useMemo(() => {
    return JSON.stringify(faqJsonLd);
  }, [faqJsonLd]);

  const onBottomSignUpClick = useMemo(() => {
    return () => trackSignUpClick("faq_bottom_cta");
  }, [trackSignUpClick]);

  const [openIndex, setOpenIndex] = useState(0);

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-[#1E1E1E]">
      {/* JSON-LD structured data for SEO (matches visible FAQ items exactly) */}
      <script type="application/ld+json">{faqJsonLdText}</script>

      <MarketingHeader />

      <main className="max-w-5xl mx-auto px-4 sm:px-8 py-10 sm:py-14">
        <header className="max-w-3xl">
          <h1 className="text-3xl sm:text-4xl font-bold text-gray-900 dark:text-gray-100 font-jetbrains-mono">
            Frequently asked questions
          </h1>
          <p className="mt-3 text-gray-600 dark:text-gray-300 font-jetbrains-mono">
            Answers to common questions about AI staging, virtual tours, coins,
            contracts, secure sharing, and pricing.
          </p>
        </header>

        <section className="mt-8">
          <h2 className="text-xl font-bold text-gray-900 dark:text-gray-100 font-jetbrains-mono">
            FAQ
          </h2>

          <div className="mt-4 rounded-xl border border-gray-200 dark:border-gray-700 bg-white dark:bg-[#262626] overflow-hidden">
            {items.map((item, idx) => {
              const isOpen = openIndex === idx;
              const btnText = item.q;

              return (
                <div
                  key={item.q}
                  className={
                    idx === 0
                      ? ""
                      : "border-t border-gray-200 dark:border-gray-700"
                  }
                >
                  <button
                    className="w-full text-left px-5 py-4 flex items-center justify-between gap-4"
                    onClick={() =>
                      setOpenIndex((prev) => (prev === idx ? -1 : idx))
                    }
                    aria-expanded={isOpen}
                  >
                    <span className="text-sm sm:text-base font-semibold text-gray-900 dark:text-gray-100 font-jetbrains-mono">
                      {btnText}
                    </span>
                    <ChevronDown
                      size={18}
                      className={
                        isOpen
                          ? "text-gray-500 dark:text-gray-300 rotate-180 transition-transform"
                          : "text-gray-500 dark:text-gray-300 transition-transform"
                      }
                    />
                  </button>

                  {isOpen ? (
                    <div className="px-5 pb-5">
                      <p className="text-sm text-gray-600 dark:text-gray-300 font-jetbrains-mono">
                        {item.a}
                      </p>
                    </div>
                  ) : null}
                </div>
              );
            })}
          </div>
        </section>

        <section className="mt-10 rounded-2xl bg-gradient-to-br from-[var(--brand)] to-[var(--brandDark)] p-8 sm:p-10">
          <h2 className="text-2xl sm:text-3xl font-bold text-white font-jetbrains-mono">
            Still have questions?
          </h2>
          <p className="mt-2 text-white/90 font-jetbrains-mono">
            Contact us and we’ll help you choose the best plan for your
            workflow.
          </p>
          <div className="mt-6 flex flex-col sm:flex-row gap-3">
            <a
              href="/contact"
              className="inline-flex items-center justify-center px-5 py-3 rounded-lg bg-white text-[var(--brandDark)] hover:bg-white/90 font-medium font-jetbrains-mono"
            >
              Contact
            </a>
            <a
              href="/signup"
              className="inline-flex items-center justify-center px-5 py-3 rounded-lg border border-white/40 text-white hover:bg-white/10 font-medium font-jetbrains-mono"
              onClick={onBottomSignUpClick}
            >
              Sign Up
            </a>
          </div>
        </section>
      </main>
    </div>
  );
}
