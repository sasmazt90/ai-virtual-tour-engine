import { useEffect } from "react";
import MarketingHeader from "@/components/MarketingHeader";

export default function AboutPage() {
  useEffect(() => {
    if (typeof window !== "undefined") {
      document.title = "About | 360 Estate Suite";
    }
  }, []);

  return (
    <div className="min-h-screen">
      <MarketingHeader />

      <main className="max-w-5xl mx-auto px-4 sm:px-8 py-10 sm:py-14">
        <h1 className="text-3xl sm:text-4xl font-bold text-gray-900 dark:text-gray-100 font-jetbrains-mono">
          About 360 Estate Suite
        </h1>
        <p className="mt-4 text-gray-600 dark:text-gray-300 font-jetbrains-mono">
          360 Estate Suite is built to help real estate professionals deliver
          higher-quality listings faster, while keeping client delivery simple
          and secure.
        </p>

        <section className="mt-10">
          <h2 className="text-xl sm:text-2xl font-bold text-gray-900 dark:text-gray-100 font-jetbrains-mono">
            Our mission
          </h2>
          <p className="mt-3 text-gray-600 dark:text-gray-300 font-jetbrains-mono">
            Make the day-to-day real estate workflow easier: fewer tools,
            clearer client delivery, and modern AI capabilities that fit how
            agents already work.
          </p>
        </section>

        <section className="mt-10">
          <h2 className="text-xl sm:text-2xl font-bold text-gray-900 dark:text-gray-100 font-jetbrains-mono">
            Product vision
          </h2>
          <p className="mt-3 text-gray-600 dark:text-gray-300 font-jetbrains-mono">
            Real estate is a coordination problem: properties, people,
            documents, and deadlines. 360 Estate Suite brings these together
            with AI staging, 3D tours, contracts with PDFs, secure share
            links, and a credit-based model so you pay for what you use.
          </p>
        </section>

        <section className="mt-10">
          <h2 className="text-xl sm:text-2xl font-bold text-gray-900 dark:text-gray-100 font-jetbrains-mono">
            Who it’s for
          </h2>
          <ul className="mt-4 space-y-2 text-gray-600 dark:text-gray-300 font-jetbrains-mono">
            <li>
              <span className="font-semibold text-gray-900 dark:text-gray-100">
                Individual agents
              </span>{" "}
              who want a faster way to prep listings.
            </li>
            <li>
              <span className="font-semibold text-gray-900 dark:text-gray-100">
                Teams
              </span>{" "}
              coordinating multiple listings and clients at once.
            </li>
            <li>
              <span className="font-semibold text-gray-900 dark:text-gray-100">
                Agencies
              </span>{" "}
              that need consistent, repeatable output across many properties.
            </li>
          </ul>
        </section>

        <section className="mt-12 rounded-2xl bg-gradient-to-br from-[var(--brand)] to-[var(--brandDark)] p-8 sm:p-10">
          <h2 className="text-2xl sm:text-3xl font-bold text-white font-jetbrains-mono">
            Try it on your next listing
          </h2>
          <p className="mt-2 text-white/90 font-jetbrains-mono">
            Create your account, add a property, and generate assets in minutes.
          </p>
          <div className="mt-6 flex flex-col sm:flex-row gap-3">
            <a
              href="/account/signup"
              className="inline-flex items-center justify-center px-5 py-3 rounded-lg bg-white text-[var(--brandDark)] hover:bg-white/90 font-medium font-jetbrains-mono"
            >
              Sign Up
            </a>
            <a
              href="/pricing"
              className="inline-flex items-center justify-center px-5 py-3 rounded-lg border border-white/40 text-white hover:bg-white/10 font-medium font-jetbrains-mono"
            >
              View Pricing
            </a>
          </div>
        </section>
      </main>
    </div>
  );
}
