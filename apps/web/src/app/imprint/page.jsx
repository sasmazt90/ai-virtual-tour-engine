import { useEffect } from "react";
import MarketingHeader from "@/components/MarketingHeader";

export default function ImprintPage() {
  useEffect(() => {
    if (typeof window !== "undefined") {
      document.title = "Imprint | 360 Estate Suite";
    }
  }, []);

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-[#1E1E1E]">
      <MarketingHeader />

      <main className="max-w-4xl mx-auto px-4 sm:px-8 py-10 sm:py-14">
        <h1 className="text-3xl sm:text-4xl font-bold text-gray-900 dark:text-gray-100 font-jetbrains-mono">
          Imprint
        </h1>
        <p className="mt-3 text-gray-600 dark:text-gray-300 font-jetbrains-mono">
          This page provides provider and contact information for 360 Estate
          Suite.
        </p>

        <section className="mt-8 rounded-xl border border-gray-200 dark:border-gray-700 bg-white dark:bg-[#262626] p-6">
          <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100 font-jetbrains-mono">
            Provider
          </h2>
          <div className="mt-3 text-sm text-gray-600 dark:text-gray-300 font-jetbrains-mono">
            360 Estate Suite
          </div>
          <div className="mt-1 text-sm text-gray-600 dark:text-gray-300 font-jetbrains-mono">
            contact@360estatesuite.com
          </div>
          <div className="mt-1 text-sm text-gray-600 dark:text-gray-300 font-jetbrains-mono">
            (Company address details can be added here.)
          </div>
        </section>

        <section className="mt-8">
          <h2 className="text-xl font-bold text-gray-900 dark:text-gray-100 font-jetbrains-mono">
            Disclaimer
          </h2>
          <p className="mt-2 text-gray-600 dark:text-gray-300 font-jetbrains-mono">
            Despite careful control, we assume no liability for external links.
            The content of linked pages is the responsibility of their
            operators.
          </p>
        </section>

        <p className="mt-10 text-xs text-gray-500 dark:text-gray-400 font-jetbrains-mono">
          Last updated: December 24, 2025.
        </p>
      </main>
    </div>
  );
}
