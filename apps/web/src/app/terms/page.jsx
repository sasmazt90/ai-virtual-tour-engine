import { useEffect } from "react";
import MarketingHeader from "@/components/MarketingHeader";

export default function TermsPage() {
  useEffect(() => {
    if (typeof window !== "undefined") {
      document.title = "Terms of Service | 360 Estate Suite";
    }
  }, []);

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-[#1E1E1E]">
      <MarketingHeader />

      <main className="max-w-4xl mx-auto px-4 sm:px-8 py-10 sm:py-14">
        <h1 className="text-3xl sm:text-4xl font-bold text-gray-900 dark:text-gray-100 font-jetbrains-mono">
          Terms of Service
        </h1>
        <p className="mt-3 text-gray-600 dark:text-gray-300 font-jetbrains-mono">
          These Terms of Service govern your access to and use of 360 Estate
          Suite. By using the service, you agree to these terms.
        </p>

        <section className="mt-8">
          <h2 className="text-xl font-bold text-gray-900 dark:text-gray-100 font-jetbrains-mono">
            Use of the service
          </h2>
          <p className="mt-2 text-gray-600 dark:text-gray-300 font-jetbrains-mono">
            You may use 360 Estate Suite to manage listings, clients, and
            documents, and to generate AI outputs such as staging images and
            virtual tours. You agree to use the service in a lawful manner and
            not to misuse it.
          </p>
        </section>

        <section className="mt-8">
          <h2 className="text-xl font-bold text-gray-900 dark:text-gray-100 font-jetbrains-mono">
            Payments and coins
          </h2>
          <p className="mt-2 text-gray-600 dark:text-gray-300 font-jetbrains-mono">
            Some features require payment and/or coins. Payments are processed
            by Stripe. Coins are used to generate AI outputs, and unused coins
            carry over while your subscription is active.
          </p>
        </section>

        <section className="mt-8">
          <h2 className="text-xl font-bold text-gray-900 dark:text-gray-100 font-jetbrains-mono">
            Content and client data
          </h2>
          <p className="mt-2 text-gray-600 dark:text-gray-300 font-jetbrains-mono">
            You are responsible for the content you upload, generate, or share,
            including property photos, client details, and contract data. Use
            client share links responsibly and only share content you are
            authorized to share.
          </p>
        </section>

        <section className="mt-8">
          <h2 className="text-xl font-bold text-gray-900 dark:text-gray-100 font-jetbrains-mono">
            Disclaimer
          </h2>
          <p className="mt-2 text-gray-600 dark:text-gray-300 font-jetbrains-mono">
            The service is provided on an “as-is” basis. AI-generated outputs
            may require review and adjustment. 360 Estate Suite does not provide
            legal advice.
          </p>
        </section>

        <section className="mt-8">
          <h2 className="text-xl font-bold text-gray-900 dark:text-gray-100 font-jetbrains-mono">
            Contact
          </h2>
          <p className="mt-2 text-gray-600 dark:text-gray-300 font-jetbrains-mono">
            For questions about these terms, contact us at
            contact@360estatesuite.com.
          </p>
        </section>

        <p className="mt-10 text-xs text-gray-500 dark:text-gray-400 font-jetbrains-mono">
          Last updated: December 24, 2025.
        </p>
      </main>
    </div>
  );
}
