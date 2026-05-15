import { useEffect } from "react";
import MarketingHeader from "@/components/MarketingHeader";

export default function PrivacyPage() {
  useEffect(() => {
    if (typeof window !== "undefined") {
      document.title = "Privacy Policy | 360 Estate Suite";
    }
  }, []);

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-[#1E1E1E]">
      <MarketingHeader />

      <main className="max-w-4xl mx-auto px-4 sm:px-8 py-10 sm:py-14">
        <h1 className="text-3xl sm:text-4xl font-bold text-gray-900 dark:text-gray-100 font-jetbrains-mono">
          Privacy Policy
        </h1>
        <p className="mt-3 text-gray-600 dark:text-gray-300 font-jetbrains-mono">
          This Privacy Policy explains how 360 Estate Suite collects, uses, and
          protects information when you use the platform.
        </p>

        <section className="mt-8">
          <h2 className="text-xl font-bold text-gray-900 dark:text-gray-100 font-jetbrains-mono">
            What we collect
          </h2>
          <ul className="mt-3 space-y-2 text-gray-600 dark:text-gray-300 font-jetbrains-mono">
            <li>Account details such as email and profile information.</li>
            <li>Property and client information you choose to store.</li>
            <li>Generated assets such as staging images and tour data.</li>
            <li>Usage and billing events required to operate the service.</li>
          </ul>
        </section>

        <section className="mt-8">
          <h2 className="text-xl font-bold text-gray-900 dark:text-gray-100 font-jetbrains-mono">
            How we use information
          </h2>
          <p className="mt-2 text-gray-600 dark:text-gray-300 font-jetbrains-mono">
            We use your information to provide the product features you request,
            to keep your account secure, to process payments through Stripe, and
            to maintain reliable operation of the platform.
          </p>
        </section>

        <section className="mt-8">
          <h2 className="text-xl font-bold text-gray-900 dark:text-gray-100 font-jetbrains-mono">
            Sharing
          </h2>
          <p className="mt-2 text-gray-600 dark:text-gray-300 font-jetbrains-mono">
            We do not sell your personal data. You control what you share with
            clients using share links. We may share information with service
            providers when needed to operate the platform (for example, payment
            processing).
          </p>
        </section>

        <section className="mt-8">
          <h2 className="text-xl font-bold text-gray-900 dark:text-gray-100 font-jetbrains-mono">
            Data security
          </h2>
          <p className="mt-2 text-gray-600 dark:text-gray-300 font-jetbrains-mono">
            We use standard security practices to protect data. However, no
            method of transmission over the internet is 100% secure. Always use
            strong passwords and protect client information.
          </p>
        </section>

        <section className="mt-8">
          <h2 className="text-xl font-bold text-gray-900 dark:text-gray-100 font-jetbrains-mono">
            Contact
          </h2>
          <p className="mt-2 text-gray-600 dark:text-gray-300 font-jetbrains-mono">
            For privacy questions, contact contact@360estatesuite.com.
          </p>
        </section>

        <p className="mt-10 text-xs text-gray-500 dark:text-gray-400 font-jetbrains-mono">
          Last updated: December 24, 2025.
        </p>
      </main>
    </div>
  );
}
