import { useEffect } from "react";
import MarketingHeader from "@/components/MarketingHeader";

export default function CookiesPage() {
  useEffect(() => {
    if (typeof window !== "undefined") {
      document.title = "Cookie Policy | 360 Estate Suite";
    }
  }, []);

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-[#1E1E1E]">
      <MarketingHeader />

      <main className="max-w-4xl mx-auto px-4 sm:px-8 py-10 sm:py-14">
        <h1 className="text-3xl sm:text-4xl font-bold text-gray-900 dark:text-gray-100 font-jetbrains-mono">
          Cookie Policy
        </h1>
        <p className="mt-3 text-gray-600 dark:text-gray-300 font-jetbrains-mono">
          This Cookie Policy explains how 360 Estate Suite uses cookies and
          similar technologies to operate the platform.
        </p>

        <section className="mt-8">
          <h2 className="text-xl font-bold text-gray-900 dark:text-gray-100 font-jetbrains-mono">
            What cookies do
          </h2>
          <p className="mt-2 text-gray-600 dark:text-gray-300 font-jetbrains-mono">
            Cookies help keep you signed in, protect your account, and improve
            reliability. They may also help us understand how the site is used
            so we can improve the experience.
          </p>
        </section>

        <section className="mt-8">
          <h2 className="text-xl font-bold text-gray-900 dark:text-gray-100 font-jetbrains-mono">
            Managing cookies
          </h2>
          <p className="mt-2 text-gray-600 dark:text-gray-300 font-jetbrains-mono">
            You can control cookies through your browser settings. Disabling
            certain cookies may affect sign-in and other core features.
          </p>
        </section>

        <p className="mt-10 text-xs text-gray-500 dark:text-gray-400 font-jetbrains-mono">
          Last updated: December 24, 2025.
        </p>
      </main>
    </div>
  );
}
