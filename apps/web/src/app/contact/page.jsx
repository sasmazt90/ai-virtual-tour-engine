import { useCallback, useEffect, useState } from "react";
import MarketingHeader from "@/components/MarketingHeader";

export default function ContactPage() {
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [message, setMessage] = useState("");
  const [error, setError] = useState(null);
  const [sent, setSent] = useState(false);

  useEffect(() => {
    if (typeof window !== "undefined") {
      document.title = "Contact | 360 Estate Suite";
    }
  }, []);

  const onSubmit = useCallback(
    (e) => {
      e.preventDefault();
      setError(null);
      setSent(false);

      const safeName = name.trim();
      const safeEmail = email.trim();
      const safeMsg = message.trim();

      if (!safeMsg) {
        setError("Please enter a short message.");
        return;
      }

      if (typeof window === "undefined") {
        return;
      }

      const to = "contact@360estatesuite.com";
      const subject = encodeURIComponent("360 Estate Suite — Contact request");
      const body = encodeURIComponent(
        `Name: ${safeName || "(not provided)"}\nEmail: ${safeEmail || "(not provided)"}\n\nMessage:\n${safeMsg}`,
      );

      window.location.href = `mailto:${to}?subject=${subject}&body=${body}`;
      setSent(true);
    },
    [email, message, name],
  );

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-[#1E1E1E]">
      <MarketingHeader />

      <main className="max-w-5xl mx-auto px-4 sm:px-8 py-10 sm:py-14">
        <h1 className="text-3xl sm:text-4xl font-bold text-gray-900 dark:text-gray-100 font-jetbrains-mono">
          Contact
        </h1>
        <p className="mt-3 text-gray-600 dark:text-gray-300 font-jetbrains-mono max-w-2xl">
          Questions about pricing, coins, AI staging, virtual tours, contracts,
          or secure sharing? Send us a note.
        </p>

        <div className="mt-8 grid grid-cols-1 lg:grid-cols-2 gap-8">
          <form
            onSubmit={onSubmit}
            className="rounded-xl border border-gray-200 dark:border-gray-700 bg-white dark:bg-[#262626] p-6"
          >
            <div className="space-y-4">
              <div>
                <label className="block text-sm text-gray-700 dark:text-gray-300 font-jetbrains-mono">
                  Name (optional)
                </label>
                <input
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  className="mt-1 w-full px-4 py-3 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-600 rounded-lg text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-[var(--brand)] font-jetbrains-mono"
                  placeholder="Your name"
                />
              </div>

              <div>
                <label className="block text-sm text-gray-700 dark:text-gray-300 font-jetbrains-mono">
                  Email (optional)
                </label>
                <input
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  type="email"
                  className="mt-1 w-full px-4 py-3 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-600 rounded-lg text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-[var(--brand)] font-jetbrains-mono"
                  placeholder="you@company.com"
                />
              </div>

              <div>
                <label className="block text-sm text-gray-700 dark:text-gray-300 font-jetbrains-mono">
                  Message
                </label>
                <textarea
                  value={message}
                  onChange={(e) => setMessage(e.target.value)}
                  rows={6}
                  className="mt-1 w-full px-4 py-3 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-600 rounded-lg text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-[var(--brand)] font-jetbrains-mono"
                  placeholder="Tell us what you’re trying to do."
                />
              </div>

              {error ? (
                <div className="text-sm text-red-600 dark:text-red-400 font-jetbrains-mono">
                  {error}
                </div>
              ) : null}

              {sent ? (
                <div className="text-sm text-gray-600 dark:text-gray-300 font-jetbrains-mono">
                  Your email app should open in a new window.
                </div>
              ) : null}

              <button
                type="submit"
                className="inline-flex items-center justify-center px-5 py-3 rounded-lg bg-[var(--brand)] hover:bg-[var(--brandHover)] text-white font-medium font-jetbrains-mono"
              >
                Send
              </button>
            </div>
          </form>

          <div className="rounded-xl border border-gray-200 dark:border-gray-700 bg-white dark:bg-[#262626] p-6">
            <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100 font-jetbrains-mono">
              Contact information
            </h2>
            <div className="mt-3 text-sm text-gray-600 dark:text-gray-300 font-jetbrains-mono">
              Email: contact@360estatesuite.com
            </div>
            <div className="mt-1 text-sm text-gray-600 dark:text-gray-300 font-jetbrains-mono">
              We aim to respond within 1–2 business days.
            </div>

            <div className="mt-6">
              <h3 className="text-sm font-semibold text-gray-900 dark:text-gray-100 font-jetbrains-mono">
                Looking for pricing?
              </h3>
              <p className="mt-2 text-sm text-gray-600 dark:text-gray-300 font-jetbrains-mono">
                Visit the Pricing page to compare Starter, Pro, and Agency.
              </p>
              <a
                href="/pricing"
                className="mt-3 inline-flex items-center justify-center px-4 py-2 rounded-lg border border-gray-200 dark:border-gray-700 text-gray-900 dark:text-gray-100 hover:bg-gray-50 dark:hover:bg-gray-800 font-jetbrains-mono"
              >
                View Pricing
              </a>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
