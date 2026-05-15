import { useMemo } from "react";

export default function MarketingFooter() {
  const logoUrl =
    "https://ucarecdn.com/f3c2cf7c-ce51-4a0c-af7b-24d29246221e/-/format/auto/";

  // Footer must stay dark in BOTH light and dark theme (like the header).
  const linkClass = useMemo(() => {
    return "block text-sm text-gray-300/90 hover:text-white transition-colors font-jetbrains-mono";
  }, []);

  return (
    <footer className="border-t border-white/10 bg-[#07080A]">
      <div className="max-w-7xl mx-auto px-4 sm:px-8 py-12">
        <div className="flex flex-col lg:flex-row lg:items-start lg:justify-between gap-10">
          {/* Left: logo + fixed 4-line brand copy (text to the right of the logo) */}
          <div className="max-w-md">
            <a href="/" className="inline-block">
              <div className="flex items-start gap-4">
                <img
                  src={logoUrl}
                  alt="360 Estate Suite logo"
                  className="h-[88px] sm:h-[96px] w-auto rounded-2xl border border-white/10 bg-black/20 object-contain"
                  draggable={false}
                />

                <div className="pt-[2px]">
                  <div className="text-base font-semibold text-gray-50 font-jetbrains-mono tracking-tight leading-[1.25]">
                    360 Estate Suite
                  </div>

                  {/* Force exactly 4 lines total (title + 3 lines of description) */}
                  <div className="mt-2 text-sm text-gray-300/90 font-jetbrains-mono leading-[1.35]">
                    <span className="block">
                      All-in-one real estate operating suite —
                    </span>
                    <span className="block">
                      AI staging, 360 tours, contracts,
                    </span>
                    <span className="block">
                      secure sharing, clients, and credits.
                    </span>
                  </div>
                </div>
              </div>
            </a>
          </div>

          {/* Right: link columns */}
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-10">
            <div>
              <div className="text-xs uppercase tracking-wider text-gray-400 font-jetbrains-mono">
                Platform
              </div>
              <div className="mt-4 space-y-3">
                <a href="/pricing" className={linkClass}>
                  Pricing
                </a>
                <a href="/faq" className={linkClass}>
                  FAQ
                </a>
                <a href="/about" className={linkClass}>
                  About
                </a>
                <a href="/contact" className={linkClass}>
                  Contact
                </a>
              </div>
            </div>

            <div>
              <div className="text-xs uppercase tracking-wider text-gray-400 font-jetbrains-mono">
                Legal
              </div>
              <div className="mt-4 space-y-3">
                <a href="/terms" className={linkClass}>
                  Terms of Service
                </a>
                <a href="/privacy" className={linkClass}>
                  Privacy Policy
                </a>
                <a href="/cookies" className={linkClass}>
                  Cookie Policy
                </a>
                <a href="/imprint" className={linkClass}>
                  Imprint
                </a>
              </div>
            </div>

            <div className="col-span-2 sm:col-span-1">
              <div className="text-xs uppercase tracking-wider text-gray-400 font-jetbrains-mono">
                Account
              </div>
              <div className="mt-4 space-y-3">
                <a href="/account/signup" className={linkClass}>
                  Sign Up
                </a>
                <a href="/account/signin" className={linkClass}>
                  Sign In
                </a>
                <a href="/properties" className={linkClass}>
                  App
                </a>
              </div>
            </div>
          </div>
        </div>

        <div className="mt-10 pt-6 border-t border-white/10 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
          <div className="text-xs text-gray-400 font-jetbrains-mono">
            © 2025 360 Estate Suite. All rights reserved.
          </div>
          <div className="text-xs text-gray-400 font-jetbrains-mono">
            Stripe-secured payments • Privacy-first architecture
          </div>
        </div>
      </div>
    </footer>
  );
}
