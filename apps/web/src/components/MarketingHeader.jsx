import { Menu, Moon, Sun, X } from "lucide-react";
import { useMemo, useState } from "react";
import useMarketingAnalytics from "@/hooks/useMarketingAnalytics";
import { useTheme } from "@/components/ThemeProvider";

export default function MarketingHeader() {
  const [open, setOpen] = useState(false);
  const { theme, toggleTheme } = useTheme();
  const { trackPricingClick, trackSignUpClick } = useMarketingAnalytics();

  const logoUrl =
    "https://ucarecdn.com/f3c2cf7c-ce51-4a0c-af7b-24d29246221e/-/format/auto/";

  // Header bar should stay dark in BOTH light and dark theme.
  const linkClass =
    "text-sm text-gray-200 hover:text-white transition-colors font-jetbrains-mono";

  const subtleLinkClass = useMemo(() => {
    return `${linkClass} opacity-90 hover:opacity-100`;
  }, [linkClass]);

  const onPricingClick = useMemo(() => {
    return () => trackPricingClick("header");
  }, [trackPricingClick]);

  const onMobilePricingClick = useMemo(() => {
    return () => {
      trackPricingClick("header_mobile");
      setOpen(false);
    };
  }, [trackPricingClick]);

  const onSignUpClick = useMemo(() => {
    return () => trackSignUpClick("header");
  }, [trackSignUpClick]);

  const onMobileSignUpClick = useMemo(() => {
    return () => {
      trackSignUpClick("header_mobile");
      setOpen(false);
    };
  }, [trackSignUpClick]);

  return (
    <header className="sticky top-0 z-50 border-b border-white/10 bg-[#07080A]">
      <div className="max-w-7xl mx-auto px-4 sm:px-8 h-16 flex items-center justify-between">
        <a href="/" className="flex items-center gap-3">
          <img
            src={logoUrl}
            alt="360 Estate Suite logo"
            className="w-9 h-9 rounded-lg border border-white/10 bg-black/20 object-contain"
            draggable={false}
          />
          <div className="text-base sm:text-lg font-semibold text-gray-50 font-jetbrains-mono tracking-tight">
            360 Estate Suite
          </div>
        </a>

        <nav className="hidden md:flex items-center gap-7">
          <a
            href="/pricing"
            className={subtleLinkClass}
            onClick={onPricingClick}
          >
            Pricing
          </a>
          <a href="/faq" className={subtleLinkClass}>
            FAQ
          </a>
          <a href="/contact" className={subtleLinkClass}>
            Contact
          </a>
        </nav>

        <div className="hidden md:flex items-center gap-3">
          <button
            type="button"
            onClick={toggleTheme}
            className="inline-flex items-center justify-center rounded-full border border-white/10 bg-white/5 hover:bg-white/10 px-3 py-2 transition-colors"
            aria-label={
              theme === "dark"
                ? "Switch to light theme"
                : "Switch to dark theme"
            }
            title={theme === "dark" ? "Light theme" : "Dark theme"}
          >
            {theme === "dark" ? (
              <Sun size={16} className="text-gray-100" />
            ) : (
              <Moon size={16} className="text-gray-100" />
            )}
          </button>

          <a href="/account/signin" className={linkClass}>
            Sign In
          </a>
          <a
            href="/account/signup"
            className="inline-flex items-center justify-center px-4 py-2 rounded-full bg-[var(--brand90)] hover:bg-[var(--brand)] text-white text-sm font-medium font-jetbrains-mono transition-colors"
            onClick={onSignUpClick}
          >
            Get Started
          </a>
        </div>

        <div className="md:hidden flex items-center gap-2">
          <button
            type="button"
            onClick={toggleTheme}
            className="p-2 rounded-lg border border-white/10 bg-white/5 hover:bg-white/10 transition-colors"
            aria-label={
              theme === "dark"
                ? "Switch to light theme"
                : "Switch to dark theme"
            }
          >
            {theme === "dark" ? (
              <Sun size={18} className="text-gray-100" />
            ) : (
              <Moon size={18} className="text-gray-100" />
            )}
          </button>

          <button
            className="p-2 text-gray-200 hover:text-white transition-colors"
            onClick={() => setOpen((v) => !v)}
            aria-label={open ? "Close menu" : "Open menu"}
          >
            {open ? <X size={20} /> : <Menu size={20} />}
          </button>
        </div>
      </div>

      {open ? (
        <div className="md:hidden border-t border-white/10 bg-[#07080A]">
          <div className="max-w-7xl mx-auto px-4 sm:px-8 py-4 flex flex-col gap-3">
            <a
              href="/pricing"
              className={linkClass}
              onClick={onMobilePricingClick}
            >
              Pricing
            </a>
            <a href="/faq" className={linkClass} onClick={() => setOpen(false)}>
              FAQ
            </a>
            <a
              href="/contact"
              className={linkClass}
              onClick={() => setOpen(false)}
            >
              Contact
            </a>

            <div className="pt-3 flex items-center gap-4">
              <a
                href="/account/signin"
                className={linkClass}
                onClick={() => setOpen(false)}
              >
                Sign In
              </a>
              <a
                href="/account/signup"
                className="inline-flex items-center justify-center px-4 py-2 rounded-full bg-[var(--brand90)] hover:bg-[var(--brand)] text-white text-sm font-medium font-jetbrains-mono transition-colors"
                onClick={onMobileSignUpClick}
              >
                Get Started
              </a>
            </div>
          </div>
        </div>
      ) : null}
    </header>
  );
}
