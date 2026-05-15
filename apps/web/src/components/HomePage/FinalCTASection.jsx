export function FinalCTASection({ onSignUpClick, onPricingClick }) {
  return (
    <section className="border-t border-black/10 dark:border-white/10">
      <div className="max-w-7xl mx-auto px-4 sm:px-8 py-14 sm:py-18">
        <div className="relative overflow-hidden rounded-3xl border border-black/10 dark:border-white/10 bg-white/70 dark:bg-[#07080A] p-8 sm:p-12">
          <div className="absolute inset-0 pointer-events-none cta-gradient" />
          {/* Keep animated grain inside CTA only in dark; in light it hurts readability. */}
          <div className="absolute inset-0 pointer-events-none hero-grain hidden dark:block opacity-60" />

          <div className="relative">
            <h2 className="text-2xl sm:text-4xl font-bold text-gray-950 dark:text-gray-50 font-jetbrains-mono">
              Ready to Scale Your Sales Performance?
            </h2>
            <p className="mt-3 text-sm sm:text-base text-gray-700 dark:text-gray-300 font-jetbrains-mono max-w-2xl">
              Ditch the multiple apps. One professional workflow to manage your
              properties from listing to contract.
            </p>

            <div className="mt-8 flex flex-col sm:flex-row gap-3">
              <a
                href="/signup"
                className="inline-flex items-center justify-center px-6 py-3 rounded-lg bg-[var(--brand90)] hover:bg-[var(--brand)] text-white font-medium font-jetbrains-mono transition-colors"
                onClick={onSignUpClick}
              >
                Sign Up
              </a>
              <a
                href="/pricing"
                className="inline-flex items-center justify-center px-6 py-3 rounded-lg border border-black/10 dark:border-white/10 bg-black/5 dark:bg-white/5 hover:bg-black/10 dark:hover:bg-white/10 text-gray-900 dark:text-gray-100 font-medium font-jetbrains-mono transition-colors"
                onClick={onPricingClick}
              >
                View Pricing
              </a>
            </div>

            <div className="mt-6 text-xs text-gray-600 dark:text-gray-400 font-jetbrains-mono">
              Secure links · Audit-friendly sharing · Credit-based usage
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
