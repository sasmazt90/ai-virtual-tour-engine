import { useEffect, useMemo } from "react";

export default function LoginRedirectPage() {
  const targetHref = useMemo(() => {
    if (typeof window === "undefined") {
      return "/account/signin";
    }

    try {
      const url = new URL(window.location.href);
      const callbackUrl = url.searchParams.get("callbackUrl");

      const target = new URL(window.location.origin + "/account/signin");
      if (callbackUrl) {
        target.searchParams.set("callbackUrl", callbackUrl);
      }

      return target.toString();
    } catch {
      return "/account/signin";
    }
  }, []);

  useEffect(() => {
    if (typeof window === "undefined") return;
    window.location.replace(targetHref);
  }, [targetHref]);

  return (
    <div className="min-h-[40vh] flex flex-col items-center justify-center gap-3 px-4">
      <p className="text-sm text-gray-600 dark:text-gray-300 font-jetbrains-mono">
        Redirecting to Login…
      </p>

      {/* Fallback if redirect is blocked */}
      <a
        href={targetHref}
        className="text-sm text-[var(--brandDark)] dark:text-[var(--brand)] hover:underline font-jetbrains-mono"
      >
        Continue
      </a>

      {/* Extra-hard fallback: run before hydration */}
      <script>{`
        (function () {
          try {
            var url = new URL(window.location.href);
            var callbackUrl = url.searchParams.get('callbackUrl');
            var target = new URL(window.location.origin + '/account/signin');
            if (callbackUrl) target.searchParams.set('callbackUrl', callbackUrl);
            window.location.replace(target.toString());
          } catch (e) {
            window.location.replace('/account/signin');
          }
        })();
      `}</script>
    </div>
  );
}
