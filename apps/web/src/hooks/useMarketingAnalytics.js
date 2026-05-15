import { useCallback } from "react";

export default function useMarketingAnalytics() {
  const track = useCallback((event, payload) => {
    if (typeof window === "undefined") {
      return;
    }

    const safePayload = payload && typeof payload === "object" ? payload : {};

    // Prep-only analytics hook (no external SDK).
    // This is intentionally a console event so future analytics can swap in a real transport.
    console.log("[marketing]", {
      event,
      ...safePayload,
      ts: new Date().toISOString(),
    });
  }, []);

  const trackSignUpClick = useCallback(
    (location) => {
      track("click_sign_up", { location });
    },
    [track],
  );

  const trackPricingClick = useCallback(
    (location) => {
      track("click_pricing", { location });
    },
    [track],
  );

  return {
    track,
    trackSignUpClick,
    trackPricingClick,
  };
}
