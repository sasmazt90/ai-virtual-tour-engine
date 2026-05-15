import { useState, useEffect } from "react";

export function useFake360HelperHint() {
  const [showHelperHint, setShowHelperHint] = useState(false);

  useEffect(() => {
    if (typeof window === "undefined") return;

    try {
      const key = "fake360_helper_hint_shown_v1";
      const already = window.localStorage.getItem(key);
      if (already) return;

      setShowHelperHint(true);
      window.localStorage.setItem(key, "1");

      const t = window.setTimeout(() => setShowHelperHint(false), 2000);
      return () => window.clearTimeout(t);
    } catch {
      // If localStorage fails, still show once per mount.
      setShowHelperHint(true);
      const t = setTimeout(() => setShowHelperHint(false), 2000);
      return () => clearTimeout(t);
    }
  }, []);

  return showHelperHint;
}
