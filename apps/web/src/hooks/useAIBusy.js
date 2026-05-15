import { useQuery } from "@tanstack/react-query";
import { useEffect, useRef, useState } from "react";

export function useAIBusy(userId, options = {}) {
  const enabled = options.enabled ?? !!userId;
  const refetchInterval = options.refetchInterval ?? 10000;

  const [isVisible, setIsVisible] = useState(() => {
    if (typeof document === "undefined") {
      return true;
    }
    return document.visibilityState === "visible";
  });
  const prevVisibleRef = useRef(isVisible);

  useEffect(() => {
    if (typeof document === "undefined") {
      return;
    }

    const update = () => {
      setIsVisible(document.visibilityState === "visible");
    };

    document.addEventListener("visibilitychange", update);
    return () => document.removeEventListener("visibilitychange", update);
  }, []);

  const pollingEnabled = !!userId && enabled && isVisible;
  const activeRefetchInterval = pollingEnabled ? refetchInterval : false;

  const query = useQuery({
    queryKey: ["ai-busy", userId],
    enabled: pollingEnabled,
    refetchInterval: activeRefetchInterval,
    queryFn: async () => {
      const res = await fetch("/api/ai/busy");
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body?.error || "Failed to fetch AI busy status");
      }
      const data = await res.json();
      return {
        busy: !!data?.busy,
        queued: Number(data?.queued || 0),
        running: Number(data?.running || 0),
        partial: !!data?.partial,
      };
    },
  });

  // Resume immediately when the tab becomes visible again (instead of waiting up
  // to the next 10s interval).
  useEffect(() => {
    const prev = prevVisibleRef.current;
    prevVisibleRef.current = isVisible;

    if (!prev && isVisible && pollingEnabled) {
      // avoid refetching during SSR
      query.refetch().catch(() => {
        // no-op: busy polling should never hard-fail the UI
      });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isVisible, pollingEnabled]);

  return query;
}
