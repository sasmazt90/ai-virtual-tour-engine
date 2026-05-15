import { useMemo } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  buildOrigin,
  isExpired,
  getLastViewedAt,
} from "@/utils/shareLinkHelpers";

export function useShareLink({ propertyId, shareClientId }) {
  const queryClient = useQueryClient();

  const { data: activeLinks = [], isLoading: activeLinkLoading } = useQuery({
    queryKey: ["share-links", "active", propertyId, shareClientId],
    queryFn: async () => {
      const url = `/api/share-links?propertyId=${encodeURIComponent(
        propertyId,
      )}&clientId=${encodeURIComponent(shareClientId)}&activeOnly=1&limit=1`;
      const res = await fetch(url);
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body?.error || "Failed to load share link");
      }
      return res.json();
    },
    enabled: !!propertyId && !!shareClientId,
  });

  const activeLink = Array.isArray(activeLinks) ? activeLinks[0] : null;

  const activeUrl = useMemo(() => {
    if (!activeLink?.slug) return null;
    const origin = buildOrigin();
    return origin
      ? `${origin}/share/${activeLink.slug}`
      : `/share/${activeLink.slug}`;
  }, [activeLink?.slug]);

  const supportSafeUrl = useMemo(() => {
    if (!activeUrl) return null;
    try {
      const origin = buildOrigin();
      const u = activeUrl.startsWith("http")
        ? new URL(activeUrl)
        : origin
          ? new URL(activeUrl, origin)
          : null;
      if (!u) return null;
      u.searchParams.set("view", "readonly");
      return u.toString();
    } catch {
      // As a last resort, return a simple hint URL (may be relative)
      if (activeUrl.includes("?")) return `${activeUrl}&view=readonly`;
      return `${activeUrl}?view=readonly`;
    }
  }, [activeUrl]);

  const activeExpired = activeLink?.expires_at
    ? isExpired(activeLink.expires_at)
    : false;

  const lastViewedIso = getLastViewedAt(activeLink?.meta);
  const lastViewedText = lastViewedIso
    ? new Date(lastViewedIso).toLocaleString()
    : "Never";

  const hasActiveLink = !!(activeLink && activeUrl);

  const disableMutation = useMutation({
    mutationFn: async ({ id }) => {
      const res = await fetch(`/api/share-links/${id}/disable`, {
        method: "POST",
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body?.error || "Failed to disable link");
      }
      return res.json();
    },
    onSuccess: async () => {
      await queryClient.invalidateQueries({
        queryKey: ["share-links", "active", propertyId, shareClientId],
      });
      await queryClient.invalidateQueries({ queryKey: ["share-links"] });
    },
  });

  const extendMutation = useMutation({
    mutationFn: async ({ id, extendDays }) => {
      const res = await fetch(`/api/share-links/${id}/extend`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ extendDays }),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body?.error || "Failed to extend expiry");
      }
      return res.json();
    },
    onSuccess: async () => {
      await queryClient.invalidateQueries({
        queryKey: ["share-links", "active", propertyId, shareClientId],
      });
      await queryClient.invalidateQueries({ queryKey: ["share-links"] });
    },
  });

  return {
    activeLink,
    activeUrl,
    supportSafeUrl,
    activeExpired,
    lastViewedText,
    hasActiveLink,
    activeLinkLoading,
    disableMutation,
    extendMutation,
  };
}
