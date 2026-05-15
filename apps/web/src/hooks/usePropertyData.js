import { useMemo } from "react";
import { useQuery } from "@tanstack/react-query";

export function usePropertyData(userId, selectedPropertyId) {
  // ---
  // Properties list (used by selector / tools pages)
  // ---
  const {
    data: properties,
    isLoading: propertiesLoading,
    error: propertiesError,
    refetch: refetchProperties,
  } = useQuery({
    queryKey: ["properties", userId || "anon"],
    queryFn: async () => {
      const res = await fetch("/api/properties");
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body?.error || "Could not load properties");
      }
      return res.json();
    },
    enabled: !!userId,
    refetchOnWindowFocus: false,
  });

  // ---
  // Single property detail (used by /properties/[id] and /properties/[id]/edit)
  // ---
  const {
    data: propertyDetail,
    isLoading: propertyLoading,
    error: propertyError,
    refetch: refetchProperty,
  } = useQuery({
    queryKey: ["property", userId || "anon", selectedPropertyId || ""],
    queryFn: async () => {
      const res = await fetch(
        `/api/properties/${encodeURIComponent(selectedPropertyId)}`,
      );
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body?.error || "Could not load this property");
      }
      return res.json();
    },
    enabled: !!userId && !!selectedPropertyId,
    refetchOnWindowFocus: false,
  });

  const selectedPropertyFromList = useMemo(() => {
    const list = Array.isArray(properties) ? properties : [];
    return (
      list.find((p) => String(p?.id) === String(selectedPropertyId)) || null
    );
  }, [properties, selectedPropertyId]);

  // Prefer the full detail payload when we have it.
  const selectedProperty = propertyDetail || selectedPropertyFromList;

  const selectedPropertyPhotoIds = useMemo(() => {
    const photos = Array.isArray(selectedProperty?.photos)
      ? selectedProperty.photos
      : [];
    return photos.map((p) => p?.id).filter(Boolean);
  }, [selectedProperty?.photos]);

  const firstPhotoId = selectedPropertyPhotoIds.length
    ? selectedPropertyPhotoIds[0]
    : null;

  const { data: customAssets } = useQuery({
    queryKey: ["custom-assets", userId || "anon", selectedPropertyId],
    queryFn: async () => {
      const res = await fetch(
        `/api/properties/${encodeURIComponent(selectedPropertyId)}/custom-assets`,
      );
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body?.error || "Could not load custom assets");
      }
      return res.json();
    },
    enabled: !!userId && !!selectedPropertyId,
    refetchOnWindowFocus: false,
  });

  const customAssetIds = useMemo(() => {
    const list = Array.isArray(customAssets) ? customAssets : [];
    return list.map((a) => a?.id).filter(Boolean);
  }, [customAssets]);

  const hasCustomAssets = customAssetIds.length > 0;

  return {
    // list
    properties,
    propertiesLoading,
    propertiesError,
    refetchProperties,

    // selected property helpers (back-compat)
    selectedProperty,
    selectedPropertyPhotoIds,
    firstPhotoId,

    // detail page API expected names
    property: selectedProperty,
    propertyLoading,
    propertyError,
    refetchProperty,

    // custom assets
    customAssets,
    customAssetIds,
    hasCustomAssets,
  };
}
