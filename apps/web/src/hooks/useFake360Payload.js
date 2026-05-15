import { useMemo } from "react";

export function useFake360Payload(tourPayload) {
  const effectivePayload = useMemo(() => {
    const p = tourPayload;

    // If the payload already provides frames/points, use it as-is.
    if (p && (Array.isArray(p.frames) || Array.isArray(p.points))) {
      return p;
    }

    const hasScenes = p && Array.isArray(p.scenes) && p.scenes.length > 0;
    if (!hasScenes) {
      return p;
    }

    // Convert legacy `scenes` graphs into points-mode where:
    // - each node is a SINGLE image (intended to be an equirectangular panorama)
    // - navigation is via hotspots, rotation is handled by the viewer (yaw/pitch)
    const points = p.scenes.map((s, idx) => {
      const hs = Array.isArray(s?.hotspots) ? s.hotspots : [];
      const panoUrl =
        s?.imageUrl ||
        s?.url ||
        s?.src ||
        s?.download_url ||
        s?.storage_path ||
        s?.storagePath;

      return {
        pointId: s?.sceneId || s?.id || `scene_${idx}`,
        panoramaUrl: panoUrl,
        initialYaw: s?.initialYaw || 0,
        hotspots: hs.map((h) => ({
          ...h,
          toPointId: h?.toSceneId || h?.toPointId || h?.toId,
        })),
      };
    });

    const initialPointId =
      p.scenes?.[0]?.sceneId || p.scenes?.[0]?.id || points?.[0]?.pointId;

    return {
      type: "virtual_tour",
      points,
      initialPointId,
    };
  }, [tourPayload]);

  return effectivePayload;
}
