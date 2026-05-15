import { useMemo } from "react";
import { fillMissingCircular, normalizeDeg } from "@/utils/fake360Utils";

export function useFake360Frames(
  activePoint,
  effectivePayload,
  isPointsMode,
  isFramesMode,
) {
  const frames = useMemo(() => {
    if (isPointsMode) {
      const local = Array.isArray(activePoint?.frames)
        ? activePoint.frames
        : [];

      // IMPORTANT:
      // If a point has too few frames (or just 1 image), we still want a full 360 rotation.
      // BUT: prefer filling from the SAME room/area cluster (not other rooms).
      if (local.length < 2) {
        const allPoints = Array.isArray(effectivePayload?.points)
          ? effectivePayload.points
          : [];

        const activeClusterId =
          activePoint?.clusterId || activePoint?.roomId || activePoint?.areaId;

        const pool = activeClusterId
          ? allPoints.filter((p) => {
              const cid = p?.clusterId || p?.roomId || p?.areaId;
              return cid && String(cid) === String(activeClusterId);
            })
          : allPoints;

        const union = [];
        const seen = new Set();
        for (const p of pool) {
          const arr = Array.isArray(p?.frames) ? p.frames : [];
          for (const item of arr) {
            const url =
              typeof item === "string"
                ? item
                : item && typeof item === "object"
                  ? item.url ||
                    item.imageUrl ||
                    item.src ||
                    item.download_url ||
                    item.storage_path ||
                    item.storagePath
                  : null;
            if (typeof url === "string" && url && !seen.has(url)) {
              seen.add(url);
              union.push(url);
            }
          }
        }

        if (union.length >= 2) {
          return union;
        }
      }

      return local;
    }

    if (isFramesMode) return effectivePayload.frames;
    return [];
  }, [
    activePoint?.frames,
    activePoint?.clusterId,
    activePoint?.roomId,
    activePoint?.areaId,
    effectivePayload?.frames,
    effectivePayload?.points,
    isFramesMode,
    isPointsMode,
  ]);

  const normalizedFrames = useMemo(() => {
    if (!Array.isArray(frames) || frames.length === 0) {
      return [];
    }

    const stepsRaw =
      (isPointsMode
        ? Number(activePoint?.steps)
        : Number(effectivePayload?.steps)) || 0;
    const desiredSteps =
      Number.isFinite(stepsRaw) && stepsRaw >= 8 ? Math.floor(stepsRaw) : 0;

    // frames as urls (optionally with null holes)
    const looksLikeUrlList = typeof frames[0] === "string" || frames[0] == null;
    if (looksLikeUrlList) {
      const cleaned = frames.map((x) => (typeof x === "string" ? x : null));

      // If caller provided an explicit "slots" array (length == steps),
      // fill only the missing slots.
      if (desiredSteps && cleaned.length === desiredSteps) {
        return fillMissingCircular(cleaned).filter(Boolean);
      }

      // Otherwise, treat the provided list as an ordered 360 sequence.
      // We do NOT upsample to `steps` here because it can create long repeated
      // runs that feel like "can't rotate".
      return cleaned.filter(Boolean);
    }

    // frames as objects with angles
    const steps = desiredSteps || 36;

    const slots = new Array(steps).fill(null);
    for (const item of frames) {
      if (!item || typeof item !== "object") continue;

      const url =
        item.url ||
        item.imageUrl ||
        item.src ||
        item.download_url ||
        item.storage_path ||
        item.storagePath;
      if (typeof url !== "string" || !url) continue;

      const angleDeg =
        Number(item.angleDeg ?? item.angle ?? item.deg ?? item.yawDeg) || 0;
      const a = normalizeDeg(angleDeg);

      const idx = Math.round((a / 360) * steps) % steps;
      slots[idx] = url;
    }

    const filled = fillMissingCircular(slots);
    return filled.filter(Boolean);
  }, [activePoint?.steps, effectivePayload?.steps, frames, isPointsMode]);

  return { frames, normalizedFrames };
}
