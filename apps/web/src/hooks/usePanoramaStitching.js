import { useState, useCallback, useEffect } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import useUpload from "@/utils/useUpload";
import {
  loadScaledBitmap,
  stitchBitmaps,
  canvasToJpegBlob,
} from "@/utils/panoramaStitching";

export function usePanoramaStitching({ propertyId, property }) {
  const queryClient = useQueryClient();

  const [panoFiles, setPanoFiles] = useState([]);
  const [panoError, setPanoError] = useState(null);
  const [panoPreviewUrl, setPanoPreviewUrl] = useState(null);
  const [panoRemoteUrl, setPanoRemoteUrl] = useState(null);
  const [panoWorking, setPanoWorking] = useState(false);

  const [overlapMode, setOverlapMode] = useState({
    mode: "auto",
    manualPct: 30,
  });

  const [upload, { loading: uploadLoading }] = useUpload();

  useEffect(() => {
    return () => {
      if (panoPreviewUrl && typeof URL !== "undefined") {
        try {
          URL.revokeObjectURL(panoPreviewUrl);
        } catch {
          // ignore
        }
      }
    };
  }, [panoPreviewUrl]);

  const createPanoramaTourMutation = useMutation({
    mutationFn: async ({ panoramaUrl }) => {
      const res = await fetch("/api/virtual-tours/from-panorama", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ propertyId, panoramaUrl }),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body?.error || "Could not create tour");
      }
      return res.json();
    },
    onSuccess: async () => {
      const userId = property?.user_id;
      await queryClient.invalidateQueries({
        queryKey: ["property", userId, propertyId],
      });
    },
  });

  const onPickPanoFiles = useCallback(
    (e) => {
      const files = Array.from(e.target.files || []).filter(Boolean);
      setPanoError(null);
      setPanoRemoteUrl(null);

      if (panoPreviewUrl && typeof URL !== "undefined") {
        try {
          URL.revokeObjectURL(panoPreviewUrl);
        } catch {
          // ignore
        }
      }
      setPanoPreviewUrl(null);

      setPanoFiles(files.slice(0, 3));
    },
    [panoPreviewUrl],
  );

  const canGenerate = panoFiles.length >= 2 && panoFiles.length <= 3;

  const onGeneratePanorama = useCallback(async () => {
    try {
      setPanoError(null);
      setPanoRemoteUrl(null);

      if (typeof createImageBitmap === "undefined") {
        throw new Error(
          "Your browser does not support createImageBitmap (needed for stitching). Try a newer Chrome/Safari.",
        );
      }

      if (!canGenerate) {
        throw new Error("Pick 2 or 3 photos (in order) to stitch");
      }

      setPanoWorking(true);

      const maxLongEdgePx = 1800;
      const scaled = [];
      for (const f of panoFiles) {
        const bm = await loadScaledBitmap(f, maxLongEdgePx);
        scaled.push(bm);
      }

      const stitchedCanvas = await stitchBitmaps({
        bitmaps: scaled,
        overlapMode,
      });

      const blob = await canvasToJpegBlob(stitchedCanvas, 0.86);

      const file = new File([blob], "panorama.jpg", { type: "image/jpeg" });

      const previewUrl =
        typeof URL !== "undefined" ? URL.createObjectURL(blob) : null;
      if (previewUrl) {
        setPanoPreviewUrl(previewUrl);
      }

      const uploaded = await upload({ file });
      if (uploaded?.error) {
        throw new Error(uploaded.error);
      }

      const url = uploaded?.url;
      if (!url) {
        throw new Error("Upload failed");
      }

      setPanoRemoteUrl(url);

      await createPanoramaTourMutation.mutateAsync({ panoramaUrl: url });

      // cleanup bitmaps best-effort
      for (const s of scaled) {
        try {
          s.bitmap.close?.();
        } catch {
          // ignore
        }
      }
    } catch (err) {
      console.error(err);
      const msg =
        err instanceof Error ? err.message : "Could not generate panorama";
      setPanoError(msg);
    } finally {
      setPanoWorking(false);
    }
  }, [canGenerate, createPanoramaTourMutation, overlapMode, panoFiles, upload]);

  const stitchBusy =
    panoWorking || uploadLoading || createPanoramaTourMutation.isPending;

  return {
    panoFiles,
    panoError,
    panoPreviewUrl,
    panoRemoteUrl,
    panoWorking,
    overlapMode,
    setOverlapMode,
    onPickPanoFiles,
    canGenerate,
    onGeneratePanorama,
    stitchBusy,
  };
}
