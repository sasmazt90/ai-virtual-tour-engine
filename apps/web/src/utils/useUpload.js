import * as React from "react";

// Add small helpers to avoid production upload failures (500s) on large images.
const DEFAULT_IMAGE_MAX_SIDE = 2400;
const DEFAULT_IMAGE_QUALITY = 0.82;
const LARGE_IMAGE_BYTES = 4 * 1024 * 1024; // ~4MB
const ALWAYS_CONVERT_NONSTANDARD_IMAGES = true;
const SMALL_IMAGE_BYTES = 1 * 1024 * 1024; // keep small images as-is when already jpeg/png/webp/gif

// New: smaller, safer settings for backend fallback uploads (base64 hits the 4.5MB body limit faster).
const FALLBACK_IMAGE_MAX_SIDE = 1600;
const FALLBACK_IMAGE_QUALITY = 0.75;

const SUPPORTED_IMAGE_MIME_TYPES = new Set([
  "image/jpeg",
  "image/jpg",
  "image/png",
  "image/webp",
  "image/gif",
]);

// We have seen the platform upload endpoint occasionally return 500 for some valid
// image files (especially PNG/WEBP/HEIC). To make property creation "never block",
// we can re-encode to a plain JPEG as a safe fallback.
const ALWAYS_CONVERT_TO_JPEG_EXCEPT_GIF = true;

async function tryReadResponseBody(response) {
  try {
    const contentType = response.headers.get("content-type") || "";
    const isJson = contentType.includes("application/json");
    if (isJson) {
      const json = await response.json();
      return typeof json === "string" ? json : JSON.stringify(json);
    }
    const text = await response.text();
    return text;
  } catch {
    return "";
  }
}

async function getImageBitmapBestEffort(file) {
  if (typeof window === "undefined") return null;
  if (!file || !(file instanceof File)) return null;

  if (typeof createImageBitmap === "function") {
    return createImageBitmap(file);
  }

  const dataUrl = await new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = () => reject(new Error("Could not read image"));
    reader.onload = () => resolve(String(reader.result || ""));
    reader.readAsDataURL(file);
  });

  const img = await new Promise((resolve, reject) => {
    const el = new Image();
    el.onload = () => resolve(el);
    el.onerror = () => reject(new Error("Could not load image"));
    el.src = dataUrl;
  });

  return img;
}

async function fileToDataUrl(file) {
  if (typeof window === "undefined") return null;
  if (!file || !(file instanceof File)) return null;

  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = () => reject(new Error("Could not read image"));
    reader.onload = () => resolve(String(reader.result || ""));
    reader.readAsDataURL(file);
  });
}

async function normalizeImageFileForUpload(
  file,
  { forceConvertToJpeg, maxSide, quality } = {},
) {
  // SSR safety
  if (typeof window === "undefined") return file;
  if (!file || !(file instanceof File)) return file;
  if (!file.type || !file.type.startsWith("image/")) return file;

  const isSupported = SUPPORTED_IMAGE_MIME_TYPES.has(file.type);
  const isGif = file.type === "image/gif";

  const shouldConvertBecauseType =
    ALWAYS_CONVERT_NONSTANDARD_IMAGES && !isSupported;

  // Even if it's supported, PNG/WEBP/etc may still fail upstream; re-encode to JPEG.
  const shouldConvertToJpegForStability =
    !!forceConvertToJpeg ||
    (ALWAYS_CONVERT_TO_JPEG_EXCEPT_GIF &&
      !isGif &&
      file.type !== "image/jpeg" &&
      file.type !== "image/jpg");

  const shouldDownscaleBecauseSize = file.size > LARGE_IMAGE_BYTES;

  // If it's already a small, common type, don't touch it.
  if (
    !shouldConvertBecauseType &&
    !shouldConvertToJpegForStability &&
    !shouldDownscaleBecauseSize &&
    file.size <= SMALL_IMAGE_BYTES
  ) {
    return file;
  }

  try {
    const bitmap = await getImageBitmapBestEffort(file);
    if (!bitmap) return file;

    const srcW = bitmap.width;
    const srcH = bitmap.height;

    if (!srcW || !srcH) return file;

    const effectiveMaxSide = Number.isFinite(maxSide)
      ? maxSide
      : DEFAULT_IMAGE_MAX_SIDE;
    const effectiveQuality = Number.isFinite(quality)
      ? quality
      : DEFAULT_IMAGE_QUALITY;

    const scale = Math.min(1, effectiveMaxSide / Math.max(srcW, srcH));
    const outW = Math.max(1, Math.round(srcW * scale));
    const outH = Math.max(1, Math.round(srcH * scale));

    const canvas = document.createElement("canvas");
    canvas.width = outW;
    canvas.height = outH;
    const ctx = canvas.getContext("2d");
    if (!ctx) return file;

    ctx.drawImage(bitmap, 0, 0, outW, outH);

    const blob = await new Promise((resolve) => {
      // Use JPEG to keep the upload pipeline stable (HEIC/HEIF can cause 500s).
      canvas.toBlob((b) => resolve(b || null), "image/jpeg", effectiveQuality);
    });

    if (!blob) return file;

    const nextName = file.name
      ? file.name.replace(/\.[^/.]+$/, "") + ".jpg"
      : "upload.jpg";

    return new File([blob], nextName, { type: "image/jpeg" });
  } catch (e) {
    // If conversion fails (often for HEIC in some browsers), show a clear error upstream.
    console.error(e);
    throw new Error(
      "This image format could not be processed for upload. Please export the photo as JPG or PNG and try again.",
    );
  }
}

async function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function useUpload() {
  const [loading, setLoading] = React.useState(false);
  const upload = React.useCallback(async (input) => {
    try {
      setLoading(true);

      // Try up to 3 times with a tiny backoff on 5xx errors (we see these sporadically on live).
      for (let attempt = 0; attempt < 3; attempt++) {
        let response;

        if ("file" in input && input.file) {
          // First attempt: best-effort normalization.
          // Subsequent attempts: force JPEG re-encode (helps when upstream rejects PNG/WEBP/etc with 500).
          const safeFile = await normalizeImageFileForUpload(input.file, {
            forceConvertToJpeg: attempt > 0,
          });
          const formData = new FormData();
          formData.append("file", safeFile);
          response = await fetch("/api/upload", {
            method: "POST",
            body: formData,
          });

          // If multipart upload returns a transient 5xx, fall back to JSON/base64.
          if (
            !response.ok &&
            response.status >= 500 &&
            response.status <= 599
          ) {
            const platformBody = await tryReadResponseBody(response);
            console.warn("Multipart upload failed, trying base64 fallback...", {
              status: response.status,
              platformBody,
            });

            const fallbackFile = await normalizeImageFileForUpload(input.file, {
              forceConvertToJpeg: true,
              maxSide: FALLBACK_IMAGE_MAX_SIDE,
              quality: FALLBACK_IMAGE_QUALITY,
            });

            const base64 = await fileToDataUrl(fallbackFile);
            if (!base64) {
              throw new Error(
                "Upload failed: Could not read image for fallback upload.",
              );
            }

            const fallbackRes = await fetch("/api/upload", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ base64 }),
            });

            if (!fallbackRes.ok) {
              const fallbackBody = await tryReadResponseBody(fallbackRes);
              const fallbackSuffix = fallbackBody ? ` — ${fallbackBody}` : "";
              throw new Error(
                `Upload failed: [${fallbackRes.status}] ${fallbackRes.statusText}${fallbackSuffix}`,
              );
            }

            const data = await fallbackRes.json();
            return { url: data.url, mimeType: data.mimeType || null };
          }
        } else if ("url" in input) {
          response = await fetch("/api/upload", {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
            },
            body: JSON.stringify({ url: input.url }),
          });
        } else if ("base64" in input) {
          response = await fetch("/api/upload", {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
            },
            body: JSON.stringify({ base64: input.base64 }),
          });
        } else {
          response = await fetch("/api/upload", {
            method: "POST",
            headers: {
              "Content-Type": "application/octet-stream",
            },
            body: input.buffer,
          });
        }

        if (!response.ok) {
          // Give the user a clear message.
          if (response.status === 413) {
            throw new Error("Upload failed: File too large.");
          }

          const bodyText = await tryReadResponseBody(response);
          const suffix = bodyText ? ` — ${bodyText}` : "";
          const err = new Error(
            `Upload failed: [${response.status}] ${response.statusText}${suffix}`,
          );

          // One retry for transient server errors.
          const isRetryable = response.status >= 500 && response.status <= 599;
          if (isRetryable && attempt < 2) {
            const backoff = attempt === 0 ? 350 : 900;
            await sleep(backoff);
            continue;
          }

          throw err;
        }

        const data = await response.json();
        return { url: data.url, mimeType: data.mimeType || null };
      }

      return { error: "Upload failed" };
    } catch (uploadError) {
      if (uploadError instanceof Error) {
        return { error: uploadError.message };
      }
      if (typeof uploadError === "string") {
        return { error: uploadError };
      }
      return { error: "Upload failed" };
    } finally {
      setLoading(false);
    }
  }, []);

  return [upload, { loading }];
}

export { useUpload };
export default useUpload;
