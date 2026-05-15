import { upload } from "@/app/api/utils/upload";
import { safeJsonParse, sleep } from "./helpers";

async function fetchWithRetry(
  url,
  options,
  { retries = 3, timeoutMs = 120000 } = {},
) {
  let lastError = null;
  for (let attempt = 0; attempt < retries; attempt++) {
    try {
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), timeoutMs);

      let res;
      try {
        res = await fetch(url, { ...options, signal: controller.signal });
      } finally {
        clearTimeout(timeout);
      }

      if (!res.ok) {
        const text = await res.text().catch(() => "");
        const err = new Error(`HTTP ${res.status}: ${text}`);
        err.status = res.status;
        err.bodyText = text;
        throw err;
      }
      return res;
    } catch (err) {
      lastError = err;

      // If we aborted due to timeout, wrap with a clearer message.
      if (err?.name === "AbortError") {
        lastError = new Error(
          `Timed out after ${Math.round(timeoutMs / 1000)}s while calling OpenAI`,
        );
      }

      const backoff = 500 * Math.pow(2, attempt);
      await sleep(backoff);
    }
  }
  throw lastError || new Error("Request failed");
}

function parsePngDimensions(bytes) {
  // PNG: width/height are in IHDR chunk starting at byte 16.
  if (!bytes || bytes.length < 24) return null;
  const isPng =
    bytes[0] === 0x89 &&
    bytes[1] === 0x50 &&
    bytes[2] === 0x4e &&
    bytes[3] === 0x47 &&
    bytes[4] === 0x0d &&
    bytes[5] === 0x0a &&
    bytes[6] === 0x1a &&
    bytes[7] === 0x0a;
  if (!isPng) return null;

  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const width = view.getUint32(16);
  const height = view.getUint32(20);
  if (!width || !height) return null;
  return { width, height };
}

function parseJpegDimensions(bytes) {
  if (!bytes || bytes.length < 4) return null;
  if (!(bytes[0] === 0xff && bytes[1] === 0xd8)) return null;

  let offset = 2;
  while (offset + 4 < bytes.length) {
    if (bytes[offset] !== 0xff) {
      offset += 1;
      continue;
    }

    const marker = bytes[offset + 1];
    const isSOF =
      marker === 0xc0 ||
      marker === 0xc1 ||
      marker === 0xc2 ||
      marker === 0xc3 ||
      marker === 0xc5 ||
      marker === 0xc6 ||
      marker === 0xc7 ||
      marker === 0xc9 ||
      marker === 0xca ||
      marker === 0xcb ||
      marker === 0xcd ||
      marker === 0xce ||
      marker === 0xcf;

    // Start of Scan / End of Image
    if (marker === 0xda || marker === 0xd9) {
      break;
    }

    const segmentLength = (bytes[offset + 2] << 8) + bytes[offset + 3];
    if (!segmentLength || offset + 2 + segmentLength > bytes.length) {
      break;
    }

    if (isSOF) {
      const height = (bytes[offset + 5] << 8) + bytes[offset + 6];
      const width = (bytes[offset + 7] << 8) + bytes[offset + 8];
      if (width && height) return { width, height };
      return null;
    }

    offset += 2 + segmentLength;
  }

  return null;
}

function getImageDimensionsFromArrayBuffer(buffer) {
  try {
    const bytes = new Uint8Array(buffer.slice(0, 512 * 1024));
    return parsePngDimensions(bytes) || parseJpegDimensions(bytes) || null;
  } catch {
    return null;
  }
}

function chooseEditSizeCandidates({ width, height }) {
  // Try to match the original photo aspect ratio first to reduce reframing artifacts.
  // If OpenAI rejects a size, we'll fall back to 1024x1024.
  if (!width || !height) return ["1024x1024"];
  const ratio = width / height;
  if (ratio > 1.15) return ["1536x1024", "1024x1024"]; // landscape
  if (ratio < 0.87) return ["1024x1536", "1024x1024"]; // portrait
  return ["1024x1024"];
}

function isInvalidSizeError(e) {
  const status = e?.status;
  if (status !== 400) return false;
  const body = String(e?.bodyText || e?.message || "").toLowerCase();
  return (
    body.includes("size") &&
    (body.includes("invalid") ||
      body.includes("supported") ||
      body.includes("must be"))
  );
}

// Helper to download a remote image into a Blob we can send via multipart/form-data
async function downloadImageAsBlob(imageUrl) {
  // IMPORTANT: keep the total wall-clock under the job heartbeat watchdog.
  // Downloading + image edit must not exceed ~10 minutes without heartbeats.
  const res = await fetchWithRetry(
    imageUrl,
    { method: "GET" },
    { retries: 2, timeoutMs: 60000 },
  );
  const headerContentType =
    res.headers.get("content-type") || "application/octet-stream";
  const buffer = await res.arrayBuffer();

  // Detect file type from magic bytes.
  const bytes = new Uint8Array(buffer.slice(0, 16));
  const isPng =
    bytes.length >= 8 &&
    bytes[0] === 0x89 &&
    bytes[1] === 0x50 &&
    bytes[2] === 0x4e &&
    bytes[3] === 0x47 &&
    bytes[4] === 0x0d &&
    bytes[5] === 0x0a &&
    bytes[6] === 0x1a &&
    bytes[7] === 0x0a;

  const isJpeg =
    bytes.length >= 3 &&
    bytes[0] === 0xff &&
    bytes[1] === 0xd8 &&
    bytes[2] === 0xff;

  const detectedType = isPng
    ? "image/png"
    : isJpeg
      ? "image/jpeg"
      : headerContentType;

  const dims = getImageDimensionsFromArrayBuffer(buffer);

  return {
    blob: new Blob([buffer], { type: detectedType }),
    mimeType: detectedType,
    width: dims?.width || null,
    height: dims?.height || null,
  };
}

// NOTE: Previously we tried to force PNG via Uploadcare transforms, but our uploads
// are served from raw.createusercontent.com which doesn't support those transforms.
// OpenAI gpt-image-* edit models accept both PNG and JPEG, so we allow JPEG input.

function getExtensionForMime(mimeType) {
  const t = String(mimeType || "").toLowerCase();
  if (t.includes("png")) return "png";
  if (t.includes("jpeg") || t.includes("jpg")) return "jpg";
  return "bin";
}

function getUploadcareBase(url) {
  // Robustly extract the canonical Uploadcare base URL:
  // https://ucarecdn.com/<uuid>/
  if (typeof url !== "string") return null;
  const match = url.match(
    /^https?:\/\/ucarecdn\.com\/([0-9a-f-]{36})(?:\/|$)/i,
  );
  if (!match) return null;
  return `https://ucarecdn.com/${match[1]}/`;
}

function isUploadcareUrl(url) {
  return !!getUploadcareBase(url);
}

function ensureUploadcarePng(url) {
  // OpenAI image edits currently rejects JPEG in our environment and expects PNG.
  // Uploadcare supports on-the-fly format conversion.
  const base = getUploadcareBase(url);
  if (!base) return url;
  return `${base}-/format/png/`;
}

function normalizeForDalle2SquarePng(url) {
  // dall-e-2 edits require a square PNG < 4MB.
  const base = getUploadcareBase(url);
  if (!base) return url;
  return `${base}-/scale_crop/1024x1024/center/-/format/png/`;
}

function toAbsolutePublicUrl(inputUrl) {
  const url = typeof inputUrl === "string" ? inputUrl.trim() : "";
  if (!url) {
    return "";
  }

  // Reject data: URLs in vision calls (we only support public URLs there)
  if (url.startsWith("data:")) {
    return "";
  }

  if (url.startsWith("http://") || url.startsWith("https://")) {
    return url;
  }

  // If a relative path sneaks in, make it absolute.
  if (url.startsWith("/")) {
    const base = String(process.env.APP_URL || "").replace(/\/$/, "");
    if (!base) {
      // No APP_URL available; return empty so caller can throw a clear error.
      return "";
    }
    return `${base}${url}`;
  }

  // Unknown/unsupported format
  return "";
}

function extractUrlCandidate(value) {
  if (!value) return "";
  if (typeof value === "string") return value;

  // Common shapes we might accidentally pass around:
  // - { url: "https://..." }
  // - { uri: "https://..." }
  // - { src: "https://..." }
  // - { href: "https://..." }
  // - { imageUrl: "https://..." }
  if (typeof value === "object") {
    const v = value;

    if (typeof v.url === "string") return v.url;
    if (typeof v.uri === "string") return v.uri;
    if (typeof v.src === "string") return v.src;
    if (typeof v.href === "string") return v.href;
    if (typeof v.imageUrl === "string") return v.imageUrl;

    // Occasionally: { url: { url: "https://..." } }
    if (v.url && typeof v.url === "object" && typeof v.url.url === "string") {
      return v.url.url;
    }
  }

  return "";
}

function normalizeImageContentPart(part) {
  if (!part || typeof part !== "object") return part;

  const t = String(part.type || "").trim();
  const isImageType = t === "image_url" || t === "input_image";
  if (!isImageType) {
    return part;
  }

  // Normalize MANY possible caller shapes into the Chat Completions contract:
  // { type: "image_url", image_url: { url: "https://..." } }
  //
  // IMPORTANT:
  // - OpenAI rejects missing `image_url.url`
  // - OpenAI rejects non-string `image_url.url`
  // - We only allow FULL public http(s) URLs (or absolute paths we can expand via APP_URL)

  const rawUrl =
    extractUrlCandidate(part.image_url) ||
    extractUrlCandidate(part.url) ||
    extractUrlCandidate(part.imageUrl) ||
    "";

  const abs = toAbsolutePublicUrl(rawUrl);
  if (!abs) {
    // Build a diagnostic message to help debug the root cause
    const partKeys = Object.keys(part).join(",");
    const imageUrlType = typeof part.image_url;
    const imageUrlPreview = part.image_url
      ? String(part.image_url).slice(0, 120)
      : "(nullish)";

    throw new Error(
      "Invalid OpenAI vision image input: expected a FULL public https URL. " +
        `Got: ${rawUrl ? String(rawUrl).slice(0, 120) : "(empty)"}. ` +
        `Part keys=[${partKeys}], image_url type=${imageUrlType}, ` +
        `image_url preview=${imageUrlPreview}`,
    );
  }

  return {
    type: "image_url",
    image_url: { url: abs },
  };
}

function normalizeChatMessagesForOpenAI(messages) {
  const list = Array.isArray(messages) ? messages : [];

  return list.map((m) => {
    if (!m || typeof m !== "object") return m;
    const content = m.content;

    if (Array.isArray(content)) {
      return {
        ...m,
        content: content.map((p) => normalizeImageContentPart(p)),
      };
    }

    // Also support accidental single-part object content
    if (content && typeof content === "object") {
      return {
        ...m,
        content: normalizeImageContentPart(content),
      };
    }

    return m;
  });
}

export async function openAiChatJson({
  openAiKey,
  model,
  messages,
  retries = 2,
}) {
  const normalizedMessages = normalizeChatMessagesForOpenAI(messages);

  const res = await fetchWithRetry(
    "https://api.openai.com/v1/chat/completions",
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${openAiKey}`,
      },
      body: JSON.stringify({
        model,
        messages: normalizedMessages,
        temperature: 0.2,
      }),
    },
    { retries, timeoutMs: 90000 },
  );

  const json = await res.json();
  const text = json?.choices?.[0]?.message?.content || "";
  const parsed = safeJsonParse(text);
  return { parsed, raw: text };
}

export async function generateImageWithOpenAI({
  openAiKey,
  prompt,
  retries = 3,
}) {
  // Keep this as a fallback path (text-only generation).
  const res = await fetchWithRetry(
    "https://api.openai.com/v1/images/generations",
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${openAiKey}`,
      },
      body: JSON.stringify({
        model: "dall-e-3",
        prompt,
        size: "1024x1024",
        quality: "hd",
        style: "natural",
        response_format: "url",
      }),
    },
    // IMPORTANT: Keep overall wall-clock under the job heartbeat timeout.
    // This endpoint is a fallback anyway.
    { retries: Math.min(2, retries), timeoutMs: 240000 },
  );

  const json = await res.json();
  const url = json?.data?.[0]?.url;
  if (!url) {
    throw new Error("OpenAI did not return an image URL");
  }

  return { kind: "url", url };
}

export async function editImageWithOpenAI({
  openAiKey,
  prompt,
  imageUrls,
  retries = 2,
}) {
  const urls = Array.isArray(imageUrls) ? imageUrls.filter(Boolean) : [];
  if (urls.length === 0) {
    throw new Error("editImageWithOpenAI requires at least 1 image URL");
  }

  // IMPORTANT:
  // We only use the first URL as the base image for now.
  const baseUrl = urls[0];

  const isOrgVerificationError = (e) => {
    const status = e?.status;
    const body = String(e?.bodyText || e?.message || "");
    if (status !== 403) return false;
    const bodyLower = body.toLowerCase();
    return (
      bodyLower.includes("must be verified") ||
      bodyLower.includes("verify organization") ||
      bodyLower.includes("organization")
    );
  };

  const makeUserFacingOrgMessage = () => {
    return (
      "OpenAI erişimi engellendi: Bu OpenAI organizasyonu doğrulanmadığı için gpt-image-1 kullanılamıyor. " +
      "OpenAI panelinden Organization Verify yapıp 10–15 dk bekleyin. Şimdilik daha düşük kaliteli bir fallback denenecek."
    );
  };

  const modelCandidates = ["gpt-image-1"]; // best-effort
  let lastErr = null;
  let blockedByOrgVerification = false;

  // Prepare the base image once so we can pick a better aspect ratio.
  let baseMeta = await downloadImageAsBlob(baseUrl);
  let blob = baseMeta.blob;

  const typeLower = String(blob?.type || "").toLowerCase();
  const looksSupported =
    typeLower.includes("image/png") ||
    typeLower.includes("image/jpeg") ||
    typeLower.includes("image/jpg");

  if (!looksSupported) {
    const up = await upload({ url: baseUrl });
    if (up?.url) {
      baseMeta = await downloadImageAsBlob(up.url);
      blob = baseMeta.blob;
    }
  }

  const finalTypeLower = String(blob?.type || "").toLowerCase();
  const isAccepted =
    finalTypeLower.includes("image/png") ||
    finalTypeLower.includes("image/jpeg") ||
    finalTypeLower.includes("image/jpg");

  if (!isAccepted) {
    throw new Error(
      `OpenAI image edits require PNG or JPEG input, but got ${blob?.type || "unknown"}. urlForModel=${String(baseUrl).slice(0, 120)}`,
    );
  }

  const ext = getExtensionForMime(blob?.type);
  const sizeCandidates = chooseEditSizeCandidates({
    width: baseMeta?.width,
    height: baseMeta?.height,
  });

  for (const model of modelCandidates) {
    for (const size of sizeCandidates) {
      try {
        const form = new FormData();
        form.append("model", model);
        form.append("prompt", prompt);
        form.append("n", "1");
        form.append("size", size);
        // IMPORTANT: Do NOT send `response_format` or `quality` for gpt-image-* edits.
        form.append("image", blob, `image.${ext}`);

        const res = await fetchWithRetry(
          "https://api.openai.com/v1/images/edits",
          {
            method: "POST",
            headers: {
              Authorization: `Bearer ${openAiKey}`,
            },
            body: form,
          },
          {
            retries,
            timeoutMs: 210000,
          },
        );

        const json = await res.json();

        const b64 = json?.data?.[0]?.b64_json;
        if (b64 && typeof b64 === "string") {
          return { kind: "b64_json", b64_json: b64, modelUsed: model };
        }

        const url = json?.data?.[0]?.url;
        if (url && typeof url === "string") {
          return { kind: "url", url, modelUsed: model };
        }

        throw new Error(
          `OpenAI did not return an edited image (no b64_json). Raw keys: ${Object.keys(json || {}).join(", ")}`,
        );
      } catch (e) {
        lastErr = e;
        if (isOrgVerificationError(e)) {
          blockedByOrgVerification = true;
        }
        // If the size is rejected, try the next candidate.
        if (isInvalidSizeError(e)) {
          continue;
        }
        // Otherwise move on (or fall through to dall-e-2 fallback if applicable).
        break;
      }
    }
  }

  // ---- Fallback: DALL·E 2 edits ----
  try {
    if (blockedByOrgVerification) {
      const up = await upload({ url: baseUrl });
      const uploadedUrl = up?.url;
      if (!uploadedUrl) {
        throw new Error(makeUserFacingOrgMessage());
      }

      const dalle2Url = normalizeForDalle2SquarePng(uploadedUrl);
      const meta = await downloadImageAsBlob(dalle2Url);
      const blob2 = meta.blob;
      const typeLower2 = String(blob2?.type || "").toLowerCase();
      if (!typeLower2.includes("image/png")) {
        throw new Error(
          `dall-e-2 requires PNG input but got ${blob2?.type || "unknown"}. urlForModel=${String(dalle2Url).slice(0, 160)}`,
        );
      }

      const form = new FormData();
      form.append("model", "dall-e-2");
      form.append("prompt", prompt);
      form.append("n", "1");
      form.append("size", "1024x1024");
      form.append("response_format", "b64_json");
      form.append("image", blob2, "image.png");

      const res = await fetchWithRetry(
        "https://api.openai.com/v1/images/edits",
        {
          method: "POST",
          headers: {
            Authorization: `Bearer ${openAiKey}`,
          },
          body: form,
        },
        { retries, timeoutMs: 210000 },
      );

      const json = await res.json();
      const b64 = json?.data?.[0]?.b64_json;
      if (b64 && typeof b64 === "string") {
        return { kind: "b64_json", b64_json: b64, modelUsed: "dall-e-2" };
      }

      const url = json?.data?.[0]?.url;
      if (url && typeof url === "string") {
        return { kind: "url", url, modelUsed: "dall-e-2" };
      }

      throw new Error(
        `OpenAI (dall-e-2) did not return an edited image. Raw keys: ${Object.keys(json || {}).join(", ")}`,
      );
    }
  } catch (e) {
    // If fallback fails, preserve the better error below.
    lastErr = e;
  }

  // If the org isn't verified, show a clear actionable message.
  if (blockedByOrgVerification) {
    throw new Error(makeUserFacingOrgMessage());
  }

  throw lastErr || new Error("Could not edit image");
}
