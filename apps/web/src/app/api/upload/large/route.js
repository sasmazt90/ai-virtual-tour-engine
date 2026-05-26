import crypto from "node:crypto";
import { auth } from "@/auth";

const VIDEO_EXTENSIONS = {
  "video/mp4": "mp4",
  "video/quicktime": "mov",
  "video/x-m4v": "m4v",
  "video/m4v": "m4v",
  "video/webm": "webm",
};

const MODEL_EXTENSIONS = {
  "model/vnd.ply": "ply",
  "application/octet-stream": "",
};

const SAFE_EXTENSIONS = new Set(["mp4", "mov", "m4v", "webm", "ply", "splat", "ksplat"]);
const VIDEO_SAFE_EXTENSIONS = new Set(["mp4", "mov", "m4v", "webm"]);
const MODEL_SAFE_EXTENSIONS = new Set(["ply", "splat", "ksplat"]);
const S3_MAX_PRESIGN_EXPIRES_SECONDS = 7 * 24 * 60 * 60;
const S3_MULTIPART_PART_BYTES = 32 * 1024 * 1024;

function getSupabaseStorageConfig() {
  const url = process.env.SUPABASE_URL;
  const serviceRoleKey = process.env.SUPABASE_SERVICE_ROLE_KEY;
  const bucket = process.env.SUPABASE_STORAGE_BUCKET || "uploads";

  if (!url || !serviceRoleKey) return null;

  return {
    url: url.replace(/\/+$/, ""),
    bucket,
    serviceRoleKey,
  };
}

function getExternalVideoStorageConfig() {
  const endpoint =
    process.env.VIDEO_UPLOAD_S3_ENDPOINT ||
    process.env.RUNPOD_S3_ENDPOINT ||
    process.env.S3_UPLOAD_ENDPOINT;
  const bucket =
    process.env.VIDEO_UPLOAD_S3_BUCKET ||
    process.env.RUNPOD_S3_BUCKET ||
    process.env.S3_UPLOAD_BUCKET;
  const accessKeyId =
    process.env.VIDEO_UPLOAD_S3_ACCESS_KEY_ID ||
    process.env.RUNPOD_S3_ACCESS_KEY_ID ||
    process.env.RUNPOD_S3_ACCESS_KEY ||
    process.env.S3_UPLOAD_ACCESS_KEY_ID;
  const secretAccessKey =
    process.env.VIDEO_UPLOAD_S3_SECRET_ACCESS_KEY ||
    process.env.RUNPOD_S3_SECRET_ACCESS_KEY ||
    process.env.RUNPOD_S3_SECRET_KEY ||
    process.env.S3_UPLOAD_SECRET_ACCESS_KEY;

  if (!endpoint || !bucket || !accessKeyId || !secretAccessKey) return null;

  return {
    endpoint: endpoint.replace(/\/+$/, ""),
    bucket,
    accessKeyId,
    secretAccessKey,
    region:
      process.env.VIDEO_UPLOAD_S3_REGION ||
      process.env.RUNPOD_S3_REGION ||
      process.env.S3_UPLOAD_REGION ||
      "us-east-1",
    publicBaseUrl:
      process.env.VIDEO_UPLOAD_S3_PUBLIC_BASE_URL ||
      process.env.RUNPOD_S3_PUBLIC_BASE_URL ||
      process.env.S3_UPLOAD_PUBLIC_BASE_URL ||
      "",
    signedGetExpiresSeconds: Math.min(
      S3_MAX_PRESIGN_EXPIRES_SECONDS,
      Math.max(
        60,
        Number(
          process.env.VIDEO_UPLOAD_S3_SIGNED_GET_EXPIRES_SECONDS ||
            process.env.RUNPOD_S3_SIGNED_GET_EXPIRES_SECONDS ||
            S3_MAX_PRESIGN_EXPIRES_SECONDS,
        ) || S3_MAX_PRESIGN_EXPIRES_SECONDS,
      ),
    ),
  };
}

function extensionFromFilename(filename) {
  const ext = String(filename || "").split(".").pop()?.toLowerCase().trim();
  return ext && SAFE_EXTENSIONS.has(ext) ? ext : "";
}

function extensionFromMime(mimeType, filename) {
  return (
    extensionFromFilename(filename) ||
    VIDEO_EXTENSIONS[String(mimeType || "").toLowerCase()] ||
    MODEL_EXTENSIONS[String(mimeType || "").toLowerCase()] ||
    ""
  );
}

function folderForExtension(ext) {
  if (VIDEO_SAFE_EXTENSIONS.has(ext)) return "video-uploads";
  if (MODEL_SAFE_EXTENSIONS.has(ext)) return "3d-scans";
  return "uploads";
}

function rfc3986(value) {
  return encodeURIComponent(String(value)).replace(/[!'()*]/g, (char) =>
    `%${char.charCodeAt(0).toString(16).toUpperCase()}`,
  );
}

function encodePath(path) {
  return String(path || "")
    .split("/")
    .map(rfc3986)
    .join("/");
}

function hmac(key, value, encoding) {
  return crypto.createHmac("sha256", key).update(value, "utf8").digest(encoding);
}

function sha256Hex(value) {
  return crypto.createHash("sha256").update(value, "utf8").digest("hex");
}

function signingKey(secretAccessKey, dateStamp, region) {
  const kDate = hmac(`AWS4${secretAccessKey}`, dateStamp);
  const kRegion = hmac(kDate, region);
  const kService = hmac(kRegion, "s3");
  return hmac(kService, "aws4_request");
}

function formatAmzDate(date = new Date()) {
  const iso = date.toISOString().replace(/[:-]|\.\d{3}/g, "");
  return {
    amzDate: iso,
    dateStamp: iso.slice(0, 8),
  };
}

function presignS3Url({
  method,
  endpoint,
  bucket,
  key,
  region,
  accessKeyId,
  secretAccessKey,
  expiresSeconds,
}) {
  const endpointUrl = new URL(endpoint);
  const { amzDate, dateStamp } = formatAmzDate();
  const credentialScope = `${dateStamp}/${region}/s3/aws4_request`;
  const canonicalUri = `${endpointUrl.pathname.replace(/\/+$/, "")}/${rfc3986(bucket)}/${encodePath(key)}`;
  const params = {
    "X-Amz-Algorithm": "AWS4-HMAC-SHA256",
    "X-Amz-Content-Sha256": "UNSIGNED-PAYLOAD",
    "X-Amz-Credential": `${accessKeyId}/${credentialScope}`,
    "X-Amz-Date": amzDate,
    "X-Amz-Expires": String(expiresSeconds),
    "X-Amz-SignedHeaders": "host",
  };
  const canonicalQuery = Object.entries(params)
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([keyName, value]) => `${rfc3986(keyName)}=${rfc3986(value)}`)
    .join("&");
  const canonicalHeaders = `host:${endpointUrl.host}\n`;
  const canonicalRequest = [
    method,
    canonicalUri,
    canonicalQuery,
    canonicalHeaders,
    "host",
    "UNSIGNED-PAYLOAD",
  ].join("\n");
  const stringToSign = [
    "AWS4-HMAC-SHA256",
    amzDate,
    credentialScope,
    sha256Hex(canonicalRequest),
  ].join("\n");
  const signature = hmac(
    signingKey(secretAccessKey, dateStamp, region),
    stringToSign,
    "hex",
  );
  return `${endpointUrl.origin}${canonicalUri}?${canonicalQuery}&X-Amz-Signature=${signature}`;
}

function canonicalQueryString(query = {}) {
  return Object.entries(query)
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([keyName, value]) => `${rfc3986(keyName)}=${rfc3986(value)}`)
    .join("&");
}

function signedS3Request({
  method,
  endpoint,
  bucket,
  key,
  region,
  accessKeyId,
  secretAccessKey,
  query = {},
}) {
  const endpointUrl = new URL(endpoint);
  const { amzDate, dateStamp } = formatAmzDate();
  const payloadHash = "UNSIGNED-PAYLOAD";
  const credentialScope = `${dateStamp}/${region}/s3/aws4_request`;
  const canonicalUri = `${endpointUrl.pathname.replace(/\/+$/, "")}/${rfc3986(bucket)}/${encodePath(key)}`;
  const canonicalQuery = canonicalQueryString(query);
  const canonicalHeaders = [
    `host:${endpointUrl.host}`,
    `x-amz-content-sha256:${payloadHash}`,
    `x-amz-date:${amzDate}`,
    "",
  ].join("\n");
  const signedHeaders = "host;x-amz-content-sha256;x-amz-date";
  const canonicalRequest = [
    method,
    canonicalUri,
    canonicalQuery,
    canonicalHeaders,
    signedHeaders,
    payloadHash,
  ].join("\n");
  const stringToSign = [
    "AWS4-HMAC-SHA256",
    amzDate,
    credentialScope,
    sha256Hex(canonicalRequest),
  ].join("\n");
  const signature = hmac(
    signingKey(secretAccessKey, dateStamp, region),
    stringToSign,
    "hex",
  );

  return {
    url: `${endpointUrl.origin}${canonicalUri}${canonicalQuery ? `?${canonicalQuery}` : ""}`,
    headers: {
      Authorization: (
        "AWS4-HMAC-SHA256 " +
        `Credential=${accessKeyId}/${credentialScope}, ` +
        `SignedHeaders=${signedHeaders}, Signature=${signature}`
      ),
      "x-amz-content-sha256": payloadHash,
      "x-amz-date": amzDate,
    },
  };
}

function publicOrSignedGetUrl(config, objectPath) {
  if (config.publicBaseUrl) {
    return `${config.publicBaseUrl.replace(/\/+$/, "")}/${encodePath(objectPath)}`;
  }

  const endpointUrl = new URL(config.endpoint);
  const canonicalUri = `${endpointUrl.pathname.replace(/\/+$/, "")}/${rfc3986(config.bucket)}/${encodePath(objectPath)}`;
  return `${endpointUrl.origin}${canonicalUri}`;
}

function uploadIdFromXml(xml) {
  const match = String(xml || "").match(/<UploadId>([^<]+)<\/UploadId>/i);
  return match?.[1] || "";
}

function xmlEscape(value) {
  return String(value || "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&apos;");
}

async function s3Fetch(config, method, objectPath, { query, body } = {}) {
  const uploadRequest = signedS3Request({
    method,
    endpoint: config.endpoint,
    bucket: config.bucket,
    key: objectPath,
    region: config.region,
    accessKeyId: config.accessKeyId,
    secretAccessKey: config.secretAccessKey,
    query,
  });
  return fetch(uploadRequest.url, {
    method,
    headers: uploadRequest.headers,
    body,
    duplex: body ? "half" : undefined,
  });
}

async function initiateMultipartUpload(config, objectPath) {
  const response = await s3Fetch(config, "POST", objectPath, {
    query: { uploads: "" },
  });
  const bodyText = await response.text().catch(() => "");
  if (!response.ok) {
    throw new Error(
      `S3 multipart upload init failed: ${response.status} ${response.statusText}${bodyText ? ` - ${bodyText}` : ""}`,
    );
  }

  const uploadId = uploadIdFromXml(bodyText);
  if (!uploadId) {
    throw new Error("S3 multipart upload init failed: missing upload ID.");
  }
  return uploadId;
}

async function uploadMultipartPart(config, objectPath, uploadId, partNumber, partBuffer) {
  const response = await s3Fetch(config, "PUT", objectPath, {
    query: { partNumber: String(partNumber), uploadId },
    body: partBuffer,
  });
  const bodyText = response.ok ? "" : await response.text().catch(() => "");
  if (!response.ok) {
    throw new Error(
      `S3 multipart upload part ${partNumber} failed: ${response.status} ${response.statusText}${bodyText ? ` - ${bodyText}` : ""}`,
    );
  }

  const etag = response.headers.get("etag");
  if (!etag) {
    throw new Error(`S3 multipart upload part ${partNumber} failed: missing ETag.`);
  }
  return { partNumber, etag };
}

async function completeMultipartUpload(config, objectPath, uploadId, parts) {
  const body = [
    "<CompleteMultipartUpload>",
    ...parts.map(
      (part) =>
        `<Part><PartNumber>${part.partNumber}</PartNumber><ETag>${xmlEscape(part.etag)}</ETag></Part>`,
    ),
    "</CompleteMultipartUpload>",
  ].join("");
  const response = await s3Fetch(config, "POST", objectPath, {
    query: { uploadId },
    body,
  });
  const bodyText = await response.text().catch(() => "");
  if (!response.ok) {
    throw new Error(
      `S3 multipart upload complete failed: ${response.status} ${response.statusText}${bodyText ? ` - ${bodyText}` : ""}`,
    );
  }
}

async function abortMultipartUpload(config, objectPath, uploadId) {
  if (!uploadId) return;
  try {
    await s3Fetch(config, "DELETE", objectPath, { query: { uploadId } });
  } catch (error) {
    console.warn("S3 multipart upload abort failed.", error);
  }
}

function concatChunks(chunks, totalBytes) {
  const result = new Uint8Array(totalBytes);
  let offset = 0;
  for (const chunk of chunks) {
    result.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return result;
}

async function uploadS3Multipart(config, objectPath, body) {
  const uploadId = await initiateMultipartUpload(config, objectPath);
  const reader = body.getReader();
  const parts = [];
  const pendingChunks = [];
  let pendingBytes = 0;
  let partNumber = 1;

  async function flushPart() {
    if (pendingBytes <= 0) return;
    const partBuffer = concatChunks(pendingChunks.splice(0), pendingBytes);
    pendingBytes = 0;
    parts.push(
      await uploadMultipartPart(config, objectPath, uploadId, partNumber, partBuffer),
    );
    partNumber += 1;
  }

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      if (!value?.byteLength) continue;
      pendingChunks.push(value);
      pendingBytes += value.byteLength;
      if (pendingBytes >= S3_MULTIPART_PART_BYTES) {
        await flushPart();
      }
    }
    await flushPart();
    if (parts.length === 0) {
      throw new Error("S3 multipart upload failed: empty upload body.");
    }
    await completeMultipartUpload(config, objectPath, uploadId, parts);
  } catch (error) {
    await abortMultipartUpload(config, objectPath, uploadId);
    throw error;
  }
}

export async function POST(request) {
  try {
    const session = await auth();
    if (!session || !session.user?.id) {
      return Response.json({ error: "Unauthorized" }, { status: 401 });
    }

    const contentLength = Number(request.headers.get("content-length") || 0);

    if (!request.body) {
      return Response.json({ error: "Missing file body." }, { status: 400 });
    }

    const filename = request.headers.get("x-filename") || "";
    const mimeType =
      request.headers.get("content-type") || "application/octet-stream";
    const ext = extensionFromMime(mimeType, filename);

    if (!SAFE_EXTENSIONS.has(ext)) {
      return Response.json(
        {
          error:
            "Supported large upload formats are .mp4, .mov, .m4v, .webm, .ply, .splat and .ksplat.",
        },
        { status: 400 },
      );
    }

    const objectPath = `${folderForExtension(ext)}/${new Date()
      .toISOString()
      .slice(0, 10)}/${crypto.randomUUID()}.${ext}`;

    const externalVideoStorage = VIDEO_SAFE_EXTENSIONS.has(ext)
      ? getExternalVideoStorageConfig()
      : null;

    if (externalVideoStorage) {
      await uploadS3Multipart(externalVideoStorage, objectPath, request.body);

      return Response.json({
        provider: "s3",
        bucket: externalVideoStorage.bucket,
        path: objectPath,
        objectPath,
        url: publicOrSignedGetUrl(externalVideoStorage, objectPath),
        mimeType,
        sizeBytes: contentLength || null,
      });
    }

    const config = getSupabaseStorageConfig();
    if (!config) {
      return Response.json(
        { error: "Storage is not configured." },
        { status: 500 },
      );
    }

    const uploadUrl = `${config.url}/storage/v1/object/${encodeURIComponent(
      config.bucket,
    )}/${objectPath.split("/").map(encodeURIComponent).join("/")}`;

    const response = await fetch(uploadUrl, {
      method: "POST",
      headers: {
        apikey: config.serviceRoleKey,
        Authorization: `Bearer ${config.serviceRoleKey}`,
        "Content-Type": mimeType,
        "x-upsert": "false",
      },
      body: request.body,
      duplex: "half",
    });

    if (!response.ok) {
      const bodyText = await response.text().catch(() => "");
      throw new Error(
        `Supabase large upload failed: ${response.status} ${response.statusText}${bodyText ? ` - ${bodyText}` : ""}`,
      );
    }

    const publicObjectPath = objectPath
      .split("/")
      .map(encodeURIComponent)
      .join("/");

    return Response.json({
      url: `${config.url}/storage/v1/object/public/${encodeURIComponent(
        config.bucket,
      )}/${publicObjectPath}`,
      mimeType,
      sizeBytes: contentLength || null,
    });
  } catch (error) {
    console.error("POST /api/upload/large error:", error);
    const message =
      error instanceof Error && error.message
        ? error.message
        : "Failed to upload large file.";
    return Response.json(
      { error: message },
      { status: 500 },
    );
  }
}
