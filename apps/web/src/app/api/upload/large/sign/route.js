import crypto from "node:crypto";
import { createClient } from "@supabase/supabase-js";
import { auth } from "@/auth";
import {
  AI_VIDEO_3D_MAX_BYTES,
  AI_VIDEO_3D_MAX_MB,
} from "@/app/api/utils/pricing";

const MAX_FILE_BYTES = AI_VIDEO_3D_MAX_BYTES;

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

function publicOrSignedGetUrl(config, objectPath) {
  if (config.publicBaseUrl) {
    return `${config.publicBaseUrl.replace(/\/+$/, "")}/${encodePath(objectPath)}`;
  }

  const endpointUrl = new URL(config.endpoint);
  const canonicalUri = `${endpointUrl.pathname.replace(/\/+$/, "")}/${rfc3986(config.bucket)}/${encodePath(objectPath)}`;
  return `${endpointUrl.origin}${canonicalUri}`;
}

function publicUrlFor({ url, bucket, objectPath }) {
  return `${url}/storage/v1/object/public/${encodeURIComponent(bucket)}/${objectPath
    .split("/")
    .map(encodeURIComponent)
    .join("/")}`;
}

export async function POST(request) {
  try {
    const session = await auth();
    if (!session || !session.user?.id) {
      return Response.json({ error: "Unauthorized" }, { status: 401 });
    }

    const config = getSupabaseStorageConfig();
    if (!config) {
      return Response.json(
        { error: "Storage is not configured." },
        { status: 500 },
      );
    }

    const body = await request.json().catch(() => ({}));
    const filename = body?.filename || "";
    const mimeType = body?.mimeType || "application/octet-stream";
    const sizeBytes = Number(body?.sizeBytes || 0);

    if (!Number.isFinite(sizeBytes) || sizeBytes <= 0) {
      return Response.json({ error: "Missing file size." }, { status: 400 });
    }

    if (Number.isFinite(MAX_FILE_BYTES) && sizeBytes > MAX_FILE_BYTES) {
      return Response.json(
        {
          error: `File is too large. Please upload ${AI_VIDEO_3D_MAX_MB} MB or less.`,
        },
        { status: 413 },
      );
    }

    const ext = extensionFromMime(mimeType, filename);
    if (!SAFE_EXTENSIONS.has(ext)) {
      return Response.json(
        {
          error:
            "Supported upload formats are .mp4, .mov, .m4v, .webm, .ply, .splat and .ksplat.",
        },
        { status: 400 },
      );
    }

    const externalVideoStorage = VIDEO_SAFE_EXTENSIONS.has(ext)
      ? getExternalVideoStorageConfig()
      : null;
    const objectPath = `${folderForExtension(ext)}/${new Date()
      .toISOString()
      .slice(0, 10)}/${crypto.randomUUID()}.${ext}`;

    if (externalVideoStorage) {
      const publicUrl = publicOrSignedGetUrl(externalVideoStorage, objectPath);

      return Response.json({
        provider: "s3",
        uploadMethod: "server-proxy",
        bucket: externalVideoStorage.bucket,
        path: objectPath,
        objectPath,
        publicUrl,
        url: publicUrl,
        mimeType,
        sizeBytes,
      });
    }

    const supabase = createClient(config.url, config.serviceRoleKey, {
      auth: {
        autoRefreshToken: false,
        persistSession: false,
      },
    });

    const { data, error } = await supabase.storage
      .from(config.bucket)
      .createSignedUploadUrl(objectPath);

    if (error || !data?.token) {
      console.error("Supabase signed upload URL failed:", error);
      return Response.json(
        {
          error:
            error?.message ||
            "Could not prepare upload. The upload will retry through the server.",
        },
        { status: 502 },
      );
    }

    return Response.json({
      bucket: config.bucket,
      path: objectPath,
      token: data.token,
      signedUrl: data.signedUrl || data.signedURL || null,
      supabaseUrl: config.url,
      publicUrl: publicUrlFor({
        url: config.url,
        bucket: config.bucket,
        objectPath,
      }),
      mimeType,
      sizeBytes,
    });
  } catch (error) {
    console.error("POST /api/upload/large/sign error:", error);
    return Response.json(
      { error: "Could not prepare upload." },
      { status: 500 },
    );
  }
}
