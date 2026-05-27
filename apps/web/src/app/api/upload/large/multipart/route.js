import crypto from "node:crypto";
import { auth } from "@/auth";

const SAFE_PREFIXES = ["video-uploads/", "3d-scans/"];

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
  };
}

function isSafeObjectPath(path) {
  const value = String(path || "");
  return (
    SAFE_PREFIXES.some((prefix) => value.startsWith(prefix)) &&
    !value.includes("..") &&
    !value.includes("\\")
  );
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

function normalizeMultipartEtag(value) {
  const trimmed = String(value || "").trim();
  const withoutWeakPrefix = trimmed.startsWith("W/") ? trimmed.slice(2) : trimmed;
  const unquoted = withoutWeakPrefix.replace(/^"+|"+$/g, "");
  return `"${unquoted}"`;
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

async function requireRequestContext(request) {
  const session = await auth();
  if (!session || !session.user?.id) {
    return { error: Response.json({ error: "Unauthorized" }, { status: 401 }) };
  }

  const config = getExternalVideoStorageConfig();
  if (!config) {
    return {
      error: Response.json(
        { error: "External video storage is not configured." },
        { status: 500 },
      ),
    };
  }

  const objectPath = request.headers.get("x-object-path") || "";
  if (!isSafeObjectPath(objectPath)) {
    return {
      error: Response.json({ error: "Invalid upload path." }, { status: 400 }),
    };
  }

  return { config, objectPath };
}

export async function POST(request) {
  const url = new URL(request.url);
  const action = url.searchParams.get("action");
  const context = await requireRequestContext(request);
  if (context.error) return context.error;

  const { config, objectPath } = context;

  try {
    if (action === "init") {
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

      return Response.json({ uploadId });
    }

    if (action === "part") {
      const uploadId = request.headers.get("x-upload-id") || "";
      const partNumber = Number(request.headers.get("x-part-number") || 0);

      if (!uploadId || !Number.isInteger(partNumber) || partNumber < 1) {
        return Response.json(
          { error: "Missing multipart upload part metadata." },
          { status: 400 },
        );
      }

      const response = await s3Fetch(config, "PUT", objectPath, {
        query: { partNumber: String(partNumber), uploadId },
        body: request.body,
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

      return Response.json({ partNumber, etag: normalizeMultipartEtag(etag) });
    }

    if (action === "complete") {
      const body = await request.json().catch(() => ({}));
      const uploadId = body?.uploadId || "";
      const parts = Array.isArray(body?.parts) ? body.parts : [];

      if (!uploadId || parts.length === 0) {
        return Response.json(
          { error: "Missing multipart completion metadata." },
          { status: 400 },
        );
      }

      const completeBody = [
        "<CompleteMultipartUpload>",
        ...parts
          .sort((a, b) => Number(a.partNumber) - Number(b.partNumber))
          .map(
            (part) =>
              `<Part><PartNumber>${Number(part.partNumber)}</PartNumber><ETag>${xmlEscape(part.etag)}</ETag></Part>`,
          ),
        "</CompleteMultipartUpload>",
      ].join("");

      const response = await s3Fetch(config, "POST", objectPath, {
        query: { uploadId },
        body: completeBody,
      });
      const bodyText = await response.text().catch(() => "");
      if (!response.ok) {
        throw new Error(
          `S3 multipart upload complete failed: ${response.status} ${response.statusText}${bodyText ? ` - ${bodyText}` : ""}`,
        );
      }

      return Response.json({ ok: true });
    }

    if (action === "abort") {
      const body = await request.json().catch(() => ({}));
      const uploadId = body?.uploadId || "";
      if (uploadId) {
        await s3Fetch(config, "DELETE", objectPath, { query: { uploadId } });
      }
      return Response.json({ ok: true });
    }

    return Response.json({ error: "Unsupported multipart action." }, { status: 400 });
  } catch (error) {
    console.error("POST /api/upload/large/multipart error:", error);
    return Response.json(
      {
        error:
          error instanceof Error && error.message
            ? error.message
            : "Multipart upload failed.",
      },
      { status: 500 },
    );
  }
}
