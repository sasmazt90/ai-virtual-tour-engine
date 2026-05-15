import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";
import crypto from "node:crypto";
import { createClient } from "@supabase/supabase-js";

const STORAGE_DIR = path.resolve(process.cwd(), "storage", "uploads");
const PUBLIC_UPLOAD_BASE = "/uploads";

const MIME_EXTENSIONS = {
  "image/jpeg": "jpg",
  "image/jpg": "jpg",
  "image/png": "png",
  "image/webp": "webp",
  "image/gif": "gif",
  "application/pdf": "pdf",
  "model/vnd.ply": "ply",
  "application/octet-stream": "bin",
};

const SAFE_FILENAME_EXTENSIONS = new Set([
  "jpg",
  "jpeg",
  "png",
  "webp",
  "gif",
  "pdf",
  "ply",
  "splat",
  "ksplat",
]);

function extensionFromFilename(filename) {
  const raw = String(filename || "");
  const ext = raw.split(".").pop()?.toLowerCase().trim();
  return ext && SAFE_FILENAME_EXTENSIONS.has(ext) ? ext : "";
}

function extensionFromMime(mimeType, filename) {
  const filenameExt = extensionFromFilename(filename);
  if (filenameExt) return filenameExt;
  return MIME_EXTENSIONS[String(mimeType || "").toLowerCase()] || "bin";
}

function getSupabaseStorageConfig() {
  const url = process.env.SUPABASE_URL;
  const serviceRoleKey = process.env.SUPABASE_SERVICE_ROLE_KEY;
  const bucket = process.env.SUPABASE_STORAGE_BUCKET || "uploads";

  if (!url || !serviceRoleKey) return null;

  return {
    bucket,
    client: createClient(url, serviceRoleKey, {
      auth: {
        persistSession: false,
        autoRefreshToken: false,
      },
    }),
  };
}

function parseDataUri(dataUri) {
  const value = String(dataUri || "");
  const match = value.match(/^data:([^;,]+)?(;base64)?,(.*)$/s);
  if (!match) {
    return {
      mimeType: "application/octet-stream",
      buffer: Buffer.from(value, "base64"),
    };
  }

  const mimeType = match[1] || "application/octet-stream";
  const isBase64 = Boolean(match[2]);
  const payload = match[3] || "";
  return {
    mimeType,
    buffer: isBase64
      ? Buffer.from(payload, "base64")
      : Buffer.from(decodeURIComponent(payload), "utf8"),
  };
}

async function downloadRemoteUrl(url) {
  const parsed = new URL(url);
  if (!["http:", "https:"].includes(parsed.protocol)) {
    throw new Error("Only http(s) URLs can be uploaded.");
  }

  const response = await fetch(parsed.toString());
  if (!response.ok) {
    throw new Error(`Could not download remote file: ${response.status}`);
  }

  const arrayBuffer = await response.arrayBuffer();
  return {
    buffer: Buffer.from(arrayBuffer),
    mimeType:
      response.headers.get("content-type")?.split(";")[0] ||
      "application/octet-stream",
  };
}

async function persistLocalUpload({ data, mimeType, filename }) {
  await mkdir(STORAGE_DIR, { recursive: true });

  const ext = extensionFromMime(mimeType, filename);
  const storedFilename = `${crypto.randomUUID()}.${ext}`;
  const absolutePath = path.join(STORAGE_DIR, storedFilename);

  await writeFile(absolutePath, data);

  return {
    url: `${PUBLIC_UPLOAD_BASE}/${storedFilename}`,
    mimeType: mimeType || null,
  };
}

async function persistSupabaseUpload({ data, mimeType, filename }) {
  const config = getSupabaseStorageConfig();
  if (!config) return null;

  const ext = extensionFromMime(mimeType, filename);
  const objectPath = `${new Date().toISOString().slice(0, 10)}/${crypto.randomUUID()}.${ext}`;

  const { error } = await config.client.storage
    .from(config.bucket)
    .upload(objectPath, data, {
      contentType: mimeType || "application/octet-stream",
      upsert: false,
    });

  if (error) {
    throw new Error(`Supabase upload failed: ${error.message}`);
  }

  const { data: publicUrl } = config.client.storage
    .from(config.bucket)
    .getPublicUrl(objectPath);

  return {
    url: publicUrl.publicUrl,
    mimeType: mimeType || null,
  };
}

async function persistUpload({ data, mimeType, filename }) {
  const supabaseUpload = await persistSupabaseUpload({
    data,
    mimeType,
    filename,
  });
  if (supabaseUpload) return supabaseUpload;

  return persistLocalUpload({ data, mimeType, filename });
}

async function upload({ url, buffer, base64, mimeType, filename }) {
  if (buffer) {
    return persistUpload({
      data: Buffer.isBuffer(buffer) ? buffer : Buffer.from(buffer),
      mimeType: mimeType || "application/octet-stream",
      filename,
    });
  }

  if (base64) {
    const parsed = parseDataUri(base64);
    return persistUpload({
      data: parsed.buffer,
      mimeType: parsed.mimeType,
      filename,
    });
  }

  if (url) {
    const remote = await downloadRemoteUrl(url);
    return persistUpload({
      data: remote.buffer,
      mimeType: remote.mimeType,
      filename: filename || new URL(url).pathname,
    });
  }

  throw new Error("Missing upload input.");
}

export { upload };
export default upload;
