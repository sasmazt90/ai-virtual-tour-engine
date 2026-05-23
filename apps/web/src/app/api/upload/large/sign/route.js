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
};

const MODEL_EXTENSIONS = {
  "model/vnd.ply": "ply",
  "application/octet-stream": "",
};

const SAFE_EXTENSIONS = new Set(["mp4", "mov", "m4v", "ply", "splat", "ksplat"]);
const VIDEO_SAFE_EXTENSIONS = new Set(["mp4", "mov", "m4v"]);
const MODEL_SAFE_EXTENSIONS = new Set(["ply", "splat", "ksplat"]);

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
            "Supported upload formats are .mp4, .mov, .m4v, .ply, .splat and .ksplat.",
        },
        { status: 400 },
      );
    }

    const objectPath = `${folderForExtension(ext)}/${new Date()
      .toISOString()
      .slice(0, 10)}/${crypto.randomUUID()}.${ext}`;

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
