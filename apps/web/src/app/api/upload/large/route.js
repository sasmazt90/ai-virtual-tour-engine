import crypto from "node:crypto";
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

    const contentLength = Number(request.headers.get("content-length") || 0);
    if (
      contentLength &&
      Number.isFinite(MAX_FILE_BYTES) &&
      contentLength > MAX_FILE_BYTES
    ) {
      return Response.json(
        {
          error: `File is too large. Please upload a file under ${AI_VIDEO_3D_MAX_MB} MB.`,
        },
        { status: 413 },
      );
    }

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
            "Supported large upload formats are .mp4, .mov, .m4v, .ply, .splat and .ksplat.",
        },
        { status: 400 },
      );
    }

    const objectPath = `${folderForExtension(ext)}/${new Date()
      .toISOString()
      .slice(0, 10)}/${crypto.randomUUID()}.${ext}`;

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
