import crypto from "node:crypto";
import { auth } from "@/auth";

const MAX_VIDEO_BYTES = 750 * 1024 * 1024;

const VIDEO_EXTENSIONS = {
  "video/mp4": "mp4",
  "video/quicktime": "mov",
  "video/x-m4v": "m4v",
  "video/m4v": "m4v",
};

const SAFE_EXTENSIONS = new Set(["mp4", "mov", "m4v"]);

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
    ""
  );
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
    if (contentLength && contentLength > MAX_VIDEO_BYTES) {
      return Response.json(
        { error: "Video is too large. Please upload a file under 750 MB." },
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
        { error: "Supported video formats are .mp4, .mov and .m4v." },
        { status: 400 },
      );
    }

    const objectPath = `video-uploads/${new Date()
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
    });
  } catch (error) {
    console.error("POST /api/upload/large error:", error);
    return Response.json(
      { error: "Failed to upload large video." },
      { status: 500 },
    );
  }
}
