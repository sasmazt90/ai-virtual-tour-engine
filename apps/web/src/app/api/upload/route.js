import { auth } from "@/auth";
import { upload } from "@/app/api/utils/upload";

export async function POST(request) {
  try {
    const session = await auth();
    if (!session || !session.user?.id) {
      return Response.json({ error: "Unauthorized" }, { status: 401 });
    }

    const contentType = request.headers.get("content-type") || "";

    if (contentType.includes("multipart/form-data")) {
      const formData = await request.formData();
      const file = formData.get("file");

      if (!file || typeof file.arrayBuffer !== "function") {
        return Response.json({ error: "Missing file" }, { status: 400 });
      }

      const result = await upload({
        buffer: Buffer.from(await file.arrayBuffer()),
        mimeType: file.type || "application/octet-stream",
      });

      return Response.json({
        url: result.url,
        mimeType: file.type || result.mimeType || null,
      });
    }

    if (!contentType.includes("application/json")) {
      return Response.json(
        { error: "Unsupported content type" },
        { status: 415 },
      );
    }

    const body = await request.json();
    const { base64, url } = body || {};

    if (!base64 && !url) {
      return Response.json({ error: "Missing base64 or url" }, { status: 400 });
    }

    const result = await upload({ base64, url });
    if (!result?.url) {
      return Response.json(
        { error: "Failed to upload image." },
        { status: 500 },
      );
    }

    return Response.json({
      url: result.url,
      mimeType: result.mimeType || null,
    });
  } catch (error) {
    console.error("POST /api/upload error:", error);

    // Best-effort: surface a clearer message for oversized requests.
    const msg = error instanceof Error ? error.message : String(error);
    const isTooLarge =
      msg.toLowerCase().includes("body") &&
      msg.toLowerCase().includes("too large");

    if (isTooLarge) {
      return Response.json(
        { error: "Upload failed: File too large." },
        { status: 413 },
      );
    }

    return Response.json({ error: "Failed to upload image." }, { status: 500 });
  }
}
