import { auth } from "@/auth";
import { upload } from "@/app/api/utils/upload";
import { editImageWithOpenAI } from "@/app/api/ai/staging/create/utils/openai";

function safeTrim(x) {
  return typeof x === "string" ? x.trim() : "";
}

export async function POST(request) {
  try {
    const session = await auth();
    if (!session || !session.user?.id) {
      return Response.json({ error: "Unauthorized" }, { status: 401 });
    }

    const body = await request.json().catch(() => ({}));
    const imageUrl = safeTrim(body?.imageUrl);

    if (!imageUrl) {
      return Response.json({ error: "imageUrl is required" }, { status: 400 });
    }

    const openAiKey = process.env.OPEN_AI_API_KEY;
    if (!openAiKey) {
      return Response.json(
        {
          error:
            "OPEN_AI_API_KEY is not set. Add it in Secrets to enable furniture cleanup.",
        },
        { status: 500 },
      );
    }

    // Goal: make reference images easier for the analyzer + verifier.
    // This does NOT guarantee the staging will include the item perfectly,
    // but it reduces failures caused by messy backgrounds / tiny items.
    const prompt =
      "You are editing a product photo for interior design reference. " +
      "Remove the background completely and keep ONLY the main furniture/decor item. " +
      "Center the item, keep its exact colors/materials, do not change its design. " +
      "Output a clean PNG with a transparent background. " +
      "If transparency is not possible, use a pure white background. " +
      "Do not add any text or watermark.";

    const edited = await editImageWithOpenAI({
      openAiKey,
      prompt,
      imageUrls: [imageUrl],
      retries: 2,
    });

    const uploaded =
      edited?.kind === "b64_json"
        ? await upload({ base64: `data:image/png;base64,${edited.b64_json}` })
        : await upload({ url: edited?.url });

    if (uploaded?.error) {
      return Response.json({ error: uploaded.error }, { status: 500 });
    }

    return Response.json({ url: uploaded?.url, modelUsed: edited?.modelUsed });
  } catch (error) {
    console.error("POST /api/ai/furniture/preprocess error:", error);
    return Response.json(
      { error: error?.message || "Internal Server Error" },
      { status: 500 },
    );
  }
}
