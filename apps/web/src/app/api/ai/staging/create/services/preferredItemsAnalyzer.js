import { openAiChatJson } from "../utils/openai";
import { buildPreferredItemAnalysisPrompt } from "../utils/prompts";

export async function analyzePreferredItemImage({
  openAiKey,
  imageUrl,
  index,
}) {
  const { system, user } = buildPreferredItemAnalysisPrompt();

  const { parsed, raw } = await openAiChatJson({
    openAiKey,
    model: "gpt-4o-mini",
    messages: [
      { role: "system", content: system },
      {
        role: "user",
        content: [
          { type: "text", text: user },
          // IMPORTANT: pass raw value; openAiChatJson normalizes image blocks
          { type: "image_url", image_url: imageUrl },
        ],
      },
    ],
    retries: 2,
  });

  if (parsed && typeof parsed === "object") {
    return { ...parsed, index };
  }

  throw new Error(
    `We couldn't understand one of the preferred item images (item #${index + 1}). Please try a clearer photo of the item.`,
  );
}

export async function analyzeAllPreferredItems({
  openAiKey,
  preferredItemUrls,
  onHeartbeat,
}) {
  if (!preferredItemUrls || preferredItemUrls.length === 0) {
    return null;
  }

  const results = [];
  for (let i = 0; i < preferredItemUrls.length; i++) {
    try {
      if (typeof onHeartbeat === "function") {
        await onHeartbeat(i);
      }
    } catch (e) {
      // Heartbeat is best-effort; never fail the job because of it.
      console.error("preferredItemsAnalyzer heartbeat error", e);
    }

    const url = preferredItemUrls[i];
    const analysis = await analyzePreferredItemImage({
      openAiKey,
      imageUrl: url,
      index: i,
    });
    results.push(analysis);
  }

  return results;
}
