import { openAiChatJson } from "../utils/openai";
import { buildVisionAnalysisPrompt } from "../utils/prompts";
import { safeJsonParse } from "../utils/helpers";

export async function analyzeRoomPhoto({ openAiKey, photoUrl }) {
  const { system, user } = buildVisionAnalysisPrompt();

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
          { type: "image_url", image_url: photoUrl },
        ],
      },
    ],
    retries: 3,
  });

  if (parsed && typeof parsed === "object") {
    return parsed;
  }

  return { raw };
}
