import { openAiChatJson } from "../utils/openai";
import { buildPreferredItemVerificationPrompt } from "../utils/prompts";

export async function verifyPreferredItemsInResult({
  openAiKey,
  generatedImageUrl,
  preferredItems,
  preferredItemUrls,
}) {
  const { system, user } = buildPreferredItemVerificationPrompt();

  const { parsed, raw } = await openAiChatJson({
    openAiKey,
    model: "gpt-4o-mini",
    messages: [
      { role: "system", content: system },
      {
        role: "user",
        content: [
          { type: "text", text: user },
          {
            type: "text",
            text: `Preferred item recognition JSON:\n${JSON.stringify(preferredItems)}`,
          },
          // IMPORTANT: pass raw values; openAiChatJson normalizes image blocks
          { type: "image_url", image_url: generatedImageUrl },
          ...preferredItemUrls.slice(0, 8).map((u) => ({
            type: "image_url",
            image_url: u,
          })),
        ],
      },
    ],
    retries: 2,
  });

  if (parsed && typeof parsed === "object") {
    return parsed;
  }

  return {
    overallPass: false,
    missing: ["Could not verify preferred items"],
    misplaced: [],
    notes: [raw ? String(raw).slice(0, 500) : ""],
  };
}
