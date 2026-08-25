// Chat plumbing: the browser talks to Ollama directly through the AI SDK.
// No server route exists in a Tauri app, so a DirectChatTransport drives a
// ToolLoopAgent in-process and streams UI message chunks to `useChat`.

import { createOpenAICompatible } from "@ai-sdk/openai-compatible";
import { DirectChatTransport, ToolLoopAgent, type ModelMessage } from "ai";

import { OLLAMA_BASE_URL } from "@/lib/api";

/** Ollama is started with OLLAMA_CONTEXT_LENGTH=16384 (see provision.sh). */
export const CONTEXT_TOKENS = 16_384;
/** Leave room for the reply and the model's own thinking. */
const OUTPUT_RESERVE_TOKENS = 4_096;
const INPUT_BUDGET_TOKENS = CONTEXT_TOKENS - OUTPUT_RESERVE_TOKENS;

const ollama = createOpenAICompatible({
  name: "ollama",
  baseURL: OLLAMA_BASE_URL,
  apiKey: "ollama",
  includeUsage: true,
});

/** Rough token count without a tokenizer: CJK is ~1 token per char, Latin ~4 chars. */
export function estimateTokens(text: string): number {
  let cjk = 0;
  for (const ch of text) {
    if (/[　-鿿가-힯豈-﫿]/.test(ch)) cjk += 1;
  }
  return cjk + Math.ceil((text.length - cjk) / 4);
}

function messageText(m: ModelMessage): string {
  if (typeof m.content === "string") return m.content;
  return m.content
    .map((part) => ("text" in part && typeof part.text === "string" ? part.text : ""))
    .join("");
}

/**
 * Keep the most recent messages that fit the input budget. The first user
 * message is always kept so the model does not lose the task framing.
 */
export function trimToBudget(messages: ModelMessage[], budget = INPUT_BUDGET_TOKENS): ModelMessage[] {
  if (messages.length === 0) return messages;
  const first = messages[0];
  let used = estimateTokens(messageText(first));
  const kept: ModelMessage[] = [];
  for (let i = messages.length - 1; i >= 1; i--) {
    const t = estimateTokens(messageText(messages[i]));
    if (used + t > budget) break;
    used += t;
    kept.push(messages[i]);
  }
  kept.push(first);
  return kept.reverse();
}

/** Total estimated tokens of a UI message list, for the context indicator. */
export function estimateConversationTokens(texts: string[]): number {
  return texts.reduce((n, t) => n + estimateTokens(t), 0);
}

export function createChatTransport(modelTag: string) {
  const agent = new ToolLoopAgent({
    model: ollama(modelTag),
    instructions:
      "You are a helpful assistant running locally on the user's machine. Answer concisely.",
    prepareCall: ({ messages, ...rest }) => ({
      ...rest,
      messages: trimToBudget(messages ?? []),
    }),
  });
  return new DirectChatTransport({
    agent,
    sendReasoning: true,
    // Surface the real failure; the default swallows it as "An error occurred."
    onError: (error) => (error instanceof Error ? error.message : String(error)),
  });
}
