import { useEffect, useMemo, useState } from "react";
import { useChat } from "@ai-sdk/react";
import type { UIMessage } from "ai";
import { Eraser, MessageSquare } from "lucide-react";

import { listInstances, onInstanceUpdate, type Instance } from "@/lib/api";
import { CONTEXT_TOKENS, createChatTransport, estimateConversationTokens } from "@/lib/chat";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription } from "@/components/ui/alert";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Conversation,
  ConversationContent,
  ConversationEmptyState,
  ConversationScrollButton,
} from "@/components/ai-elements/conversation";
import { Message, MessageContent, MessageResponse } from "@/components/ai-elements/message";
import {
  PromptInput,
  PromptInputBody,
  PromptInputFooter,
  type PromptInputMessage,
  PromptInputSubmit,
  PromptInputTextarea,
  PromptInputTools,
} from "@/components/ai-elements/prompt-input";
import { Reasoning, ReasoningContent, ReasoningTrigger } from "@/components/ai-elements/reasoning";

function isRunningModel(instance: Instance): boolean {
  return instance.kind === "model" && instance.status === "running";
}

function formatTokens(n: number): string {
  return n >= 1000 ? `${(n / 1000).toFixed(1)}k` : String(n);
}

/** One message: reasoning (collapsed, consolidated) then the text parts as Markdown. */
function MessageParts({
  message,
  isLastMessage,
  isStreaming,
}: {
  message: UIMessage;
  isLastMessage: boolean;
  isStreaming: boolean;
}) {
  const reasoningParts = message.parts.filter((p) => p.type === "reasoning");
  const reasoningText = reasoningParts.map((p) => p.text).join("\n\n");
  const lastPart = message.parts.at(-1);
  const reasoningStreaming = isLastMessage && isStreaming && lastPart?.type === "reasoning";

  return (
    <>
      {reasoningParts.length > 0 && (
        <Reasoning className="w-full" isStreaming={reasoningStreaming}>
          <ReasoningTrigger />
          <ReasoningContent>{reasoningText}</ReasoningContent>
        </Reasoning>
      )}
      {message.parts.map((part, i) =>
        part.type === "text" ? (
          <MessageResponse key={`${message.id}-${i}`}>{part.text}</MessageResponse>
        ) : null,
      )}
    </>
  );
}

/** A conversation bound to one running model. Remounted (via `key`) when the model changes. */
function ChatSession({ instance }: { instance: Instance }) {
  const modelTag = instance.model_tag ?? "";
  const transport = useMemo(() => createChatTransport(modelTag), [modelTag]);
  const { messages, sendMessage, status, stop, setMessages, error, clearError } = useChat({
    id: instance.id,
    transport,
  });

  const isStreaming = status === "streaming";
  const busy = status === "submitted" || status === "streaming";
  const usedTokens = useMemo(
    () =>
      estimateConversationTokens(
        messages.flatMap((m) => m.parts.map((p) => ("text" in p ? String(p.text) : ""))),
      ),
    [messages],
  );

  function handleSubmit(message: PromptInputMessage) {
    const text = message.text.trim();
    if (!text || busy) return;
    void sendMessage({ text });
  }

  function clear() {
    stop();
    clearError();
    setMessages([]);
  }

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-3">
      <div className="flex items-center gap-2 text-sm text-muted-foreground">
        <Badge variant="outline">{instance.display_name}</Badge>
        <span>
          context {formatTokens(usedTokens)} / {formatTokens(CONTEXT_TOKENS)}
        </span>
        <span className="flex-1" />
        <Button variant="ghost" size="sm" onClick={clear} disabled={messages.length === 0 && !error}>
          <Eraser className="mr-1 size-4" /> Clear
        </Button>
      </div>

      <Conversation className="min-h-0 flex-1 rounded-xl border border-border">
        <ConversationContent>
          {messages.length === 0 ? (
            <ConversationEmptyState
              icon={<MessageSquare className="size-10" />}
              title="Start a conversation"
              description={`Messages go straight to ${instance.display_name} on this machine.`}
            />
          ) : (
            messages.map((message, index) => (
              <Message from={message.role} key={message.id}>
                <MessageContent>
                  <MessageParts
                    message={message}
                    isLastMessage={index === messages.length - 1}
                    isStreaming={isStreaming}
                  />
                </MessageContent>
              </Message>
            ))
          )}
        </ConversationContent>
        <ConversationScrollButton />
      </Conversation>

      {error && (
        <Alert variant="destructive">
          <AlertDescription>{error.message}</AlertDescription>
        </Alert>
      )}

      <PromptInput onSubmit={handleSubmit}>
        <PromptInputBody>
          <PromptInputTextarea placeholder="Message the model. Enter sends, Shift+Enter for a new line." />
        </PromptInputBody>
        <PromptInputFooter>
          <PromptInputTools />
          <PromptInputSubmit status={status} onStop={stop} />
        </PromptInputFooter>
      </PromptInput>
    </div>
  );
}

export function Chat({ initialInstanceId }: { initialInstanceId?: string }) {
  const [instances, setInstances] = useState<Instance[]>([]);
  const [selectedId, setSelectedId] = useState<string | undefined>(initialInstanceId);

  useEffect(() => {
    let cancelled = false;
    listInstances().then((res) => {
      if (!cancelled) setInstances(res.filter(isRunningModel));
    });
    let unlisten: (() => void) | undefined;
    onInstanceUpdate((instance) => {
      if (instance.kind !== "model") return;
      setInstances((prev) => {
        const withoutIt = prev.filter((i) => i.id !== instance.id);
        return instance.status === "running" ? [instance, ...withoutIt] : withoutIt;
      });
    }).then((fn) => {
      if (cancelled) fn();
      else unlisten = fn;
    });
    return () => {
      cancelled = true;
      unlisten?.();
    };
  }, []);

  useEffect(() => {
    if (selectedId && instances.some((i) => i.id === selectedId)) return;
    setSelectedId(instances[0]?.id);
  }, [instances, selectedId]);

  const selected = instances.find((i) => i.id === selectedId);

  return (
    <div className="flex h-full flex-col gap-3 p-6">
      <div className="flex items-center gap-2">
        <Select
          value={selectedId ?? null}
          onValueChange={(v) => setSelectedId(v ?? undefined)}
          items={Object.fromEntries(instances.map((i) => [i.id, i.display_name]))}
        >
          <SelectTrigger className="w-72">
            <SelectValue placeholder="Select a running model" />
          </SelectTrigger>
          <SelectContent>
            {instances.map((instance) => (
              <SelectItem key={instance.id} value={instance.id}>
                {instance.display_name}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        {instances.length === 0 && (
          <p className="text-sm text-muted-foreground">No running models. Launch one from Browse.</p>
        )}
      </div>
      {selected && <ChatSession key={selected.id} instance={selected} />}
    </div>
  );
}
