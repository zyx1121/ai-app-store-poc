import { useEffect, useRef, useState, type KeyboardEvent } from "react";
import { Send, Square } from "lucide-react";
import {
  listInstances,
  onInstanceUpdate,
  OLLAMA_BASE_URL,
  type Instance,
} from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { ScrollArea } from "@/components/ui/scroll-area";
import { cn } from "@/lib/utils";

type ChatMessage = { role: "user" | "assistant"; content: string };

function isRunningModel(instance: Instance): boolean {
  return instance.kind === "model" && instance.status === "running";
}

/** Strip `<think>...</think>` reasoning blocks (including an unclosed trailing one) from display. */
function stripThink(text: string): string {
  const closed = text.replace(/<think>[\s\S]*?<\/think>/g, "");
  const openIdx = closed.indexOf("<think>");
  return openIdx === -1 ? closed : closed.slice(0, openIdx);
}

export function Chat({ initialInstanceId }: { initialInstanceId?: string }) {
  const [instances, setInstances] = useState<Instance[]>([]);
  const [selectedId, setSelectedId] = useState<string | undefined>(initialInstanceId);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const abortRef = useRef<AbortController | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

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

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ block: "end" });
  }, [messages]);

  async function send() {
    const instance = instances.find((i) => i.id === selectedId);
    const text = input.trim();
    if (!instance || !instance.model_tag || !text || sending) return;

    const history = [...messages, { role: "user" as const, content: text }];
    setMessages([...history, { role: "assistant", content: "" }]);
    setInput("");
    setSending(true);

    const assistantIndex = history.length;
    const controller = new AbortController();
    abortRef.current = controller;
    let raw = "";

    try {
      const res = await fetch(`${OLLAMA_BASE_URL}/chat/completions`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: instance.model_tag,
          messages: history.map(({ role, content }) => ({ role, content })),
          stream: true,
        }),
        signal: controller.signal,
      });

      if (!res.ok || !res.body) throw new Error(`request failed (${res.status})`);

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";

        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed.startsWith("data:")) continue;
          const data = trimmed.slice("data:".length).trim();
          if (data === "" || data === "[DONE]") continue;

          try {
            const parsed = JSON.parse(data);
            const delta = parsed.choices?.[0]?.delta?.content;
            if (typeof delta === "string") {
              raw += delta;
              const display = stripThink(raw);
              setMessages((prev) => {
                const next = prev.slice();
                next[assistantIndex] = { role: "assistant", content: display };
                return next;
              });
            }
          } catch {
            // ignore malformed SSE chunk
          }
        }
      }
    } catch (e) {
      if ((e as Error).name !== "AbortError") {
        setMessages((prev) => {
          const next = prev.slice();
          const prior = next[assistantIndex]?.content ?? "";
          next[assistantIndex] = {
            role: "assistant",
            content: `${prior}\n[error: ${String(e)}]`,
          };
          return next;
        });
      }
    } finally {
      setSending(false);
      abortRef.current = null;
    }
  }

  function stop() {
    abortRef.current?.abort();
  }

  function handleKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      void send();
    }
  }

  return (
    <div className="flex h-full flex-col gap-4 p-6">
      <div className="flex items-center gap-2">
        <Select
          value={selectedId ?? null}
          onValueChange={(v) => setSelectedId(v ?? undefined)}
          items={Object.fromEntries(instances.map((i) => [i.id, i.display_name]))}
        >
          <SelectTrigger className="w-64">
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
          <p className="text-sm text-muted-foreground">
            No running models. Launch one from Browse.
          </p>
        )}
      </div>

      <ScrollArea className="flex-1 rounded-xl border border-border">
        <div className="flex flex-col gap-3 p-4">
          {messages.map((message, i) => (
            <div
              key={i}
              className={cn("flex", message.role === "user" ? "justify-end" : "justify-start")}
            >
              <div
                className={cn(
                  "max-w-[75%] whitespace-pre-wrap rounded-xl px-3 py-2 text-sm",
                  message.role === "user"
                    ? "bg-primary text-primary-foreground"
                    : "bg-muted text-foreground",
                )}
              >
                {message.content}
              </div>
            </div>
          ))}
          <div ref={bottomRef} />
        </div>
      </ScrollArea>

      <div className="flex items-end gap-2">
        <Textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Message the model..."
          disabled={!selectedId}
          className="flex-1"
        />
        {sending ? (
          <Button variant="outline" onClick={stop}>
            <Square /> Stop
          </Button>
        ) : (
          <Button onClick={() => void send()} disabled={!selectedId || !input.trim()}>
            <Send /> Send
          </Button>
        )}
      </div>
    </div>
  );
}
