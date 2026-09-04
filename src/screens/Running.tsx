import { useEffect, useState } from "react";
import { ChevronDown } from "lucide-react";
import {
  launchSpace,
  listInstances,
  onInstanceUpdate,
  openUrl,
  removeInstance,
  stopInstance,
  type Instance,
} from "@/lib/api";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { StatusBadge } from "@/components/StatusBadge";
import { LogTail } from "@/components/LogTail";
import { GpuMemoryCard } from "@/components/GpuMemoryCard";

const LIVE_STATUSES: Instance["status"][] = ["pulling", "building", "starting", "running", "error"];

/**
 * Merge an update into the list. A relaunch of the same repo starts a fresh
 * instance with a new id; drop any stopped card for that repo so the old one
 * does not pile up next to the new run.
 */
function upsert(list: Instance[], next: Instance): Instance[] {
  const withoutStaleStopped = list.filter(
    (i) =>
      i.id === next.id ||
      !(i.repo === next.repo && i.kind === next.kind && i.status === "stopped" && next.status !== "stopped"),
  );
  const idx = withoutStaleStopped.findIndex((i) => i.id === next.id);
  if (idx === -1) return [next, ...withoutStaleStopped];
  const copy = withoutStaleStopped.slice();
  copy[idx] = next;
  return copy;
}

/** `NAME=value` per line, one env var each; parsing is forgiving (blank lines,
 * no `=`, are just skipped) since this is a small manual retry form, not a form
 * with per-field validation (#57). */
function parseEnvLines(text: string): Record<string, string> {
  const env: Record<string, string> = {};
  for (const line of text.split("\n")) {
    const idx = line.indexOf("=");
    if (idx <= 0) continue;
    const name = line.slice(0, idx).trim();
    const value = line.slice(idx + 1).trim();
    if (name) env[name] = value;
  }
  return env;
}

/** A Space that failed to launch (missing secret, gated model without a
 * token) can be retried here with corrected values, without going back to
 * the Store (#57). */
function EnvRetry({ onRetry }: { onRetry: (env: Record<string, string>) => void }) {
  const [text, setText] = useState("");
  const [retrying, setRetrying] = useState(false);

  async function submit() {
    setRetrying(true);
    try {
      await onRetry(parseEnvLines(text));
    } finally {
      setRetrying(false);
    }
  }

  return (
    <div className="flex flex-col gap-1.5 rounded-lg border border-border p-2">
      <label className="text-xs text-muted-foreground">
        Secrets this app reads from its environment, one NAME=value per line
      </label>
      <Textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        rows={2}
        className="font-mono text-xs"
        placeholder="tryon_url=https://..."
      />
      <Button size="sm" variant="outline" onClick={submit} disabled={retrying} className="self-start">
        {retrying ? "Retrying..." : "Retry with these values"}
      </Button>
    </div>
  );
}

function InstanceCard({
  instance,
  onOpenChat,
  onStop,
  onRemove,
  onRetry,
}: {
  instance: Instance;
  onOpenChat: (instanceId: string) => void;
  onStop: (id: string) => void;
  onRemove: (id: string) => void;
  onRetry: (repo: string, env: Record<string, string>) => void;
}) {
  const canOpen = instance.kind === "space" && instance.status === "running" && instance.url;
  const canChat = instance.kind === "model" && instance.status === "running";
  const canStop = instance.status !== "stopped";
  const canRemove = instance.status === "stopped" || instance.status === "error";
  const canRetry = instance.kind === "space" && instance.status === "error";

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Badge variant="outline">{instance.kind}</Badge>
          <span className="truncate">{instance.display_name}</span>
        </CardTitle>
        <CardDescription>{instance.repo}</CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col gap-2">
        <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
          <StatusBadge status={instance.status} />
          {instance.local_build && <Badge variant="outline">Built locally</Badge>}
          {instance.port && <span>port {instance.port}</span>}
          {instance.url && <span className="truncate">{instance.url}</span>}
          {instance.status === "pulling" && instance.pull_size_mb !== null && (
            <span>{(instance.pull_size_mb / 1024).toFixed(1)} GB image</span>
          )}
          {instance.status === "pulling" && instance.progress_pct !== null && (
            <span className="flex min-w-24 flex-1 items-center gap-1.5">
              <span className="h-1 flex-1 overflow-hidden rounded-full bg-muted">
                <span
                  className="block h-full rounded-full bg-primary transition-[width]"
                  style={{ width: `${instance.progress_pct}%` }}
                />
              </span>
              {instance.progress_pct}%
            </span>
          )}
        </div>
        {instance.error && <p className="text-xs text-destructive">{instance.error}</p>}
        <LogTail lines={instance.log_tail} />
        {canRetry && <EnvRetry onRetry={(env) => onRetry(instance.repo, env)} />}
      </CardContent>
      <CardFooter className="flex gap-2">
        {canOpen && (
          <Button variant="outline" onClick={() => openUrl(instance.url!)}>
            Open
          </Button>
        )}
        {canChat && <Button onClick={() => onOpenChat(instance.id)}>Chat</Button>}
        {canStop && (
          <Button variant="outline" onClick={() => onStop(instance.id)}>
            Stop
          </Button>
        )}
        {canRemove && (
          <Button variant="destructive" onClick={() => onRemove(instance.id)}>
            Remove
          </Button>
        )}
      </CardFooter>
    </Card>
  );
}

export function Running({ onOpenChat }: { onOpenChat: (instanceId: string) => void }) {
  const [instances, setInstances] = useState<Instance[]>([]);

  useEffect(() => {
    let cancelled = false;
    listInstances().then((res) => {
      if (!cancelled) setInstances(res);
    });

    let unlisten: (() => void) | undefined;
    onInstanceUpdate((instance) => {
      setInstances((prev) => upsert(prev, instance));
    }).then((fn) => {
      if (cancelled) fn();
      else unlisten = fn;
    });

    return () => {
      cancelled = true;
      unlisten?.();
    };
  }, []);

  async function handleStop(id: string) {
    await stopInstance(id);
  }

  async function handleRemove(id: string) {
    await removeInstance(id);
    setInstances((prev) => prev.filter((i) => i.id !== id));
  }

  async function handleRemoveAll(ids: string[]) {
    await Promise.all(ids.map((id) => removeInstance(id)));
    setInstances((prev) => prev.filter((i) => !ids.includes(i.id)));
  }

  /** Relaunches the same repo with the env values the user just supplied
   * (#57); the instance update event refreshes this card as usual. */
  async function handleRetry(repo: string, env: Record<string, string>) {
    await launchSpace(repo, env);
  }

  if (instances.length === 0) {
    return (
      <div className="flex flex-col gap-3 p-6">
        <GpuMemoryCard />
        <p className="text-sm text-muted-foreground">
          No instances yet. Launch an app or model from the Store.
        </p>
      </div>
    );
  }

  const live = instances.filter((i) => LIVE_STATUSES.includes(i.status));
  const stopped = instances.filter((i) => i.status === "stopped");

  return (
    <div className="flex flex-col gap-3 p-6">
      <GpuMemoryCard />
      {live.length === 0 && (
        <p className="text-sm text-muted-foreground">
          No active instances. Launch an app or model from the Store.
        </p>
      )}
      {live.map((instance) => (
        <InstanceCard
          key={instance.id}
          instance={instance}
          onOpenChat={onOpenChat}
          onStop={handleStop}
          onRemove={handleRemove}
          onRetry={handleRetry}
        />
      ))}

      {stopped.length > 0 && (
        <Collapsible className="flex flex-col gap-3 rounded-xl border border-border p-3">
          <div className="flex items-center gap-2">
            <CollapsibleTrigger
              render={
                <button
                  type="button"
                  className="group flex flex-1 items-center gap-2 text-left text-sm font-medium"
                />
              }
            >
              <ChevronDown className="size-4 shrink-0 transition-transform group-data-[panel-open]:rotate-180" />
              Recent ({stopped.length})
            </CollapsibleTrigger>
            <Button
              variant="outline"
              size="sm"
              onClick={() => handleRemoveAll(stopped.map((i) => i.id))}
            >
              Remove all
            </Button>
          </div>
          <CollapsibleContent className="flex flex-col gap-3">
            {stopped.map((instance) => (
              <InstanceCard
                key={instance.id}
                instance={instance}
                onOpenChat={onOpenChat}
                onStop={handleStop}
                onRemove={handleRemove}
                onRetry={handleRetry}
              />
            ))}
          </CollapsibleContent>
        </Collapsible>
      )}
    </div>
  );
}
