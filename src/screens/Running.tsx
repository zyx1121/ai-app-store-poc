import { useEffect, useState } from "react";
import {
  listInstances,
  onInstanceUpdate,
  openUrl,
  removeInstance,
  stopInstance,
  type Instance,
} from "@/lib/api";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { StatusBadge } from "@/components/StatusBadge";
import { LogTail } from "@/components/LogTail";

function upsert(list: Instance[], next: Instance): Instance[] {
  const idx = list.findIndex((i) => i.id === next.id);
  if (idx === -1) return [next, ...list];
  const copy = list.slice();
  copy[idx] = next;
  return copy;
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

  if (instances.length === 0) {
    return (
      <div className="p-6">
        <p className="text-sm text-muted-foreground">
          No instances yet. Launch an app or model from Browse.
        </p>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-3 p-6">
      {instances.map((instance) => {
        const canOpen = instance.kind === "space" && instance.status === "running" && instance.url;
        const canChat = instance.kind === "model" && instance.status === "running";
        const canStop = instance.status !== "stopped";
        const canRemove = instance.status === "stopped" || instance.status === "error";

        return (
          <Card key={instance.id}>
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
                {instance.port && <span>port {instance.port}</span>}
                {instance.url && <span className="truncate">{instance.url}</span>}
              </div>
              {instance.error && <p className="text-xs text-destructive">{instance.error}</p>}
              <LogTail lines={instance.log_tail} />
            </CardContent>
            <CardFooter className="flex gap-2">
              {canOpen && (
                <Button variant="outline" onClick={() => openUrl(instance.url!)}>
                  Open
                </Button>
              )}
              {canChat && <Button onClick={() => onOpenChat(instance.id)}>Chat</Button>}
              {canStop && (
                <Button variant="outline" onClick={() => handleStop(instance.id)}>
                  Stop
                </Button>
              )}
              {canRemove && (
                <Button variant="destructive" onClick={() => handleRemove(instance.id)}>
                  Remove
                </Button>
              )}
            </CardFooter>
          </Card>
        );
      })}
    </div>
  );
}
