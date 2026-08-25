import { useEffect, useRef, useState } from "react";
import {
  onProvisionProgress,
  provisionRuntime,
  runtimeStatus,
  type ProvisionEvent,
  type RuntimeStatus,
} from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { CircleCheck, CircleX, TriangleAlert } from "lucide-react";
import { cn } from "@/lib/utils";

type Row = { label: string; ok: boolean; detail?: string };

function rows(status: RuntimeStatus): Row[] {
  return [
    {
      label: "WSL2",
      ok: status.wsl_installed,
      detail: status.wsl_version ?? undefined,
    },
    {
      label: "Runtime distro",
      ok: status.distro_present && status.distro_running,
    },
    { label: "Docker", ok: status.docker_ok },
    {
      label: "NVIDIA GPU",
      ok: status.gpu_ok,
      detail: status.gpu_name
        ? `${status.gpu_name}${status.vram_mb ? ` (${(status.vram_mb / 1024).toFixed(1)} GB VRAM)` : ""}`
        : undefined,
    },
    { label: "Ollama", ok: status.ollama_ok },
  ];
}

function eventTone(status: ProvisionEvent["status"]) {
  switch (status) {
    case "error":
      return "text-destructive";
    case "ok":
      return "text-foreground";
    case "start":
      return "mt-2 font-semibold text-foreground first:mt-0";
    default:
      return "pl-2 text-muted-foreground";
  }
}

export function Setup({
  status,
  onStatusChange,
}: {
  status: RuntimeStatus | null;
  onStatusChange: (status: RuntimeStatus) => void;
}) {
  const [provisioning, setProvisioning] = useState(false);
  const [events, setEvents] = useState<ProvisionEvent[]>([]);
  const [rechecking, setRechecking] = useState(false);
  const logRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    let unlisten: (() => void) | undefined;
    let cancelled = false;
    onProvisionProgress((event) => {
      setEvents((prev) => [...prev, event]);
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
    logRef.current?.scrollTo({ top: logRef.current.scrollHeight });
  }, [events]);

  async function install() {
    setProvisioning(true);
    setEvents([]);
    try {
      const result = await provisionRuntime();
      onStatusChange(result);
    } finally {
      setProvisioning(false);
    }
  }

  async function recheck() {
    setRechecking(true);
    try {
      onStatusChange(await runtimeStatus());
    } finally {
      setRechecking(false);
    }
  }

  return (
    <div className="mx-auto flex max-w-xl flex-col gap-4 p-6">
      <div>
        <h1 className="font-heading text-lg font-medium">Set up the runtime</h1>
        <p className="text-sm text-muted-foreground">
          AI App Store runs apps and models locally through WSL2, Docker and Ollama.
        </p>
      </div>

      <div className="flex flex-col gap-1.5 rounded-xl border border-border p-3">
        {status ? (
          rows(status).map((row) => (
            <div key={row.label} className="flex items-center gap-2 py-1 text-sm">
              {row.ok ? (
                <CircleCheck className="size-4 shrink-0 text-foreground" />
              ) : (
                <CircleX className="size-4 shrink-0 text-destructive" />
              )}
              <span className="flex-1">{row.label}</span>
              {row.detail && (
                <span className="text-xs text-muted-foreground">{row.detail}</span>
              )}
            </div>
          ))
        ) : (
          <p className="py-2 text-sm text-muted-foreground">Checking runtime status...</p>
        )}
      </div>

      {status?.reboot_required && (
        <Alert variant="destructive">
          <TriangleAlert className="size-4" />
          <AlertTitle>Reboot required</AlertTitle>
          <AlertDescription>
            Windows needs to restart to finish enabling WSL2. Reboot, then reopen AI App Store.
          </AlertDescription>
        </Alert>
      )}

      <div className="flex gap-2">
        <Button onClick={install} disabled={provisioning}>
          {provisioning ? "Installing..." : "Install runtime"}
        </Button>
        <Button variant="outline" onClick={recheck} disabled={rechecking || provisioning}>
          {rechecking ? "Checking..." : "Re-check"}
        </Button>
      </div>

      {(provisioning || events.length > 0) && (
        <div
          ref={logRef}
          className="max-h-64 overflow-y-auto rounded-lg border border-border bg-muted/50 p-3 font-mono text-xs"
        >
          {events.length === 0 ? (
            <p className="text-muted-foreground">Starting...</p>
          ) : (
            events.map((event, i) => (
              <div key={i} className={cn(eventTone(event.status))}>
                {event.status === "start" ? event.step : event.message}
              </div>
            ))
          )}
        </div>
      )}
    </div>
  );
}
