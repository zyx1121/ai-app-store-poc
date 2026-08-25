import { Fragment, useEffect, useRef, useState } from "react";
import {
  APP_VERSION,
  checkForUpdate,
  installUpdate,
  onProvisionProgress,
  provisionRuntime,
  relaunch,
  runtimeStatus,
  type AvailableUpdate,
  type Gpu,
  type HardwareProfile,
  type ProvisionEvent,
  type RuntimeStatus,
  type Vendor,
} from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Cpu, CircleCheck, CircleX, Microchip, TriangleAlert } from "lucide-react";
import { cn } from "@/lib/utils";

type Row = { label: string; ok: boolean; detail?: string };

const VENDOR_LABEL: Record<Vendor, string> = {
  nvidia: "NVIDIA",
  amd: "AMD",
  intel: "Intel",
  cpu: "CPU",
};

function runtimePlan(vendor: Vendor): string[] {
  if (vendor === "nvidia") {
    return [
      "LLM: Ollama in WSL (CUDA)",
      "Speech: Speaches (CUDA)",
      "Images: ComfyUI (CUDA)",
      "Detection: Triton Inference Server (CUDA)",
      "Spaces: pull CUDA images",
    ];
  }
  return [
    "LLM: Ollama on Windows (ROCm / Vulkan)",
    "Speech: Speaches (CPU)",
    "Images: ComfyUI (CPU, slow)",
    "Detection: OpenVINO Model Server (CPU)",
    "Spaces: pull, or build locally for CPU",
  ];
}

function gpuVram(gpu: Gpu): string {
  return gpu.vram_mb ? `${(gpu.vram_mb / 1024).toFixed(1)} GB` : "-";
}

function HardwareCard({ hardware, vendor }: { hardware: HardwareProfile; vendor: Vendor }) {
  return (
    <div className="flex flex-col gap-3 rounded-xl border border-border p-3">
      <div className="flex items-center gap-2">
        <Cpu className="size-4 shrink-0 text-foreground" />
        <span className="flex-1 text-sm font-medium">Hardware</span>
        <Badge>{VENDOR_LABEL[vendor]}</Badge>
      </div>

      {hardware.gpus.length > 0 && (
        <div className="flex flex-col gap-1">
          <div className="grid grid-cols-[minmax(0,1fr)_auto_auto_auto] gap-x-3 gap-y-1 text-xs">
            <span className="text-muted-foreground">Name</span>
            <span className="text-muted-foreground">VRAM</span>
            <span className="text-muted-foreground"></span>
            <span className="text-muted-foreground">Driver</span>
            {hardware.gpus.map((gpu) => (
              <Fragment key={gpu.name}>
                <span className="truncate">{gpu.name}</span>
                <span>{gpuVram(gpu)}</span>
                <span>{gpu.integrated && <Badge variant="outline">integrated</Badge>}</span>
                <span className="text-muted-foreground">{gpu.driver ?? "-"}</span>
              </Fragment>
            ))}
          </div>
        </div>
      )}

      {hardware.npus.length > 0 && (
        <div className="flex flex-col gap-1.5 border-t border-border pt-2">
          {hardware.npus.map((npu) => (
            <div key={npu.name} className="flex items-center gap-2 text-sm">
              <Microchip className="size-4 shrink-0 text-muted-foreground" />
              <span>{npu.name}</span>
              <span className="text-xs text-muted-foreground">({npu.vendor})</span>
            </div>
          ))}
          <p className="text-xs text-muted-foreground">
            NPU detected; acceleration not used yet
          </p>
        </div>
      )}

      <div className="flex flex-col gap-1 border-t border-border pt-2 text-sm">
        <span className="font-medium">Runtime plan</span>
        <ul className="list-disc pl-5 text-xs text-muted-foreground">
          {runtimePlan(vendor).map((line) => (
            <li key={line}>{line}</li>
          ))}
        </ul>
      </div>

      <p className="text-xs text-muted-foreground">
        GPU visible to WSL2: {hardware.wsl_gpu ? "yes" : "no"}
      </p>
    </div>
  );
}

function UpdateCard() {
  const [checking, setChecking] = useState(false);
  const [checked, setChecked] = useState(false);
  const [update, setUpdate] = useState<AvailableUpdate | null>(null);
  const [checkError, setCheckError] = useState<string | null>(null);
  const [installing, setInstalling] = useState(false);
  const [installError, setInstallError] = useState<string | null>(null);
  const [progress, setProgress] = useState<{ downloaded: number; total: number | null }>({
    downloaded: 0,
    total: null,
  });

  async function handleCheck() {
    setChecking(true);
    setCheckError(null);
    try {
      const result = await checkForUpdate();
      setUpdate(result);
      setChecked(true);
    } catch (e) {
      setCheckError(String(e));
    } finally {
      setChecking(false);
    }
  }

  async function handleInstall() {
    setInstalling(true);
    setInstallError(null);
    setProgress({ downloaded: 0, total: null });
    try {
      await installUpdate((downloaded, total) => setProgress({ downloaded, total }));
      await relaunch();
    } catch (e) {
      setInstallError(String(e));
    } finally {
      setInstalling(false);
    }
  }

  const progressPct =
    progress.total && progress.total > 0
      ? Math.min(100, Math.round((progress.downloaded / progress.total) * 100))
      : null;

  return (
    <div className="flex flex-col gap-3 rounded-xl border border-border p-3">
      <div className="flex items-center gap-2">
        <span className="flex-1 text-sm font-medium">Update</span>
        <span className="text-xs text-muted-foreground">v{APP_VERSION}</span>
      </div>

      {checkError && (
        <Alert variant="destructive">
          <AlertTitle>Could not check for updates</AlertTitle>
          <AlertDescription>{checkError}</AlertDescription>
        </Alert>
      )}

      {checked && !update && !checkError && (
        <p className="text-sm text-muted-foreground">Up to date.</p>
      )}

      {update && (
        <div className="flex flex-col gap-1 text-sm">
          <span>
            Version <span className="font-medium">{update.version}</span>
            {update.date && <span className="text-xs text-muted-foreground"> · {update.date}</span>}
          </span>
          {update.notes && (
            <p className="whitespace-pre-wrap text-xs text-muted-foreground">{update.notes}</p>
          )}
        </div>
      )}

      {installError && (
        <Alert variant="destructive">
          <AlertTitle>Could not install update</AlertTitle>
          <AlertDescription>{installError}</AlertDescription>
        </Alert>
      )}

      {installing && (
        <Progress value={progressPct}>
          <span className="text-xs text-muted-foreground">
            {progress.total
              ? `${(progress.downloaded / 1024 / 1024).toFixed(1)} / ${(progress.total / 1024 / 1024).toFixed(1)} MB`
              : `${(progress.downloaded / 1024 / 1024).toFixed(1)} MB`}
          </span>
        </Progress>
      )}

      <div className="flex gap-2">
        <Button variant="outline" onClick={handleCheck} disabled={checking || installing}>
          {checking ? "Checking..." : "Check for updates"}
        </Button>
        {update && (
          <Button onClick={handleInstall} disabled={installing}>
            {installing ? "Installing..." : "Install and restart"}
          </Button>
        )}
      </div>
    </div>
  );
}

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

      {status && <HardwareCard hardware={status.hardware} vendor={status.vendor} />}

      <UpdateCard />

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
