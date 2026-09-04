import { Fragment, useEffect, useRef, useState } from "react";
import {
  APP_VERSION,
  checkForUpdate,
  hfTokenStatus,
  installUpdate,
  onProvisionProgress,
  provisionRuntime,
  relaunch,
  runtimeStatus,
  setHfToken,
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
import { Input } from "@/components/ui/input";
import { Progress } from "@/components/ui/progress";
import { StorageCard } from "@/components/StorageCard";
import { Cpu, CircleCheck, CircleX, Microchip, Minus, TriangleAlert } from "lucide-react";
import { cn } from "@/lib/utils";

type Row = { label: string; ok: boolean; detail?: string; /** not needed on this vendor */ na?: boolean };

const VENDOR_LABEL: Record<Vendor, string> = {
  nvidia: "NVIDIA",
  amd: "AMD",
  intel: "Intel",
  cpu: "CPU",
};

function runtimePlan(vendor: Vendor): string[] {
  switch (vendor) {
    case "nvidia":
      return [
        "Chat (language models): Ollama in WSL2, CUDA",
        "Speech to text, text to speech: Speaches, CUDA",
        "Text to image, image to image: ComfyUI, CUDA",
        "Object detection: Triton Inference Server, CUDA",
        "Apps (Hugging Face Spaces): prebuilt CUDA images",
      ];
    case "amd":
      return [
        "Chat (language models): Ollama on Windows, ROCm with Vulkan fallback",
        "Speech to text: whisper.cpp on Windows, Vulkan",
        "Text to speech: Speaches, CPU",
        "Text to image, image to image: ComfyUI portable on Windows, ROCm",
        "Object detection: OpenVINO Model Server, CPU",
        "Apps (Hugging Face Spaces): prebuilt images, or build locally for CPU",
      ];
    case "intel":
      return [
        "Chat (language models): Ollama on Windows, Vulkan",
        "Speech to text: whisper.cpp on Windows, Vulkan",
        "Text to speech: Speaches, CPU",
        "Text to image, image to image: ComfyUI portable on Windows, XPU",
        "Object detection: OpenVINO Model Server, CPU (NPU when present)",
        "Apps (Hugging Face Spaces): prebuilt images, or build locally for CPU",
      ];
    case "cpu":
      return [
        "Chat (language models): Ollama on Windows, CPU",
        "Speech to text, text to speech: Speaches, CPU",
        "Text to image, image to image: ComfyUI, CPU (slow)",
        "Object detection: OpenVINO Model Server, CPU",
        "Apps (Hugging Face Spaces): prebuilt images, or build locally for CPU",
      ];
  }
}

const gb = (mb: number) => `${(mb / 1024).toFixed(1)} GB`;

/** Discrete: the VRAM. Unified: the shared budget verdicts use, with the carve-out the driver reports. */
function gpuMemory(gpu: Gpu): string {
  if (gpu.memory_model === "unified") {
    const budget = gpu.effective_memory_mb ? `${gb(gpu.effective_memory_mb)} shared` : "shared";
    return gpu.vram_mb ? `${budget} (${gb(gpu.vram_mb)} reserved)` : budget;
  }
  return gpu.vram_mb ? gb(gpu.vram_mb) : "-";
}

function HardwareCard({
  hardware,
  vendor,
  forced,
}: {
  hardware: HardwareProfile;
  vendor: Vendor;
  forced: boolean;
}) {
  return (
    <div className="flex flex-col gap-3 rounded-xl border border-border p-3">
      <div className="flex items-center gap-2">
        <Cpu className="size-4 shrink-0 text-foreground" />
        <span className="flex-1 text-sm font-medium">Hardware</span>
        {forced && <Badge variant="outline">forced via AIAS_VENDOR</Badge>}
        <Badge>{VENDOR_LABEL[vendor]}</Badge>
      </div>

      {hardware.gpus.length > 0 && (
        <div className="flex flex-col gap-1">
          <div className="grid grid-cols-[minmax(0,1fr)_auto_auto_auto] gap-x-3 gap-y-1 text-xs">
            <span className="text-muted-foreground">Name</span>
            <span className="text-muted-foreground">Memory</span>
            <span className="text-muted-foreground"></span>
            <span className="text-muted-foreground">Driver</span>
            {hardware.gpus.map((gpu) => (
              <Fragment key={gpu.name}>
                <span className="break-words" title={gpu.name}>
                  {gpu.name}
                </span>
                <span>{gpuMemory(gpu)}</span>
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
              <span className="text-xs text-muted-foreground">
                ({npu.vendor}
                {npu.effective_memory_mb ? `, ${gb(npu.effective_memory_mb)} shared` : ""})
              </span>
            </div>
          ))}
          <p className="text-xs text-muted-foreground">
            NPU detected. Used for object detection on Intel; chat, speech and image generation
            do not use it yet.
          </p>
        </div>
      )}

      <div className="flex flex-col gap-1 border-t border-border pt-2 text-sm">
        <span className="font-medium">What runs where on this machine</span>
        <ul className="list-disc pl-5 text-xs text-muted-foreground">
          {runtimePlan(vendor).map((line) => (
            <li key={line}>{line}</li>
          ))}
        </ul>
      </div>

      <p className="text-xs text-muted-foreground">
        GPU visible to WSL2: {hardware.wsl_gpu ? "yes" : "no"}
      </p>
      {hardware.primary_gpu?.memory_model === "unified" && hardware.primary_gpu.effective_memory_mb && (
        <p className="text-xs text-muted-foreground">
          Model budget: {gb(hardware.primary_gpu.effective_memory_mb)}
          {hardware.total_ram_mb ? ` (unified memory, half of ${gb(hardware.total_ram_mb)} RAM)` : ""}
          ; the reserved figure is the driver carve-out, not the ceiling
        </p>
      )}

      {(() => {
        const v = hardware.virtualization;
        const usable = v.hypervisor_present || (v.vt_supported && v.vt_firmware_enabled);
        if (usable) return null;
        const msg = !v.vt_supported
          ? "This CPU has no hardware virtualization (VT-x / AMD-V), which WSL2 requires."
          : "Virtualization is turned off in the UEFI firmware. Reboot into firmware setup (Del / F2 / F10), turn on Intel Virtualization Technology (VT-x) and VT-d, or AMD SVM, then save and reboot.";
        return (
          <Alert variant="destructive">
            <TriangleAlert />
            <AlertTitle>Virtualization must be enabled first</AlertTitle>
            <AlertDescription>{msg}</AlertDescription>
          </Alert>
        );
      })()}
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

/** Optional Hugging Face token (#60): unlocks gated models/Spaces and, once #57
 * lands its env editor, Spaces that read HF_TOKEN themselves. */
function TokenCard() {
  const [hasToken, setHasToken] = useState<boolean | null>(null);
  const [value, setValue] = useState("");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    hfTokenStatus()
      .then(setHasToken)
      .catch(() => setHasToken(false));
  }, []);

  async function save() {
    setSaving(true);
    setError(null);
    try {
      const trimmed = value.trim();
      await setHfToken(trimmed.length > 0 ? trimmed : null);
      setHasToken(trimmed.length > 0);
      setValue("");
    } catch (e) {
      setError(String(e));
    } finally {
      setSaving(false);
    }
  }

  async function clear() {
    setSaving(true);
    setError(null);
    try {
      await setHfToken(null);
      setHasToken(false);
      setValue("");
    } catch (e) {
      setError(String(e));
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className="flex flex-col gap-3 rounded-xl border border-border p-3">
      <div className="flex items-center gap-2">
        <span className="flex-1 text-sm font-medium">Hugging Face token</span>
        {hasToken !== null && <Badge variant={hasToken ? "default" : "outline"}>{hasToken ? "set" : "not set"}</Badge>}
      </div>
      <p className="text-xs text-muted-foreground">
        Optional. Needed for gated models and Spaces that require an HF_TOKEN secret.
      </p>

      {error && (
        <Alert variant="destructive">
          <AlertTitle>Could not save token</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      <div className="flex gap-2">
        <Input
          type="password"
          value={value}
          onChange={(e) => setValue(e.target.value)}
          placeholder={hasToken ? "Replace token" : "hf_..."}
          className="max-w-sm"
        />
        <Button onClick={save} disabled={saving || value.trim().length === 0}>
          {saving ? "Saving..." : "Save"}
        </Button>
        {hasToken && (
          <Button variant="outline" onClick={clear} disabled={saving}>
            Clear
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
      label: "WSL2 distro (ai-app-store)",
      ok: status.distro_present && status.distro_running,
    },
    {
      label: "Virtualization (UEFI)",
      ok: status.virtualization.hypervisor_present ||
        (status.virtualization.vt_supported && status.virtualization.vt_firmware_enabled),
      detail: status.virtualization.hypervisor_present
        ? "hypervisor running"
        : !status.virtualization.vt_supported
          ? "not supported by this CPU"
          : status.virtualization.vt_firmware_enabled
            ? "enabled"
            : "turned off in firmware",
    },
    { label: "Docker in WSL2", ok: status.docker_ok },
    {
      label: "GPU in WSL2",
      ok: status.gpu_ok,
      na: status.vendor !== "nvidia",
      detail:
        status.vendor !== "nvidia"
          ? "not needed: models run natively on Windows for this GPU"
          : status.gpu_name
            ? `${status.gpu_name}${status.vram_mb ? ` (${(status.vram_mb / 1024).toFixed(1)} GB VRAM)` : ""}`
            : undefined,
    },
    {
      label: status.vendor === "nvidia" ? "Ollama (WSL2)" : "Ollama (Windows)",
      ok: status.ollama_ok,
    },
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
  const [installError, setInstallError] = useState<string | null>(null);
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
    setInstallError(null);
    try {
      const result = await provisionRuntime();
      onStatusChange(result);
    } catch (e) {
      setInstallError(e instanceof Error ? e.message : String(e));
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
    <div className="mx-auto flex w-full max-w-3xl flex-col gap-4 p-6">
      <div>
        <h1 className="font-heading text-lg font-medium">Setup</h1>
        <p className="text-sm text-muted-foreground">
          Hardware, runtime, updates and storage. Apps and models run locally through WSL2, Docker
          and Ollama.
        </p>
      </div>

      {status && (
        <HardwareCard hardware={status.hardware} vendor={status.vendor} forced={status.vendor_forced} />
      )}

      <UpdateCard />

      <TokenCard />

      <div className="flex flex-col gap-1.5 rounded-xl border border-border p-3">
        {status ? (
          rows(status).map((row) => (
            <div key={row.label} className="flex items-center gap-2 py-1 text-sm">
              {row.ok ? (
                <CircleCheck className="size-4 shrink-0 text-foreground" />
              ) : row.na ? (
                <Minus className="size-4 shrink-0 text-muted-foreground" />
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

      {installError && (
        <Alert variant="destructive">
          <TriangleAlert className="size-4" />
          <AlertTitle>Could not install the runtime</AlertTitle>
          <AlertDescription>{installError}</AlertDescription>
        </Alert>
      )}

      <div className="flex gap-2">
        <Button onClick={install} disabled={provisioning}>
          {provisioning ? "Installing..." : installError ? "Retry" : "Install runtime"}
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

      <StorageCard />
    </div>
  );
}
