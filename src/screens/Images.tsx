import { useEffect, useRef, useState } from "react";
import { nanoid } from "nanoid";
import {
  Ban,
  Copy,
  Dices,
  Download,
  ImagePlus,
  RefreshCw,
  Upload,
  Wand2,
  X,
} from "lucide-react";

import {
  COMFYUI_BASE_URL,
  installServiceModel,
  onServiceUpdate,
  serviceModels,
  serviceStatus,
  startService,
  stopService,
  type ServiceModel,
  type ServiceStatus,
} from "@/lib/api";
import { Button } from "@/components/ui/button";
import { useGpuGate } from "@/components/GpuGate";
import { Badge } from "@/components/ui/badge";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Textarea } from "@/components/ui/textarea";
import { Input } from "@/components/ui/input";
import { Spinner } from "@/components/ui/spinner";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { LogTail } from "@/components/LogTail";
import { ServiceStateBadge } from "@/components/ServiceStateBadge";
import { cn } from "@/lib/utils";

const SERVICE_ID = "comfyui";
const CHECKPOINT_MODEL_NAME = "sd-turbo";
const CHECKPOINT_FILE = "sd_turbo.safetensors";

const SIZE_OPTIONS = [
  { value: "512x512", label: "512 x 512", width: 512, height: 512 },
  { value: "640x384", label: "640 x 384", width: 640, height: 384 },
  { value: "384x640", label: "384 x 640", width: 384, height: 640 },
];

const DEFAULT_STRENGTH = 0.6;
const HISTORY_LIMIT = 12;

type SourceImage = {
  blob: Blob;
  url: string;
};

type ImageResult = {
  id: string;
  url: string;
  blob: Blob;
  prompt: string;
  negativePrompt: string;
  width: number;
  height: number;
  steps: number;
  seed: number;
  usedSource: boolean;
  strength: number | null;
};

type ComfyPromptResponse = {
  prompt_id: string;
  number: number;
  node_errors: Record<string, unknown>;
};

type ComfyHistoryEntry = {
  status: { status_str: string; completed: boolean };
  outputs: Record<string, { images?: { filename: string; subfolder: string; type: string }[] }>;
};

function randomSeed(): number {
  return Math.floor(Math.random() * 2 ** 31);
}

function nodeErrorSummary(errors: Record<string, unknown>): string {
  const parts = Object.entries(errors).map(([node, err]) => {
    if (err && typeof err === "object" && "errors" in err) {
      const inner = (err as { errors?: { message?: string }[] }).errors ?? [];
      return `node ${node}: ${inner.map((e) => e.message ?? "error").join(", ")}`;
    }
    return `node ${node}: error`;
  });
  return parts.join("; ");
}

export function Images() {
  const [status, setStatus] = useState<ServiceStatus | null>(null);
  const [serviceBusy, setServiceBusy] = useState(false);
  const { gate, dialog: gpuDialog } = useGpuGate();
  const [serviceError, setServiceError] = useState<string | null>(null);

  const [models, setModels] = useState<ServiceModel[]>([]);
  const [modelsError, setModelsError] = useState<string | null>(null);
  const [installing, setInstalling] = useState(false);

  const [checkpointListed, setCheckpointListed] = useState(false);
  const [checkpointChecked, setCheckpointChecked] = useState(false);

  const [prompt, setPrompt] = useState("a cozy reading nook, warm light");
  const [negativePrompt, setNegativePrompt] = useState("");
  const [size, setSize] = useState(SIZE_OPTIONS[0].value);
  const [steps, setSteps] = useState(2);
  const [seed, setSeed] = useState(() => randomSeed());
  const [strength, setStrength] = useState(DEFAULT_STRENGTH);

  const [sourceImage, setSourceImage] = useState<SourceImage | null>(null);

  const [generating, setGenerating] = useState(false);
  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  const [genError, setGenError] = useState<string | null>(null);

  const [result, setResult] = useState<ImageResult | null>(null);
  const [history, setHistory] = useState<ImageResult[]>([]);

  const clientIdRef = useRef(nanoid());
  const cancelledRef = useRef(false);
  const timerRef = useRef<number | null>(null);
  const sourceUrlRef = useRef<string | null>(null);
  const resultUrlsRef = useRef<Set<string>>(new Set());

  const running = status?.state === "running";
  const checkpointModel = models.find((m) => m.name === CHECKPOINT_MODEL_NAME);
  const modelInstalled = checkpointModel?.installed ?? false;
  const ready = running && modelInstalled && checkpointListed;

  useEffect(() => {
    let cancelled = false;
    serviceStatus(SERVICE_ID).then((s) => {
      if (!cancelled) setStatus(s);
    });
    let unlisten: (() => void) | undefined;
    onServiceUpdate((s) => {
      if (s.id !== SERVICE_ID) return;
      setStatus(s);
    }).then((fn) => {
      if (cancelled) fn();
      else unlisten = fn;
    });
    return () => {
      cancelled = true;
      unlisten?.();
    };
  }, []);

  async function refreshModels() {
    try {
      const list = await serviceModels(SERVICE_ID);
      setModels(list);
      setModelsError(null);
    } catch (err) {
      setModelsError(err instanceof Error ? err.message : String(err));
    }
  }

  async function refreshCheckpoint() {
    try {
      const res = await fetch(`${COMFYUI_BASE_URL}/object_info/CheckpointLoaderSimple`);
      if (!res.ok) throw new Error(`Failed to read object info (${res.status})`);
      const body = (await res.json()) as {
        CheckpointLoaderSimple?: { input?: { required?: { ckpt_name?: [string[]] } } };
      };
      const names = body.CheckpointLoaderSimple?.input?.required?.ckpt_name?.[0] ?? [];
      setCheckpointListed(names.includes(CHECKPOINT_FILE));
    } catch {
      setCheckpointListed(false);
    } finally {
      setCheckpointChecked(true);
    }
  }

  useEffect(() => {
    if (!running) {
      setCheckpointChecked(false);
      return;
    }
    void refreshModels();
    void refreshCheckpoint();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [running]);

  useEffect(() => {
    return () => {
      if (timerRef.current !== null) window.clearInterval(timerRef.current);
      if (sourceUrlRef.current) URL.revokeObjectURL(sourceUrlRef.current);
      resultUrlsRef.current.forEach((u) => URL.revokeObjectURL(u));
    };
  }, []);

  async function handleStart() {
    setServiceBusy(true);
    setServiceError(null);
    try {
      // Image generation needs the card to itself next to a chat LLM on 10 GB;
      // ask what must be unloaded first (issue #10).
      const result = await gate({ kind: "service", id: SERVICE_ID }, "ComfyUI");
      if (!result.proceed) return;
      const s = await startService(SERVICE_ID, result.leaseId);
      setStatus(s);
    } catch (err) {
      setServiceError(err instanceof Error ? err.message : String(err));
    } finally {
      setServiceBusy(false);
    }
  }

  async function handleStop() {
    setServiceBusy(true);
    setServiceError(null);
    try {
      await stopService(SERVICE_ID);
    } catch (err) {
      setServiceError(err instanceof Error ? err.message : String(err));
    } finally {
      setServiceBusy(false);
    }
  }

  async function handleRestart() {
    setServiceBusy(true);
    setServiceError(null);
    try {
      await stopService(SERVICE_ID);
      // Stopping just freed ComfyUI's own memory, but the start still needs a
      // valid lease (#70); re-plan rather than assume nothing else changed.
      const result = await gate({ kind: "service", id: SERVICE_ID }, "ComfyUI");
      if (!result.proceed) return;
      const s = await startService(SERVICE_ID, result.leaseId);
      setStatus(s);
    } catch (err) {
      setServiceError(err instanceof Error ? err.message : String(err));
    } finally {
      setServiceBusy(false);
    }
  }

  async function handleInstallModel(name: string) {
    setInstalling(true);
    setModelsError(null);
    try {
      await installServiceModel(SERVICE_ID, name);
      await refreshModels();
      await refreshCheckpoint();
    } catch (err) {
      setModelsError(err instanceof Error ? err.message : String(err));
    } finally {
      setInstalling(false);
    }
  }

  function handleFile(file: File) {
    setGenError(null);
    if (sourceUrlRef.current) URL.revokeObjectURL(sourceUrlRef.current);
    const url = URL.createObjectURL(file);
    sourceUrlRef.current = url;
    setSourceImage({ blob: file, url });
  }

  function handleFileInputChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (file) handleFile(file);
    e.target.value = "";
  }

  function handleDrop(e: React.DragEvent<HTMLDivElement>) {
    e.preventDefault();
    const file = e.dataTransfer.files?.[0];
    if (file) handleFile(file);
  }

  function handleDragOver(e: React.DragEvent<HTMLDivElement>) {
    e.preventDefault();
  }

  function removeSourceImage() {
    if (sourceUrlRef.current) {
      URL.revokeObjectURL(sourceUrlRef.current);
      sourceUrlRef.current = null;
    }
    setSourceImage(null);
  }

  useEffect(() => {
    function onPaste(e: ClipboardEvent) {
      const item = Array.from(e.clipboardData?.items ?? []).find((it) =>
        it.type.startsWith("image/"),
      );
      const file = item?.getAsFile();
      if (file) handleFile(file);
    }
    window.addEventListener("paste", onPaste);
    return () => window.removeEventListener("paste", onPaste);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function uploadSourceImage(): Promise<string> {
    if (!sourceImage) throw new Error("No source image.");
    const form = new FormData();
    form.append("image", sourceImage.blob, "source.png");
    form.append("overwrite", "true");
    const res = await fetch(`${COMFYUI_BASE_URL}/upload/image`, { method: "POST", body: form });
    if (!res.ok) throw new Error(`Image upload failed (${res.status})`);
    const body = (await res.json()) as { name: string };
    return body.name;
  }

  function buildWorkflow(opts: {
    width: number;
    height: number;
    uploadedName: string | null;
  }): Record<string, unknown> {
    const workflow: Record<string, unknown> = {
      "1": { class_type: "CheckpointLoaderSimple", inputs: { ckpt_name: CHECKPOINT_FILE } },
      "2": { class_type: "CLIPTextEncode", inputs: { clip: ["1", 1], text: prompt } },
      "3": { class_type: "CLIPTextEncode", inputs: { clip: ["1", 1], text: negativePrompt } },
      "6": { class_type: "VAEDecode", inputs: { samples: ["5", 0], vae: ["1", 2] } },
      "7": {
        class_type: "SaveImage",
        inputs: { images: ["6", 0], filename_prefix: "aias" },
      },
    };

    if (opts.uploadedName) {
      workflow["4a"] = { class_type: "LoadImage", inputs: { image: opts.uploadedName } };
      workflow["4"] = { class_type: "VAEEncode", inputs: { pixels: ["4a", 0], vae: ["1", 2] } };
    } else {
      workflow["4"] = {
        class_type: "EmptyLatentImage",
        inputs: { width: opts.width, height: opts.height, batch_size: 1 },
      };
    }

    workflow["5"] = {
      class_type: "KSampler",
      inputs: {
        model: ["1", 0],
        positive: ["2", 0],
        negative: ["3", 0],
        latent_image: ["4", 0],
        seed,
        steps,
        cfg: 1.0,
        sampler_name: "euler_ancestral",
        scheduler: "sgm_uniform",
        denoise: opts.uploadedName ? strength : 1.0,
      },
    };

    return workflow;
  }

  // Same cap style as services.rs::wait_healthy / instances.rs::wait_for_http:
  // 600 ticks at 1 s each bounds a lost prompt to 10 minutes instead of forever.
  const POLL_HISTORY_MAX_ATTEMPTS = 600;

  async function pollHistory(promptId: string): Promise<ComfyHistoryEntry> {
    for (let attempt = 0; attempt < POLL_HISTORY_MAX_ATTEMPTS; attempt++) {
      if (cancelledRef.current) throw new Error("Cancelled.");
      await new Promise((resolve) => setTimeout(resolve, 1000));
      const res = await fetch(`${COMFYUI_BASE_URL}/history/${promptId}`);
      if (!res.ok) throw new Error(`Failed to read history (${res.status})`);
      const body = (await res.json()) as Record<string, ComfyHistoryEntry>;
      const entry = body[promptId];
      if (!entry) continue;
      if (entry.status.status_str === "error") {
        throw new Error("Generation failed on the ComfyUI side.");
      }
      if (entry.status.completed) return entry;
    }
    throw new Error("ComfyUI did not finish the job.");
  }

  async function handleGenerate() {
    const sizeOption = SIZE_OPTIONS.find((s) => s.value === size) ?? SIZE_OPTIONS[0];
    setGenError(null);
    setGenerating(true);
    setElapsedSeconds(0);
    cancelledRef.current = false;
    const startedAt = Date.now();
    timerRef.current = window.setInterval(() => {
      setElapsedSeconds(Math.floor((Date.now() - startedAt) / 1000));
    }, 1000);

    try {
      let uploadedName: string | null = null;
      if (sourceImage) {
        uploadedName = await uploadSourceImage();
      }

      const workflow = buildWorkflow({
        width: sizeOption.width,
        height: sizeOption.height,
        uploadedName,
      });

      const res = await fetch(`${COMFYUI_BASE_URL}/prompt`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt: workflow, client_id: clientIdRef.current }),
      });
      const body = (await res.json()) as ComfyPromptResponse & { error?: unknown };
      if (!res.ok || (body.node_errors && Object.keys(body.node_errors).length > 0)) {
        const summary = body.node_errors ? nodeErrorSummary(body.node_errors) : `Request failed (${res.status})`;
        throw new Error(summary || "Generation failed to start.");
      }

      const entry = await pollHistory(body.prompt_id);
      if (cancelledRef.current) return;

      const images = entry.outputs["7"]?.images ?? [];
      const image = images[0];
      if (!image) throw new Error("No image was produced.");

      const params = new URLSearchParams({
        filename: image.filename,
        subfolder: image.subfolder,
        type: image.type,
      });
      const viewRes = await fetch(`${COMFYUI_BASE_URL}/view?${params.toString()}`);
      if (!viewRes.ok) throw new Error(`Failed to fetch image (${viewRes.status})`);
      const blob = await viewRes.blob();
      const url = URL.createObjectURL(blob);
      resultUrlsRef.current.add(url);

      const imageResult: ImageResult = {
        id: body.prompt_id,
        url,
        blob,
        prompt,
        negativePrompt,
        width: sizeOption.width,
        height: sizeOption.height,
        steps,
        seed,
        usedSource: Boolean(uploadedName),
        strength: uploadedName ? strength : null,
      };
      setResult(imageResult);
      setHistory((prev) => {
        const next = [imageResult, ...prev];
        const overflow = next.splice(HISTORY_LIMIT);
        overflow.forEach((r) => {
          URL.revokeObjectURL(r.url);
          resultUrlsRef.current.delete(r.url);
        });
        return next;
      });
    } catch (err) {
      if (!cancelledRef.current) {
        setGenError(err instanceof Error ? err.message : String(err));
      }
    } finally {
      if (timerRef.current !== null) {
        window.clearInterval(timerRef.current);
        timerRef.current = null;
      }
      setGenerating(false);
    }
  }

  async function handleCancel() {
    cancelledRef.current = true;
    try {
      await fetch(`${COMFYUI_BASE_URL}/interrupt`, { method: "POST" });
    } catch {
      // best effort
    }
    setGenerating(false);
    if (timerRef.current !== null) {
      window.clearInterval(timerRef.current);
      timerRef.current = null;
    }
  }

  function useAsSource(r: ImageResult) {
    if (sourceUrlRef.current) URL.revokeObjectURL(sourceUrlRef.current);
    const url = URL.createObjectURL(r.blob);
    sourceUrlRef.current = url;
    setSourceImage({ blob: r.blob, url });
  }

  function handleSave(r: ImageResult) {
    const a = document.createElement("a");
    a.href = r.url;
    a.download = `${r.id}.png`;
    a.click();
  }

  async function handleCopyPrompt(r: ImageResult) {
    try {
      await navigator.clipboard.writeText(r.prompt);
    } catch {
      // best effort
    }
  }

  return (
    <div className="flex h-full flex-col gap-3 overflow-y-auto p-6">
      {gpuDialog}
      <Card className="shrink-0">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <span>ComfyUI</span>
            {status && <ServiceStateBadge state={status.state} />}
            {status && <Badge variant="outline">{status.backend}</Badge>}
          </CardTitle>
          <CardDescription>{status?.url ?? COMFYUI_BASE_URL}</CardDescription>
          {status?.runtime === "native" && (
            <p className="text-xs text-muted-foreground">
              Official ComfyUI portable running natively on Windows ({status.backend}); the first
              start unpacks a 1.8 GB download
            </p>
          )}
          {status?.backend === "cpu" && (
            <p className="text-xs text-muted-foreground">
              Running on CPU: expect about a minute per image
            </p>
          )}
        </CardHeader>
        <CardContent className="flex flex-col gap-2">
          {status?.error && <p className="text-xs text-destructive">{status.error}</p>}
          {serviceError && (
            <Alert variant="destructive">
              <AlertDescription>{serviceError}</AlertDescription>
            </Alert>
          )}
          {status && <LogTail lines={status.log_tail} />}
        </CardContent>
        <CardFooter className="flex gap-2">
          <Button
            onClick={handleStart}
            disabled={
              serviceBusy || status?.state === "pulling" || status?.state === "starting" || running
            }
          >
            {status?.state === "pulling"
              ? "Pulling..."
              : status?.state === "starting"
                ? "Starting..."
                : "Start"}
          </Button>
          <Button
            variant="outline"
            onClick={handleStop}
            disabled={serviceBusy || !status || status.state === "stopped" || status.state === "missing"}
          >
            Stop
          </Button>
        </CardFooter>
      </Card>

      <Card className="shrink-0">
        <CardHeader>
          <CardTitle>Models</CardTitle>
        </CardHeader>
        <CardContent className="flex flex-col gap-2">
          {modelsError && (
            <Alert variant="destructive">
              <AlertDescription>{modelsError}</AlertDescription>
            </Alert>
          )}
          {!running && (
            <p className="text-sm text-muted-foreground">Start the service to see models.</p>
          )}
          {running &&
            models.map((m) => (
              <div key={m.name} className="flex items-center gap-2">
                <span className="text-sm">{m.name}</span>
                <span className="flex-1" />
                {m.installed ? (
                  <Badge variant="outline">Installed</Badge>
                ) : (
                  <Button
                    variant="outline"
                    size="sm"
                    disabled={installing}
                    onClick={() => void handleInstallModel(m.name)}
                  >
                    {installing ? <Spinner /> : <Download />}
                    {installing ? "Downloading..." : "Download (5 GB)"}
                  </Button>
                )}
              </div>
            ))}
          {running && modelInstalled && checkpointChecked && !checkpointListed && (
            <Alert>
              <AlertDescription className="flex items-center justify-between gap-2">
                <span>The checkpoint was downloaded after ComfyUI started; restart the service to pick it up.</span>
                <Button variant="outline" size="sm" disabled={serviceBusy} onClick={() => void handleRestart()}>
                  <RefreshCw /> Restart
                </Button>
              </AlertDescription>
            </Alert>
          )}
        </CardContent>
      </Card>

      <fieldset disabled={!ready} className="flex shrink-0 flex-col gap-3 disabled:opacity-50">
        <div className="grid grid-cols-1 gap-3 lg:grid-cols-[360px_minmax(0,1fr)]">
          <Card className="flex flex-col">
            <CardHeader>
              <CardTitle>Controls</CardTitle>
            </CardHeader>
            <CardContent className="flex flex-col gap-3">
              <div className="flex flex-col gap-1.5">
                <label htmlFor="canvas-prompt" className="text-xs text-muted-foreground">
                  Prompt
                </label>
                <Textarea
                  id="canvas-prompt"
                  value={prompt}
                  onChange={(e) => setPrompt(e.target.value)}
                  placeholder="a cozy reading nook, warm light"
                  className="min-h-20"
                />
              </div>
              <div className="flex flex-col gap-1.5">
                <label htmlFor="canvas-negative" className="text-xs text-muted-foreground">
                  Negative prompt
                </label>
                <Input
                  id="canvas-negative"
                  value={negativePrompt}
                  onChange={(e) => setNegativePrompt(e.target.value)}
                  placeholder="blurry, low quality"
                />
              </div>
              <div className="flex items-center gap-2">
                <Select
                  value={size}
                  onValueChange={(v) => v && setSize(v)}
                  items={Object.fromEntries(SIZE_OPTIONS.map((o) => [o.value, o.label]))}
                >
                  <SelectTrigger className="flex-1">
                    <SelectValue placeholder="Size" />
                  </SelectTrigger>
                  <SelectContent>
                    {SIZE_OPTIONS.map((o) => (
                      <SelectItem key={o.value} value={o.value}>
                        {o.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <div className="flex items-center gap-2">
                <label htmlFor="canvas-steps" className="w-14 text-xs text-muted-foreground">
                  Steps
                </label>
                <Input
                  id="canvas-steps"
                  type="number"
                  min={1}
                  max={4}
                  value={steps}
                  onChange={(e) =>
                    setSteps(Math.min(4, Math.max(1, Number(e.target.value) || 1)))
                  }
                  className="w-20"
                />
                <label htmlFor="canvas-seed" className="ml-2 w-10 text-xs text-muted-foreground">
                  Seed
                </label>
                <Input
                  id="canvas-seed"
                  type="number"
                  value={seed}
                  onChange={(e) => setSeed(Number(e.target.value) || 0)}
                  className="w-28"
                />
                <Button
                  variant="outline"
                  size="icon-sm"
                  type="button"
                  onClick={() => setSeed(randomSeed())}
                  title="Random seed"
                >
                  <Dices />
                </Button>
              </div>
              {sourceImage && (
                <div className="flex flex-col gap-1.5">
                  <label htmlFor="canvas-strength" className="text-xs text-muted-foreground">
                    Strength ({strength.toFixed(2)})
                  </label>
                  <input
                    id="canvas-strength"
                    type="range"
                    min={0.3}
                    max={0.9}
                    step={0.05}
                    value={strength}
                    onChange={(e) => setStrength(Number(e.target.value))}
                    className="h-1.5 w-full cursor-pointer appearance-none rounded-full bg-muted accent-primary"
                  />
                </div>
              )}
              <div className="flex gap-2">
                <Button onClick={() => void handleGenerate()} disabled={generating || !prompt.trim()}>
                  {generating ? <Spinner /> : <Wand2 />}
                  {generating ? `Generating... ${elapsedSeconds}s` : "Generate"}
                </Button>
                <Button variant="outline" onClick={() => void handleCancel()} disabled={!generating}>
                  <Ban /> Cancel
                </Button>
              </div>
              {genError && (
                <Alert variant="destructive">
                  <AlertDescription>{genError}</AlertDescription>
                </Alert>
              )}
            </CardContent>
          </Card>

          <Card className="flex flex-col">
            <CardHeader>
              <CardTitle>Source image (optional)</CardTitle>
            </CardHeader>
            <CardContent className="flex flex-col gap-3">
              {sourceImage ? (
                <div className="flex items-start gap-3">
                  <img
                    src={sourceImage.url}
                    alt="Source"
                    className="h-32 w-32 rounded-lg border border-border object-cover"
                  />
                  <Button variant="outline" size="sm" onClick={removeSourceImage}>
                    <X /> Remove
                  </Button>
                </div>
              ) : (
                <div
                  onDrop={handleDrop}
                  onDragOver={handleDragOver}
                  className="flex flex-col items-center justify-center gap-2 rounded-lg border border-dashed border-border p-6 text-center text-sm text-muted-foreground"
                >
                  <Upload className="size-5" />
                  <p>Drag &amp; drop an image, paste from clipboard, or choose a file.</p>
                  <Input type="file" accept="image/*" onChange={handleFileInputChange} className="w-auto" />
                </div>
              )}
            </CardContent>
          </Card>
        </div>

        <Card className="flex flex-col">
          <CardHeader>
            <CardTitle>Result</CardTitle>
          </CardHeader>
          <CardContent className="flex flex-col gap-3">
            <div className="flex min-h-48 items-center justify-center overflow-hidden rounded-lg border border-border bg-muted">
              {result ? (
                <img
                  src={result.url}
                  alt={result.prompt}
                  className="max-h-[480px] max-w-full object-contain"
                />
              ) : (
                <span className="text-sm text-muted-foreground">No image yet</span>
              )}
            </div>
            {result && (
              <div className="flex flex-wrap gap-2">
                <Button variant="outline" size="sm" onClick={() => useAsSource(result)}>
                  <ImagePlus /> Use as source
                </Button>
                <Button variant="outline" size="sm" onClick={() => handleSave(result)}>
                  <Download /> Save
                </Button>
                <Button variant="outline" size="sm" onClick={() => void handleCopyPrompt(result)}>
                  <Copy /> Copy prompt
                </Button>
              </div>
            )}
            {history.length > 0 && (
              <div className="flex gap-2 overflow-x-auto pt-1">
                {history.map((r) => (
                  <button
                    key={r.id}
                    type="button"
                    onClick={() => setResult(r)}
                    className={cn(
                      "shrink-0 overflow-hidden rounded-lg border-2 transition-colors",
                      result?.id === r.id ? "border-primary" : "border-transparent",
                    )}
                  >
                    <img src={r.url} alt={r.prompt} className="size-16 object-cover" />
                  </button>
                ))}
              </div>
            )}
          </CardContent>
        </Card>
      </fieldset>
    </div>
  );
}
