import { useCallback, useEffect, useRef, useState } from "react";
import { generateText, streamText } from "ai";
import {
  Camera,
  Download,
  ScanSearch,
  Send,
  Square,
  Trash2,
  TriangleAlert,
  Upload,
} from "lucide-react";

import {
  cvDetect,
  installServiceModel,
  launchModel,
  listInstances,
  onInstanceUpdate,
  onServiceUpdate,
  serviceModels,
  serviceStatus,
  startService,
  stopService,
  type Instance,
  type ServiceModel,
  type ServiceState,
  type ServiceStatus,
} from "@/lib/api";
import { ollama } from "@/lib/chat";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
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
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { StatusBadge } from "@/components/StatusBadge";
import { LogTail } from "@/components/LogTail";
import { MessageResponse } from "@/components/ai-elements/message";

const VISION_TAG_PREFIXES = ["qwen2.5vl", "gemma3", "llava", "minicpm-v", "moondream"];
const MAX_SIDE = 1024;
const CV_SERVICE = "cv";
/** Detector choice meaning "ask the vision-language model for boxes". */
const VLM_DETECTOR = "vlm";
const CV_MIN_SCORE = 0.25;

type Dimensions = { width: number; height: number };

type Detection = { label: string; box: [number, number, number, number]; score?: number };

const SERVICE_STATE_LABEL: Record<ServiceState, string> = {
  missing: "Not installed",
  pulling: "Pulling image",
  starting: "Starting",
  running: "Running",
  stopped: "Stopped",
  error: "Error",
};

const SERVICE_STATE_VARIANT: Record<ServiceState, "default" | "secondary" | "destructive" | "outline"> = {
  missing: "outline",
  pulling: "secondary",
  starting: "secondary",
  running: "default",
  stopped: "outline",
  error: "destructive",
};

function ServiceStateBadge({ state }: { state: ServiceState }) {
  const animated = state === "pulling" || state === "starting";
  return (
    <Badge variant={SERVICE_STATE_VARIANT[state]} className={cn(animated && "animate-pulse")}>
      {SERVICE_STATE_LABEL[state]}
    </Badge>
  );
}

function isVisionModelTag(tag: string | null): boolean {
  if (!tag) return false;
  const t = tag.toLowerCase();
  return VISION_TAG_PREFIXES.some((prefix) => t.startsWith(prefix)) || t.includes("vl");
}

function upsert(list: Instance[], next: Instance): Instance[] {
  const idx = list.findIndex((i) => i.id === next.id);
  if (idx === -1) return [next, ...list];
  const copy = list.slice();
  copy[idx] = next;
  return copy;
}

/** Longest side capped at `MAX_SIDE`, preserving aspect ratio. */
function scaledSize(width: number, height: number): Dimensions {
  if (width <= MAX_SIDE && height <= MAX_SIDE) return { width, height };
  const scale = MAX_SIDE / Math.max(width, height);
  return { width: Math.round(width * scale), height: Math.round(height * scale) };
}

function isValidBox(box: unknown): box is [number, number, number, number] {
  return Array.isArray(box) && box.length === 4 && box.every((n) => typeof n === "number");
}

/** Lenient parse: pull the first `{...}` block out of the reply and validate its shape. */
/** Return the first complete JSON value (object or array) found in free text. */
function extractJson(text: string): unknown {
  const start = text.search(/[[{]/);
  if (start === -1) throw new Error("Model did not return JSON.");
  let depth = 0;
  let inString = false;
  for (let i = start; i < text.length; i++) {
    const ch = text[i];
    if (inString) {
      if (ch === "\\") i++;
      else if (ch === '"') inString = false;
      continue;
    }
    if (ch === '"') inString = true;
    else if (ch === "{" || ch === "[") depth++;
    else if (ch === "}" || ch === "]") {
      depth--;
      if (depth === 0) return JSON.parse(text.slice(start, i + 1));
    }
  }
  throw new Error("Model returned truncated JSON.");
}

/** Accept `{objects:[...]}`, `{detections:[...]}` or a bare array of `{label, box}`. */
function parseDetections(text: string): Detection[] {
  const parsed = extractJson(text) as
    | Array<{ label?: unknown; box?: unknown }>
    | { objects?: unknown; detections?: unknown };
  const list = Array.isArray(parsed)
    ? parsed
    : Array.isArray(parsed.objects)
      ? parsed.objects
      : Array.isArray(parsed.detections)
        ? parsed.detections
        : [];
  return (list as Array<{ label?: unknown; box?: unknown }>)
    .filter(
      (o): o is { label: string; box: [number, number, number, number] } =>
        typeof o.label === "string" && isValidBox(o.box),
    )
    .map((o) => ({ label: o.label, box: o.box }));
}

export function Vision() {
  const [instances, setInstances] = useState<Instance[]>([]);
  const [launching, setLaunching] = useState(false);
  const [launchError, setLaunchError] = useState<string | null>(null);

  const [sourceTab, setSourceTab] = useState<"camera" | "image">("camera");
  const [modeTab, setModeTab] = useState<"ask" | "detect">("ask");

  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [imageBytes, setImageBytes] = useState<Uint8Array | null>(null);
  const [imageDims, setImageDims] = useState<Dimensions | null>(null);
  const [imageError, setImageError] = useState<string | null>(null);
  const [cameraError, setCameraError] = useState<string | null>(null);

  const [askPrompt, setAskPrompt] = useState("Describe this image in detail.");
  const [asking, setAsking] = useState(false);
  const [askError, setAskError] = useState<string | null>(null);
  const [askAnswer, setAskAnswer] = useState("");

  const [detectQuery, setDetectQuery] = useState("person and vehicle");
  const [detecting, setDetecting] = useState(false);
  const [detectError, setDetectError] = useState<string | null>(null);
  const [detections, setDetections] = useState<Detection[]>([]);
  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  const [detectSummary, setDetectSummary] = useState<string | null>(null);

  const [cvStatus, setCvStatus] = useState<ServiceStatus | null>(null);
  const [cvBusy, setCvBusy] = useState(false);
  const [cvError, setCvError] = useState<string | null>(null);
  const [cvModels, setCvModels] = useState<ServiceModel[]>([]);
  const [installing, setInstalling] = useState<string | null>(null);
  const [detector, setDetector] = useState<string>(VLM_DETECTOR);

  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const previewContainerRef = useRef<HTMLDivElement>(null);
  const overlayCanvasRef = useRef<HTMLCanvasElement>(null);
  const imageUrlRef = useRef<string | null>(null);
  const askAbortRef = useRef<AbortController | null>(null);
  const detectTimerRef = useRef<number | null>(null);
  /** the user picked a detector by hand; stop auto-switching to the CV server */
  const detectorPickedRef = useRef(false);

  useEffect(() => {
    let cancelled = false;
    listInstances().then((res) => {
      if (!cancelled) setInstances(res);
    });
    let unlisten: (() => void) | undefined;
    onInstanceUpdate((instance) => {
      if (instance.kind !== "model") return;
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

  useEffect(() => {
    return () => {
      if (imageUrlRef.current) URL.revokeObjectURL(imageUrlRef.current);
      streamRef.current?.getTracks().forEach((t) => t.stop());
      askAbortRef.current?.abort();
      if (detectTimerRef.current !== null) window.clearInterval(detectTimerRef.current);
    };
  }, []);

  useEffect(() => {
    let cancelled = false;
    serviceStatus(CV_SERVICE).then((s) => {
      if (!cancelled) setCvStatus(s);
    });
    refreshCvModels();
    let unlisten: (() => void) | undefined;
    onServiceUpdate((s) => {
      if (s.id !== CV_SERVICE) return;
      setCvStatus(s);
    }).then((fn) => {
      if (cancelled) fn();
      else unlisten = fn;
    });
    return () => {
      cancelled = true;
      unlisten?.();
    };
  }, []);

  const visionCandidates = instances.filter((i) => isVisionModelTag(i.model_tag));
  const visionInstance = visionCandidates.find((i) => i.status === "running");
  const pendingVision = visionCandidates.find((i) => i.status !== "running") ?? visionCandidates[0];
  const cvRunning = cvStatus?.state === "running";
  const installedCvModels = cvModels.filter((m) => m.installed);
  const cvUsable = cvRunning && installedCvModels.length > 0;
  const ready = Boolean(visionInstance) || cvUsable;
  const usingCv = detector !== VLM_DETECTOR;

  // Prefer the CV server once it can answer; fall back to the VLM when it cannot.
  useEffect(() => {
    if (usingCv && !(cvUsable && installedCvModels.some((m) => m.name === detector))) {
      setDetector(cvUsable ? installedCvModels[0].name : VLM_DETECTOR);
    } else if (!usingCv && cvUsable && (!visionInstance || !detectorPickedRef.current)) {
      setDetector(installedCvModels[0].name);
    }
  }, [cvUsable, installedCvModels, detector, usingCv, visionInstance]);

  function refreshCvModels() {
    serviceModels(CV_SERVICE)
      .then(setCvModels)
      .catch((err) => setCvError(err instanceof Error ? err.message : String(err)));
  }

  async function handleCvStart() {
    setCvBusy(true);
    setCvError(null);
    try {
      setCvStatus(await startService(CV_SERVICE));
    } catch (err) {
      setCvError(err instanceof Error ? err.message : String(err));
    } finally {
      setCvBusy(false);
    }
  }

  async function handleCvStop() {
    setCvBusy(true);
    setCvError(null);
    try {
      await stopService(CV_SERVICE);
    } catch (err) {
      setCvError(err instanceof Error ? err.message : String(err));
    } finally {
      setCvBusy(false);
    }
  }

  async function handleInstallCvModel(name: string) {
    setInstalling(name);
    setCvError(null);
    try {
      await installServiceModel(CV_SERVICE, name);
      refreshCvModels();
    } catch (err) {
      setCvError(err instanceof Error ? err.message : String(err));
    } finally {
      setInstalling(null);
    }
  }

  async function handleStartVisionModel() {
    setLaunching(true);
    setLaunchError(null);
    try {
      const instance = await launchModel("qwen2.5vl", "7b");
      setInstances((prev) => upsert(prev, instance));
    } catch (err) {
      setLaunchError(err instanceof Error ? err.message : String(err));
    } finally {
      setLaunching(false);
    }
  }

  const redrawOverlay = useCallback(() => {
    const canvas = overlayCanvasRef.current;
    const container = previewContainerRef.current;
    if (!canvas || !container) return;
    const containerW = container.clientWidth;
    const containerH = container.clientHeight;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.max(1, Math.round(containerW * dpr));
    canvas.height = Math.max(1, Math.round(containerH * dpr));
    canvas.style.width = `${containerW}px`;
    canvas.style.height = `${containerH}px`;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, containerW, containerH);
    if (!imageDims || detections.length === 0) return;

    const scale = Math.min(containerW / imageDims.width, containerH / imageDims.height);
    const offsetX = (containerW - imageDims.width * scale) / 2;
    const offsetY = (containerH - imageDims.height * scale) / 2;

    const rootStyle = getComputedStyle(document.documentElement);
    const primary = rootStyle.getPropertyValue("--primary").trim() || "red";
    const foreground = rootStyle.getPropertyValue("--primary-foreground").trim() || "white";

    ctx.lineWidth = 2;
    ctx.font = "12px sans-serif";
    ctx.textBaseline = "bottom";

    for (const d of detections) {
      const [x1, y1, x2, y2] = d.box;
      const bx = offsetX + x1 * scale;
      const by = offsetY + y1 * scale;
      const bw = (x2 - x1) * scale;
      const bh = (y2 - y1) * scale;
      ctx.strokeStyle = primary;
      ctx.strokeRect(bx, by, bw, bh);
      const labelY = Math.max(by, 12);
      const text = d.score === undefined ? d.label : `${d.label} ${Math.round(d.score * 100)}%`;
      const textWidth = ctx.measureText(text).width;
      ctx.fillStyle = primary;
      ctx.fillRect(bx, labelY - 12, textWidth + 6, 14);
      ctx.fillStyle = foreground;
      ctx.fillText(text, bx + 3, labelY + 1);
    }
  }, [detections, imageDims]);

  useEffect(() => {
    redrawOverlay();
  }, [redrawOverlay]);

  useEffect(() => {
    const container = previewContainerRef.current;
    if (!container) return;
    const observer = new ResizeObserver(() => redrawOverlay());
    observer.observe(container);
    return () => observer.disconnect();
  }, [redrawOverlay]);

  useEffect(() => {
    if (sourceTab !== "camera" || !ready) {
      streamRef.current?.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
      return;
    }
    let cancelled = false;
    navigator.mediaDevices
      .getUserMedia({ video: true })
      .then((stream) => {
        if (cancelled) {
          stream.getTracks().forEach((t) => t.stop());
          return;
        }
        streamRef.current = stream;
        if (videoRef.current) videoRef.current.srcObject = stream;
        setCameraError(null);
      })
      .catch((err) => setCameraError(err instanceof Error ? err.message : String(err)));
    return () => {
      cancelled = true;
      streamRef.current?.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    };
  }, [sourceTab, ready]);

  useEffect(() => {
    if (sourceTab !== "image") return;
    function onPaste(e: ClipboardEvent) {
      const item = Array.from(e.clipboardData?.items ?? []).find((it) => it.type.startsWith("image/"));
      const file = item?.getAsFile();
      if (file) void loadImageFile(file);
    }
    window.addEventListener("paste", onPaste);
    return () => window.removeEventListener("paste", onPaste);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sourceTab]);

  async function setImageFromSource(source: CanvasImageSource, naturalWidth: number, naturalHeight: number) {
    setImageError(null);
    try {
      const { width, height } = scaledSize(naturalWidth, naturalHeight);
      const canvas = document.createElement("canvas");
      canvas.width = width;
      canvas.height = height;
      const ctx = canvas.getContext("2d");
      if (!ctx) throw new Error("Canvas is not supported.");
      ctx.drawImage(source, 0, 0, width, height);
      const blob = await new Promise<Blob>((resolve, reject) => {
        canvas.toBlob((b) => (b ? resolve(b) : reject(new Error("Failed to encode image."))), "image/jpeg", 0.9);
      });
      const bytes = new Uint8Array(await blob.arrayBuffer());
      if (imageUrlRef.current) URL.revokeObjectURL(imageUrlRef.current);
      const url = URL.createObjectURL(blob);
      imageUrlRef.current = url;
      setImageUrl(url);
      setImageBytes(bytes);
      setImageDims({ width, height });
      setDetections([]);
      setAskAnswer("");
    } catch (err) {
      setImageError(err instanceof Error ? err.message : String(err));
    }
  }

  function handleCapture() {
    const video = videoRef.current;
    if (!video || video.videoWidth === 0) return;
    void setImageFromSource(video, video.videoWidth, video.videoHeight);
  }

  async function loadImageFile(file: File) {
    try {
      const bitmap = await createImageBitmap(file);
      await setImageFromSource(bitmap, bitmap.width, bitmap.height);
      bitmap.close();
    } catch (err) {
      setImageError(err instanceof Error ? err.message : String(err));
    }
  }

  function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (file) void loadImageFile(file);
    e.target.value = "";
  }

  function handleDrop(e: React.DragEvent<HTMLDivElement>) {
    e.preventDefault();
    const file = e.dataTransfer.files?.[0];
    if (file) void loadImageFile(file);
  }

  function handleDragOver(e: React.DragEvent<HTMLDivElement>) {
    e.preventDefault();
  }

  async function handleAsk() {
    if (!visionInstance || !imageBytes) return;
    setAsking(true);
    setAskError(null);
    setAskAnswer("");
    const controller = new AbortController();
    askAbortRef.current = controller;
    try {
      const result = streamText({
        model: ollama(visionInstance.model_tag ?? ""),
        abortSignal: controller.signal,
        messages: [
          {
            role: "user",
            content: [
              { type: "text", text: askPrompt.trim() || "Describe this image in detail." },
              { type: "file", mediaType: "image/jpeg", data: imageBytes },
            ],
          },
        ],
      });
      for await (const delta of result.textStream) {
        setAskAnswer((prev) => prev + delta);
      }
    } catch (err) {
      if (!(err instanceof Error && err.name === "AbortError")) {
        setAskError(err instanceof Error ? err.message : String(err));
      }
    } finally {
      setAsking(false);
      askAbortRef.current = null;
    }
  }

  function handleStopAsk() {
    askAbortRef.current?.abort();
  }

  async function handleDetect() {
    if (!imageBytes || !imageDims) return;
    if (usingCv ? !cvUsable : !visionInstance) return;
    setDetecting(true);
    setDetectError(null);
    setDetectSummary(null);
    setElapsedSeconds(0);
    const startedAt = Date.now();
    detectTimerRef.current = window.setInterval(() => {
      setElapsedSeconds(Math.floor((Date.now() - startedAt) / 1000));
    }, 1000);
    try {
      if (usingCv) {
        const result = await cvDetect(imageBytes, detector, CV_MIN_SCORE);
        setDetections(result.detections.map((d) => ({ label: d.label, box: d.box, score: d.score })));
        setDetectSummary(
          `${result.detections.length} objects in ${result.total_ms} ms (${result.model} on ${result.backend})`,
        );
        return;
      }
      if (!visionInstance) return;
      const prompt =
        `Detect every ${detectQuery.trim() || "object"} in this image. ` +
        `Reply with JSON only: {"objects": [{"label": str, "box": [x1, y1, x2, y2]}]} ` +
        `with pixel coordinates of the original ${imageDims.width}x${imageDims.height} image. No prose.`;
      const result = await generateText({
        model: ollama(visionInstance.model_tag ?? ""),
        temperature: 0,
        messages: [
          {
            role: "user",
            content: [
              { type: "text", text: prompt },
              { type: "file", mediaType: "image/jpeg", data: imageBytes },
            ],
          },
        ],
      });
      setDetections(parseDetections(result.text));
      setDetectSummary(`${Math.round((Date.now() - startedAt) / 1000)} s via ${visionInstance.display_name}`);
    } catch (err) {
      setDetectError(err instanceof Error ? err.message : String(err));
    } finally {
      if (detectTimerRef.current !== null) {
        window.clearInterval(detectTimerRef.current);
        detectTimerRef.current = null;
      }
      setDetecting(false);
    }
  }

  function handleClearBoxes() {
    setDetections([]);
  }

  return (
    <div className="flex h-full flex-col gap-3 overflow-y-auto p-6">
      {!visionInstance ? (
        <Card className="shrink-0">
          <CardHeader>
            <CardTitle>No vision model running</CardTitle>
            <CardDescription>
              Ask needs a vision-language model like qwen2.5vl, gemma3, llava, minicpm-v, or moondream.
              Detect can use the CV server below instead.
            </CardDescription>
          </CardHeader>
          <CardContent className="flex flex-col gap-2">
            {pendingVision && (
              <div className="flex items-center gap-2">
                <StatusBadge status={pendingVision.status} />
                <span className="text-sm text-muted-foreground">{pendingVision.display_name}</span>
              </div>
            )}
            {pendingVision?.error && <p className="text-xs text-destructive">{pendingVision.error}</p>}
            {pendingVision && <LogTail lines={pendingVision.log_tail} />}
            {launchError && (
              <Alert variant="destructive">
                <TriangleAlert />
                <AlertDescription>{launchError}</AlertDescription>
              </Alert>
            )}
          </CardContent>
          <CardFooter>
            <Button
              onClick={() => void handleStartVisionModel()}
              disabled={
                launching ||
                (pendingVision !== undefined &&
                  pendingVision.status !== "error" &&
                  pendingVision.status !== "stopped")
              }
            >
              {launching || pendingVision?.status === "pulling" || pendingVision?.status === "starting" ? (
                <Spinner />
              ) : (
                <Camera />
              )}
              Start qwen2.5vl:7b
            </Button>
          </CardFooter>
        </Card>
      ) : (
        <Card size="sm" className="shrink-0">
          <CardContent className="flex items-center gap-2">
            <StatusBadge status="running" />
            <span className="text-sm">{visionInstance?.display_name}</span>
          </CardContent>
        </Card>
      )}

      <Card size="sm" className="shrink-0">
        <CardContent className="flex flex-col gap-2">
          <div className="flex flex-wrap items-center gap-2">
            {cvStatus && <ServiceStateBadge state={cvStatus.state} />}
            <span className="text-sm">{cvStatus?.display_name ?? "CV server"}</span>
            {cvStatus && (
              <Badge variant="outline" className="font-mono text-xs">
                {cvStatus.backend}
              </Badge>
            )}
            <span className="flex-1" />
            <Button
              size="sm"
              onClick={() => void handleCvStart()}
              disabled={cvBusy || cvStatus?.state === "pulling" || cvStatus?.state === "starting" || cvRunning}
            >
              {cvStatus?.state === "pulling" || cvStatus?.state === "starting" ? <Spinner /> : null}
              {cvStatus?.state === "pulling" ? "Pulling..." : cvStatus?.state === "starting" ? "Starting..." : "Start"}
            </Button>
            <Button
              size="sm"
              variant="outline"
              onClick={() => void handleCvStop()}
              disabled={cvBusy || !cvStatus || cvStatus.state === "stopped" || cvStatus.state === "missing"}
            >
              Stop
            </Button>
          </div>
          <div className="flex flex-wrap items-center gap-x-4 gap-y-1">
            {cvModels.map((m) => (
              <div key={m.name} className="flex items-center gap-2 text-sm">
                <span>{m.display_name}</span>
                {m.installed ? (
                  <Badge variant="outline">Installed</Badge>
                ) : (
                  <Button
                    variant="outline"
                    size="sm"
                    disabled={installing !== null}
                    onClick={() => void handleInstallCvModel(m.name)}
                  >
                    {installing === m.name ? <Spinner /> : <Download />}
                    {installing === m.name ? "Downloading..." : "Download"}
                  </Button>
                )}
              </div>
            ))}
          </div>
          {(cvStatus?.error || cvError) && (
            <Alert variant="destructive">
              <TriangleAlert />
              <AlertDescription>{cvStatus?.error ?? cvError}</AlertDescription>
            </Alert>
          )}
          {cvStatus && cvStatus.state !== "running" && cvStatus.log_tail.length > 0 && (
            <LogTail lines={cvStatus.log_tail} />
          )}
        </CardContent>
      </Card>

      <fieldset disabled={!ready} className="grid shrink-0 grid-cols-1 gap-3 disabled:opacity-50 lg:grid-cols-[minmax(0,1fr)_360px]">
        <Card className="shrink-0">
          <CardHeader>
            <CardTitle>Source</CardTitle>
            <CardDescription>Capture from a camera or pick an image.</CardDescription>
          </CardHeader>
          <CardContent className="flex flex-col gap-3">
            <Tabs value={sourceTab} onValueChange={(v) => v && setSourceTab(v as "camera" | "image")}>
              <TabsList>
                <TabsTrigger value="camera">Camera</TabsTrigger>
                <TabsTrigger value="image">Image</TabsTrigger>
              </TabsList>
              <TabsContent value="camera" className="mt-3 flex flex-col gap-2">
                {cameraError && (
                  <Alert variant="destructive">
                    <TriangleAlert />
                    <AlertDescription>{cameraError}</AlertDescription>
                  </Alert>
                )}
                <video ref={videoRef} autoPlay muted playsInline className="aspect-video w-full rounded-lg bg-muted object-contain" />
                <Button onClick={handleCapture} disabled={sourceTab !== "camera"}>
                  <Camera /> Capture
                </Button>
              </TabsContent>
              <TabsContent value="image" className="mt-3 flex flex-col gap-2">
                <div
                  onDrop={handleDrop}
                  onDragOver={handleDragOver}
                  className="flex flex-col items-center justify-center gap-2 rounded-lg border border-dashed border-border p-6 text-center text-sm text-muted-foreground"
                >
                  <Upload className="size-5" />
                  <p>Drag &amp; drop an image, paste from clipboard, or choose a file.</p>
                  <Input type="file" accept="image/*" onChange={handleFileChange} className="w-auto" />
                </div>
              </TabsContent>
            </Tabs>

            {imageError && (
              <Alert variant="destructive">
                <TriangleAlert />
                <AlertDescription>{imageError}</AlertDescription>
              </Alert>
            )}

            <div ref={previewContainerRef} className="relative h-[420px] shrink-0 overflow-hidden rounded-lg border border-border bg-muted">
              {imageUrl ? (
                <>
                  <img src={imageUrl} alt="Captured" className="size-full object-contain" onLoad={redrawOverlay} />
                  <canvas ref={overlayCanvasRef} className="pointer-events-none absolute inset-0 size-full" />
                </>
              ) : (
                <div className="flex size-full items-center justify-center text-sm text-muted-foreground">
                  No image yet
                </div>
              )}
            </div>
          </CardContent>
        </Card>

        <Card className="shrink-0">
          <CardHeader>
            <CardTitle>Ask &amp; Detect</CardTitle>
          </CardHeader>
          <CardContent className="flex flex-col gap-3">
            <Tabs value={modeTab} onValueChange={(v) => v && setModeTab(v as "ask" | "detect")}>
              <TabsList>
                <TabsTrigger value="ask">Ask</TabsTrigger>
                <TabsTrigger value="detect">Detect</TabsTrigger>
              </TabsList>

              <TabsContent value="ask" className="mt-3 flex flex-col gap-3">
                <Textarea
                  value={askPrompt}
                  onChange={(e) => setAskPrompt(e.target.value)}
                  placeholder="Describe this image in detail."
                  className="min-h-20"
                />
                <div className="flex gap-2">
                  <Button onClick={() => void handleAsk()} disabled={!imageBytes || asking || !visionInstance}>
                    {asking ? <Spinner /> : <Send />} Ask
                  </Button>
                  <Button variant="outline" onClick={handleStopAsk} disabled={!asking}>
                    <Square /> Stop
                  </Button>
                </div>
                {askError && (
                  <Alert variant="destructive">
                    <TriangleAlert />
                    <AlertDescription>{askError}</AlertDescription>
                  </Alert>
                )}
                {askAnswer && (
                  <div className="rounded-xl border border-border p-3">
                    <MessageResponse>{askAnswer}</MessageResponse>
                  </div>
                )}
              </TabsContent>

              <TabsContent value="detect" className="mt-3 flex flex-col gap-3">
                <div className="flex flex-col gap-1.5">
                  <label htmlFor="detector" className="text-xs text-muted-foreground">
                    Detector
                  </label>
                  <Select
                    value={detector}
                    onValueChange={(v) => {
                      if (!v) return;
                      detectorPickedRef.current = true;
                      setDetector(v);
                    }}
                    items={[
                      ...installedCvModels.map((m) => ({ value: m.name, label: `${m.display_name} via CV server` })),
                      { value: VLM_DETECTOR, label: "Vision-language model (asks for JSON boxes)" },
                    ]}
                  >
                    <SelectTrigger id="detector" className="w-full">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {installedCvModels.map((m) => (
                        <SelectItem key={m.name} value={m.name} disabled={!cvRunning}>
                          {m.display_name} via CV server
                        </SelectItem>
                      ))}
                      <SelectItem value={VLM_DETECTOR} disabled={!visionInstance}>
                        Vision-language model (asks for JSON boxes)
                      </SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                {usingCv ? (
                  <p className="text-xs text-muted-foreground">
                    Finds the 80 COCO classes (people, vehicles, animals, everyday objects) with real boxes and scores.
                  </p>
                ) : (
                  <div className="flex flex-col gap-1.5">
                    <label htmlFor="detect-query" className="text-xs text-muted-foreground">
                      What to look for
                    </label>
                    <Input
                      id="detect-query"
                      value={detectQuery}
                      onChange={(e) => setDetectQuery(e.target.value)}
                      placeholder="person and vehicle"
                    />
                  </div>
                )}
                <div className="flex items-center gap-2">
                  <Button
                    onClick={() => void handleDetect()}
                    disabled={!imageBytes || detecting || (usingCv ? !cvUsable : !visionInstance)}
                  >
                    {detecting ? <Spinner /> : <ScanSearch />} Detect objects
                  </Button>
                  <Button variant="outline" onClick={handleClearBoxes} disabled={detections.length === 0}>
                    <Trash2 /> Clear boxes
                  </Button>
                  {detecting && <span className="text-xs text-muted-foreground">{elapsedSeconds}s</span>}
                </div>
                {!detecting && detectSummary && (
                  <p className="text-xs text-muted-foreground">{detectSummary}</p>
                )}
                {detectError && (
                  <Alert variant="destructive">
                    <TriangleAlert />
                    <AlertDescription>{detectError}</AlertDescription>
                  </Alert>
                )}
                {detections.length > 0 && (
                  <div className="overflow-hidden rounded-lg border border-border">
                    <table className="w-full text-sm">
                      <thead className="bg-muted text-muted-foreground">
                        <tr>
                          <th className="px-2.5 py-1.5 text-left font-medium">Label</th>
                          <th className="px-2.5 py-1.5 text-left font-medium">Score</th>
                          <th className="px-2.5 py-1.5 text-left font-medium">Box (x1, y1, x2, y2)</th>
                        </tr>
                      </thead>
                      <tbody>
                        {detections.map((d, i) => (
                          <tr key={i} className="border-t border-border">
                            <td className="px-2.5 py-1.5">{d.label}</td>
                            <td className="px-2.5 py-1.5 font-mono text-xs">
                              {d.score === undefined ? "-" : `${Math.round(d.score * 100)}%`}
                            </td>
                            <td className="px-2.5 py-1.5 font-mono text-xs">
                              {d.box.map((n) => Math.round(n)).join(", ")}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </TabsContent>
            </Tabs>
          </CardContent>
        </Card>
      </fieldset>
    </div>
  );
}
