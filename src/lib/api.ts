// Typed contract between the React frontend and the Rust backend.
// Every Tauri command and event used by the UI is declared here; nothing else
// in `src/` may call `invoke` or `listen` directly.

import { invoke } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";
import { check as checkUpdate, type DownloadEvent } from "@tauri-apps/plugin-updater";
import { relaunch as relaunchApp } from "@tauri-apps/plugin-process";

// ---------------------------------------------------------------------------
// Runtime (WSL2 distro + Docker + NVIDIA + Ollama)
// ---------------------------------------------------------------------------

export type Vendor = "nvidia" | "amd" | "intel" | "cpu";

/** `dedicated`: a discrete card, `vram_mb` is the budget. `unified`: shares system RAM. */
export type MemoryModel = "dedicated" | "unified";

export type Gpu = {
  name: string;
  vendor: Vendor;
  /** dedicated memory as the driver reports it; on a unified part only the small carve-out */
  vram_mb: number | null;
  /** integrated graphics (shares system RAM); a discrete card is preferred */
  integrated: boolean;
  driver: string | null;
  memory_model: MemoryModel;
  /** what a model may occupy here: `vram_mb` on a discrete card, half of system RAM on a unified part */
  effective_memory_mb: number | null;
};

export type Npu = {
  name: string;
  vendor: "intel" | "amd" | "qualcomm" | "unknown";
  /** NPUs share system RAM: same budget as a unified GPU */
  effective_memory_mb: number | null;
};

export type Virtualization = {
  /** the CPU has VT-x / AMD-V */
  vt_supported: boolean;
  /** virtualization is enabled in the UEFI firmware (the BIOS switch) */
  vt_firmware_enabled: boolean;
  /** the Windows hypervisor is already running */
  hypervisor_present: boolean;
};

export type HardwareProfile = {
  gpus: Gpu[];
  npus: Npu[];
  primary_gpu: Gpu | null;
  /** vendor the runtime is built around; selects service implementations */
  vendor: Vendor;
  /** only NVIDIA can be handed to WSL2 containers; other vendors run natively on Windows */
  wsl_gpu: boolean;
  virtualization: Virtualization;
  /** physical RAM; sizes the budget of unified-memory accelerators */
  total_ram_mb: number | null;
};

export type RuntimeStatus = {
  /** wsl.exe is present and WSL2 is functional */
  wsl_installed: boolean;
  wsl_version: string | null;
  /** our own distro (`ai-app-store`) is registered */
  distro_present: boolean;
  distro_running: boolean;
  docker_ok: boolean;
  /** `docker run --gpus all` works inside the distro */
  gpu_ok: boolean;
  ollama_ok: boolean;
  gpu_name: string | null;
  /** dedicated memory of the primary GPU (display) */
  vram_mb: number | null;
  /** budget compatibility verdicts use: `vram_mb` on a discrete card, half of RAM on a unified part */
  effective_memory_mb: number | null;
  /** everything above that is required is true */
  ready: boolean;
  /** a Windows reboot is required before continuing (WSL feature just enabled) */
  reboot_required: boolean;
  /** firmware / hypervisor state that decides whether WSL2 can run at all */
  virtualization: Virtualization;
  hardware: HardwareProfile;
  vendor: Vendor;
  /** `vendor` was forced with the `AIAS_VENDOR` environment variable (testing another vendor's path) */
  vendor_forced: boolean;
};

export type ProvisionEvent = {
  step: string;
  status: "start" | "ok" | "error" | "log";
  message: string;
};

export const runtimeStatus = () => invoke<RuntimeStatus>("runtime_status");

/** Long-running. Progress arrives on `provision://progress`; resolves with the final status. */
export const provisionRuntime = () => invoke<RuntimeStatus>("provision_runtime");

export const onProvisionProgress = (cb: (e: ProvisionEvent) => void): Promise<UnlistenFn> =>
  listen<ProvisionEvent>("provision://progress", (ev) => cb(ev.payload));

// ---------------------------------------------------------------------------
// Hugging Face token (#60). Optional; unlocks gated models and Spaces that
// read HF_TOKEN / HUGGING_FACE_HUB_TOKEN. Stored on disk by the Rust side;
// the value never round-trips back to the frontend, only whether one is set.
// ---------------------------------------------------------------------------

export const setHfToken = (token: string | null) => invoke<void>("set_hf_token", { token });

export const hfTokenStatus = () => invoke<boolean>("hf_token_status");

// ---------------------------------------------------------------------------
// Hugging Face catalog
// ---------------------------------------------------------------------------

export type Compat = "ready" | "maybe" | "incompatible";

export type SpaceSummary = {
  /** `owner/name` */
  id: string;
  author: string;
  name: string;
  sdk: string | null;
  sdk_version: string | null;
  likes: number;
  /** HF hardware tier the Space asks for, e.g. `cpu-basic`, `zero-a10g`, `t4-small` */
  hardware: string | null;
  /** port the container listens on (Gradio default 7860, Streamlit 8501) */
  app_port: number;
  /** entry file for gradio/streamlit Spaces, default `app.py` */
  app_file: string;
  title: string | null;
  emoji: string | null;
  compat: Compat;
  compat_reason: string | null;
  /** env var names the Space's source reads that the user must supply (#57) */
  secrets: string[];
  /** HF's one-line summary ("Generate images from text prompts"); only when the
   * result came from a query or category search */
  description: string | null;
  /** HF's category label ("Image Generation"); same source as `description` */
  category: string | null;
};

/** One page of Store results; `next_cursor` is opaque, `null` on the last page. */
export type SpacePage = { items: SpaceSummary[]; next_cursor: string | null };
export type ModelPage = { items: ModelSummary[]; next_cursor: string | null };

/** Space categories the Store offers, a subset of what HF's semantic search
 * accepts (see `SPACE_CATEGORIES` in hf.rs), in the order shown. */
export const SPACE_CATEGORIES: { slug: string; label: string }[] = [
  { slug: "image-generation", label: "Image generation" },
  { slug: "image-editing", label: "Image editing" },
  { slug: "video-generation", label: "Video generation" },
  { slug: "chatbots", label: "Chatbots" },
  { slug: "text-generation", label: "Text generation" },
  { slug: "speech-synthesis", label: "Speech synthesis" },
  { slug: "voice-cloning", label: "Voice cloning" },
  { slug: "music-generation", label: "Music generation" },
  { slug: "object-detection", label: "Object detection" },
  { slug: "image-captioning", label: "Image captioning" },
  { slug: "visual-qa", label: "Visual QA" },
  { slug: "ocr", label: "OCR" },
  { slug: "document-analysis", label: "Document analysis" },
  { slug: "background-removal", label: "Background removal" },
  { slug: "image-upscaling", label: "Image upscaling" },
  { slug: "style-transfer", label: "Style transfer" },
  { slug: "3d-modeling", label: "3D modeling" },
  { slug: "code-generation", label: "Code generation" },
  { slug: "language-translation", label: "Translation" },
  { slug: "text-summarization", label: "Summarization" },
];

/** Model pipelines the Store can filter on; both are what Ollama serves. */
export const MODEL_PIPELINES: { slug: string; label: string }[] = [
  { slug: "text-generation", label: "Chat (text)" },
  { slug: "image-text-to-text", label: "Chat with images" },
];

export type GgufFile = {
  filename: string;
  /** e.g. `Q4_K_M`, `IQ4_XS`, `BF16` */
  quant: string;
  size_bytes: number;
  /** whether the file fits the local GPU: fully, with CPU offload, or not at all */
  fits: "gpu" | "partial" | "no";
};

export type ModelSummary = {
  /** `owner/name` */
  id: string;
  author: string;
  name: string;
  downloads: number;
  likes: number;
  pipeline_tag: string | null;
  compat: Compat;
  compat_reason: string | null;
};

/** Empty query and no category: HF's most-liked listing, paged by the Hub.
 * Otherwise HF's semantic search (which also yields `description`), paged locally. */
export const searchSpaces = (
  query: string,
  opts: { category?: string | null; cursor?: string | null; limit?: number } = {},
) =>
  invoke<SpacePage>("search_spaces", {
    query,
    category: opts.category ?? null,
    cursor: opts.cursor ?? null,
    limit: opts.limit ?? 24,
  });

export const searchModels = (
  query: string,
  opts: { pipeline?: string | null; cursor?: string | null; limit?: number } = {},
) =>
  invoke<ModelPage>("search_models", {
    query,
    pipeline: opts.pipeline ?? null,
    cursor: opts.cursor ?? null,
    limit: opts.limit ?? 24,
  });

/** GGUF files of one model repo, sorted by size ascending, with fit annotations. */
export const modelFiles = (repo: string) => invoke<GgufFile[]>("model_files", { repo });

// ---------------------------------------------------------------------------
// Instances (things we launched)
// ---------------------------------------------------------------------------

export type InstanceKind = "space" | "model";

export type InstanceStatus = "pulling" | "building" | "starting" | "running" | "error" | "stopped";

export type Instance = {
  /** stable id, safe for DOM keys and container names */
  id: string;
  kind: InstanceKind;
  /** HF repo id (`owner/name`) */
  repo: string;
  /** for models: `hf.co/owner/name:QUANT`, the name Ollama knows it by */
  model_tag: string | null;
  display_name: string;
  status: InstanceStatus;
  /** Windows-side port the thing is reachable on */
  port: number | null;
  /** http URL to open in the browser (Spaces) or the API base (models) */
  url: string | null;
  error: string | null;
  /** last few log lines, most recent last */
  log_tail: string[];
  started_at: string;
  /** image was built on this machine instead of pulled from the Hub */
  local_build: boolean;
  /** container was started with the GPU; CPU-tier Spaces run without it */
  gpu: boolean;
  /** size of the image being (or about to be) pulled, from the registry manifest; null when unknown */
  pull_size_mb: number | null;
  /** aggregate `docker pull` progress across layers, 0..100; null outside `pulling` */
  progress_pct: number | null;
};

/**
 * `env` supplies values for `SpaceSummary.secrets` (#57); kept in memory only.
 * `leaseId` comes from `GpuPlan.lease_id` (via `useGpuGate`'s `gate()`) when the
 * launch was gated; required whenever this machine has a GPU budget to schedule
 * against, so two launches cannot double-book VRAM (#70).
 */
export const launchSpace = (id: string, env?: Record<string, string>, leaseId?: string | null) =>
  invoke<Instance>("launch_space", { id, env, leaseId });

/**
 * Clone the Space and build its image here, then run it. For GPUs the Hub never
 * built for (AMD, Intel, CPU) and for Spaces without an image. Non-NVIDIA builds
 * target the CPU; Spaces needing CUDA-only packages are refused with the reason.
 *
 * `useRepoDockerfile` opts into running the Space repo's own Dockerfile verbatim
 * (network access, arbitrary `RUN` steps) instead of the generated one; Store.tsx
 * gates it behind a one-time consent dialog and defaults it off.
 */
export const buildSpace = (
  id: string,
  useRepoDockerfile: boolean,
  env?: Record<string, string>,
  leaseId?: string | null,
) => invoke<Instance>("build_space", { id, useRepoDockerfile, env, leaseId });

/**
 * `repo` is a Hugging Face GGUF repo (`owner/name`) with `quant` the file's quant tag,
 * or a bare Ollama library name (`qwen2.5vl`) with `quant` the library tag (`7b`).
 */
export const launchModel = (repo: string, quant: string, leaseId?: string | null) =>
  invoke<Instance>("launch_model", { repo, quant, leaseId });

export const listInstances = () => invoke<Instance[]>("list_instances");

export const stopInstance = (id: string) => invoke<void>("stop_instance", { id });

export const removeInstance = (id: string) => invoke<void>("remove_instance", { id });

export const openUrl = (url: string) => invoke<void>("open_url", { url });

/** Fires whenever an instance changes status or gains log lines. */
export const onInstanceUpdate = (cb: (i: Instance) => void): Promise<UnlistenFn> =>
  listen<Instance>("instance://update", (ev) => cb(ev.payload));

// ---------------------------------------------------------------------------
// Platform services: extra inference servers the store runs as containers
// inside the distro (Speaches for STT/TTS, ComfyUI for images, a KServe v2
// CV server for detection).
// ---------------------------------------------------------------------------

export type ServiceId = "speaches" | "comfyui" | "cv" | "whisper";

/** `unavailable`: this machine's GPU vendor has no implementation of the service. */
export type ServiceState =
  | "missing"
  | "pulling"
  | "starting"
  | "running"
  | "stopped"
  | "error"
  | "unavailable";

export type ServiceStatus = {
  id: ServiceId;
  display_name: string;
  state: ServiceState;
  /** Windows-side port; the service's OpenAI-style API lives at `http://localhost:{port}` */
  port: number;
  url: string;
  /** the image (container) or unpacked program (native) is present locally */
  image_present: boolean;
  /** which implementation this machine got: `cuda`, `cpu`, `triton`, `openvino`, `vulkan`, `rocm`, `xpu` */
  backend: string;
  /** `container` inside the WSL2 distro, `native` Windows process, or `none` when unavailable */
  runtime: "container" | "native" | "none";
  error: string | null;
  log_tail: string[];
};

export const serviceStatus = (id: ServiceId) => invoke<ServiceStatus>("service_status", { id });

/**
 * Pulls the image if needed, starts the container, waits for health. Progress on
 * `service://update`. `leaseId` (from `gate()`) is required for ComfyUI, the one
 * service heavy enough to contend for the shared GPU budget (#70).
 */
export const startService = (id: ServiceId, leaseId?: string | null) =>
  invoke<ServiceStatus>("start_service", { id, leaseId });

export const stopService = (id: ServiceId) => invoke<void>("stop_service", { id });

/**
 * Tell the store this service was just used, for services whose calls go
 * straight from the webview (Speaches STT/TTS, whisper.cpp): the idle-stop
 * timer (15 min) resets from here instead of firing under an active session.
 */
export const serviceTouch = (id: ServiceId) => invoke<void>("service_touch", { id });

export const onServiceUpdate = (cb: (s: ServiceStatus) => void): Promise<UnlistenFn> =>
  listen<ServiceStatus>("service://update", (ev) => cb(ev.payload));

/** A model file a service needs beyond its image (e.g. a checkpoint for ComfyUI). */
export type ServiceModel = { name: string; display_name: string; path: string; installed: boolean };

/**
 * Speaches / ComfyUI: requires the service container to be running (the probe runs inside it).
 * CV: reads the shared model store, works before the service is started.
 */
export const serviceModels = (id: ServiceId) => invoke<ServiceModel[]>("service_models", { id });

/** Long-running download; progress lines arrive on `service://update` log_tail. */
export const installServiceModel = (id: ServiceId, name: string) =>
  invoke<ServiceModel>("install_service_model", { id, name });

// ---------------------------------------------------------------------------
// CV (object detection). The service speaks KServe v2 (Triton on NVIDIA,
// OpenVINO Model Server elsewhere); the Rust side is the client so the UI is
// the same whichever server answers. Boxes come back in pixels of the image sent.
// ---------------------------------------------------------------------------

export const CV_PORT = 8900;

export type CvDetection = {
  label: string;
  /** 0..1 confidence */
  score: number;
  /** x1, y1, x2, y2 in pixels of the image that was sent */
  box: [number, number, number, number];
};

export type CvDetectResult = {
  model: string;
  /** which server answered: `triton` or `openvino` */
  backend: string;
  detections: CvDetection[];
  inference_ms: number;
  total_ms: number;
};

/** `image` is an encoded JPEG or PNG; `model` is a `serviceModels("cv")` name. */
export const cvDetect = (image: Uint8Array, model: string, minScore = 0.25) =>
  invoke<CvDetectResult>("cv_detect", image, {
    headers: { "x-model": model, "x-min-score": String(minScore) },
  });

// ---------------------------------------------------------------------------
// GPU memory scheduling. One card holds one modality's model family at a time;
// before a launch the UI asks what must be unloaded, shows it, and releases it.
// ---------------------------------------------------------------------------

export type ResidentKind = "model" | "service" | "space";

export type Resident = {
  kind: ResidentKind;
  /** model tag, ServiceId, or instance id */
  id: string;
  name: string;
  /** memory held now; null when the runtime cannot report it (Space containers) */
  vram_mb: number | null;
  /** holds a whole model family; only heavy residents are ever evicted */
  heavy: boolean;
};

export type GpuMemory = {
  /** the accelerator's budget (dedicated VRAM or the unified share); null on CPU machines */
  budget_mb: number | null;
  /** device-wide memory in use when a runtime reports it */
  used_mb: number | null;
  residents: Resident[];
};

export type GpuRequest =
  | { kind: "model"; tag: string; size_bytes: number | null }
  | { kind: "service"; id: ServiceId }
  /** `id` is the Space's `owner/name` */
  | { kind: "space"; id: string };

export type GpuPlan = {
  need_mb: number | null;
  budget_mb: number | null;
  /** memory held by what stays */
  resident_mb: number;
  /** unload these, in order, before the launch */
  evict: Resident[];
  fits_without_eviction: boolean;
  /**
   * Held for 120s from this plan through the matching launch; pass it to
   * `launchSpace` / `launchModel` / `startService`. `null` on a machine with no
   * GPU budget, where nothing needs a lock (#70).
   */
  lease_id: string | null;
};

export const gpuMemory = () => invoke<GpuMemory>("gpu_memory");

export const gpuPlan = (request: GpuRequest) => invoke<GpuPlan>("gpu_plan", { request });

/** Unload one resident: `keep_alive: 0` for models, `/free` for ComfyUI, stop for the rest. */
export const gpuRelease = (resident: Resident) => invoke<void>("gpu_release", { resident });

/** ComfyUI HTTP API (`/prompt`, `/history/{id}`, `/view`, `/upload/image`). */
export const COMFYUI_PORT = 8188;
export const COMFYUI_BASE_URL = `http://localhost:${COMFYUI_PORT}`;

/** Speaches (OpenAI-compatible speech API). Same base for STT and TTS. */
export const SPEACHES_PORT = 8880;
export const SPEACHES_BASE_URL = `http://localhost:${SPEACHES_PORT}/v1`;

/**
 * whisper.cpp server (AMD / Intel GPUs via Vulkan, a native Windows process).
 * `POST {base}/audio/transcriptions` takes the same multipart form as Speaches
 * but only WAV audio; the reply is `{ text }`.
 */
export const WHISPER_PORT = 8881;
export const WHISPER_BASE_URL = `http://localhost:${WHISPER_PORT}/v1`;

// ---------------------------------------------------------------------------
// Chat (base UI for text models). Ollama speaks the OpenAI API on this base.
// The frontend calls it directly with fetch + streaming; no Rust round trip.
// ---------------------------------------------------------------------------

export const OLLAMA_BASE_URL = "http://localhost:11434/v1";

// ---------------------------------------------------------------------------
// Storage: Docker images, build cache, the HF cache volume, our local build
// directory, and the WSL virtual disk on Windows (#63).
// ---------------------------------------------------------------------------

export type StorageUsage = {
  images_total_mb: number;
  images_reclaimable_mb: number;
  build_cache_mb: number;
  build_cache_reclaimable_mb: number;
  hf_cache_mb: number;
  build_dir_mb: number;
  /** size of the WSL virtual disk on Windows; `null` when it cannot be located */
  vhdx_mb: number | null;
};

export type StorageCleanupOptions = {
  dangling_images: boolean;
  build_cache: boolean;
  /** `registry.hf.space/*` / `aias-local/*` images no running or starting instance uses */
  unused_space_images: boolean;
  /** trim HF cache entries older than this many days; omit to skip the step */
  hf_cache_older_than_days?: number;
  /** shrinks the WSL virtual disk; stops the distro (Docker, Ollama, every instance) while it runs */
  compact_vhd: boolean;
};

export type StorageCleanupResult = {
  freed_mb: number;
  usage: StorageUsage;
};

export type StorageProgressEvent = {
  step: string;
  status: "start" | "ok" | "error" | "log";
  message: string;
};

export const storageUsage = () => invoke<StorageUsage>("storage_usage");

/** Long-running. Progress arrives on `storage://progress`; resolves with the freed total and fresh usage. */
export const storageCleanup = (options: StorageCleanupOptions) =>
  invoke<StorageCleanupResult>("storage_cleanup", { options });

export const onStorageProgress = (cb: (e: StorageProgressEvent) => void): Promise<UnlistenFn> =>
  listen<StorageProgressEvent>("storage://progress", (ev) => cb(ev.payload));

// ---------------------------------------------------------------------------
// Store self-update (tauri-plugin-updater). The endpoint is the repository's
// latest GitHub Release; the MSI is verified against the public key in
// tauri.conf.json before it is installed.
// ---------------------------------------------------------------------------

export type AvailableUpdate = {
  version: string;
  currentVersion: string;
  date: string | null;
  notes: string | null;
};

export const APP_VERSION = __APP_VERSION__;

let pendingUpdate: Awaited<ReturnType<typeof checkUpdate>> = null;

/** Resolves to null when this build is current or the endpoint is unreachable (error is thrown then). */
export async function checkForUpdate(): Promise<AvailableUpdate | null> {
  pendingUpdate = await checkUpdate();
  if (!pendingUpdate) return null;
  return {
    version: pendingUpdate.version,
    currentVersion: pendingUpdate.currentVersion,
    date: pendingUpdate.date ?? null,
    notes: pendingUpdate.body ?? null,
  };
}

/** Download and install the update found by `checkForUpdate`; progress in bytes. */
export async function installUpdate(
  onProgress?: (downloaded: number, total: number | null) => void,
): Promise<void> {
  if (!pendingUpdate) throw new Error("call checkForUpdate first");
  let downloaded = 0;
  let total: number | null = null;
  await pendingUpdate.downloadAndInstall((ev: DownloadEvent) => {
    if (ev.event === "Started") total = ev.data.contentLength ?? null;
    else if (ev.event === "Progress") downloaded += ev.data.chunkLength;
    onProgress?.(downloaded, total);
  });
}

export const relaunch = () => relaunchApp();
