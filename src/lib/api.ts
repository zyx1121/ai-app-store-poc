// Typed contract between the React frontend and the Rust backend.
// Every Tauri command and event used by the UI is declared here; nothing else
// in `src/` may call `invoke` or `listen` directly.

import { invoke } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";

// ---------------------------------------------------------------------------
// Runtime (WSL2 distro + Docker + NVIDIA + Ollama)
// ---------------------------------------------------------------------------

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
  vram_mb: number | null;
  /** everything above that is required is true */
  ready: boolean;
  /** a Windows reboot is required before continuing (WSL feature just enabled) */
  reboot_required: boolean;
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
// Hugging Face catalog
// ---------------------------------------------------------------------------

export type Compat = "ready" | "maybe" | "incompatible";

export type SpaceSummary = {
  /** `owner/name` */
  id: string;
  author: string;
  name: string;
  sdk: string | null;
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
};

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

export const searchSpaces = (query: string, limit = 24) =>
  invoke<SpaceSummary[]>("search_spaces", { query, limit });

export const searchModels = (query: string, limit = 24) =>
  invoke<ModelSummary[]>("search_models", { query, limit });

/** GGUF files of one model repo, sorted by size ascending, with fit annotations. */
export const modelFiles = (repo: string) => invoke<GgufFile[]>("model_files", { repo });

// ---------------------------------------------------------------------------
// Instances (things we launched)
// ---------------------------------------------------------------------------

export type InstanceKind = "space" | "model";

export type InstanceStatus = "pulling" | "starting" | "running" | "error" | "stopped";

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
};

export const launchSpace = (id: string) => invoke<Instance>("launch_space", { id });

export const launchModel = (repo: string, quant: string) =>
  invoke<Instance>("launch_model", { repo, quant });

export const listInstances = () => invoke<Instance[]>("list_instances");

export const stopInstance = (id: string) => invoke<void>("stop_instance", { id });

export const removeInstance = (id: string) => invoke<void>("remove_instance", { id });

export const openUrl = (url: string) => invoke<void>("open_url", { url });

/** Fires whenever an instance changes status or gains log lines. */
export const onInstanceUpdate = (cb: (i: Instance) => void): Promise<UnlistenFn> =>
  listen<Instance>("instance://update", (ev) => cb(ev.payload));

// ---------------------------------------------------------------------------
// Chat (base UI for text models). Ollama speaks the OpenAI API on this base.
// The frontend calls it directly with fetch + streaming; no Rust round trip.
// ---------------------------------------------------------------------------

export const OLLAMA_BASE_URL = "http://localhost:11434/v1";
