# Architecture

This PoC is one vertical slice of a unified edge AI runtime: a store that
installs a runtime on heterogeneous devices and runs apps and models from
public hubs on it. The slice covers Windows + NVIDIA + Hugging Face. This
document records the full layer model so the slice stays aligned with it.

## Layer model

```text
  control plane   fleet, OTA, registry mirror, accounts        (not in PoC)
  ─────────────────────────────────────────────────────────────────────────
  app layer       OCI image + Compose-style manifest            Spaces
                  base UIs per modality: chat / voice / camera  Chat only
  model layer     model artifact + inference API                GGUF via Ollama
                  runtime chosen per hardware                   OpenAI-compatible
  driver layer    kernel driver on host + CDI for containers    NVIDIA toolkit
  host            Windows + WSL2 distro / Linux                 WSL2 distro
```

Two rules make "write once, deploy many" real:

1. **Apps declare, the platform resolves.** An app never bundles a runtime
   tied to a GPU vendor. It declares what it needs (accelerator class, VRAM,
   a model by id) and the platform picks the runtime build for the device.
2. **The unified surface is the API, not the binary.** Apps talk to models
   over an OpenAI-compatible (LLM) or KServe v2 (CV) endpoint on localhost.
   Behind it the platform runs vLLM, Ollama, OpenVINO Model Server, or an NPU
   vendor stack. Arbitrary CUDA code cannot be translated to an NPU; a model
   behind an API can be.

## Compatibility verdicts

The store shows, per device, whether an item can run before the user clicks.
Verdicts are built from three sources of increasing confidence:

| Source | Example | Confidence |
|--------|---------|------------|
| Adapter inference | Space SDK `static`, HF hardware tier `zero-a10g`, GGUF size vs VRAM | low |
| Developer manifest | declared accelerator, VRAM, ports (Compose `deploy.resources`, `models:`) | medium |
| CI on reference devices | the image actually ran on this hardware class | high |

The PoC implements the first row: `ready` / `maybe` / `incompatible` with a
reason string. An `incompatible` item cannot be launched.

## What the PoC implements

| Piece | Implementation |
|-------|----------------|
| Runtime detection | `wsl.exe --status`, distro list, one probe script inside the distro (Docker, NVIDIA runtime, Ollama, GPU name and VRAM); a host probe for firmware virtualization (WSL2 needs the hypervisor) |
| Provisioning | `wsl --install --no-distribution`, `wsl --install -d Ubuntu-24.04 --name ai-app-store --no-launch`, then `provision.sh` over stdin, then a distro restart so systemd owns Docker and Ollama |
| Keepalive | WSL stops a distro seconds after its last client exits; the app holds `wsl -d ai-app-store -- sleep infinity` while it runs |
| Spaces | `docker pull registry.hf.space/<owner>-<name>:latest`, `docker run -p <free port>:<app_port> --gpus all`, poll the port until it answers; labels `aias.*` let a restarted app re-adopt containers |
| Models | `ollama pull hf.co/<repo>:<quant>`, warm load through `/api/generate`, served on `localhost:11434/v1`; the Chat screen streams from it directly |
| Contract | every command and event is typed once in `src/lib/api.ts`; the Rust side mirrors it with serde |

## Hardware vendors

The store probes the host (PowerShell: video controllers with VRAM from the
driver registry key, NPU devices) and picks a primary vendor. Everything below
the UI keys off it; the four base UIs only ever talk to `localhost` APIs.

| | NVIDIA | AMD / Intel / CPU |
|---|---|---|
| LLM | Ollama inside WSL2 (CUDA) | Ollama for Windows, started headless by the store (ROCm on supported Radeon, Vulkan otherwise) |
| Speech | Speaches CUDA container | AMD / Intel: whisper.cpp (Vulkan) as a native Windows process for STT, Speaches CPU container for TTS. CPU: Speaches CPU container |
| Images | ComfyUI CUDA container | AMD / Intel: ComfyUI's official portable build as a native Windows process (ROCm / XPU). CPU: ComfyUI CPU container |
| Detection | Triton Inference Server container (ONNX Runtime on CUDA) | OpenVINO Model Server container; same KServe v2 API and ONNX files. On an Intel machine with an NPU it targets the NPU (`target_device: NPU` in `ovms.json`), otherwise the CPU |
| Spaces | pull the Hub's CUDA image | pull, or build locally for the CPU |
| NPU | detected and shown; not used for acceleration yet | same |

WSL2 only exposes NVIDIA GPUs to containers, which is why on AMD and Intel
every GPU-accelerated service (Ollama, whisper.cpp, ComfyUI) is a native
Windows process and only CPU-bound services stay in containers. `services.rs`
models both: a spec is either a `Container` (image, ports, volumes, command)
or `Native` (archives to download, program, arguments); the store fetches,
starts, health-checks and stops them the same way, and the four base UIs only
ever see `localhost` ports. Native programs unpack into
`%LOCALAPPDATA%\ai-app-store\services\<id>\` with the system `tar.exe`
(zip and 7z). whisper.cpp has no upstream Vulkan Windows build, so
`runtimes.yml` builds it and publishes `whisper-server-vulkan-x64.zip` on the
`runtimes` release. NPUs are not visible inside WSL2 at all.

## CV serving

Detection does not go through the VLM. The store runs a KServe v2 model server
as a platform service on `localhost:8900`: Triton on NVIDIA, OpenVINO Model
Server elsewhere. Both read the same model repository, a named volume laid out
as `repo/<model>/1/model.onnx`; the store downloads ONNX exports from the Hub
into it (YOLOv10 to start) and rewrites `ovms.json` so OpenVINO picks them up,
while Triton polls the directory. The Rust side is the only client: it
letterboxes the image, sends one FP32 tensor with the KServe binary extension,
and maps the detector's rows back to image pixels. The Vision screen therefore
draws real boxes with scores in well under a second on a GPU, and the same
screen works unchanged when the server underneath is swapped. The VLM path
stays for open-ended questions about an image.

## Local build

The Hub builds Space images on NVIDIA tiers only. When a Space has no image, or
the GPU is not NVIDIA, the store clones the Space into the distro, generates a
Dockerfile for gradio / streamlit Spaces (slim Python, SDK pinned to
`sdk_version`, requirements, `app_file` as entrypoint) or uses the Space's own
Dockerfile, and builds `aias-local/<slug>`. Dependencies are resolved with
`uv --exclude-newer <SDK release date + 7 days>` so an old Space gets the
dependency set of its era (a 2023 gradio with today's starlette does not start).
Non-NVIDIA builds pin `torch` to the CPU wheel index. Requirements that only ship CUDA builds (flash-attn, xformers,
bitsandbytes, custom kernels) are refused up front with the reason instead of
failing fifteen minutes in.

## Store updates

`tauri-plugin-updater` reads `latest.json` from the repository's latest GitHub
Release and verifies the MSI against the public key in `tauri.conf.json`.
`release.yml` builds and signs on `v*` tags. The runtime inside the distro is
updated by re-running the idempotent `provision.sh`. Note: GitHub release assets
of a private repository are not reachable by the updater; publish the repo or
mirror `latest.json` and the MSI to a public HTTPS host.

## What a product adds

- **Installer**: MSI (Tauri bundler) that also ships a pre-baked distro rootfs
  (`wsl --import`) instead of provisioning over the network; silent install
  for system integrators (`msiexec /qn`), offline bundle, EV signing.
- **Updates on three channels**: app via MSI upgrade or `tauri-plugin-updater`;
  distro via A/B import and switch; apps and models via new image tags.
  The agent is the channel that updates the other two, so it ships first.
- **Fleet**: device profiles reported to a control plane, staged rollouts,
  rollback, on-prem registry mirror for sites without internet.
- **Heterogeneous hardware**: OpenVINO (Intel), ROCm (AMD), NPU vendor
  backends behind the same inference API; CDI spec files per accelerator.
- **Private hub**: Harbor for images and model artifacts, a catalog frontend,
  CI that builds one image tag per hardware class. Imports from Hugging Face
  and GitHub go through the same CI so devices only ever pull.
- **Base UIs**: voice (STT/TTS) and camera/video (CV) next to Chat, chosen by
  the model's `pipeline_tag`.
