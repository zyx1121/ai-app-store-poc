```
 █████╗ ██╗     █████╗ ██████╗ ██████╗     ███████╗████████╗ ██████╗ ██████╗ ███████╗
██╔══██╗██║    ██╔══██╗██╔══██╗██╔══██╗    ██╔════╝╚══██╔══╝██╔═══██╗██╔══██╗██╔════╝
███████║██║    ███████║██████╔╝██████╔╝    ███████╗   ██║   ██║   ██║██████╔╝█████╗  
██╔══██║██║    ██╔══██║██╔═══╝ ██╔═══╝     ╚════██║   ██║   ██║   ██║██╔══██╗██╔══╝  
██║  ██║██║    ██║  ██║██║     ██║         ███████║   ██║   ╚██████╔╝██║  ██║███████╗
╚═╝  ╚═╝╚═╝    ╚═╝  ╚═╝╚═╝     ╚═╝         ╚══════╝   ╚═╝    ╚═════╝ ╚═╝  ╚═╝╚══════╝
                                                                                     
```

# ai-app-store-poc

> Browse Hugging Face like an app store, click Run, and it is serving on your own GPU box a minute later.

`Tauri 2` · `Rust` · `WSL2 + Docker` · `Ollama` · `Hugging Face Hub`

[![CI](https://github.com/zyx1121/ai-app-store-poc/actions/workflows/ci.yml/badge.svg)](https://github.com/zyx1121/ai-app-store-poc/actions/workflows/ci.yml) &nbsp;[![version](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fraw.githubusercontent.com%2Fzyx1121%2Fai-app-store-poc%2Fmain%2Fsrc-tauri%2Ftauri.conf.json&query=%24.version&label=version&color=111111)](src-tauri/tauri.conf.json) &nbsp;[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](#license)

Edge AI boxes ship with a GPU and nothing to run on it. Getting one Hugging Face demo onto such a machine means a Linux VM, Docker, the right CUDA toolkit, and an afternoon. This PoC is the "store" half of a unified edge runtime: one Windows installer sets up the runtime, one window browses Spaces and GGUF models, and one click runs them locally with an honest compatibility verdict first.

```text
   ┌──────────────────────────── Windows ─────────────────────────────┐
   │  AI App Store (Tauri)                                            │
   │   Store ─ Running ─ Chat                 Rust agent              │
   │      │  invoke                              │ wsl.exe            │
   │      ▼                                      ▼                    │
   │  ┌───────────────────── WSL2 distro "ai-app-store" ───────────┐ │
   │  │  Docker Engine + NVIDIA toolkit        Ollama (GPU)         │ │
   │  │   registry.hf.space/<space>  ◀──pull   hf.co/<repo>:<quant> │ │
   │  │   container :7860 ──▶ localhost:<port> ──▶ browser          │ │
   │  └─────────────────────────────────────────────────────────────┘ │
   └──────────────────────────────────────────────────────────────────┘
```

<sub>Apps are OCI images that Hugging Face already builds for every Space; models are GGUF files Ollama pulls straight from the Hub. The store never builds on the device.</sub>

## Features

- **Install the runtime in one click**: WSL2, an isolated Ubuntu distro, Docker Engine, NVIDIA container toolkit and Ollama, with a live log and a reboot prompt when Windows needs one.
- **Store: Spaces and GGUF models live**: search the Hub by text or category, scroll without end, see likes, SDK, hardware tier and a compatibility badge (ready / maybe / incompatible, with the reason) before you run anything.
- **Run and use**: a Space becomes a container on a free local port and opens in the browser; a model is pulled into Ollama and served on an OpenAI-compatible API with a built-in chat UI.

## Install

Windows 11 with virtualization enabled. NVIDIA GPUs get the full CUDA path; AMD and Intel GPUs run LLMs natively through Ollama (ROCm / Vulkan) with speech and image services on the CPU; NPUs are detected but not used yet. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#hardware-vendors).

1. Download the latest `.msi` from [Releases](https://github.com/zyx1121/ai-app-store-poc/releases) and install it.
2. Open **AI App Store**. The Setup screen lists what is missing; click **Install runtime**. If WSL was just enabled, reboot when asked and open the app again.
3. Search, click **Run**, and open the result from **Running**.

Uninstall from Settings > Apps. The WSL distro and its images stay until you run `wsl --unregister ai-app-store`.

## Develop

Needs Rust ([rustup](https://rustup.rs)), [Bun](https://bun.sh), and on Windows the MSVC Build Tools plus WebView2 (preinstalled on Windows 11).

```bash
git clone https://github.com/zyx1121/ai-app-store-poc && cd ai-app-store-poc
bun install
bun tauri dev            # desktop window with hot reload
bun tauri build          # src-tauri/target/release/bundle/msi/*.msi
```

`cargo test` and `cargo clippy --all-targets -- -D warnings` run inside `src-tauri/`; the frontend type-checks with `bun run build`.

### Release

The version lives in `package.json`, `src-tauri/tauri.conf.json`, `src-tauri/Cargo.toml` and `src-tauri/Cargo.lock`; CI fails when they disagree. To ship a version:

```bash
bun run bump 0.3.0       # rewrites every version file
```

Open a PR with the bump and merge it. The `release` workflow tags `v0.3.0`, builds and signs the MSI, and publishes it with `latest.json`, which the app's Setup page checks against. Setup shows the version and the short commit hash of the running build.

## How it works

| Layer | What | Where |
|-------|------|-------|
| UI | React 19, Tailwind v4, shadcn/ui; talks to Rust only through [`src/lib/api.ts`](src/lib/api.ts) | `src/` |
| Agent | Tauri commands: runtime detection and provisioning, Hub search with compatibility verdicts, instance lifecycle | `src-tauri/src/` |
| Runtime | WSL2 distro `ai-app-store` with systemd, Docker Engine, NVIDIA container toolkit, Ollama; provisioned by [`provision.sh`](src-tauri/provision.sh) | inside WSL |
| Catalog | Hugging Face Hub API; Spaces run from `registry.hf.space`, models from `hf.co/<repo>:<quant>` via Ollama | remote |

The layer model this PoC is a slice of, and what a production version adds (fleet, OTA, NPU backends, private registry), is in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

How new AI PCs are supported (the accelerator adapter model and the device onboarding SOP) is in [docs/adr/0001-accelerator-adapters.md](docs/adr/0001-accelerator-adapters.md).

## Roadmap

- [ ] Pre-baked distro rootfs shipped with the installer instead of provisioning on first run
- [x] Auto-update for the app (tauri-plugin-updater; latest.json on the latest GitHub Release)
- [x] Hardware profile and per-vendor runtime (NVIDIA / AMD / Intel / CPU)
- [x] Build Space images locally when the Hub has none for this GPU
- [x] Native ROCm / XPU ComfyUI and Vulkan whisper.cpp on AMD / Intel (code complete; verified for the download, unpack and process plumbing until an AMD or Intel box runs it)
- [ ] NPU backends
- [ ] Space secrets and HF token for gated repos
- [x] Voice and vision base UIs next to Chat
- [x] Real CV serving (KServe v2: Triton on NVIDIA, OpenVINO Model Server elsewhere) behind the Vision screen
- [ ] Segmentation, pose and OCR models on the CV server
- [ ] Linux host support (no WSL layer)

## Contributing

Issues and PRs welcome: start with [CONTRIBUTING.md](https://github.com/zyx1121/.github/blob/main/CONTRIBUTING.md).

## License

[MIT](LICENSE) · it is a store where everything costs one click
