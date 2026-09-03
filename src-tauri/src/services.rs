//! Platform services: inference servers the store itself runs next to Ollama.
//! Speaches (STT/TTS), ComfyUI (images), a KServe v2 CV server (Triton on
//! NVIDIA, OpenVINO Model Server elsewhere) and whisper.cpp (GPU speech to text
//! on AMD / Intel). Each service is a fixed spec; the store only fetches,
//! starts, health-checks and stops it.
//!
//! Two runtimes: containers inside the distro (the default), and native
//! Windows processes for the GPU backends WSL2 cannot see (ROCm, XPU, Vulkan),
//! mirroring how `ollama.rs` already runs natively on non-NVIDIA machines.

use std::path::PathBuf;
use std::pin::Pin;
use std::process::Stdio;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Emitter, Manager};
use tokio::io::{AsyncBufReadExt, AsyncRead, BufReader};
use tokio::process::Command;

use crate::cv;
use crate::error::{Error, Result};
use crate::fetch;
use crate::hardware::Vendor;
use crate::state::AppState;
use crate::wsl;

const UPDATE_EVENT: &str = "service://update";
const LOG_TAIL: usize = 20;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "lowercase")]
pub enum ServiceId {
    Speaches,
    Comfyui,
    /// Object detection and other CV models behind the KServe v2 API.
    Cv,
    /// whisper.cpp server: GPU speech to text where Speaches can only use the CPU.
    Whisper,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ServiceState {
    Missing,
    Pulling,
    Starting,
    Running,
    Stopped,
    Error,
    /// This machine's vendor has no implementation of the service.
    Unavailable,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceStatus {
    pub id: ServiceId,
    pub display_name: String,
    pub state: ServiceState,
    pub port: u16,
    pub url: String,
    /// the image (container) or the unpacked program (native) is present locally
    pub image_present: bool,
    /// which implementation this machine got, e.g. `cuda`, `cpu`, `triton`, `openvino`, `vulkan`, `rocm`
    pub backend: String,
    /// `container` inside the distro, or `native` Windows process
    pub runtime: &'static str,
    pub error: Option<String>,
    pub log_tail: Vec<String>,
}

/// How a service runs on this machine.
#[derive(Clone, Copy)]
enum Runtime {
    /// A Docker container inside the distro, published on `host_port`.
    Container {
        image: &'static str,
        container: &'static str,
        container_port: u16,
        gpu: bool,
        env: &'static [(&'static str, &'static str)],
        /// (named volume, container path)
        volumes: &'static [(&'static str, &'static str)],
        /// arguments appended after the image (the container's command)
        cmd: &'static str,
        /// Model files live in this (named volume, container mount) and are written from the
        /// distro side, so they can be installed before the service runs and survive a
        /// backend swap. Without it, files are fetched with `docker exec` inside the container.
        model_store: Option<(&'static str, &'static str)>,
    },
    /// A Windows process unpacked into `%LOCALAPPDATA%\ai-app-store\services\<id>\`.
    Native {
        /// archives to download and unpack into the install dir, in order: (url,
        /// archive file name, expected SHA-256). `None` means no compile-time
        /// digest is embedded; `run_native` then falls back to the `SHA256SUMS`
        /// file this repository's own `runtimes` release publishes beside its
        /// assets (see `fetch::release_checksum`), and fails the download if
        /// even that yields nothing.
        downloads: &'static [(&'static str, &'static str, Option<&'static str>)],
        /// program to run, relative to the install dir
        exe: &'static str,
        args: &'static [&'static str],
        env: &'static [(&'static str, &'static str)],
        /// working directory, relative to the install dir
        cwd: &'static str,
        /// the program cannot start without every model file present
        models_required: bool,
    },
}

#[derive(Clone, Copy)]
struct ServiceSpec {
    id: ServiceId,
    display_name: &'static str,
    /// Windows-side port. Fixed so the frontend can hardcode the base URL.
    host_port: u16,
    /// shown to the user as the implementation this machine got
    backend: &'static str,
    /// path that answers 2xx once the service is usable
    health_path: &'static str,
    /// downloadable model files: (name shown to the user, URL, path). For containers the
    /// path is inside the container; for native services it is relative to the install dir.
    models: &'static [(&'static str, &'static str, &'static str)],
    runtime: Runtime,
}

const SPEACHES: ServiceSpec = ServiceSpec {
    id: ServiceId::Speaches,
    display_name: "Speaches (speech to text, text to speech)",
    // 8000 is taken by other WSL distros on the dev box; all distros share one network namespace.
    host_port: 8880,
    backend: "cuda",
    health_path: "/v1/models",
    models: &[],
    runtime: Runtime::Container {
        image: "ghcr.io/speaches-ai/speaches:latest-cuda",
        container: "aias-svc-speaches",
        container_port: 8000,
        gpu: true,
        env: &[
            (
                "ALLOW_ORIGINS",
                r#"["http://tauri.localhost","https://tauri.localhost","http://localhost:1420"]"#,
            ),
            ("ENABLE_UI", "false"),
        ],
        volumes: &[("aias-speaches-cache", "/home/ubuntu/.cache/huggingface/hub")],
        cmd: "",
        model_store: None,
    },
};

/// CPU build for machines whose GPU WSL2 cannot see (AMD, Intel) or that have none.
/// On AMD and Intel it keeps doing TTS while `WHISPER` takes over STT on the GPU.
const SPEACHES_CPU: ServiceSpec = ServiceSpec {
    backend: "cpu",
    runtime: Runtime::Container {
        image: "ghcr.io/speaches-ai/speaches:latest-cpu",
        container: "aias-svc-speaches",
        container_port: 8000,
        gpu: false,
        env: &[
            (
                "ALLOW_ORIGINS",
                r#"["http://tauri.localhost","https://tauri.localhost","http://localhost:1420"]"#,
            ),
            ("ENABLE_UI", "false"),
        ],
        volumes: &[("aias-speaches-cache", "/home/ubuntu/.cache/huggingface/hub")],
        cmd: "",
        model_store: None,
    },
    ..SPEACHES
};

const COMFYUI: ServiceSpec = ServiceSpec {
    id: ServiceId::Comfyui,
    display_name: "ComfyUI (image generation)",
    host_port: 8188,
    backend: "cuda",
    health_path: "/system_stats",
    models: &[(
        "sd-turbo",
        "https://huggingface.co/stabilityai/sd-turbo/resolve/main/sd_turbo.safetensors",
        "/root/ComfyUI/models/checkpoints/sd_turbo.safetensors",
    )],
    runtime: Runtime::Container {
        image: "yanwk/comfyui-boot:cu126-slim",
        container: "aias-svc-comfyui",
        container_port: 8188,
        gpu: true,
        // The boot image reads its flags from CLI_ARGS; --lowvram keeps a 10 GB card usable next to an LLM.
        env: &[(
            "CLI_ARGS",
            "--enable-cors-header http://tauri.localhost --lowvram",
        )],
        // The image installs ComfyUI under /root on first start; one volume keeps app, models and outputs.
        volumes: &[("aias-comfyui-root", "/root")],
        cmd: "",
        model_store: None,
    },
};

const COMFYUI_CPU: ServiceSpec = ServiceSpec {
    backend: "cpu",
    runtime: Runtime::Container {
        image: "yanwk/comfyui-boot:cpu",
        container: "aias-svc-comfyui",
        container_port: 8188,
        gpu: false,
        env: &[("CLI_ARGS", "--enable-cors-header --cpu")],
        volumes: &[("aias-comfyui-root", "/root")],
        cmd: "",
        model_store: None,
    },
    ..COMFYUI
};

/// ComfyUI's official Windows portable builds run natively, where PyTorch sees the
/// AMD (ROCm, RDNA 3 and newer) or Intel (XPU) GPU that WSL2 hides from containers.
const COMFYUI_PORTABLE_ARGS: &[&str] = &[
    "-s",
    "ComfyUI\\main.py",
    "--listen",
    "127.0.0.1",
    "--port",
    "8188",
    "--enable-cors-header",
];
/// The embedded Python encodes a redirected stderr in the console code page (cp950
/// on a zh-TW machine), so tqdm's block characters arrived as non-UTF-8 bytes;
/// force UTF-8 so the log stays readable whatever the locale.
const COMFYUI_PORTABLE_ENV: &[(&str, &str)] = &[("PYTHONUTF8", "1"), ("PYTHONIOENCODING", "utf-8")];
const COMFYUI_PORTABLE_MODELS: &[(&str, &str, &str)] = &[(
    "sd-turbo",
    "https://huggingface.co/stabilityai/sd-turbo/resolve/main/sd_turbo.safetensors",
    "ComfyUI_windows_portable\\ComfyUI\\models\\checkpoints\\sd_turbo.safetensors",
)];

const COMFYUI_ROCM: ServiceSpec = ServiceSpec {
    backend: "rocm",
    models: COMFYUI_PORTABLE_MODELS,
    runtime: Runtime::Native {
        // Pinned to v0.34.0 (Comfy-Org/ComfyUI, published 2026-08-26); upstream
        // publishes no checksum file, so the digest below is `shasum -a 256` on
        // the file downloaded here on 2026-09-03 (1,817,392,344 bytes).
        downloads: &[(
            "https://github.com/comfyanonymous/ComfyUI/releases/download/v0.34.0/ComfyUI_windows_portable_amd.7z",
            "ComfyUI_windows_portable_amd.7z",
            Some("da9317b62eab26865563b0529012799fd1f63e604d4ce81432c4e98eb6008b3f"),
        )],
        exe: "ComfyUI_windows_portable\\python_embeded\\python.exe",
        args: COMFYUI_PORTABLE_ARGS,
        env: COMFYUI_PORTABLE_ENV,
        cwd: "ComfyUI_windows_portable",
        models_required: false,
    },
    ..COMFYUI
};

const COMFYUI_XPU: ServiceSpec = ServiceSpec {
    backend: "xpu",
    models: COMFYUI_PORTABLE_MODELS,
    runtime: Runtime::Native {
        // Pinned to v0.34.0, same release as the ROCm build above; digest is
        // `shasum -a 256` on the file downloaded here on 2026-09-03
        // (1,734,410,473 bytes), upstream publishes no checksum file.
        downloads: &[(
            "https://github.com/comfyanonymous/ComfyUI/releases/download/v0.34.0/ComfyUI_windows_portable_intel.7z",
            "ComfyUI_windows_portable_intel.7z",
            Some("7dd41db69b53b4db120ce617d310c785e9fe9c0c9d634a3ea57cd877cab89e9e"),
        )],
        exe: "ComfyUI_windows_portable\\python_embeded\\python.exe",
        args: COMFYUI_PORTABLE_ARGS,
        env: COMFYUI_PORTABLE_ENV,
        cwd: "ComfyUI_windows_portable",
        models_required: false,
    },
    ..COMFYUI
};

/// Model repository shared by both CV servers: `repo/<model>/<version>/model.onnx`
/// is the layout Triton and OpenVINO Model Server both read.
const CV_STORE: (&str, &str) = ("aias-cv-models", "/models");

/// NVIDIA: Triton with the ONNX Runtime backend on CUDA. Polls the repository so a
/// detector downloaded while it runs is picked up without a restart.
const CV_TRITON: ServiceSpec = ServiceSpec {
    id: ServiceId::Cv,
    display_name: "CV server (object detection)",
    host_port: cv::PORT,
    backend: "triton",
    health_path: "/v2/health/ready",
    models: &[],
    runtime: Runtime::Container {
        image: "nvcr.io/nvidia/tritonserver:26.07-py3",
        container: "aias-svc-cv",
        container_port: 8000,
        gpu: true,
        env: &[],
        volumes: &[CV_STORE],
        cmd: "tritonserver --model-repository=/models/repo --model-control-mode=poll \
              --repository-poll-secs=5 --allow-grpc=false --allow-metrics=false",
        model_store: Some(CV_STORE),
    },
};

/// Everyone else: OpenVINO Model Server on the CPU. Same KServe v2 API, same ONNX
/// files; it re-reads `ovms.json` whenever the store rewrites it after a download.
const CV_OVMS: ServiceSpec = ServiceSpec {
    backend: "openvino",
    runtime: Runtime::Container {
        image: "openvino/model_server:latest",
        container: "aias-svc-cv",
        container_port: 8000,
        gpu: false,
        env: &[],
        volumes: &[CV_STORE],
        cmd: "--config_path /models/ovms.json --rest_port 8000 --port 9000 \
              --file_system_poll_wait_seconds 2",
        model_store: Some(CV_STORE),
    },
    ..CV_TRITON
};

/// Intel machines with an NPU: OpenVINO Model Server as a native Windows process.
/// WSL2 exposes no NPU device (`/dev/dxg` only, no `/dev/accel`), so the container
/// build cannot reach it; OVMS runs natively where the NPU driver lives. Same
/// KServe v2 API and ONNX detectors as the CPU build, but the model repo and
/// `ovms.json` (with `target_device: NPU`) live on the Windows side, written by
/// the store. Sits above the CPU rung; a machine with no NPU falls through to it.
const CV_OVMS_NPU: ServiceSpec = ServiceSpec {
    backend: "openvino-npu",
    runtime: Runtime::Native {
        // Digest from upstream's own published
        // `ovms_windows_2026.3.0_python_off.zip.sha256` (release published
        // 2026-08-04), cross-checked here with `shasum -a 256` on 2026-09-03
        // (108,901,047 bytes, matched).
        downloads: &[(
            "https://github.com/openvinotoolkit/model_server/releases/download/v2026.3/ovms_windows_2026.3.0_python_off.zip",
            "ovms.zip",
            Some("29ae9bda6f86544be14673397f1625d0161e9a3a0ff71d80e197b0d32e168fd9"),
        )],
        exe: "ovms\\ovms.exe",
        // Both listeners bind loopback: OVMS defaults to 0.0.0.0, which makes Windows
        // Defender Firewall raise its "allow this app" dialog the first time ovms.exe
        // listens. Nothing outside this machine needs the port. Every native service
        // in this file binds 127.0.0.1 for the same reason (see the test below).
        args: &[
            "--config_path",
            "ovms.json",
            "--rest_port",
            "8900",
            "--rest_bind_address",
            "127.0.0.1",
            "--port",
            "9000",
            "--grpc_bind_address",
            "127.0.0.1",
            "--file_system_poll_wait_seconds",
            "2",
        ],
        env: &[],
        cwd: "",
        models_required: false,
    },
    ..CV_OVMS
};

/// Windows-side port of the whisper.cpp server.
pub const WHISPER_PORT: u16 = 8881;

/// whisper.cpp built with the Vulkan backend by this repository's `runtimes.yml`
/// (upstream ships no Vulkan Windows binary). `--inference-path` makes its
/// multipart endpoint answer where an OpenAI client expects transcriptions; the
/// response `{"text": ...}` already matches. It only accepts WAV, so the Audio
/// screen converts recordings before sending.
const WHISPER_VULKAN: ServiceSpec = ServiceSpec {
    id: ServiceId::Whisper,
    display_name: "whisper.cpp (speech to text on the GPU)",
    host_port: WHISPER_PORT,
    backend: "vulkan",
    health_path: "/",
    models: &[(
        "ggml-small-q5_1",
        "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small-q5_1.bin",
        "models\\ggml-small-q5_1.bin",
    )],
    runtime: Runtime::Native {
        // This repository builds and publishes the asset itself (no upstream
        // Vulkan Windows build exists), so there is no third-party digest to
        // pin here; `runtimes.yml` now cuts a versioned `runtimes-vN` release
        // (never overwritten) and uploads a `SHA256SUMS` beside the zip, which
        // `run_native` fetches and checks against instead (`sha256: None`
        // means "verify against the release's own SHA256SUMS", not "skip").
        downloads: &[(
            "https://github.com/zyx1121/ai-app-store-poc/releases/download/runtimes-v1/whisper-server-vulkan-x64.zip",
            "whisper-server-vulkan-x64.zip",
            None,
        )],
        exe: "whisper-server.exe",
        args: &[
            "--host",
            "127.0.0.1",
            "--port",
            "8881",
            "-m",
            "models\\ggml-small-q5_1.bin",
            "--inference-path",
            "/v1/audio/transcriptions",
            "-l",
            "auto",
        ],
        env: &[],
        cwd: "",
        models_required: true,
    },
};

/// One rung of a modality's runtime resolution chain: the first rung whose
/// `applies` matches this machine wins. Specialised rungs come first, the generic
/// fallback last, so a machine with no special stack still resolves to something
/// that runs (or to nothing, where a modality has no generic fallback).
///
/// `applies` keys off `Vendor` for now. When capability adapters land (NPU,
/// aiDAPTIV) it widens to the full `HardwareProfile`, and a device-profile rung
/// for shipped SKUs is prepended; the ordered-chain shape here is that model.
/// See docs/adr/0001-accelerator-adapters.md.
/// What the resolution chain keys off. `vendor` today, plus capability flags as
/// adapters land (the NPU flag is the first). Widening this, not adding a new
/// selection path, is how a capability adapter plugs into the chain.
#[derive(Clone, Copy)]
struct Selector {
    vendor: Vendor,
    has_npu: bool,
}

struct Rung {
    applies: fn(Selector) -> bool,
    spec: &'static ServiceSpec,
}

const fn any(_: Selector) -> bool {
    true
}
const fn is_nvidia(s: Selector) -> bool {
    matches!(s.vendor, Vendor::Nvidia)
}
const fn is_amd(s: Selector) -> bool {
    matches!(s.vendor, Vendor::Amd)
}
const fn is_intel(s: Selector) -> bool {
    matches!(s.vendor, Vendor::Intel)
}
const fn amd_or_intel(s: Selector) -> bool {
    matches!(s.vendor, Vendor::Amd | Vendor::Intel)
}
const fn intel_with_npu(s: Selector) -> bool {
    matches!(s.vendor, Vendor::Intel) && s.has_npu
}

/// The resolution chain for a modality, specialised rungs first, generic last.
fn chain(id: ServiceId) -> &'static [Rung] {
    match id {
        ServiceId::Speaches => &[
            Rung {
                applies: is_nvidia,
                spec: &SPEACHES,
            },
            Rung {
                applies: any,
                spec: &SPEACHES_CPU,
            },
        ],
        ServiceId::Comfyui => &[
            Rung {
                applies: is_nvidia,
                spec: &COMFYUI,
            },
            Rung {
                applies: is_amd,
                spec: &COMFYUI_ROCM,
            },
            Rung {
                applies: is_intel,
                spec: &COMFYUI_XPU,
            },
            Rung {
                applies: any,
                spec: &COMFYUI_CPU,
            },
        ],
        ServiceId::Cv => &[
            Rung {
                applies: is_nvidia,
                spec: &CV_TRITON,
            },
            Rung {
                applies: intel_with_npu,
                spec: &CV_OVMS_NPU,
            },
            Rung {
                applies: any,
                spec: &CV_OVMS,
            },
        ],
        // Speaches already serves STT on CUDA and CPU; whisper.cpp only earns its
        // place where the GPU is reachable only through Vulkan. No generic rung:
        // NVIDIA and CPU resolve to nothing and keep using Speaches for STT.
        ServiceId::Whisper => &[Rung {
            applies: amd_or_intel,
            spec: &WHISPER_VULKAN,
        }],
    }
}

/// The implementation of a service for this machine, the first rung of the
/// modality's chain that applies, or `None` when the chain has no fallback here.
fn spec(id: ServiceId, sel: Selector) -> Option<&'static ServiceSpec> {
    chain(id)
        .iter()
        .find(|rung| (rung.applies)(sel))
        .map(|rung| rung.spec)
}

/// The selector for the current machine.
fn selector(state: &AppState) -> Selector {
    Selector {
        vendor: state.vendor(),
        has_npu: state.has_npu(),
    }
}

fn require_spec(id: ServiceId, sel: Selector) -> Result<&'static ServiceSpec> {
    spec(id, sel).ok_or_else(|| {
        Error::Other(format!(
            "{id:?} has no implementation for {:?} on this machine",
            sel.vendor
        ))
    })
}

/// Status of a service this vendor does not get.
fn unavailable(id: ServiceId) -> ServiceStatus {
    let (display_name, port) = match id {
        ServiceId::Whisper => (WHISPER_VULKAN.display_name, WHISPER_PORT),
        ServiceId::Speaches => (SPEACHES.display_name, SPEACHES.host_port),
        ServiceId::Comfyui => (COMFYUI.display_name, COMFYUI.host_port),
        ServiceId::Cv => (CV_TRITON.display_name, CV_TRITON.host_port),
    };
    ServiceStatus {
        id,
        display_name: display_name.to_string(),
        state: ServiceState::Unavailable,
        port,
        url: format!("http://localhost:{port}"),
        image_present: false,
        backend: "none".into(),
        runtime: "none",
        error: None,
        log_tail: vec![],
    }
}

/// Where a native service is unpacked.
fn native_dir(id: ServiceId) -> Result<PathBuf> {
    let local = std::env::var_os("LOCALAPPDATA")
        .ok_or_else(|| Error::Other("LOCALAPPDATA is not set".into()))?;
    let name = match id {
        ServiceId::Speaches => "speaches",
        ServiceId::Comfyui => "comfyui",
        ServiceId::Cv => "cv",
        ServiceId::Whisper => "whisper",
    };
    Ok(PathBuf::from(local)
        .join("ai-app-store")
        .join("services")
        .join(name))
}

/// (name, URL, path) of every model file a service can install.
fn model_files(s: &ServiceSpec) -> Vec<(String, String, String)> {
    let mut out: Vec<(String, String, String)> = s
        .models
        .iter()
        .map(|(n, u, p)| (n.to_string(), u.to_string(), p.to_string()))
        .collect();
    if s.id == ServiceId::Cv {
        match s.runtime {
            // Native OVMS keeps its model repo on the Windows side, relative to the
            // install dir; the container build keeps it in the shared distro volume.
            Runtime::Native { .. } => out.extend(cv::DETECTORS.iter().map(|d| {
                (
                    d.name.to_string(),
                    d.url.to_string(),
                    format!("models\\repo\\{}\\1\\model.onnx", d.name),
                )
            })),
            Runtime::Container { .. } => {
                let (_, mount) = CV_STORE;
                out.extend(cv::DETECTORS.iter().map(|d| {
                    (
                        d.name.to_string(),
                        d.url.to_string(),
                        format!("{mount}/repo/{}/1/model.onnx", d.name),
                    )
                }));
            }
        }
    }
    out
}

/// Write `ovms.json` for a native OVMS install: every detector whose ONNX is
/// present, with an absolute forward-slash base_path and this spec's target
/// device. OVMS re-reads it on its file poll, so it works before or during a run.
async fn write_native_cv_config(id: ServiceId, s: &ServiceSpec) -> Result<()> {
    let dir = native_dir(id)?;
    let target = ov_target_device(s);
    let mut items: Vec<String> = Vec::new();
    for d in cv::DETECTORS {
        let onnx = dir.join(format!("models\\repo\\{}\\1\\model.onnx", d.name));
        let present = tokio::fs::metadata(&onnx)
            .await
            .map(|m| m.len() > 0)
            .unwrap_or(false);
        if !present {
            continue;
        }
        let base = dir
            .join(format!("models\\repo\\{}", d.name))
            .to_string_lossy()
            .replace('\\', "/");
        items.push(format!(
            r#"{{"config":{{"name":"{}","base_path":"{}","target_device":"{}"}}}}"#,
            d.name, base, target
        ));
    }
    let json = format!(r#"{{"model_config_list":[{}]}}"#, items.join(","));
    tokio::fs::create_dir_all(&dir).await?;
    tokio::fs::write(dir.join("ovms.json"), json).await?;
    Ok(())
}

fn display_name(id: ServiceId, name: &str) -> String {
    match id {
        ServiceId::Cv => cv::detector(name)
            .map(|d| d.display_name.to_string())
            .unwrap_or_else(|| name.to_string()),
        _ => name.to_string(),
    }
}

/// Shell that resolves the store volume's directory on the distro and puts it in `$mp`
/// (creating the volume so it exists before the first `docker run`).
fn store_prelude(volume: &str) -> String {
    format!(
        "docker volume create {volume} >/dev/null 2>&1; \
         mp=$(docker volume inspect -f '{{{{.Mountpoint}}}}' {volume}) && [ -n \"$mp\" ] || exit 90; "
    )
}

/// Container path -> path under `$mp` for a file in the model store.
fn store_relative(mount: &str, container_path: &str) -> String {
    container_path
        .strip_prefix(mount)
        .unwrap_or(container_path)
        .to_string()
}

/// Regenerate the files a server reads to learn which models exist. Triton lists the
/// repository itself; OpenVINO Model Server wants an explicit `ovms.json`.
fn store_index_script(target: &str) -> String {
    let (volume, _) = CV_STORE;
    format!(
        "{prelude} mkdir -p \"$mp/repo\"; \
         {{ printf '{{\"model_config_list\":['; first=1; \
           for d in \"$mp\"/repo/*/; do n=$(basename \"$d\"); [ -s \"$d/1/model.onnx\" ] || continue; \
             [ $first = 1 ] || printf ','; first=0; \
             printf '{{\"config\":{{\"name\":\"%s\",\"base_path\":\"/models/repo/%s\",\"target_device\":\"{target}\"}}}}' \"$n\" \"$n\"; \
           done; printf ']}}\\n'; }} > \"$mp/ovms.json.tmp\" && mv \"$mp/ovms.json.tmp\" \"$mp/ovms.json\"; \
         chmod -R a+rX \"$mp\"",
        prelude = store_prelude(volume)
    )
}

/// The OpenVINO device the CV server should target for a spec (only OVMS reads it).
fn ov_target_device(spec: &ServiceSpec) -> &'static str {
    if spec.backend == "openvino-npu" {
        "NPU"
    } else {
        "CPU"
    }
}

/// The serde name of a service id (`speaches`, `comfyui`, `cv`, `whisper`).
pub fn id_str(id: ServiceId) -> &'static str {
    match id {
        ServiceId::Speaches => "speaches",
        ServiceId::Comfyui => "comfyui",
        ServiceId::Cv => "cv",
        ServiceId::Whisper => "whisper",
    }
}

pub fn parse_id(s: &str) -> Option<ServiceId> {
    match s {
        "speaches" => Some(ServiceId::Speaches),
        "comfyui" => Some(ServiceId::Comfyui),
        "cv" => Some(ServiceId::Cv),
        "whisper" => Some(ServiceId::Whisper),
        _ => None,
    }
}

/// Windows-side port of a service (the same on every vendor).
pub fn port(id: ServiceId) -> u16 {
    match id {
        ServiceId::Speaches => SPEACHES.host_port,
        ServiceId::Comfyui => COMFYUI.host_port,
        ServiceId::Cv => CV_TRITON.host_port,
        ServiceId::Whisper => WHISPER_PORT,
    }
}

/// Does this machine's implementation of the service hold GPU memory? A CUDA
/// container only when the distro can see the GPU; native builds whenever their
/// backend is a GPU one (Vulkan, ROCm, XPU, the NPU).
pub fn uses_gpu(app: &AppHandle, id: ServiceId) -> bool {
    let state = app.state::<AppState>();
    let Some(s) = spec(id, selector(&state)) else {
        return false;
    };
    match s.runtime {
        Runtime::Container { gpu, .. } => gpu && state.has_gpu(),
        Runtime::Native { .. } => {
            matches!(s.backend, "vulkan" | "rocm" | "xpu" | "openvino-npu")
        }
    }
}

/// Name of the implementation this machine gets for a service, without probing it.
pub fn backend_name(app: &AppHandle, id: ServiceId) -> String {
    spec(id, selector(&app.state::<AppState>()))
        .map(|s| s.backend.to_string())
        .unwrap_or_else(|| "none".into())
}

/// Files the service needs that are not part of its image, with whether they are present.
#[derive(Debug, Clone, Serialize)]
pub struct ServiceModel {
    pub name: String,
    pub display_name: String,
    pub path: String,
    pub installed: bool,
}

async fn model_installed(s: &ServiceSpec, id: ServiceId, path: &str) -> bool {
    match s.runtime {
        Runtime::Native { .. } => match native_dir(id) {
            Ok(dir) => tokio::fs::metadata(dir.join(path))
                .await
                .map(|m| m.len() > 0)
                .unwrap_or(false),
            Err(_) => false,
        },
        Runtime::Container {
            model_store: Some((volume, mount)),
            ..
        } => {
            let probe = format!(
                "{} test -s \"$mp{}\" && echo YES",
                store_prelude(volume),
                store_relative(mount, path)
            );
            wsl::sh(&probe)
                .await
                .map(|o| o.stdout.contains("YES"))
                .unwrap_or(false)
        }
        Runtime::Container { container, .. } => {
            let probe = format!("docker exec {container} test -s '{path}' && echo YES");
            wsl::sh(&probe)
                .await
                .map(|o| o.stdout.contains("YES"))
                .unwrap_or(false)
        }
    }
}

pub async fn models(app: &AppHandle, id: ServiceId) -> Result<Vec<ServiceModel>> {
    let s = require_spec(id, selector(&app.state::<AppState>()))?;
    let mut out = Vec::new();
    for (name, _url, path) in model_files(s) {
        let installed = model_installed(s, id, &path).await;
        out.push(ServiceModel {
            display_name: display_name(s.id, &name),
            name,
            path,
            installed,
        });
    }
    Ok(out)
}

/// Download one of the service's model files, streaming progress into the
/// service log. Blocks until done. Goes through the model store or the native
/// install dir when the service has one (no running container needed), else
/// through `docker exec`.
pub async fn install_model(app: AppHandle, id: ServiceId, name: String) -> Result<ServiceModel> {
    let state = app.state::<AppState>();
    let s = require_spec(id, selector(&state))?;
    let Some((_, url, path)) = model_files(s).into_iter().find(|(n, _, _)| *n == name) else {
        return Err(Error::Other(format!(
            "`{name}` is not a known model for {id:?}"
        )));
    };
    push_log(&app, id, format!("downloading {name}"));
    match s.runtime {
        Runtime::Native { .. } => {
            let dest = native_dir(id)?.join(&path);
            let http = state.http.clone();
            let app2 = app.clone();
            // Model files (safetensors, GGUF) come from the Hub, not this store's
            // pinned native runtimes; #37 only covers the latter, so no digest.
            fetch::download(&http, &url, &dest, None, &mut |l| push_log(&app2, id, l)).await?;
            // A native CV server (OVMS) reads its model list from ovms.json; refresh it
            // so a running server picks the new detector up on its file poll.
            if id == ServiceId::Cv {
                write_native_cv_config(id, s).await?;
            }
        }
        Runtime::Container {
            model_store,
            container,
            ..
        } => {
            let cmd = match model_store {
                Some((volume, mount)) => format!(
                    "{prelude} f=\"$mp{rel}\"; mkdir -p \"$(dirname \"$f\")\" && \
                     curl -L --fail --progress-bar -o \"$f.part\" {url} 2>&1 && mv \"$f.part\" \"$f\" && {index}",
                    prelude = store_prelude(volume),
                    rel = store_relative(mount, &path),
                    index = store_index_script(ov_target_device(s)),
                ),
                None => format!(
                    "docker exec {container} sh -c 'mkdir -p \"$(dirname {path})\" && curl -L --fail --progress-bar -o {path}.part {url} 2>&1 && mv {path}.part {path}' 2>&1"
                ),
            };
            let child = wsl::spawn_sh(&cmd)?;
            let mut last = String::new();
            let code = wsl::stream_lines(child, |l| {
                if l != last {
                    last = l.clone();
                    push_log(&app, id, l);
                }
            })
            .await?;
            if code != 0 {
                return Err(Error::Other(format!("download failed ({code}); see log")));
            }
        }
    }
    push_log(&app, id, format!("{name} installed"));
    Ok(ServiceModel {
        display_name: display_name(id, &name),
        name,
        path,
        installed: true,
    })
}

fn base_status(s: &ServiceSpec) -> ServiceStatus {
    ServiceStatus {
        id: s.id,
        display_name: s.display_name.to_string(),
        state: ServiceState::Missing,
        port: s.host_port,
        url: format!("http://localhost:{}", s.host_port),
        image_present: false,
        backend: s.backend.to_string(),
        runtime: match s.runtime {
            Runtime::Container { .. } => "container",
            Runtime::Native { .. } => "native",
        },
        error: None,
        log_tail: vec![],
    }
}

fn get(state: &AppState, id: ServiceId) -> Option<ServiceStatus> {
    state.services.lock().ok().and_then(|m| m.get(&id).cloned())
}

fn update(app: &AppHandle, id: ServiceId, f: impl FnOnce(&mut ServiceStatus)) {
    let state = app.state::<AppState>();
    let sel = selector(&state);
    let snapshot = {
        let mut map = match state.services.lock() {
            Ok(m) => m,
            Err(_) => return,
        };
        let entry = map.entry(id).or_insert_with(|| match spec(id, sel) {
            Some(s) => base_status(s),
            None => unavailable(id),
        });
        f(entry);
        entry.clone()
    };
    let _ = app.emit(UPDATE_EVENT, snapshot);
}

fn push_log(app: &AppHandle, id: ServiceId, line: String) {
    update(app, id, |s| {
        if s.log_tail.len() >= LOG_TAIL {
            s.log_tail.remove(0);
        }
        s.log_tail.push(line);
    });
}

fn fail(app: &AppHandle, id: ServiceId, msg: String) {
    log::error!("service {id:?}: {msg}");
    update(app, id, |s| {
        s.state = ServiceState::Error;
        s.error = Some(msg);
    });
}

async fn health_ok(port: u16, path: &str) -> bool {
    let Ok(http) = reqwest::Client::builder()
        .no_proxy()
        .timeout(Duration::from_secs(3))
        .build()
    else {
        return false;
    };
    match http
        .get(format!("http://localhost:{port}{path}"))
        .send()
        .await
    {
        Ok(r) => r.status().is_success(),
        Err(_) => false,
    }
}

/// Is the native child we started still alive?
fn native_child_alive(state: &AppState, id: ServiceId) -> bool {
    state
        .native_services
        .lock()
        .ok()
        .map(|mut m| match m.get_mut(&id) {
            Some(child) => matches!(child.try_wait(), Ok(None)),
            None => false,
        })
        .unwrap_or(false)
}

/// Probe the real state and store it; the UI's `service_status` command.
pub async fn status(app: &AppHandle, id: ServiceId) -> Result<ServiceStatus> {
    let state = app.state::<AppState>();
    let Some(s) = spec(id, selector(&state)) else {
        let st = unavailable(id);
        if let Ok(mut m) = state.services.lock() {
            m.insert(id, st.clone());
        }
        return Ok(st);
    };
    let busy = get(&state, id)
        .map(|st| matches!(st.state, ServiceState::Pulling | ServiceState::Starting))
        .unwrap_or(false);
    if busy {
        // A start task owns the status while it runs; do not clobber its progress.
        return get(&state, id).ok_or_else(|| Error::Other("service status missing".into()));
    }
    let (present, running, healthy) = match s.runtime {
        Runtime::Container {
            image, container, ..
        } => {
            // The published port is what the distro sees; the container port is private to it.
            let probe = format!(
                "docker image inspect {image} >/dev/null 2>&1 && echo IMAGE; \
                 docker inspect -f '{{{{.State.Running}}}}' {container} 2>/dev/null; \
                 curl -sf -m 3 -o /dev/null http://127.0.0.1:{port}{health} && echo HEALTHY",
                port = s.host_port,
                health = s.health_path,
            );
            let out = wsl::sh(&probe).await?;
            (
                out.stdout.lines().any(|l| l.trim() == "IMAGE"),
                out.stdout.lines().any(|l| l.trim() == "true"),
                out.stdout.lines().any(|l| l.trim() == "HEALTHY"),
            )
        }
        Runtime::Native { exe, .. } => {
            let present = native_dir(id)
                .map(|d| d.join(exe).exists())
                .unwrap_or(false);
            let healthy = health_ok(s.host_port, s.health_path).await;
            // A server left over from a previous store session still counts as running.
            (present, native_child_alive(&state, id) || healthy, healthy)
        }
    };
    update(app, id, |st| {
        st.image_present = present;
        st.backend = s.backend.to_string();
        st.error = None;
        st.state = if running && healthy {
            ServiceState::Running
        } else if running {
            ServiceState::Starting
        } else if present {
            ServiceState::Stopped
        } else {
            ServiceState::Missing
        };
    });
    get(&state, id).ok_or_else(|| Error::Other("service status missing".into()))
}

/// Fetch (if needed), run, and wait for health. Returns immediately with the
/// in-progress status; progress arrives on `service://update`.
pub async fn start(app: AppHandle, id: ServiceId) -> Result<ServiceStatus> {
    let state = app.state::<AppState>();
    if !state.is_ready() {
        return Err(Error::NotReady("install the runtime first".into()));
    }
    let s = require_spec(id, selector(&state))?;
    let current = status(&app, id).await?;
    if matches!(
        current.state,
        ServiceState::Pulling | ServiceState::Starting | ServiceState::Running
    ) {
        return Ok(current);
    }
    if let Runtime::Native {
        models_required: true,
        ..
    } = s.runtime
    {
        for (name, _, path) in model_files(s) {
            if !model_installed(s, id, &path).await {
                return Err(Error::Other(format!(
                    "download the `{name}` model first; the server loads it at start"
                )));
            }
        }
    }
    let gpu = state.has_gpu();
    update(&app, id, |st| {
        st.state = if st.image_present {
            ServiceState::Starting
        } else {
            ServiceState::Pulling
        };
        st.error = None;
    });
    let app2 = app.clone();
    tokio::spawn(async move {
        let r = match s.runtime {
            Runtime::Container { .. } => run_container(&app2, s, gpu).await,
            Runtime::Native { .. } => run_native(&app2, s).await,
        };
        if let Err(e) = r {
            fail(&app2, id, e.to_string());
        }
    });
    get(&state, id).ok_or_else(|| Error::Other("service status missing".into()))
}

/// Poll the health URL until it answers 2xx. `alive` reports whether the process is
/// still there so a crash is reported instead of waiting out the timeout.
async fn wait_healthy<F, Fut>(
    app: &AppHandle,
    s: &ServiceSpec,
    ticks: u32,
    mut alive: F,
) -> Result<()>
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = bool>,
{
    for tick in 0..ticks {
        tokio::time::sleep(Duration::from_secs(2)).await;
        if health_ok(s.host_port, s.health_path).await {
            update(app, s.id, |st| st.state = ServiceState::Running);
            push_log(app, s.id, "healthy".into());
            return Ok(());
        }
        if tick % 5 == 4 && !alive().await {
            return Err(Error::Other(
                "the service exited before becoming healthy; see log".into(),
            ));
        }
    }
    Err(Error::Other(
        "timed out waiting for the service to become healthy".into(),
    ))
}

async fn run_container(app: &AppHandle, s: &ServiceSpec, gpu: bool) -> Result<()> {
    let Runtime::Container {
        image,
        container,
        container_port,
        gpu: wants_gpu,
        env,
        volumes,
        cmd,
        model_store,
    } = s.runtime
    else {
        unreachable!("run_container on a native spec");
    };
    let id = s.id;
    let has_image = get(&app.state::<AppState>(), id)
        .map(|st| st.image_present)
        .unwrap_or(false);
    if !has_image {
        push_log(app, id, format!("docker pull {image}"));
        let child = wsl::spawn_sh(&format!("docker pull {image} 2>&1"))?;
        let mut last = String::new();
        let code = wsl::stream_lines(child, |l| {
            if l != last {
                last = l.clone();
                push_log(app, id, l);
            }
        })
        .await?;
        if code != 0 {
            return Err(Error::Other(format!("image pull failed ({code}); see log")));
        }
        update(app, id, |st| {
            st.image_present = true;
            st.state = ServiceState::Starting;
        });
    }

    if model_store.is_some() {
        // The server must find a valid (possibly empty) index on its first start.
        wsl::sh(&store_index_script(ov_target_device(s)))
            .await?
            .require("prepare model store")?;
    }

    let gpu_flag = if gpu && wants_gpu { "--gpus all" } else { "" };
    let env: String = env
        .iter()
        .map(|(k, v)| format!("-e {k}='{v}'"))
        .collect::<Vec<_>>()
        .join(" ");
    let volumes: String = volumes
        .iter()
        .map(|(name, path)| format!("-v {name}:{path}"))
        .collect::<Vec<_>>()
        .join(" ");
    let publish = wsl::publish(s.host_port, container_port);
    let run = format!(
        "docker rm -f {container} >/dev/null 2>&1; \
         docker run -d --name {container} --restart unless-stopped {gpu_flag} \
           {} {publish} {env} {volumes} --label aias.kind=service {image} {cmd}",
        wsl::HARDEN,
    );
    push_log(app, id, format!("docker run {publish} {image}"));
    wsl::sh(&run).await?.require("docker run")?;

    wait_healthy(app, s, 90, || async {
        let probe = format!(
            "docker inspect -f '{{{{.State.Running}}}}' {container} 2>/dev/null; docker logs --tail 3 {container} 2>&1"
        );
        match wsl::sh(&probe).await {
            Ok(o) => {
                let mut lines = o.stdout.lines();
                let running = lines.next().map(str::trim) == Some("true");
                for l in lines {
                    push_log(app, id, l.to_string());
                }
                running
            }
            Err(_) => true,
        }
    })
    .await
}

async fn run_native(app: &AppHandle, s: &ServiceSpec) -> Result<()> {
    let Runtime::Native {
        downloads,
        exe,
        args,
        env,
        cwd,
        ..
    } = s.runtime
    else {
        unreachable!("run_native on a container spec");
    };
    let id = s.id;
    let state = app.state::<AppState>();
    let dir = native_dir(id)?;
    let exe_path = dir.join(exe);
    if !exe_path.exists() {
        tokio::fs::create_dir_all(&dir).await?;
        for (url, archive, sha256) in downloads {
            let dest = dir.join(archive);
            push_log(app, id, format!("downloading {url}"));
            let sha256 = match sha256 {
                Some(s) => Some((*s).to_string()),
                None => {
                    let checked =
                        fetch::release_checksum(&state.http, url, &mut |l| push_log(app, id, l))
                            .await;
                    if checked.is_none() {
                        return Err(Error::Other(format!(
                            "no SHA-256 available for {archive} (no compile-time digest, and \
                             the release published no SHA256SUMS); refusing to download it"
                        )));
                    }
                    checked
                }
            };
            fetch::download(&state.http, url, &dest, sha256.as_deref(), &mut |l| {
                push_log(app, id, l)
            })
            .await?;
            push_log(app, id, format!("unpacking {archive}"));
            fetch::extract(&dest, &dir).await?;
            let _ = tokio::fs::remove_file(&dest).await;
        }
        if !exe_path.exists() {
            return Err(Error::Other(format!("the archive did not contain {exe}")));
        }
        update(app, id, |st| {
            st.image_present = true;
            st.state = ServiceState::Starting;
        });
    }

    // A native OVMS server needs a valid ovms.json before it starts; write it from
    // whatever detectors are installed (an empty list is valid and hot-fills later).
    if id == ServiceId::Cv {
        write_native_cv_config(id, s).await?;
    }

    // Anything already listening on our port (a server left over from an earlier
    // session) would make the new process exit; stop it first.
    stop_native_process(&state, id, s.host_port).await;

    push_log(
        app,
        id,
        format!("{} {}", exe_path.display(), args.join(" ")),
    );
    let mut cmd = Command::new(&exe_path);
    cmd.args(args)
        .current_dir(if cwd.is_empty() {
            dir.clone()
        } else {
            dir.join(cwd)
        })
        .envs(env.iter().copied())
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    #[cfg(windows)]
    cmd.creation_flags(0x0800_0000);
    let mut child = cmd
        .spawn()
        .map_err(|e| Error::Spawn(exe_path.to_string_lossy().to_string(), e.to_string()))?;
    for reader in [
        child
            .stdout
            .take()
            .map(|o| Box::pin(o) as Pin<Box<dyn AsyncRead + Send>>),
        child
            .stderr
            .take()
            .map(|e| Box::pin(e) as Pin<Box<dyn AsyncRead + Send>>),
    ]
    .into_iter()
    .flatten()
    {
        let app = app.clone();
        tokio::spawn(async move {
            // Read raw bytes and decode lossily: `lines()` fails on the first byte
            // that is not UTF-8 (the embedded Python printing tqdm's block characters
            // in the console code page), the task would end, the pipe would close and
            // every later write to stderr in the service would fail with EINVAL. That
            // is how ComfyUI's second generation on the Intel laptop died.
            let mut reader = BufReader::new(reader);
            let mut buf = Vec::new();
            loop {
                buf.clear();
                match reader.read_until(b'\n', &mut buf).await {
                    Ok(0) | Err(_) => break,
                    Ok(_) => {
                        if let Some(l) = log_line(&buf) {
                            push_log(&app, id, l);
                        }
                    }
                }
            }
        });
    }
    if let Ok(mut m) = state.native_services.lock() {
        m.insert(id, child);
    }

    // Portable ComfyUI unpacks and imports a lot on first start; allow six minutes.
    wait_healthy(app, s, 180, || async { native_child_alive(&state, id) }).await
}

/// One log line from raw process output: lossy UTF-8, and for a progress bar
/// that redraws with `\r` only the last frame.
fn log_line(bytes: &[u8]) -> Option<String> {
    let text = String::from_utf8_lossy(bytes);
    let last = text.trim_end().rsplit('\r').next().unwrap_or("").trim();
    (!last.is_empty()).then(|| last.to_string())
}

/// Kill the native process we own, or whatever still listens on the port.
async fn stop_native_process(state: &AppState, id: ServiceId, port: u16) {
    let child = state
        .native_services
        .lock()
        .ok()
        .and_then(|mut m| m.remove(&id));
    if let Some(mut child) = child {
        let _ = child.start_kill();
        let _ = child.wait().await;
    }
    if health_ok(port, "/").await
        || tokio::net::TcpStream::connect(("127.0.0.1", port))
            .await
            .is_ok()
    {
        let script = format!(
            "Get-NetTCPConnection -LocalPort {port} -State Listen -ErrorAction SilentlyContinue | \
             ForEach-Object {{ Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }}"
        );
        let _ = wsl::run(
            "powershell.exe",
            &["-NoProfile", "-NonInteractive", "-Command", &script],
        )
        .await;
    }
}

pub async fn stop(app: AppHandle, id: ServiceId) -> Result<()> {
    let state = app.state::<AppState>();
    let s = require_spec(id, selector(&state))?;
    match s.runtime {
        Runtime::Container { container, .. } => {
            let _ = wsl::sh(&format!("docker rm -f {container} >/dev/null 2>&1")).await;
        }
        Runtime::Native { .. } => stop_native_process(&state, id, s.host_port).await,
    }
    update(&app, id, |st| {
        st.state = if st.image_present {
            ServiceState::Stopped
        } else {
            ServiceState::Missing
        };
        st.error = None;
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sel(v: Vendor) -> Selector {
        Selector {
            vendor: v,
            has_npu: false,
        }
    }
    fn sel_npu(v: Vendor) -> Selector {
        Selector {
            vendor: v,
            has_npu: true,
        }
    }

    #[test]
    fn every_vendor_gets_the_same_ports_per_service() {
        for id in [ServiceId::Speaches, ServiceId::Comfyui, ServiceId::Cv] {
            let ports: Vec<u16> = [Vendor::Nvidia, Vendor::Amd, Vendor::Intel, Vendor::Cpu]
                .into_iter()
                .filter_map(|v| spec(id, sel(v)))
                .map(|s| s.host_port)
                .collect();
            assert_eq!(ports.len(), 4, "{id:?} missing a vendor");
            assert!(
                ports.windows(2).all(|w| w[0] == w[1]),
                "{id:?} ports differ"
            );
        }
    }

    #[test]
    fn whisper_only_where_speaches_cannot_use_the_gpu() {
        assert!(spec(ServiceId::Whisper, sel(Vendor::Nvidia)).is_none());
        assert!(spec(ServiceId::Whisper, sel(Vendor::Cpu)).is_none());
        assert_eq!(
            spec(ServiceId::Whisper, sel(Vendor::Amd)).unwrap().backend,
            "vulkan"
        );
        assert_eq!(
            spec(ServiceId::Whisper, sel(Vendor::Intel))
                .unwrap()
                .backend,
            "vulkan"
        );
    }

    #[test]
    fn intel_npu_routes_cv_to_openvino_npu() {
        // Intel + NPU picks the NPU rung; Intel without NPU and other vendors do not.
        assert_eq!(
            spec(ServiceId::Cv, sel_npu(Vendor::Intel)).unwrap().backend,
            "openvino-npu"
        );
        assert_eq!(
            spec(ServiceId::Cv, sel(Vendor::Intel)).unwrap().backend,
            "openvino"
        );
        assert_eq!(
            spec(ServiceId::Cv, sel_npu(Vendor::Amd)).unwrap().backend,
            "openvino"
        );
        assert_eq!(
            spec(ServiceId::Cv, sel_npu(Vendor::Nvidia))
                .unwrap()
                .backend,
            "triton"
        );
        // The NPU spec still serves the same port and API as the CPU one.
        assert_eq!(
            spec(ServiceId::Cv, sel_npu(Vendor::Intel))
                .unwrap()
                .host_port,
            spec(ServiceId::Cv, sel(Vendor::Intel)).unwrap().host_port
        );
    }

    #[test]
    fn chain_has_a_generic_fallback_except_whisper() {
        // Every vendor resolves for the always-on modalities; the last rung is the
        // generic one (SPEACHES_CPU / COMFYUI_CPU / CV_OVMS), reached by cpu.
        for id in [ServiceId::Speaches, ServiceId::Comfyui, ServiceId::Cv] {
            for v in [Vendor::Nvidia, Vendor::Amd, Vendor::Intel, Vendor::Cpu] {
                assert!(spec(id, sel(v)).is_some(), "{id:?} has no rung for {v:?}");
            }
        }
        // Whisper is specialised-only: it resolves on AMD/Intel and nowhere else.
        assert!(spec(ServiceId::Whisper, sel(Vendor::Amd)).is_some());
        assert!(spec(ServiceId::Whisper, sel(Vendor::Intel)).is_some());
        assert!(spec(ServiceId::Whisper, sel(Vendor::Nvidia)).is_none());
        assert!(spec(ServiceId::Whisper, sel(Vendor::Cpu)).is_none());
    }

    #[test]
    fn chain_is_specialised_before_generic() {
        // The generic rung matches any vendor, so it must be last or it would
        // shadow the specialised rungs above it.
        for id in [ServiceId::Speaches, ServiceId::Comfyui, ServiceId::Cv] {
            let rungs = chain(id);
            let generic = rungs.iter().position(|r| {
                (r.applies)(sel(Vendor::Nvidia))
                    && (r.applies)(sel(Vendor::Amd))
                    && (r.applies)(sel(Vendor::Intel))
                    && (r.applies)(sel(Vendor::Cpu))
            });
            assert_eq!(
                generic,
                Some(rungs.len() - 1),
                "{id:?} generic rung not last"
            );
        }
    }

    #[test]
    fn non_nvidia_comfyui_runs_natively_on_a_gpu_vendor() {
        assert!(matches!(
            spec(ServiceId::Comfyui, sel(Vendor::Amd)).unwrap().runtime,
            Runtime::Native { .. }
        ));
        assert!(matches!(
            spec(ServiceId::Comfyui, sel(Vendor::Intel))
                .unwrap()
                .runtime,
            Runtime::Native { .. }
        ));
        assert!(matches!(
            spec(ServiceId::Comfyui, sel(Vendor::Cpu)).unwrap().runtime,
            Runtime::Container { .. }
        ));
    }

    /// Every native Windows service must listen on loopback only. A listener on
    /// 0.0.0.0 makes Windows Defender Firewall raise a modal "allow this app"
    /// dialog the first time the executable binds, which breaks unattended
    /// provisioning on a shipped machine (issue #22). Ollama is covered in
    /// `ollama.rs` (`OLLAMA_HOST=127.0.0.1`).
    #[test]
    fn native_services_bind_loopback_only() {
        let bind_flags = [
            "--host",
            "--listen",
            "--rest_bind_address",
            "--grpc_bind_address",
        ];
        let mut native = 0;
        for id in [
            ServiceId::Speaches,
            ServiceId::Comfyui,
            ServiceId::Cv,
            ServiceId::Whisper,
        ] {
            for rung in chain(id) {
                let Runtime::Native { args, .. } = rung.spec.runtime else {
                    continue;
                };
                native += 1;
                let binds: Vec<&str> = args
                    .windows(2)
                    .filter(|w| bind_flags.contains(&w[0]))
                    .map(|w| w[1])
                    .collect();
                assert!(
                    !binds.is_empty(),
                    "{:?} ({}) has no bind address flag; it would listen on 0.0.0.0",
                    id,
                    rung.spec.backend
                );
                for b in binds {
                    assert_eq!(
                        b, "127.0.0.1",
                        "{:?} ({}) binds {b}, not loopback",
                        id, rung.spec.backend
                    );
                }
            }
        }
        // OVMS has two listeners (REST and gRPC); both must be pinned.
        let Runtime::Native { args, .. } = CV_OVMS_NPU.runtime else {
            unreachable!()
        };
        assert!(args.contains(&"--rest_bind_address"));
        assert!(args.contains(&"--grpc_bind_address"));
        assert!(native >= 4, "expected whisper, two ComfyUI builds and OVMS");
    }

    #[test]
    fn log_lines_survive_bad_bytes_and_keep_the_last_progress_frame() {
        // cp950-encoded block character, not UTF-8: must not break the reader.
        assert_eq!(
            log_line(b"loading \xa2\x60 50%\n").unwrap(),
            "loading \u{fffd}` 50%"
        );
        assert_eq!(log_line(b"10%\r20%\r30%\n").unwrap(), "30%");
        assert_eq!(log_line(b"\r\n"), None);
        assert_eq!(log_line(b"healthy\n").unwrap(), "healthy");
    }

    #[test]
    fn portable_comfyui_forces_utf8_python_io() {
        for s in [&COMFYUI_ROCM, &COMFYUI_XPU] {
            let Runtime::Native { env, .. } = s.runtime else {
                unreachable!()
            };
            assert!(env.contains(&("PYTHONUTF8", "1")), "{}", s.backend);
        }
    }

    #[test]
    fn store_paths_are_relative_to_the_mount() {
        assert_eq!(
            store_relative("/models", "/models/repo/yolov10n/1/model.onnx"),
            "/repo/yolov10n/1/model.onnx"
        );
    }
}
