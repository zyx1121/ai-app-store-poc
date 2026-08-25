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
        /// archives to download and unpack into the install dir, in order
        downloads: &'static [(&'static str, &'static str)],
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
        env: &[("ALLOW_ORIGINS", r#"["*"]"#), ("ENABLE_UI", "false")],
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
        env: &[("ALLOW_ORIGINS", r#"["*"]"#), ("ENABLE_UI", "false")],
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
        env: &[("CLI_ARGS", "--enable-cors-header --lowvram")],
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
const COMFYUI_PORTABLE_MODELS: &[(&str, &str, &str)] = &[(
    "sd-turbo",
    "https://huggingface.co/stabilityai/sd-turbo/resolve/main/sd_turbo.safetensors",
    "ComfyUI_windows_portable\\ComfyUI\\models\\checkpoints\\sd_turbo.safetensors",
)];

const COMFYUI_ROCM: ServiceSpec = ServiceSpec {
    backend: "rocm",
    models: COMFYUI_PORTABLE_MODELS,
    runtime: Runtime::Native {
        downloads: &[(
            "https://github.com/comfyanonymous/ComfyUI/releases/latest/download/ComfyUI_windows_portable_amd.7z",
            "ComfyUI_windows_portable_amd.7z",
        )],
        exe: "ComfyUI_windows_portable\\python_embeded\\python.exe",
        args: COMFYUI_PORTABLE_ARGS,
        env: &[],
        cwd: "ComfyUI_windows_portable",
        models_required: false,
    },
    ..COMFYUI
};

const COMFYUI_XPU: ServiceSpec = ServiceSpec {
    backend: "xpu",
    models: COMFYUI_PORTABLE_MODELS,
    runtime: Runtime::Native {
        downloads: &[(
            "https://github.com/comfyanonymous/ComfyUI/releases/latest/download/ComfyUI_windows_portable_intel.7z",
            "ComfyUI_windows_portable_intel.7z",
        )],
        exe: "ComfyUI_windows_portable\\python_embeded\\python.exe",
        args: COMFYUI_PORTABLE_ARGS,
        env: &[],
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
        downloads: &[(
            "https://github.com/zyx1121/ai-app-store-poc/releases/download/runtimes/whisper-server-vulkan-x64.zip",
            "whisper-server-vulkan-x64.zip",
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

/// The implementation of a service for this machine's GPU vendor, if it has one.
fn spec(id: ServiceId, vendor: Vendor) -> Option<&'static ServiceSpec> {
    Some(match (id, vendor) {
        (ServiceId::Speaches, Vendor::Nvidia) => &SPEACHES,
        (ServiceId::Speaches, _) => &SPEACHES_CPU,
        (ServiceId::Comfyui, Vendor::Nvidia) => &COMFYUI,
        (ServiceId::Comfyui, Vendor::Amd) => &COMFYUI_ROCM,
        (ServiceId::Comfyui, Vendor::Intel) => &COMFYUI_XPU,
        (ServiceId::Comfyui, Vendor::Cpu) => &COMFYUI_CPU,
        (ServiceId::Cv, Vendor::Nvidia) => &CV_TRITON,
        (ServiceId::Cv, _) => &CV_OVMS,
        // Speaches already runs speech on CUDA; on a CPU-only box Vulkan has nothing to drive.
        (ServiceId::Whisper, Vendor::Amd | Vendor::Intel) => &WHISPER_VULKAN,
        (ServiceId::Whisper, Vendor::Nvidia | Vendor::Cpu) => return None,
    })
}

fn require_spec(id: ServiceId, vendor: Vendor) -> Result<&'static ServiceSpec> {
    spec(id, vendor).ok_or_else(|| {
        Error::Other(format!(
            "{id:?} has no implementation for {vendor:?} on this machine"
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
        let (_, mount) = CV_STORE;
        out.extend(cv::DETECTORS.iter().map(|d| {
            (
                d.name.to_string(),
                d.url.to_string(),
                format!("{mount}/repo/{}/1/model.onnx", d.name),
            )
        }));
    }
    out
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
fn store_index_script() -> String {
    let (volume, _) = CV_STORE;
    format!(
        "{prelude} mkdir -p \"$mp/repo\"; \
         {{ printf '{{\"model_config_list\":['; first=1; \
           for d in \"$mp\"/repo/*/; do n=$(basename \"$d\"); [ -s \"$d/1/model.onnx\" ] || continue; \
             [ $first = 1 ] || printf ','; first=0; \
             printf '{{\"config\":{{\"name\":\"%s\",\"base_path\":\"/models/repo/%s\"}}}}' \"$n\" \"$n\"; \
           done; printf ']}}\\n'; }} > \"$mp/ovms.json.tmp\" && mv \"$mp/ovms.json.tmp\" \"$mp/ovms.json\"; \
         chmod -R a+rX \"$mp\"",
        prelude = store_prelude(volume)
    )
}

/// Name of the implementation this machine gets for a service, without probing it.
pub fn backend_name(app: &AppHandle, id: ServiceId) -> String {
    spec(id, app.state::<AppState>().vendor())
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
    let s = require_spec(id, app.state::<AppState>().vendor())?;
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
    let s = require_spec(id, state.vendor())?;
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
            fetch::download(&http, &url, &dest, &mut |l| push_log(&app2, id, l)).await?;
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
                    index = store_index_script(),
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
    let vendor = state.vendor();
    let snapshot = {
        let mut map = match state.services.lock() {
            Ok(m) => m,
            Err(_) => return,
        };
        let entry = map.entry(id).or_insert_with(|| match spec(id, vendor) {
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
    let Some(s) = spec(id, state.vendor()) else {
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
    let s = require_spec(id, state.vendor())?;
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
        wsl::sh(&store_index_script())
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
    let run = format!(
        "docker rm -f {container} >/dev/null 2>&1; \
         docker run -d --name {container} --restart unless-stopped {gpu_flag} \
           -p {hp}:{container_port} {env} {volumes} --label aias.kind=service {image} {cmd}",
        hp = s.host_port,
    );
    push_log(
        app,
        id,
        format!("docker run -p {}:{container_port} {image}", s.host_port),
    );
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
        for (url, archive) in downloads {
            let dest = dir.join(archive);
            push_log(app, id, format!("downloading {url}"));
            fetch::download(&state.http, url, &dest, &mut |l| push_log(app, id, l)).await?;
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
            let mut lines = BufReader::new(reader).lines();
            while let Ok(Some(l)) = lines.next_line().await {
                let l = l.trim_end().to_string();
                if !l.is_empty() {
                    push_log(&app, id, l);
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
    let s = require_spec(id, state.vendor())?;
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

    #[test]
    fn every_vendor_gets_the_same_ports_per_service() {
        for id in [ServiceId::Speaches, ServiceId::Comfyui, ServiceId::Cv] {
            let ports: Vec<u16> = [Vendor::Nvidia, Vendor::Amd, Vendor::Intel, Vendor::Cpu]
                .into_iter()
                .filter_map(|v| spec(id, v))
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
        assert!(spec(ServiceId::Whisper, Vendor::Nvidia).is_none());
        assert!(spec(ServiceId::Whisper, Vendor::Cpu).is_none());
        assert_eq!(
            spec(ServiceId::Whisper, Vendor::Amd).unwrap().backend,
            "vulkan"
        );
        assert_eq!(
            spec(ServiceId::Whisper, Vendor::Intel).unwrap().backend,
            "vulkan"
        );
    }

    #[test]
    fn non_nvidia_comfyui_runs_natively_on_a_gpu_vendor() {
        assert!(matches!(
            spec(ServiceId::Comfyui, Vendor::Amd).unwrap().runtime,
            Runtime::Native { .. }
        ));
        assert!(matches!(
            spec(ServiceId::Comfyui, Vendor::Intel).unwrap().runtime,
            Runtime::Native { .. }
        ));
        assert!(matches!(
            spec(ServiceId::Comfyui, Vendor::Cpu).unwrap().runtime,
            Runtime::Container { .. }
        ));
    }

    #[test]
    fn store_paths_are_relative_to_the_mount() {
        assert_eq!(
            store_relative("/models", "/models/repo/yolov10n/1/model.onnx"),
            "/repo/yolov10n/1/model.onnx"
        );
    }
}
