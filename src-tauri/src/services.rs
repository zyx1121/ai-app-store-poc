//! Platform services: inference servers the store itself runs as containers
//! inside the distro, next to Ollama: Speaches (STT/TTS), ComfyUI (images) and
//! a KServe v2 CV server (Triton on NVIDIA, OpenVINO Model Server elsewhere).
//! Each service is a fixed spec (image, ports, env, volumes); the store only
//! pulls, starts, health-checks and stops it.

use std::time::Duration;

use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Emitter, Manager};

use crate::cv;
use crate::error::{Error, Result};
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
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceStatus {
    pub id: ServiceId,
    pub display_name: String,
    pub state: ServiceState,
    pub port: u16,
    pub url: String,
    pub image_present: bool,
    /// which implementation this machine got, e.g. `cuda`, `cpu`, `triton`, `openvino`
    pub backend: String,
    pub error: Option<String>,
    pub log_tail: Vec<String>,
}

#[derive(Clone, Copy)]
struct ServiceSpec {
    id: ServiceId,
    display_name: &'static str,
    image: &'static str,
    container: &'static str,
    /// Windows-side port. Fixed so the frontend can hardcode the base URL.
    host_port: u16,
    container_port: u16,
    gpu: bool,
    /// shown to the user as the implementation this machine got
    backend: &'static str,
    env: &'static [(&'static str, &'static str)],
    /// (named volume, container path)
    volumes: &'static [(&'static str, &'static str)],
    /// arguments appended after the image (the container's command)
    cmd: &'static str,
    /// path that answers 2xx once the service is usable
    health_path: &'static str,
    /// downloadable model files: (name shown to the user, URL, path inside the container)
    models: &'static [(&'static str, &'static str, &'static str)],
    /// Model files live in this (named volume, container mount) and are written from the
    /// distro side, so they can be installed before the service runs and survive a
    /// backend swap. Without it, files are fetched with `docker exec` inside the container.
    model_store: Option<(&'static str, &'static str)>,
}

const SPEACHES: ServiceSpec = ServiceSpec {
    id: ServiceId::Speaches,
    display_name: "Speaches (speech to text, text to speech)",
    image: "ghcr.io/speaches-ai/speaches:latest-cuda",
    container: "aias-svc-speaches",
    // 8000 is taken by other WSL distros on the dev box; all distros share one network namespace.
    host_port: 8880,
    container_port: 8000,
    gpu: true,
    backend: "cuda",
    env: &[("ALLOW_ORIGINS", r#"["*"]"#), ("ENABLE_UI", "false")],
    volumes: &[("aias-speaches-cache", "/home/ubuntu/.cache/huggingface/hub")],
    cmd: "",
    health_path: "/v1/models",
    models: &[],
    model_store: None,
};

const COMFYUI: ServiceSpec = ServiceSpec {
    id: ServiceId::Comfyui,
    display_name: "ComfyUI (image generation)",
    image: "yanwk/comfyui-boot:cu126-slim",
    container: "aias-svc-comfyui",
    host_port: 8188,
    container_port: 8188,
    gpu: true,
    backend: "cuda",
    // The boot image reads its flags from CLI_ARGS; --lowvram keeps a 10 GB card usable next to an LLM.
    env: &[("CLI_ARGS", "--enable-cors-header --lowvram")],
    // The image installs ComfyUI under /root on first start; one volume keeps app, models and outputs.
    volumes: &[("aias-comfyui-root", "/root")],
    cmd: "",
    health_path: "/system_stats",
    models: &[(
        "sd-turbo",
        "https://huggingface.co/stabilityai/sd-turbo/resolve/main/sd_turbo.safetensors",
        "/root/ComfyUI/models/checkpoints/sd_turbo.safetensors",
    )],
    model_store: None,
};

/// Model repository shared by both CV servers: `repo/<model>/<version>/model.onnx`
/// is the layout Triton and OpenVINO Model Server both read.
const CV_STORE: (&str, &str) = ("aias-cv-models", "/models");

/// NVIDIA: Triton with the ONNX Runtime backend on CUDA. Polls the repository so a
/// detector downloaded while it runs is picked up without a restart.
const CV_TRITON: ServiceSpec = ServiceSpec {
    id: ServiceId::Cv,
    display_name: "CV server (object detection)",
    image: "nvcr.io/nvidia/tritonserver:26.07-py3",
    container: "aias-svc-cv",
    host_port: cv::PORT,
    container_port: 8000,
    gpu: true,
    backend: "triton",
    env: &[],
    volumes: &[CV_STORE],
    cmd: "tritonserver --model-repository=/models/repo --model-control-mode=poll \
          --repository-poll-secs=5 --allow-grpc=false --allow-metrics=false",
    health_path: "/v2/health/ready",
    models: &[],
    model_store: Some(CV_STORE),
};

/// Everyone else: OpenVINO Model Server on the CPU. Same KServe v2 API, same ONNX
/// files; it re-reads `ovms.json` whenever the store rewrites it after a download.
const CV_OVMS: ServiceSpec = ServiceSpec {
    image: "openvino/model_server:latest",
    gpu: false,
    backend: "openvino",
    cmd: "--config_path /models/ovms.json --rest_port 8000 --port 9000 \
          --file_system_poll_wait_seconds 2",
    ..CV_TRITON
};

/// CPU builds for machines whose GPU WSL2 cannot see (AMD, Intel) or that have none.
const SPEACHES_CPU: ServiceSpec = ServiceSpec {
    image: "ghcr.io/speaches-ai/speaches:latest-cpu",
    gpu: false,
    backend: "cpu",
    ..SPEACHES
};

const COMFYUI_CPU: ServiceSpec = ServiceSpec {
    image: "yanwk/comfyui-boot:cpu",
    gpu: false,
    backend: "cpu",
    env: &[("CLI_ARGS", "--enable-cors-header --cpu")],
    ..COMFYUI
};

/// The implementation of a service for this machine's GPU vendor.
fn spec(id: ServiceId, vendor: Vendor) -> &'static ServiceSpec {
    match (id, vendor) {
        (ServiceId::Speaches, Vendor::Nvidia) => &SPEACHES,
        (ServiceId::Speaches, _) => &SPEACHES_CPU,
        (ServiceId::Comfyui, Vendor::Nvidia) => &COMFYUI,
        (ServiceId::Comfyui, _) => &COMFYUI_CPU,
        (ServiceId::Cv, Vendor::Nvidia) => &CV_TRITON,
        (ServiceId::Cv, _) => &CV_OVMS,
    }
}

/// (name, URL, path inside the container) of every model file a service can install.
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
fn store_relative(s: &ServiceSpec, container_path: &str) -> String {
    match s.model_store {
        Some((_, mount)) => container_path
            .strip_prefix(mount)
            .unwrap_or(container_path)
            .to_string(),
        None => container_path.to_string(),
    }
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
        .backend
        .to_string()
}

/// Files the service needs that are not part of its image, with whether they are present.
#[derive(Debug, Clone, Serialize)]
pub struct ServiceModel {
    pub name: String,
    pub display_name: String,
    pub path: String,
    pub installed: bool,
}

pub async fn models(app: &AppHandle, id: ServiceId) -> Result<Vec<ServiceModel>> {
    let s = spec(id, app.state::<AppState>().vendor());
    let mut out = Vec::new();
    for (name, _url, path) in model_files(s) {
        let probe = match s.model_store {
            Some((volume, _)) => format!(
                "{} test -s \"$mp{}\" && echo YES",
                store_prelude(volume),
                store_relative(s, &path)
            ),
            None => format!("docker exec {} test -s '{}' && echo YES", s.container, path),
        };
        let installed = wsl::sh(&probe)
            .await
            .map(|o| o.stdout.contains("YES"))
            .unwrap_or(false);
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
/// service log. Blocks until done. Goes through the model store when the service
/// has one (no running container needed), else through `docker exec`.
pub async fn install_model(app: AppHandle, id: ServiceId, name: String) -> Result<ServiceModel> {
    let s = spec(id, app.state::<AppState>().vendor());
    let Some((_, url, path)) = model_files(s).into_iter().find(|(n, _, _)| *n == name) else {
        return Err(Error::Other(format!(
            "`{name}` is not a known model for {id:?}"
        )));
    };
    push_log(&app, id, format!("downloading {name}"));
    let cmd = match s.model_store {
        Some((volume, _)) => format!(
            "{prelude} f=\"$mp{rel}\"; mkdir -p \"$(dirname \"$f\")\" && \
             curl -L --fail --progress-bar -o \"$f.part\" {url} 2>&1 && mv \"$f.part\" \"$f\" && {index}",
            prelude = store_prelude(volume),
            rel = store_relative(s, &path),
            index = store_index_script(),
        ),
        None => format!(
            "docker exec {c} sh -c 'mkdir -p \"$(dirname {path})\" && curl -L --fail --progress-bar -o {path}.part {url} 2>&1 && mv {path}.part {path}' 2>&1",
            c = s.container
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
        let entry = map
            .entry(id)
            .or_insert_with(|| base_status(spec(id, vendor)));
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

/// Probe docker for the real state and store it; the UI's `service_status` command.
pub async fn status(app: &AppHandle, id: ServiceId) -> Result<ServiceStatus> {
    let state = app.state::<AppState>();
    let s = spec(id, state.vendor());
    let busy = get(&state, id)
        .map(|st| matches!(st.state, ServiceState::Pulling | ServiceState::Starting))
        .unwrap_or(false);
    if busy {
        // A start task owns the status while it runs; do not clobber its progress.
        return get(&state, id).ok_or_else(|| Error::Other("service status missing".into()));
    }
    // The published port is what the distro sees; the container port is private to it.
    let probe = format!(
        "docker image inspect {image} >/dev/null 2>&1 && echo IMAGE; \
         docker inspect -f '{{{{.State.Running}}}}' {container} 2>/dev/null; \
         curl -sf -m 3 -o /dev/null http://127.0.0.1:{port}{health} && echo HEALTHY",
        image = s.image,
        container = s.container,
        port = s.host_port,
        health = s.health_path,
    );
    let out = wsl::sh(&probe).await?;
    let image_present = out.stdout.lines().any(|l| l.trim() == "IMAGE");
    let running = out.stdout.lines().any(|l| l.trim() == "true");
    let healthy = out.stdout.lines().any(|l| l.trim() == "HEALTHY");
    update(app, id, |st| {
        st.image_present = image_present;
        st.backend = s.backend.to_string();
        st.error = None;
        st.state = if running && healthy {
            ServiceState::Running
        } else if running {
            ServiceState::Starting
        } else if image_present {
            ServiceState::Stopped
        } else {
            ServiceState::Missing
        };
    });
    get(&state, id).ok_or_else(|| Error::Other("service status missing".into()))
}

/// Pull (if needed), run, and wait for health. Returns immediately with the
/// in-progress status; progress arrives on `service://update`.
pub async fn start(app: AppHandle, id: ServiceId) -> Result<ServiceStatus> {
    let state = app.state::<AppState>();
    if !state.is_ready() {
        return Err(Error::NotReady("install the runtime first".into()));
    }
    let current = status(&app, id).await?;
    if matches!(
        current.state,
        ServiceState::Pulling | ServiceState::Starting | ServiceState::Running
    ) {
        return Ok(current);
    }
    let gpu = state.has_gpu();
    update(&app, id, |s| {
        s.state = if s.image_present {
            ServiceState::Starting
        } else {
            ServiceState::Pulling
        };
        s.error = None;
    });
    let app2 = app.clone();
    tokio::spawn(async move {
        if let Err(e) = run(&app2, id, gpu).await {
            fail(&app2, id, e.to_string());
        }
    });
    get(&state, id).ok_or_else(|| Error::Other("service status missing".into()))
}

async fn run(app: &AppHandle, id: ServiceId, gpu: bool) -> Result<()> {
    let s = spec(id, app.state::<AppState>().vendor());
    let has_image = get(&app.state::<AppState>(), id)
        .map(|st| st.image_present)
        .unwrap_or(false);
    if !has_image {
        push_log(app, id, format!("docker pull {}", s.image));
        let child = wsl::spawn_sh(&format!("docker pull {} 2>&1", s.image))?;
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

    if s.model_store.is_some() {
        // The server must find a valid (possibly empty) index on its first start.
        wsl::sh(&store_index_script())
            .await?
            .require("prepare model store")?;
    }

    let gpu_flag = if gpu && s.gpu { "--gpus all" } else { "" };
    let env: String = s
        .env
        .iter()
        .map(|(k, v)| format!("-e {k}='{v}'"))
        .collect::<Vec<_>>()
        .join(" ");
    let volumes: String = s
        .volumes
        .iter()
        .map(|(name, path)| format!("-v {name}:{path}"))
        .collect::<Vec<_>>()
        .join(" ");
    let cmd = format!(
        "docker rm -f {c} >/dev/null 2>&1; \
         docker run -d --name {c} --restart unless-stopped {gpu_flag} \
           -p {hp}:{cp} {env} {volumes} --label aias.kind=service {image} {args}",
        c = s.container,
        hp = s.host_port,
        cp = s.container_port,
        image = s.image,
        args = s.cmd,
    );
    push_log(
        app,
        id,
        format!(
            "docker run -p {}:{} {}",
            s.host_port, s.container_port, s.image
        ),
    );
    wsl::sh(&cmd).await?.require("docker run")?;

    let http = reqwest::Client::builder()
        .no_proxy()
        .timeout(Duration::from_secs(3))
        .build()?;
    let url = format!("http://localhost:{}{}", s.host_port, s.health_path);
    for tick in 0..90u32 {
        tokio::time::sleep(Duration::from_secs(2)).await;
        if let Ok(resp) = http.get(&url).send().await {
            if resp.status().is_success() {
                update(app, id, |st| st.state = ServiceState::Running);
                push_log(app, id, "healthy".into());
                return Ok(());
            }
        }
        if tick % 5 == 4 {
            let probe = format!(
                "docker inspect -f '{{{{.State.Running}}}}' {c} 2>/dev/null; docker logs --tail 3 {c} 2>&1",
                c = s.container
            );
            if let Ok(o) = wsl::sh(&probe).await {
                let mut lines = o.stdout.lines();
                let running = lines.next().map(str::trim) == Some("true");
                for l in lines {
                    push_log(app, id, l.to_string());
                }
                if !running {
                    return Err(Error::Other(
                        "container exited before becoming healthy; see log".into(),
                    ));
                }
            }
        }
    }
    Err(Error::Other(
        "timed out waiting for the service to become healthy".into(),
    ))
}

pub async fn stop(app: AppHandle, id: ServiceId) -> Result<()> {
    let s = spec(id, app.state::<AppState>().vendor());
    let _ = wsl::sh(&format!("docker rm -f {} >/dev/null 2>&1", s.container)).await;
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
