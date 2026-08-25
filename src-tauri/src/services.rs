//! Platform services: inference servers the store itself runs as containers
//! inside the distro, next to Ollama. Speaches (STT/TTS) today; ComfyUI later.
//! Each service is a fixed spec (image, ports, env, volumes); the store only
//! pulls, starts, health-checks and stops it.

use std::time::Duration;

use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Emitter, Manager};

use crate::error::{Error, Result};
use crate::state::AppState;
use crate::wsl;

const UPDATE_EVENT: &str = "service://update";
const LOG_TAIL: usize = 20;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "lowercase")]
pub enum ServiceId {
    Speaches,
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
    pub error: Option<String>,
    pub log_tail: Vec<String>,
}

struct ServiceSpec {
    id: ServiceId,
    display_name: &'static str,
    image: &'static str,
    container: &'static str,
    /// Windows-side port. Fixed so the frontend can hardcode the base URL.
    host_port: u16,
    container_port: u16,
    gpu: bool,
    env: &'static [(&'static str, &'static str)],
    /// (named volume, container path)
    volumes: &'static [(&'static str, &'static str)],
    /// path that answers 2xx once the service is usable
    health_path: &'static str,
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
    env: &[("ALLOW_ORIGINS", r#"["*"]"#), ("ENABLE_UI", "false")],
    volumes: &[("aias-speaches-cache", "/home/ubuntu/.cache/huggingface/hub")],
    health_path: "/v1/models",
};

fn spec(id: ServiceId) -> &'static ServiceSpec {
    match id {
        ServiceId::Speaches => &SPEACHES,
    }
}

fn base_status(s: &ServiceSpec) -> ServiceStatus {
    ServiceStatus {
        id: s.id,
        display_name: s.display_name.to_string(),
        state: ServiceState::Missing,
        port: s.host_port,
        url: format!("http://localhost:{}", s.host_port),
        image_present: false,
        error: None,
        log_tail: vec![],
    }
}

fn get(state: &AppState, id: ServiceId) -> Option<ServiceStatus> {
    state.services.lock().ok().and_then(|m| m.get(&id).cloned())
}

fn update(app: &AppHandle, id: ServiceId, f: impl FnOnce(&mut ServiceStatus)) {
    let state = app.state::<AppState>();
    let snapshot = {
        let mut map = match state.services.lock() {
            Ok(m) => m,
            Err(_) => return,
        };
        let entry = map.entry(id).or_insert_with(|| base_status(spec(id)));
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
    let s = spec(id);
    let state = app.state::<AppState>();
    let busy = get(&state, id)
        .map(|st| matches!(st.state, ServiceState::Pulling | ServiceState::Starting))
        .unwrap_or(false);
    if busy {
        // A start task owns the status while it runs; do not clobber its progress.
        return get(&state, id).ok_or_else(|| Error::Other("service status missing".into()));
    }
    let probe = format!(
        "docker image inspect {image} >/dev/null 2>&1 && echo IMAGE; \
         docker inspect -f '{{{{.State.Running}}}}' {container} 2>/dev/null; \
         curl -sf -m 3 -o /dev/null http://127.0.0.1:{port}{health} && echo HEALTHY",
        image = s.image,
        container = s.container,
        port = s.container_port,
        health = s.health_path,
    );
    let out = wsl::sh(&probe).await?;
    let image_present = out.stdout.lines().any(|l| l.trim() == "IMAGE");
    let running = out.stdout.lines().any(|l| l.trim() == "true");
    let healthy = out.stdout.lines().any(|l| l.trim() == "HEALTHY");
    update(app, id, |st| {
        st.image_present = image_present;
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
    let s = spec(id);
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
           -p {hp}:{cp} {env} {volumes} --label aias.kind=service {image}",
        c = s.container,
        hp = s.host_port,
        cp = s.container_port,
        image = s.image,
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
    let s = spec(id);
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
