//! Launch, track and stop the things the store runs: Spaces as Docker
//! containers, GGUF models inside Ollama. All long work runs in spawned tasks
//! that publish `instance://update` events.

use std::time::Duration;

use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Emitter, Manager};

use crate::error::{Error, Result};
use crate::hf;
use crate::state::AppState;
use crate::wsl::{self, slug};

const UPDATE_EVENT: &str = "instance://update";
const LOG_TAIL: usize = 20;
const CONTAINER_PREFIX: &str = "aias-";
pub const OLLAMA_PORT: u16 = 11434;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Kind {
    Space,
    Model,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Status {
    Pulling,
    Starting,
    Running,
    Error,
    Stopped,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Instance {
    pub id: String,
    pub kind: Kind,
    pub repo: String,
    pub model_tag: Option<String>,
    pub display_name: String,
    pub status: Status,
    pub port: Option<u16>,
    pub url: Option<String>,
    pub error: Option<String>,
    pub log_tail: Vec<String>,
    pub started_at: String,
}

fn now() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    // ISO-ish without pulling in chrono: the UI only needs ordering.
    format!("{secs}")
}

fn container_name(id: &str) -> String {
    format!("{CONTAINER_PREFIX}{}", id.trim_start_matches("space-"))
}

fn free_port() -> Result<u16> {
    let l = std::net::TcpListener::bind(("127.0.0.1", 0))?;
    Ok(l.local_addr()?.port())
}

/// Apply `f` to the instance, then broadcast it.
fn update(app: &AppHandle, id: &str, f: impl FnOnce(&mut Instance)) {
    let state = app.state::<AppState>();
    let snapshot = {
        let mut map = match state.instances.lock() {
            Ok(m) => m,
            Err(_) => return,
        };
        let Some(inst) = map.get_mut(id) else { return };
        f(inst);
        inst.clone()
    };
    let _ = app.emit(UPDATE_EVENT, snapshot);
}

fn push_log(app: &AppHandle, id: &str, line: String) {
    update(app, id, |i| {
        if i.log_tail.len() >= LOG_TAIL {
            i.log_tail.remove(0);
        }
        i.log_tail.push(line);
    });
}

fn fail(app: &AppHandle, id: &str, msg: String) {
    log::error!("{id}: {msg}");
    update(app, id, |i| {
        i.status = Status::Error;
        i.error = Some(msg);
    });
}

pub fn list(state: &AppState) -> Vec<Instance> {
    state
        .instances
        .lock()
        .map(|m| m.values().cloned().collect())
        .unwrap_or_default()
}

fn insert(state: &AppState, inst: Instance) {
    if let Ok(mut m) = state.instances.lock() {
        m.insert(inst.id.clone(), inst);
    }
}

fn get(state: &AppState, id: &str) -> Option<Instance> {
    state.instances.lock().ok().and_then(|m| m.get(id).cloned())
}

fn require_ready(state: &AppState) -> Result<()> {
    if state.is_ready() {
        Ok(())
    } else {
        Err(Error::NotReady("install the runtime first".into()))
    }
}

// ---- Spaces ---------------------------------------------------------------

pub async fn launch_space(app: AppHandle, id: String) -> Result<Instance> {
    let state = app.state::<AppState>();
    require_ready(&state)?;
    let space = hf::space(&state.http, &id, state.has_gpu()).await?;
    if space.compat == hf::Compat::Incompatible {
        return Err(Error::Other(
            space
                .compat_reason
                .unwrap_or_else(|| "Space is not compatible".into()),
        ));
    }

    let inst_id = format!("space-{}", slug(&id));
    if let Some(existing) = get(&state, &inst_id) {
        if matches!(
            existing.status,
            Status::Pulling | Status::Starting | Status::Running
        ) {
            return Ok(existing);
        }
    }
    let inst = Instance {
        id: inst_id.clone(),
        kind: Kind::Space,
        repo: id.clone(),
        model_tag: None,
        display_name: space.title.clone().unwrap_or_else(|| space.name.clone()),
        status: Status::Pulling,
        port: None,
        url: None,
        error: None,
        log_tail: vec![],
        started_at: now(),
    };
    insert(&state, inst.clone());

    let image = format!("registry.hf.space/{}:latest", slug(&id));
    let gpu = state.has_gpu();
    let app2 = app.clone();
    tokio::spawn(async move {
        if let Err(e) = run_space(&app2, &inst_id, &image, &space, gpu).await {
            fail(&app2, &inst_id, e.to_string());
        }
    });
    Ok(inst)
}

/// HF Space images carry no CMD; the Hub starts them with a command derived
/// from the SDK. Mirror that, but leave images that define their own command alone.
fn space_command(sdk: Option<&str>, app_file: &str, app_port: u16, image_has_cmd: bool) -> String {
    if image_has_cmd {
        return String::new();
    }
    match sdk {
        Some("gradio") => format!("python {app_file}"),
        Some("streamlit") => format!(
            "streamlit run {app_file} --server.port {app_port} --server.address 0.0.0.0 --server.headless true"
        ),
        _ => String::new(),
    }
}

async fn run_space(
    app: &AppHandle,
    id: &str,
    image: &str,
    space: &hf::SpaceSummary,
    gpu: bool,
) -> Result<()> {
    let cname = container_name(id);
    let app_port = space.app_port;
    push_log(app, id, format!("docker pull {image}"));
    let child = wsl::spawn_sh(&format!("docker pull {image} 2>&1"))?;
    let code = wsl::stream_lines(child, |l| push_log(app, id, l)).await?;
    if code != 0 {
        return Err(Error::Other(format!(
            "image pull failed ({code}); the Space may be private, gated, or have no image"
        )));
    }

    let inspect = wsl::sh(&format!(
        "docker inspect -f '{{{{len .Config.Cmd}}}} {{{{len .Config.Entrypoint}}}}' {image}"
    ))
    .await?;
    let image_has_cmd = inspect
        .stdout
        .split_whitespace()
        .any(|n| n.parse::<u32>().map(|v| v > 0).unwrap_or(false));
    let command = space_command(
        space.sdk.as_deref(),
        &space.app_file,
        app_port,
        image_has_cmd,
    );
    if command.is_empty() && !image_has_cmd {
        return Err(Error::Other(format!(
            "image has no start command and SDK `{}` has no default; cannot launch",
            space.sdk.as_deref().unwrap_or("unknown")
        )));
    }

    let port = free_port()?;
    let gpu_flag = if gpu { "--gpus all" } else { "" };
    let repo = &space.id;
    let run = format!(
        "docker rm -f {cname} >/dev/null 2>&1; \
         docker run -d --name {cname} {gpu_flag} -p {port}:{app_port} \
           --label aias.kind=space --label aias.repo='{repo}' --label aias.port={port} \
           -e PORT={app_port} -e GRADIO_SERVER_NAME=0.0.0.0 -e GRADIO_SERVER_PORT={app_port} \
           {image} {command}"
    );
    update(app, id, |i| {
        i.status = Status::Starting;
        i.port = Some(port);
    });
    push_log(
        app,
        id,
        format!("docker run -p {port}:{app_port} {gpu_flag} {image} {command}"),
    );
    wsl::sh(&run).await?.require("docker run")?;

    wait_for_http(app, id, port, &cname).await?;
    update(app, id, |i| {
        i.status = Status::Running;
        i.url = Some(format!("http://localhost:{port}"));
    });
    Ok(())
}

/// Poll the published port until something answers, or the container dies.
async fn wait_for_http(app: &AppHandle, id: &str, port: u16, cname: &str) -> Result<()> {
    let http = reqwest::Client::builder()
        .no_proxy()
        .timeout(Duration::from_secs(3))
        .build()?;
    let url = format!("http://localhost:{port}/");
    for tick in 0..150u32 {
        tokio::time::sleep(Duration::from_secs(2)).await;
        if http.get(&url).send().await.is_ok() {
            return Ok(());
        }
        if tick % 5 == 4 {
            let probe = format!(
                "docker inspect -f '{{{{.State.Running}}}}' {cname} 2>/dev/null; docker logs --tail 5 {cname} 2>&1"
            );
            if let Ok(o) = wsl::sh(&probe).await {
                let mut lines = o.stdout.lines();
                let running = lines.next().map(str::trim) == Some("true");
                for l in lines {
                    push_log(app, id, l.to_string());
                }
                if !running {
                    return Err(Error::Other(
                        "container exited before serving; see log".into(),
                    ));
                }
            }
        }
    }
    Err(Error::Other(
        "timed out waiting for the app to answer on its port".into(),
    ))
}

// ---- Models ---------------------------------------------------------------

pub async fn launch_model(app: AppHandle, repo: String, quant: String) -> Result<Instance> {
    let state = app.state::<AppState>();
    require_ready(&state)?;
    hf::validate_repo(&repo)?;
    if quant.is_empty() || !quant.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
        return Err(Error::Other(format!("`{quant}` is not a quant tag")));
    }
    let tag = format!("hf.co/{repo}:{quant}");
    let inst_id = format!("model-{}", slug(&format!("{repo}-{quant}")));
    if let Some(existing) = get(&state, &inst_id) {
        if matches!(
            existing.status,
            Status::Pulling | Status::Starting | Status::Running
        ) {
            return Ok(existing);
        }
    }
    let (_, name) = hf::split_repo(&repo);
    let inst = Instance {
        id: inst_id.clone(),
        kind: Kind::Model,
        repo: repo.clone(),
        model_tag: Some(tag.clone()),
        display_name: format!("{name} ({quant})"),
        status: Status::Pulling,
        port: Some(OLLAMA_PORT),
        url: None,
        error: None,
        log_tail: vec![],
        started_at: now(),
    };
    insert(&state, inst.clone());

    let app2 = app.clone();
    tokio::spawn(async move {
        if let Err(e) = run_model(&app2, &inst_id, &tag).await {
            fail(&app2, &inst_id, e.to_string());
        }
    });
    Ok(inst)
}

async fn run_model(app: &AppHandle, id: &str, tag: &str) -> Result<()> {
    push_log(app, id, format!("ollama pull {tag}"));
    let child = wsl::spawn_sh(&format!("ollama pull {tag} 2>&1"))?;
    let mut last = String::new();
    let code = wsl::stream_lines(child, |l| {
        // Progress bars repeat the same prefix hundreds of times; keep changes only.
        if l != last {
            last = l.clone();
            push_log(app, id, l);
        }
    })
    .await?;
    if code != 0 {
        return Err(Error::Other(format!(
            "ollama pull failed ({code}); see log"
        )));
    }

    update(app, id, |i| i.status = Status::Starting);
    push_log(app, id, "loading model into memory".into());
    let load = format!(
        "curl -sf -m 600 http://127.0.0.1:{OLLAMA_PORT}/api/generate -d '{{\"model\":\"{tag}\",\"keep_alive\":\"30m\"}}' >/dev/null && ollama ps | grep -F '{tag}'"
    );
    let o = wsl::sh(&load).await?.require("ollama load")?;
    push_log(app, id, o.stdout.trim().to_string());
    update(app, id, |i| {
        i.status = Status::Running;
        i.url = Some(format!("http://localhost:{OLLAMA_PORT}/v1"));
    });
    Ok(())
}

// ---- lifecycle ------------------------------------------------------------

pub async fn stop(app: AppHandle, id: String) -> Result<()> {
    let state = app.state::<AppState>();
    let Some(inst) = get(&state, &id) else {
        return Err(Error::Other(format!("no instance `{id}`")));
    };
    match inst.kind {
        Kind::Space => {
            let cname = container_name(&id);
            let _ = wsl::sh(&format!("docker rm -f {cname} >/dev/null 2>&1")).await;
        }
        Kind::Model => {
            if let Some(tag) = &inst.model_tag {
                let _ = wsl::sh(&format!("ollama stop '{tag}' >/dev/null 2>&1")).await;
            }
        }
    }
    update(&app, &id, |i| {
        i.status = Status::Stopped;
        i.url = None;
    });
    Ok(())
}

pub async fn remove(app: AppHandle, id: String) -> Result<()> {
    let state = app.state::<AppState>();
    if let Some(inst) = get(&state, &id) {
        if inst.kind == Kind::Space {
            let cname = container_name(&id);
            let _ = wsl::sh(&format!("docker rm -f {cname} >/dev/null 2>&1")).await;
        }
        if let Ok(mut m) = state.instances.lock() {
            m.remove(&id);
        }
        let mut gone = inst;
        gone.status = Status::Stopped;
        let _ = app.emit(UPDATE_EVENT, gone);
    }
    Ok(())
}

/// Re-adopt containers left from a previous session (labels tell us what they are).
pub async fn discover(app: &AppHandle) {
    let state = app.state::<AppState>();
    {
        let mut done = match state.discovered.lock() {
            Ok(d) => d,
            Err(_) => return,
        };
        if *done || !state.is_ready() {
            return;
        }
        *done = true;
    }
    let list = "docker ps --filter label=aias.kind=space --format '{{.Names}}\t{{.Label \"aias.repo\"}}\t{{.Label \"aias.port\"}}\t{{.Status}}'";
    let Ok(o) = wsl::sh(list).await else { return };
    for line in o.stdout.lines() {
        let parts: Vec<&str> = line.split('\t').collect();
        if parts.len() < 4 {
            continue;
        }
        let cname = parts[0];
        let repo = parts[1].to_string();
        let port: Option<u16> = parts[2].parse().ok();
        let id = format!("space-{}", cname.trim_start_matches(CONTAINER_PREFIX));
        if get(&state, &id).is_some() {
            continue;
        }
        let (_, name) = hf::split_repo(&repo);
        insert(
            &state,
            Instance {
                id: id.clone(),
                kind: Kind::Space,
                repo,
                model_tag: None,
                display_name: name,
                status: Status::Running,
                port,
                url: port.map(|p| format!("http://localhost:{p}")),
                error: None,
                log_tail: vec![format!("adopted running container ({})", parts[3])],
                started_at: now(),
            },
        );
        if let Some(inst) = get(&state, &id) {
            let _ = app.emit(UPDATE_EVENT, inst);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::space_command;

    #[test]
    fn derives_start_command() {
        assert_eq!(
            space_command(Some("gradio"), "app.py", 7860, false),
            "python app.py"
        );
        assert!(space_command(Some("streamlit"), "main.py", 8501, false)
            .starts_with("streamlit run main.py"));
        assert_eq!(space_command(Some("docker"), "app.py", 7860, true), "");
        assert_eq!(space_command(Some("docker"), "app.py", 7860, false), "");
    }
}
