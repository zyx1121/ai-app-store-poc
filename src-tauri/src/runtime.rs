//! Detect and provision the local runtime: WSL2, our distro, Docker, NVIDIA,
//! Ollama. This is the piece that a future MSI/OTA agent grows out of.

use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Emitter};

use crate::error::{Error, Result};
use crate::hardware::{self, HardwareProfile, Vendor};
use crate::ollama;
use crate::state::AppState;
use crate::wsl::{self, DISTRO};

const PROVISION_SCRIPT: &str = include_str!("../provision.sh");
const PROGRESS_EVENT: &str = "provision://progress";

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RuntimeStatus {
    pub wsl_installed: bool,
    pub wsl_version: Option<String>,
    pub distro_present: bool,
    pub distro_running: bool,
    pub docker_ok: bool,
    pub gpu_ok: bool,
    pub ollama_ok: bool,
    pub gpu_name: Option<String>,
    pub vram_mb: Option<u64>,
    pub ready: bool,
    pub reboot_required: bool,
    /// GPUs and NPUs on the host and the vendor the runtime is built around.
    pub hardware: HardwareProfile,
    pub vendor: Vendor,
}

#[derive(Debug, Clone, Serialize)]
pub struct ProvisionEvent {
    pub step: String,
    pub status: &'static str,
    pub message: String,
}

fn emit(app: &AppHandle, step: &str, status: &'static str, message: impl Into<String>) {
    let ev = ProvisionEvent {
        step: step.to_string(),
        status,
        message: message.into(),
    };
    log::info!("[{}] {} {}", ev.step, ev.status, ev.message);
    let _ = app.emit(PROGRESS_EVENT, ev);
}

/// Probe everything. Never fails: a missing piece is a `false`, not an error.
pub async fn status() -> RuntimeStatus {
    let mut s = RuntimeStatus::default();
    s.hardware = hardware::probe().await;
    s.vendor = s.hardware.vendor;
    if let Some(g) = &s.hardware.primary_gpu {
        s.gpu_name = Some(g.name.clone());
        s.vram_mb = g.vram_mb;
    }

    match wsl::wsl(&["--status"]).await {
        Ok(o) if o.ok() => s.wsl_installed = true,
        _ => return finish(s),
    }
    if let Ok(o) = wsl::wsl(&["--version"]).await {
        s.wsl_version = o
            .stdout
            .lines()
            .next()
            .and_then(|l| l.split_whitespace().last())
            .map(str::to_string);
    }

    if let Ok(o) = wsl::wsl(&["-l", "-q"]).await {
        s.distro_present = o.stdout.lines().any(|l| l.trim() == DISTRO);
    }
    if !s.distro_present {
        return finish(s);
    }
    if let Ok(o) = wsl::wsl(&["-l", "-q", "--running"]).await {
        s.distro_running = o.stdout.lines().any(|l| l.trim() == DISTRO);
    }

    // One round trip into the distro for all the service checks.
    let probe = r#"
        systemctl is-active --quiet docker && docker info >/dev/null 2>&1 && echo DOCKER_OK
        docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q nvidia && echo GPU_RUNTIME
        if [ -x /usr/lib/wsl/lib/nvidia-smi ]; then
          echo "GPU_INFO $(/usr/lib/wsl/lib/nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)"
        fi
    "#;
    if let Ok(o) = wsl::sh(probe).await {
        let mut gpu_runtime = false;
        for line in o.stdout.lines() {
            let line = line.trim();
            match line {
                "DOCKER_OK" => s.docker_ok = true,
                "GPU_RUNTIME" => gpu_runtime = true,
                l if l.starts_with("GPU_INFO ") => {
                    let rest = &l["GPU_INFO ".len()..];
                    let mut parts = rest.rsplitn(2, ',');
                    let mem = parts
                        .next()
                        .map(str::trim)
                        .and_then(|m| m.parse::<u64>().ok());
                    let name = parts.next().map(|n| n.trim().to_string());
                    if let (Some(name), Some(mem)) = (name, mem) {
                        s.gpu_name = Some(name);
                        s.vram_mb = Some(mem);
                    }
                }
                _ => {}
            }
        }
        s.gpu_ok = gpu_runtime && s.vendor == Vendor::Nvidia;
        s.distro_running = true;
    }
    // Ollama answers on localhost whether it runs in WSL (NVIDIA) or natively (others).
    s.ollama_ok = ollama::is_up().await;
    finish(s)
}

fn finish(mut s: RuntimeStatus) -> RuntimeStatus {
    s.ready = s.wsl_installed && s.distro_present && s.docker_ok && s.ollama_ok;
    s
}

/// Refresh status into shared state and keep the distro alive if it exists.
pub async fn refresh(state: &AppState) -> RuntimeStatus {
    let s = status().await;
    if s.distro_present {
        state.ensure_keepalive();
    }
    if let Ok(mut g) = state.runtime.lock() {
        *g = Some(s.clone());
    }
    s
}

/// Bring the machine from "nothing" to "ready", emitting progress events.
/// Returns early with `reboot_required` when Windows needs a restart.
pub async fn provision(app: &AppHandle, state: &AppState) -> Result<RuntimeStatus> {
    let mut s = status().await;

    if !s.wsl_installed {
        emit(app, "wsl", "start", "Installing WSL2 (no distribution)");
        let child = wsl::spawn_win("wsl.exe", &["--install", "--no-distribution"])?;
        let code = wsl::stream_lines(child, |l| emit(app, "wsl", "log", l)).await?;
        if code != 0 {
            emit(
                app,
                "wsl",
                "error",
                format!("wsl --install exited with {code}"),
            );
            return Err(Error::Other(
                "WSL install failed; run `wsl --install --no-distribution` in an elevated terminal"
                    .into(),
            ));
        }
        s = status().await;
        if !s.wsl_installed {
            s.reboot_required = true;
            emit(
                app,
                "wsl",
                "ok",
                "WSL installed. Reboot Windows, then open this app again.",
            );
            return Ok(s);
        }
        emit(app, "wsl", "ok", "WSL2 ready");
    }

    if !s.distro_present {
        emit(
            app,
            "distro",
            "start",
            format!("Installing Ubuntu 24.04 as `{DISTRO}`"),
        );
        let child = wsl::spawn_win(
            "wsl.exe",
            &[
                "--install",
                "-d",
                "Ubuntu-24.04",
                "--name",
                DISTRO,
                "--no-launch",
            ],
        )?;
        let code = wsl::stream_lines(child, |l| emit(app, "distro", "log", l)).await?;
        let present = wsl::wsl(&["-l", "-q"])
            .await
            .map(|o| o.stdout.lines().any(|l| l.trim() == DISTRO))
            .unwrap_or(false);
        if code != 0 || !present {
            emit(
                app,
                "distro",
                "error",
                format!("distro install exited with {code}"),
            );
            return Err(Error::Other("could not install the Ubuntu distro".into()));
        }
        emit(app, "distro", "ok", "distro registered");
    }

    emit(
        app,
        "provision",
        "start",
        "Installing Docker, NVIDIA toolkit, Ollama inside the distro",
    );
    // The Rust side knows the vendor; the script gates the WSL Ollama install on it.
    // A Windows checkout may have turned the script into CRLF; bash rejects that.
    let script = format!(
        "export AIAS_VENDOR={}\n{}",
        match s.vendor {
            Vendor::Nvidia => "nvidia",
            Vendor::Amd => "amd",
            Vendor::Intel => "intel",
            Vendor::Cpu => "cpu",
        },
        PROVISION_SCRIPT.replace("\r\n", "\n")
    );
    let child = wsl::spawn_script(&script).await?;
    let code = wsl::stream_lines(child, |l| emit(app, "provision", "log", l)).await?;
    if code != 0 {
        emit(
            app,
            "provision",
            "error",
            format!("provision script exited with {code}"),
        );
        return Err(Error::Other("provisioning failed; see log".into()));
    }
    emit(app, "provision", "ok", "packages installed");

    emit(
        app,
        "restart",
        "start",
        "Restarting the distro so systemd takes over",
    );
    if let Ok(mut g) = state.keepalive.lock() {
        if let Some(mut child) = g.take() {
            let _ = child.start_kill();
        }
    }
    let _ = wsl::wsl(&["--terminate", DISTRO]).await;
    tokio::time::sleep(std::time::Duration::from_secs(2)).await;
    state.ensure_keepalive();
    let boot = if s.vendor == Vendor::Nvidia {
        wsl::sh(
            "systemctl is-system-running --wait >/dev/null 2>&1; systemctl enable --now docker ollama >/dev/null 2>&1; \
             for i in $(seq 1 30); do curl -sf -m 2 http://127.0.0.1:11434/api/version >/dev/null && break; sleep 1; done; \
             systemctl is-active docker ollama",
        )
        .await?
    } else {
        wsl::sh(
            "systemctl is-system-running --wait >/dev/null 2>&1; systemctl enable --now docker >/dev/null 2>&1; systemctl is-active docker",
        )
        .await?
    };
    emit(app, "restart", "log", boot.stdout.trim().to_string());
    emit(app, "restart", "ok", "services up");

    if s.vendor != Vendor::Nvidia {
        emit(
            app,
            "ollama",
            "start",
            "Ollama runs natively on Windows for this GPU",
        );
        // The store cannot see the hardware profile from inside the distro; store the
        // fresh status first so `ensure_native` knows the vendor.
        if let Ok(mut g) = state.runtime.lock() {
            *g = Some(s.clone());
        }
        match ollama::ensure_native(state, |l| emit(app, "ollama", "log", l)).await {
            Ok(()) => emit(app, "ollama", "ok", "Ollama up on localhost:11434"),
            Err(e) => emit(app, "ollama", "error", e.to_string()),
        }
    }

    let s = refresh(state).await;
    if s.ready {
        emit(app, "done", "ok", "Runtime ready");
    } else {
        emit(app, "done", "error", format!("still not ready: {s:?}"));
    }
    Ok(s)
}
