//! Detect and provision the local runtime: WSL2, our distro, Docker, NVIDIA,
//! Ollama. This is the piece that a future MSI/OTA agent grows out of.

use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Emitter};

use crate::error::{Error, Result};
use crate::hardware::{self, HardwareProfile, Vendor, Virtualization};
use crate::ollama;
use crate::state::AppState;
use crate::wsl::{self, DISTRO};

const PROVISION_SCRIPT: &str = include_str!("../provision.sh");
const PROGRESS_EVENT: &str = "provision://progress";
/// Set to `nvidia` / `amd` / `intel` / `cpu` to exercise another vendor's runtime
/// path on this machine (per-vendor services, native Ollama, CPU builds). For
/// testing only: the GPU itself does not change, so CUDA still works underneath.
pub const VENDOR_OVERRIDE_ENV: &str = "AIAS_VENDOR";

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
    /// Dedicated memory of the primary GPU as the driver reports it (display).
    pub vram_mb: Option<u64>,
    /// What a model may occupy on the primary accelerator: `vram_mb` on a
    /// discrete card, half of system RAM on a unified part. Verdicts read this.
    pub effective_memory_mb: Option<u64>,
    pub ready: bool,
    pub reboot_required: bool,
    /// virtualization must be enabled in UEFI firmware before WSL2 can run
    pub virtualization: Virtualization,
    /// GPUs and NPUs on the host and the vendor the runtime is built around.
    pub hardware: HardwareProfile,
    pub vendor: Vendor,
    /// `vendor` came from `AIAS_VENDOR`, not from the probe.
    pub vendor_forced: bool,
}

/// The vendor named by `AIAS_VENDOR`, if it is set to a known value.
fn forced_vendor() -> Option<Vendor> {
    let v = std::env::var(VENDOR_OVERRIDE_ENV).ok()?;
    match v.trim().to_ascii_lowercase().as_str() {
        "nvidia" => Some(Vendor::Nvidia),
        "amd" => Some(Vendor::Amd),
        "intel" => Some(Vendor::Intel),
        "cpu" => Some(Vendor::Cpu),
        other => {
            log::warn!("{VENDOR_OVERRIDE_ENV}={other:?} is not a vendor; ignoring");
            None
        }
    }
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

/// Emit an `error` progress event before an early exit that would otherwise be
/// silent: a failed spawn or a broken read has no business-logic message of its
/// own, so the UI would just see the button flip back with an empty log.
fn emit_err<T>(app: &AppHandle, step: &str, result: Result<T>) -> Result<T> {
    if let Err(e) = &result {
        emit(app, step, "error", e.to_string());
    }
    result
}

/// Probe everything. Never fails: a missing piece is a `false`, not an error.
pub async fn status() -> RuntimeStatus {
    let mut s = RuntimeStatus {
        hardware: hardware::probe().await,
        ..Default::default()
    };
    if let Some(v) = forced_vendor() {
        log::warn!(
            "{VENDOR_OVERRIDE_ENV}: treating this machine as {v:?} (probe said {:?})",
            s.hardware.vendor
        );
        s.hardware.vendor = v;
        s.hardware.wsl_gpu = v == Vendor::Nvidia;
        s.vendor_forced = true;
    }
    s.vendor = s.hardware.vendor;
    s.virtualization = s.hardware.virtualization;
    if let Some(g) = &s.hardware.primary_gpu {
        s.gpu_name = Some(g.name.clone());
        s.vram_mb = g.vram_mb;
        s.effective_memory_mb = g.effective_memory_mb;
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
                        // nvidia-smi is authoritative for a CUDA card: dedicated memory.
                        s.gpu_name = Some(name);
                        s.vram_mb = Some(mem);
                        s.effective_memory_mb = Some(mem);
                    }
                }
                _ => {}
            }
        }
        s.gpu_ok = s.vendor.gpu_ok(gpu_runtime);
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

    // WSL2 needs the Windows hypervisor. If the CPU cannot virtualize, or VT is
    // switched off in the UEFI firmware, `wsl --install` "succeeds" but the VM
    // never boots. There is no reliable way to flip the firmware switch from
    // Windows, so stop here with the one instruction the user must act on.
    let v = s.virtualization;
    if !s.wsl_installed && !v.usable() {
        if !v.vt_supported {
            emit(
                app,
                "virtualization",
                "error",
                "This CPU has no hardware virtualization (VT-x / AMD-V); WSL2 cannot run here.",
            );
            return Err(Error::Other(
                "This CPU does not support hardware virtualization, which WSL2 requires.".into(),
            ));
        }
        emit(
            app,
            "virtualization",
            "error",
            "Virtualization is turned off in the UEFI firmware.",
        );
        return Err(Error::Other(
            "Enable virtualization (Intel VT-x / VT-d or AMD SVM) in the UEFI firmware, then reboot and try again.              On most machines: reboot, open firmware setup (Del / F2 / F10 at boot), turn on Intel Virtualization Technology (VT-x) and VT-d, save and exit."
                .into(),
        ));
    }

    if !s.wsl_installed {
        // Enable the two Windows features WSL2 needs, deterministically. `wsl --install`
        // is avoided here: on a machine whose inbox WSL is stale it drops to an
        // interactive "press a key to update WSL" prompt that a spawned process cannot
        // answer. DISM never prompts and returns 3010 when a reboot is required.
        emit(app, "wsl", "start", "Enabling the WSL2 Windows features");
        let mut need_reboot = false;
        for feature in [
            "Microsoft-Windows-Subsystem-Linux",
            "VirtualMachinePlatform",
        ] {
            let out = emit_err(
                app,
                "wsl",
                wsl::run(
                    "dism.exe",
                    &[
                        "/online",
                        "/enable-feature",
                        &format!("/featurename:{feature}"),
                        "/all",
                        "/norestart",
                    ],
                )
                .await,
            )?;
            emit(
                app,
                "wsl",
                "log",
                format!("{feature}: dism exit {}", out.status),
            );
            match out.status {
                0 => {}
                // 3010: the feature was enabled but Windows must restart to activate it.
                3010 => need_reboot = true,
                code => {
                    emit(
                        app,
                        "wsl",
                        "error",
                        format!("enabling {feature} failed ({code})"),
                    );
                    return Err(Error::Other(format!(
                        "could not enable the {feature} Windows feature (dism exit {code})"
                    )));
                }
            }
        }
        if need_reboot {
            s.reboot_required = true;
            emit(
                app,
                "wsl",
                "ok",
                "WSL2 features enabled. Reboot Windows, then open this app again.",
            );
            return Ok(s);
        }

        // Features are on; install the modern WSL app. `--web-download` takes it from
        // GitHub instead of the Store, so it works on machines without the Store.
        emit(app, "wsl", "start", "Installing the WSL app");
        let child = emit_err(
            app,
            "wsl",
            wsl::spawn_win("wsl.exe", &["--update", "--web-download"]),
        )?;
        let code = emit_err(
            app,
            "wsl",
            wsl::stream_lines(child, |l| emit(app, "wsl", "log", l)).await,
        )?;
        if code != 0 {
            emit(
                app,
                "wsl",
                "error",
                format!("wsl --update exited with {code}"),
            );
            return Err(Error::Other(
                "installing the WSL app failed; see log".into(),
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

    // Global to every WSL distro; only touched when absent, so a reboot into
    // firmware setup and a second `provision` call never clobbers the user's
    // own tuning. Best-effort: a failure here does not block provisioning.
    match write_wslconfig(s.hardware.total_ram_mb) {
        WslConfigOutcome::Written(path) => emit(
            app,
            "memory",
            "ok",
            format!("wrote {path} (WSL2 memory ceiling)"),
        ),
        WslConfigOutcome::Kept => emit(app, "memory", "ok", "existing .wslconfig kept"),
        WslConfigOutcome::Error(e) => emit(app, "memory", "error", e),
    }

    if !s.distro_present {
        emit(
            app,
            "distro",
            "start",
            format!("Installing Ubuntu 24.04 as `{DISTRO}`"),
        );
        let child = emit_err(
            app,
            "distro",
            wsl::spawn_win(
                "wsl.exe",
                &[
                    "--install",
                    "-d",
                    "Ubuntu-24.04",
                    "--name",
                    DISTRO,
                    "--no-launch",
                ],
            ),
        )?;
        let code = emit_err(
            app,
            "distro",
            wsl::stream_lines(child, |l| emit(app, "distro", "log", l)).await,
        )?;
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
        s.vendor.env_name(),
        PROVISION_SCRIPT.replace("\r\n", "\n")
    );
    let child = emit_err(app, "provision", wsl::spawn_script(&script).await)?;
    let code = emit_err(
        app,
        "provision",
        wsl::stream_lines(child, |l| emit(app, "provision", "log", l)).await,
    )?;
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
    let boot = emit_err(
        app,
        "restart",
        if s.vendor == Vendor::Nvidia {
            wsl::sh(
                "systemctl is-system-running --wait >/dev/null 2>&1; systemctl enable --now docker ollama >/dev/null 2>&1; \
                 for i in $(seq 1 30); do curl -sf -m 2 http://127.0.0.1:11434/api/version >/dev/null && break; sleep 1; done; \
                 systemctl is-active docker ollama",
            )
            .await
        } else {
            wsl::sh(
                "systemctl is-system-running --wait >/dev/null 2>&1; systemctl enable --now docker >/dev/null 2>&1; systemctl is-active docker",
            )
            .await
        },
    )?;
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

// ---- Resource limits (#64) -------------------------------------------------
//
// WSL2's VM keeps page cache and returns it slowly (or not at all) without a
// memory ceiling of its own: `vmmemWSL` can hold most of the host's RAM after
// a day of image pulls. `write_wslconfig` gives the whole VM one ceiling;
// Space containers additionally get a per-container `--memory` cap sized off
// the same figure (`state::container_memory_cap_mb`, set from `wsl_cap_mb`
// below at startup and read by `instances::run_space`).

const MIN_WSL_MEMORY_GB: u64 = 6;
/// Fallback when `total_ram_mb` could not be probed: conservative, but never
/// leaves the VM uncapped.
const DEFAULT_TOTAL_RAM_MB: u64 = 16 * 1024;

/// The WSL2 VM's memory ceiling: half of physical RAM, floored at 6 GB.
pub fn wsl_cap_mb(total_ram_mb: Option<u64>) -> u64 {
    let total = total_ram_mb.unwrap_or(DEFAULT_TOTAL_RAM_MB);
    std::cmp::max(MIN_WSL_MEMORY_GB * 1024, total / 2)
}

/// `--memory` cap for a single Space container: 75% of the VM's own ceiling,
/// so one container cannot alone push the VM to its limit.
pub fn container_memory_cap_mb(total_ram_mb: Option<u64>) -> u64 {
    wsl_cap_mb(total_ram_mb) * 3 / 4
}

#[derive(Debug, Clone)]
pub enum WslConfigOutcome {
    /// wrote a new file at this path
    Written(String),
    /// an existing `.wslconfig` was left alone
    Kept,
    /// could not determine `%USERPROFILE%` or write the file
    Error(String),
}

/// Write `%USERPROFILE%\.wslconfig` with a memory ceiling for the whole WSL2
/// VM, but only if the file does not exist yet: it is global to every distro
/// on the machine, so an existing file is the user's own tuning and is never
/// touched.
pub fn write_wslconfig(total_ram_mb: Option<u64>) -> WslConfigOutcome {
    let profile = match std::env::var("USERPROFILE") {
        Ok(p) => p,
        Err(_) => return WslConfigOutcome::Error("%USERPROFILE% is not set".into()),
    };
    let path = std::path::PathBuf::from(profile).join(".wslconfig");
    if path.exists() {
        log::info!("existing .wslconfig kept");
        return WslConfigOutcome::Kept;
    }
    let cap_gb = wsl_cap_mb(total_ram_mb) / 1024;
    let contents =
        format!("[wsl2]\nmemory={cap_gb}GB\nautoMemoryReclaim=gradual\nsparseVhd=true\n");
    match std::fs::write(&path, contents) {
        Ok(()) => WslConfigOutcome::Written(path.display().to_string()),
        Err(e) => WslConfigOutcome::Error(e.to_string()),
    }
}

#[cfg(test)]
mod resource_limit_tests {
    use super::*;

    #[test]
    fn wsl_cap_is_half_of_ram_floored_at_six_gb() {
        assert_eq!(wsl_cap_mb(Some(32 * 1024)), 16 * 1024);
        assert_eq!(
            wsl_cap_mb(Some(8 * 1024)),
            6 * 1024,
            "half of 8GB is below the floor"
        );
        assert_eq!(wsl_cap_mb(None), DEFAULT_TOTAL_RAM_MB / 2);
    }

    #[test]
    fn container_cap_is_three_quarters_of_the_wsl_cap() {
        assert_eq!(container_memory_cap_mb(Some(32 * 1024)), 12 * 1024);
    }
}
