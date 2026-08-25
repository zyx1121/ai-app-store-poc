//! Where Ollama lives depends on the GPU vendor. NVIDIA hands its GPU to WSL2,
//! so Ollama runs inside the distro next to Docker. AMD and Intel GPUs are
//! invisible inside WSL2, so on those machines Ollama runs natively on Windows
//! (ROCm or Vulkan backend) and the store keeps a `ollama serve` child alive.
//! Either way the API is `http://localhost:11434`, which both sides share.

use std::process::Stdio;
use std::time::Duration;

use tokio::process::{Child, Command};

use crate::error::{Error, Result};
use crate::hardware::Vendor;
use crate::state::AppState;
use crate::wsl;

pub const PORT: u16 = 11434;

fn native_exe() -> Option<std::path::PathBuf> {
    if let Ok(local) = std::env::var("LOCALAPPDATA") {
        let p = std::path::PathBuf::from(local).join("Programs\\Ollama\\ollama.exe");
        if p.exists() {
            return Some(p);
        }
    }
    // Fall back to PATH.
    std::env::var_os("PATH").and_then(|paths| {
        std::env::split_paths(&paths)
            .map(|d| d.join("ollama.exe"))
            .find(|p| p.exists())
    })
}

/// Candidate base URLs. WSL2's localhost forwarding listens on `[::1]` on the
/// Windows side while a native `ollama serve` binds `127.0.0.1`; try both.
pub const BASES: [&str; 3] = [
    "http://localhost:11434",
    "http://[::1]:11434",
    "http://127.0.0.1:11434",
];

/// The first base URL that answers, if any.
pub async fn reachable_base() -> Option<&'static str> {
    let http = reqwest::Client::builder()
        .no_proxy()
        .timeout(Duration::from_secs(3))
        .build()
        .ok()?;
    for base in BASES {
        if let Ok(r) = http.get(format!("{base}/api/version")).send().await {
            if r.status().is_success() {
                return Some(base);
            }
        }
    }
    None
}

/// Does anything answer on the Ollama port (from the Windows side, which also
/// sees WSL-published ports)?
pub async fn is_up() -> bool {
    reachable_base().await.is_some()
}

fn native_env(cmd: &mut Command) {
    cmd.env("OLLAMA_HOST", format!("127.0.0.1:{PORT}"))
        .env("OLLAMA_ORIGINS", "*")
        .env("OLLAMA_CONTEXT_LENGTH", "16384")
        .env("OLLAMA_FLASH_ATTENTION", "1")
        .env("OLLAMA_KV_CACHE_TYPE", "q8_0");
}

/// Run an `ollama ...` command wherever Ollama lives and wait for it.
pub async fn run(state: &AppState, args: &[&str]) -> Result<wsl::Output> {
    match state.vendor() {
        Vendor::Nvidia => {
            let joined = args
                .iter()
                .map(|a| format!("'{}'", a.replace('\'', "'\\''")))
                .collect::<Vec<_>>()
                .join(" ");
            wsl::sh(&format!("ollama {joined}")).await
        }
        _ => {
            let exe = native_exe()
                .ok_or_else(|| Error::NotReady("Ollama is not installed on Windows".into()))?;
            wsl::run(&exe.to_string_lossy(), args).await
        }
    }
}

/// Spawn a long `ollama ...` command (pulls) for line streaming.
pub fn spawn(state: &AppState, args: &[&str]) -> Result<Child> {
    match state.vendor() {
        Vendor::Nvidia => {
            let joined = args
                .iter()
                .map(|a| format!("'{}'", a.replace('\'', "'\\''")))
                .collect::<Vec<_>>()
                .join(" ");
            wsl::spawn_sh(&format!("ollama {joined} 2>&1"))
        }
        _ => {
            let exe = native_exe()
                .ok_or_else(|| Error::NotReady("Ollama is not installed on Windows".into()))?;
            wsl::spawn_win(&exe.to_string_lossy(), args)
        }
    }
}

/// Non-NVIDIA machines: install Ollama for Windows if missing and keep a
/// headless `ollama serve` running with CORS open. Idempotent.
pub async fn ensure_native(state: &AppState, mut log: impl FnMut(String)) -> Result<()> {
    if state.vendor() == Vendor::Nvidia {
        return Ok(());
    }
    if native_exe().is_none() {
        log("installing Ollama for Windows (winget)".into());
        let out = wsl::run(
            "winget",
            &[
                "install",
                "-e",
                "--id",
                "Ollama.Ollama",
                "--silent",
                "--accept-source-agreements",
                "--accept-package-agreements",
                "--disable-interactivity",
            ],
        )
        .await?;
        if native_exe().is_none() {
            return Err(Error::Other(format!(
                "Ollama install did not produce ollama.exe: {}",
                out.stderr.trim().chars().take(300).collect::<String>()
            )));
        }
        log("Ollama installed".into());
    }
    if is_up().await {
        return Ok(());
    }
    let exe = native_exe().expect("checked above");
    log(format!("starting {} serve", exe.display()));
    let mut cmd = Command::new(&exe);
    cmd.arg("serve")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    native_env(&mut cmd);
    #[cfg(windows)]
    cmd.creation_flags(0x0800_0000);
    let child = cmd
        .spawn()
        .map_err(|e| Error::Spawn(exe.to_string_lossy().to_string(), e.to_string()))?;
    if let Ok(mut g) = state.native_ollama.lock() {
        *g = Some(child);
    }
    for _ in 0..30 {
        tokio::time::sleep(Duration::from_secs(1)).await;
        if is_up().await {
            log("Ollama is up".into());
            return Ok(());
        }
    }
    Err(Error::Other(
        "ollama serve did not come up within 30 s".into(),
    ))
}
