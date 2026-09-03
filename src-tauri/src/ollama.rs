//! Where Ollama lives depends on the GPU vendor. NVIDIA hands its GPU to WSL2,
//! so Ollama runs inside the distro next to Docker. AMD and Intel GPUs are
//! invisible inside WSL2, so on those machines Ollama runs natively on Windows
//! (ROCm or Vulkan backend) and the store keeps a `ollama serve` child alive.
//! Either way the API is `http://localhost:11434`, which both sides share.

use std::path::PathBuf;
use std::process::Stdio;
use std::time::Duration;

use tokio::process::{Child, Command};

use crate::error::{Error, Result};
use crate::fetch;
use crate::hardware::Vendor;
use crate::state::AppState;
use crate::wsl;

pub const PORT: u16 = 11434;

/// Standalone CLI zips (Ollama's documented path for embedding it in another
/// application). The base zip carries the CPU and CUDA backends; AMD adds ROCm.
/// The interactive installer has no working silent mode (ollama/ollama#7969).
const STANDALONE_BASE: &str = "https://ollama.com/download/";
const STANDALONE_ZIP: &str = "ollama-windows-amd64.zip";
const STANDALONE_ROCM_ZIP: &str = "ollama-windows-amd64-rocm.zip";

/// Where the store keeps its own copy of Ollama for Windows.
fn managed_dir() -> Option<PathBuf> {
    std::env::var_os("LOCALAPPDATA").map(|l| PathBuf::from(l).join("ai-app-store").join("ollama"))
}

fn native_exe() -> Option<PathBuf> {
    if let Ok(local) = std::env::var("LOCALAPPDATA") {
        // A copy the user installed with OllamaSetup.exe wins; it keeps itself updated.
        let p = PathBuf::from(local).join("Programs\\Ollama\\ollama.exe");
        if p.exists() {
            return Some(p);
        }
    }
    if let Some(p) = managed_dir().map(|d| d.join("ollama.exe")) {
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

/// Fetch the standalone zips for this vendor into the managed directory and
/// unpack them with the system `tar.exe` (bsdtar reads zip). Returns the exe.
async fn install_standalone(
    http: &reqwest::Client,
    vendor: Vendor,
    log: &mut impl FnMut(String),
) -> Result<PathBuf> {
    let dir = managed_dir().ok_or_else(|| Error::Other("LOCALAPPDATA is not set".into()))?;
    tokio::fs::create_dir_all(&dir).await?;
    let mut zips = vec![STANDALONE_ZIP];
    if vendor == Vendor::Amd {
        zips.push(STANDALONE_ROCM_ZIP);
    }
    for zip in zips {
        let url = format!("{STANDALONE_BASE}{zip}");
        let dest = dir.join(zip);
        log(format!("downloading {url}"));
        fetch::download(http, &url, &dest, log).await?;
        log(format!("unpacking {zip}"));
        wsl::run(
            "tar.exe",
            &["-xf", &dest.to_string_lossy(), "-C", &dir.to_string_lossy()],
        )
        .await?
        .require("tar -xf")?;
        let _ = tokio::fs::remove_file(&dest).await;
    }
    let exe = dir.join("ollama.exe");
    if !exe.exists() {
        return Err(Error::Other(format!(
            "{zip} did not contain ollama.exe",
            zip = STANDALONE_ZIP
        )));
    }
    Ok(exe)
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

/// Origins allowed to call the local inference servers: the Tauri webview on
/// each platform plus the vite dev server. Never `*`: any web page open in
/// the user's browser could otherwise drive Ollama / ComfyUI on localhost.
/// Ollama panics on any scheme other than http(s), so no `tauri://` entry.
pub const WEBVIEW_ORIGINS: &str =
    "http://tauri.localhost,https://tauri.localhost,http://localhost:1420";

fn native_env(cmd: &mut Command) {
    cmd.env("OLLAMA_HOST", format!("127.0.0.1:{PORT}"))
        .env("OLLAMA_ORIGINS", WEBVIEW_ORIGINS)
        .env("OLLAMA_CONTEXT_LENGTH", "16384")
        .env("OLLAMA_FLASH_ATTENTION", "1")
        .env("OLLAMA_KV_CACHE_TYPE", "q8_0")
        // Native Ollama only runs on non-NVIDIA machines here, where the GPU is an
        // AMD or Intel part reached through Vulkan. Ollama ships the Vulkan backend
        // but leaves it off by default, and it drops integrated GPUs (the Intel Arc
        // iGPU, AMD Radeon graphics) unless told to keep them. Without both flags it
        // silently runs on the CPU; with them a Core Ultra's Arc iGPU takes 100% of
        // the inference. Verified on an Intel Core Ultra 5 225H (Arc 130T).
        .env("OLLAMA_VULKAN", "1")
        .env("OLLAMA_IGPU_ENABLE", "1");
}

/// Run an `ollama ...` command wherever Ollama lives and wait for it.
pub async fn run(state: &AppState, args: &[&str]) -> Result<wsl::Output> {
    match state.vendor() {
        Vendor::Nvidia => {
            let joined = args
                .iter()
                .map(|a| wsl::quote(a))
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
                .map(|a| wsl::quote(a))
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

/// Non-NVIDIA machines: install Ollama for Windows if missing (standalone zip
/// into the store's own directory) and keep a headless `ollama serve` running
/// with CORS open. Idempotent.
pub async fn ensure_native(state: &AppState, mut log: impl FnMut(String)) -> Result<()> {
    let vendor = state.vendor();
    if vendor == Vendor::Nvidia {
        return Ok(());
    }
    let exe = match native_exe() {
        Some(exe) => exe,
        None => {
            log("installing Ollama for Windows (standalone zip)".into());
            let exe = install_standalone(&state.http, vendor, &mut log).await?;
            log(format!("Ollama installed at {}", exe.display()));
            exe
        }
    };
    if is_up().await {
        return Ok(());
    }
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
