//! Downloads and archive extraction on the Windows side, for runtimes the
//! store installs natively (Ollama zips, whisper.cpp, ComfyUI portable) and
//! the model files they need.

use std::path::Path;

use tokio::io::AsyncWriteExt;

use crate::error::{Error, Result};
use crate::wsl;

/// Optional token for GitHub release assets of a private repository
/// (development only; a shipped store downloads from a public host).
const GITHUB_TOKEN_ENV: &str = "AIAS_GITHUB_TOKEN";

/// Stream a download to `dest` (via `dest.part`), reporting every 32 MB.
pub async fn download(
    http: &reqwest::Client,
    url: &str,
    dest: &Path,
    log: &mut impl FnMut(String),
) -> Result<()> {
    let name = dest
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_default();
    let mut req = http.get(url);
    if url.starts_with("https://github.com/") || url.starts_with("https://api.github.com/") {
        if let Ok(token) = std::env::var(GITHUB_TOKEN_ENV) {
            if !token.trim().is_empty() {
                req = req
                    .header("Authorization", format!("Bearer {}", token.trim()))
                    .header("Accept", "application/octet-stream");
            }
        }
    }
    let mut resp = req
        .send()
        .await?
        .error_for_status()
        .map_err(|e| Error::Other(format!("download {url}: {e}")))?;
    let total = resp.content_length();
    if let Some(parent) = dest.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }
    let part = dest.with_extension("part");
    let mut file = tokio::fs::File::create(&part).await?;
    let mut done: u64 = 0;
    let mut next_report: u64 = 32 << 20;
    while let Some(chunk) = resp.chunk().await? {
        file.write_all(&chunk).await?;
        done += chunk.len() as u64;
        if done >= next_report {
            next_report += 32 << 20;
            log(match total {
                Some(t) => format!("{name}: {} / {} MB", done >> 20, t >> 20),
                None => format!("{name}: {} MB", done >> 20),
            });
        }
    }
    file.flush().await?;
    drop(file);
    tokio::fs::rename(&part, dest).await?;
    Ok(())
}

/// Unpack a zip or 7z with the system `tar.exe` (bsdtar with liblzma on Windows 11).
pub async fn extract(archive: &Path, into: &Path) -> Result<()> {
    tokio::fs::create_dir_all(into).await?;
    wsl::run(
        "tar.exe",
        &[
            "-xf",
            &archive.to_string_lossy(),
            "-C",
            &into.to_string_lossy(),
        ],
    )
    .await?
    .require("tar -xf")?;
    Ok(())
}
