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

/// `https://github.com/{owner}/{repo}/releases/download/{tag}/{asset}` -> (owner/repo, tag, asset).
fn parse_release_url(url: &str) -> Option<(&str, &str, &str)> {
    let rest = url.strip_prefix("https://github.com/")?;
    let (repo, rest) = rest.split_once("/releases/download/")?;
    let (tag, asset) = rest.split_once('/')?;
    (!repo.contains('/') || repo.matches('/').count() == 1)
        .then_some((repo, tag, asset))
        .filter(|_| !tag.is_empty() && !asset.is_empty())
}

/// Release assets of a private repository are not served from the web URL even
/// with a token; the API asset endpoint is. Resolve it when a token is present.
async fn github_asset_api_url(http: &reqwest::Client, url: &str, token: &str) -> Option<String> {
    let (repo, tag, asset) = parse_release_url(url)?;
    let release: serde_json::Value = http
        .get(format!(
            "https://api.github.com/repos/{repo}/releases/tags/{tag}"
        ))
        .header("Authorization", format!("Bearer {token}"))
        .header("Accept", "application/vnd.github+json")
        .send()
        .await
        .ok()?
        .error_for_status()
        .ok()?
        .json()
        .await
        .ok()?;
    let id = release
        .get("assets")?
        .as_array()?
        .iter()
        .find(|a| a.get("name").and_then(|n| n.as_str()) == Some(asset))?
        .get("id")?
        .as_u64()?;
    Some(format!(
        "https://api.github.com/repos/{repo}/releases/assets/{id}"
    ))
}

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
    let token = std::env::var(GITHUB_TOKEN_ENV)
        .ok()
        .map(|t| t.trim().to_string())
        .filter(|t| !t.is_empty());
    let mut url = url.to_string();
    let mut req = http.get(&url);
    if let Some(token) = token.as_deref() {
        if url.starts_with("https://github.com/") || url.starts_with("https://api.github.com/") {
            if let Some(api) = github_asset_api_url(http, &url, token).await {
                log(format!("private release asset, using {api}"));
                url = api;
            }
            req = http
                .get(&url)
                .header("Authorization", format!("Bearer {token}"))
                .header("Accept", "application/octet-stream");
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_release_download_urls() {
        assert_eq!(
            parse_release_url(
                "https://github.com/zyx1121/ai-app-store-poc/releases/download/runtimes/whisper-server-vulkan-x64.zip"
            ),
            Some(("zyx1121/ai-app-store-poc", "runtimes", "whisper-server-vulkan-x64.zip"))
        );
        assert_eq!(
            parse_release_url(
                "https://github.com/comfyanonymous/ComfyUI/releases/latest/download/x.7z"
            ),
            None
        );
        assert_eq!(
            parse_release_url("https://huggingface.co/a/b/resolve/main/c.bin"),
            None
        );
    }
}
