//! Downloads and archive extraction on the Windows side, for runtimes the
//! store installs natively (Ollama zips, whisper.cpp, ComfyUI portable) and
//! the model files they need.

use std::path::Path;

use sha2::{Digest, Sha256};
use tokio::io::AsyncWriteExt;

use crate::error::{Error, Result};
use crate::wsl;

/// Optional token for GitHub release assets of a private repository
/// (development only; a shipped store downloads from a public host).
const GITHUB_TOKEN_ENV: &str = "AIAS_GITHUB_TOKEN";

/// Name of the checksums file this repository's `runtimes.yml` publishes
/// alongside its own release assets, for the one runtime (whisper-server)
/// that this project builds itself and cannot ship a compile-time digest for.
const SHA256SUMS_NAME: &str = "SHA256SUMS";

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

/// Resolve a GitHub URL to a ready-to-send request, switching to the API asset
/// endpoint (with the dev token) when the plain web URL would 404 on a
/// private repository. Returns the URL actually used, for logging and errors.
async fn github_request(
    http: &reqwest::Client,
    url: &str,
    log: &mut impl FnMut(String),
) -> (String, reqwest::RequestBuilder) {
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
    (url, req)
}

/// Look up `asset` in a `sha256sum`-style listing (`<digest>  <name>`, one per
/// line, an optional leading `*` on the name for binary mode). Pure so it can
/// be unit-tested without a network round trip.
fn find_sha256sums_digest(text: &str, asset: &str) -> Option<String> {
    text.lines().find_map(|line| {
        let mut parts = line.split_whitespace();
        let digest = parts.next()?;
        let name = parts.next()?.trim_start_matches('*');
        (name == asset && digest.len() == 64 && digest.bytes().all(|b| b.is_ascii_hexdigit()))
            .then(|| digest.to_ascii_lowercase())
    })
}

/// Fetch the `SHA256SUMS` file this repository's own `runtimes` release
/// publishes beside its assets (see `.github/workflows/runtimes.yml`) and
/// look up the digest for `asset`. Used only where a compile-time digest
/// cannot be embedded because CI builds the asset fresh rather than
/// vendoring it from a third party. Returns `None` on any failure (no
/// network, an older release with no `SHA256SUMS`, no matching line) so the
/// caller can turn that into "no digest available" rather than a panic.
pub async fn release_checksum(
    http: &reqwest::Client,
    url: &str,
    log: &mut impl FnMut(String),
) -> Option<String> {
    let (repo, tag, asset) = parse_release_url(url)?;
    let sums_url = format!("https://github.com/{repo}/releases/download/{tag}/{SHA256SUMS_NAME}");
    let (_, req) = github_request(http, &sums_url, log).await;
    let text = req
        .send()
        .await
        .ok()?
        .error_for_status()
        .ok()?
        .text()
        .await
        .ok()?;
    find_sha256sums_digest(&text, asset)
}

/// Stream a download to `dest` (via `dest.part`), reporting every 32 MB. When
/// `sha256` is `Some`, the streamed bytes are hashed as they arrive and
/// compared once the transfer finishes; on a mismatch the partial file is
/// deleted and `dest` is left untouched, so a corrupted or tampered download
/// never becomes the file another part of the store trusts and executes.
pub async fn download(
    http: &reqwest::Client,
    url: &str,
    dest: &Path,
    sha256: Option<&str>,
    log: &mut impl FnMut(String),
) -> Result<()> {
    let name = dest
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_default();
    let (url, req) = github_request(http, url, log).await;
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
    let mut hasher = Sha256::new();
    let mut done: u64 = 0;
    let mut next_report: u64 = 32 << 20;
    while let Some(chunk) = resp.chunk().await? {
        file.write_all(&chunk).await?;
        hasher.update(&chunk);
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
    if let Some(expected) = sha256 {
        let got = format!("{:x}", hasher.finalize());
        if !got.eq_ignore_ascii_case(expected) {
            let _ = tokio::fs::remove_file(&part).await;
            return Err(Error::Other(format!(
                "{name}: SHA-256 mismatch (expected {expected}, got {got}); refusing to install it"
            )));
        }
    }
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

    #[test]
    fn finds_the_matching_line_in_a_sha256sums_listing() {
        let sums = "\
2439cbea65310b1aadf7d8fc41d7faf5d033f920d42e00a476c58bf9bff6950e  ollama-windows-amd64.zip
c132d2d4dd3ab58ae39e2251b2335c234f8e861209a65979c12c155c4bab8c40 *whisper-server-vulkan-x64.zip
";
        assert_eq!(
            find_sha256sums_digest(sums, "whisper-server-vulkan-x64.zip").as_deref(),
            Some("c132d2d4dd3ab58ae39e2251b2335c234f8e861209a65979c12c155c4bab8c40")
        );
        assert_eq!(
            find_sha256sums_digest(sums, "ollama-windows-amd64.zip").as_deref(),
            Some("2439cbea65310b1aadf7d8fc41d7faf5d033f920d42e00a476c58bf9bff6950e")
        );
        assert_eq!(find_sha256sums_digest(sums, "missing.zip"), None);
        assert_eq!(find_sha256sums_digest("", "anything"), None);
        // A line with a short or non-hex first field (corrupt file, header
        // row) never matches, whatever the second field says.
        assert_eq!(
            find_sha256sums_digest(
                "not-a-digest  ollama-windows-amd64.zip",
                "ollama-windows-amd64.zip"
            ),
            None
        );
    }
}
