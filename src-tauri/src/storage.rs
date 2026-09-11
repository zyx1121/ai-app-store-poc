//! Disk usage and cleanup for the WSL2 distro: Docker images, the build
//! cache, the shared HF cache volume, our local clone directory, and the WSL
//! virtual disk on Windows (#63). Everything here is best-effort: a piece
//! that cannot be measured or cleaned reports zero rather than failing the
//! whole call.

use std::collections::HashSet;

use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Emitter};

use crate::build;
use crate::error::Result;
use crate::instances::{Kind, Status};
use crate::services::ServiceState;
use crate::state::AppState;
use crate::wsl::{self, DISTRO};

const PROGRESS_EVENT: &str = "storage://progress";
const HF_CACHE_VOLUME: &str = "aias-hf-cache";

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct StorageUsage {
    pub images_total_mb: u64,
    pub images_reclaimable_mb: u64,
    pub build_cache_mb: u64,
    pub build_cache_reclaimable_mb: u64,
    pub hf_cache_mb: u64,
    pub build_dir_mb: u64,
    /// size of the WSL virtual disk on Windows; `None` when it cannot be located
    pub vhdx_mb: Option<u64>,
}

#[derive(Debug, Clone, Deserialize)]
struct DfRow {
    #[serde(rename = "Type")]
    kind: String,
    #[serde(rename = "Size", default)]
    size: String,
    #[serde(rename = "Reclaimable", default)]
    reclaimable: String,
}

/// Parse `docker system df --format '{{json .}}'` output: one JSON object per
/// line (not a JSON array), one line per resource type. Only the two types
/// the store can act on are kept.
pub fn parse_system_df(stdout: &str) -> StorageUsage {
    let mut usage = StorageUsage::default();
    for line in stdout.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let Ok(row) = serde_json::from_str::<DfRow>(line) else {
            continue;
        };
        let size_mb = parse_size_mb(&row.size);
        let reclaim_mb = parse_size_mb(&row.reclaimable);
        match row.kind.as_str() {
            "Images" => {
                usage.images_total_mb = size_mb;
                usage.images_reclaimable_mb = reclaim_mb;
            }
            "Build Cache" => {
                usage.build_cache_mb = size_mb;
                usage.build_cache_reclaimable_mb = reclaim_mb;
            }
            _ => {}
        }
    }
    usage
}

/// `"79.1GB (29%)"` / `"267GB"` / `"341MB"` / `"0B"` -> MB. Docker labels these
/// binary units with SI-looking suffixes (`GB` means GiB); the percentage in
/// parentheses is dropped.
fn parse_size_mb(s: &str) -> u64 {
    let s = s.split('(').next().unwrap_or(s).trim();
    if s.is_empty() {
        return 0;
    }
    let split_at = s.find(|c: char| c.is_ascii_alphabetic()).unwrap_or(s.len());
    let (num, unit) = s.split_at(split_at);
    let n: f64 = num.trim().parse().unwrap_or(0.0);
    let mb = match unit.trim().to_ascii_uppercase().as_str() {
        "TB" => n * 1024.0 * 1024.0,
        "GB" => n * 1024.0,
        "MB" => n,
        "KB" => n / 1024.0,
        "B" | "" => n / (1024.0 * 1024.0),
        _ => n,
    };
    mb.round().max(0.0) as u64
}

/// The image tag a Space instance runs under, mirroring `run_space` (pulled)
/// and `build.rs` (local build).
pub fn space_image(repo: &str, local_build: bool) -> String {
    let slug = wsl::slug(repo);
    if local_build {
        build::local_image_name(&slug)
    } else {
        format!("registry.hf.space/{slug}:latest")
    }
}

/// Both images a library entry may run under. The entry records the repo, not
/// how it was started, so a repo in the library keeps its Hub image and its
/// local build; Remove is what makes them collectable.
fn entry_images(repo: &str) -> [String; 2] {
    [space_image(repo, false), space_image(repo, true)]
}

/// Images something still needs: a Space instance that is pulling, building,
/// starting or running, or any Space in the library (added items keep their
/// image so Run does not have to pull it again).
///
/// `None` when the state cannot be read: a poisoned lock must never lead to a
/// destructive `docker rmi` (#63).
fn referenced_images(state: &AppState) -> Option<HashSet<String>> {
    let mut keep: HashSet<String> = state
        .instances
        .lock()
        .ok()?
        .values()
        .filter(|i| i.kind == Kind::Space)
        .filter(|i| {
            matches!(
                i.status,
                Status::Pulling | Status::Building | Status::Starting | Status::Running
            )
        })
        .map(|i| space_image(&i.repo, i.local_build))
        .collect();
    for entry in state.library.lock().ok()?.iter() {
        if entry.kind == Kind::Space {
            keep.extend(entry_images(&entry.repo));
        }
    }
    Some(keep)
}

/// Images present on disk that are not in `keep`; pure so it is unit-testable
/// without a Docker daemon.
fn images_to_remove(present: &[String], keep: &HashSet<String>) -> Vec<String> {
    present
        .iter()
        .filter(|img| !keep.contains(img.as_str()))
        .cloned()
        .collect()
}

async fn list_images(pattern: &str) -> Vec<String> {
    let out = wsl::sh(&format!(
        "docker images --format '{{{{.Repository}}}}:{{{{.Tag}}}}' --filter 'reference={pattern}' 2>/dev/null"
    ))
    .await;
    out.map(|o| {
        o.stdout
            .lines()
            .map(str::trim)
            .filter(|l| !l.is_empty())
            .map(str::to_string)
            .collect()
    })
    .unwrap_or_default()
}

/// After a Space is unsubscribed, drop its images unless another library
/// entry or a live instance still references them. Call it once the entry is
/// out of the library, or it keeps its own image alive. Ollama models are not
/// images; `instances::remove` deletes those through `ollama rm`.
pub async fn maybe_remove_image(state: &AppState, removed_repo: &str) {
    let Some(keep) = referenced_images(state) else {
        return;
    };
    for image in images_to_remove(&entry_images(removed_repo), &keep) {
        let _ = wsl::sh(&format!("docker rmi {image} >/dev/null 2>&1")).await;
    }
}

async fn volume_size_mb(name: &str) -> u64 {
    let script = format!(
        "m=$(docker volume inspect -f '{{{{.Mountpoint}}}}' {name} 2>/dev/null); [ -n \"$m\" ] && du -sm \"$m\" 2>/dev/null | cut -f1"
    );
    wsl::sh(&script)
        .await
        .ok()
        .and_then(|o| {
            o.stdout
                .split_whitespace()
                .next()
                .and_then(|n| n.parse().ok())
        })
        .unwrap_or(0)
}

async fn dir_size_mb(path: &str) -> u64 {
    let script = format!("du -sm {} 2>/dev/null | cut -f1", wsl::quote(path));
    wsl::sh(&script)
        .await
        .ok()
        .and_then(|o| {
            o.stdout
                .split_whitespace()
                .next()
                .and_then(|n| n.parse().ok())
        })
        .unwrap_or(0)
}

/// Locate `ai-app-store`'s `ext4.vhdx` through the registry (the WSL CLI has
/// no "give me the disk path" command) and read its size.
async fn vhdx_size_mb() -> Option<u64> {
    let script = format!(
        "$ErrorActionPreference = 'SilentlyContinue'; \
         $key = Get-ChildItem 'HKCU:\\Software\\Microsoft\\Windows\\CurrentVersion\\Lxss' | \
           Where-Object {{ (Get-ItemProperty $_.PSPath).DistributionName -eq '{DISTRO}' }} | \
           Select-Object -First 1; \
         if ($key) {{ \
           $base = (Get-ItemProperty $key.PSPath).BasePath; \
           $vhdx = Join-Path $base 'ext4.vhdx'; \
           if (Test-Path $vhdx) {{ (Get-Item $vhdx).Length }} \
         }}"
    );
    let out = wsl::run(
        "powershell.exe",
        &["-NoProfile", "-NonInteractive", "-Command", &script],
    )
    .await
    .ok()?;
    let bytes: u64 = out.stdout.trim().parse().ok()?;
    Some(bytes / (1024 * 1024))
}

pub async fn usage() -> StorageUsage {
    let mut usage = wsl::sh("docker system df --format '{{json .}}' 2>/dev/null")
        .await
        .ok()
        .filter(|o| o.ok())
        .map(|o| parse_system_df(&o.stdout))
        .unwrap_or_default();

    usage.hf_cache_mb = volume_size_mb(HF_CACHE_VOLUME).await;
    usage.build_dir_mb = dir_size_mb(build::BUILD_ROOT).await;
    usage.vhdx_mb = vhdx_size_mb().await;
    usage
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct CleanupOptions {
    #[serde(default)]
    pub dangling_images: bool,
    #[serde(default)]
    pub build_cache: bool,
    #[serde(default)]
    pub unused_space_images: bool,
    /// remove HF cache entries whose top-level directory is older than this
    /// many days; `None` skips the step
    #[serde(default)]
    pub hf_cache_older_than_days: Option<u32>,
    #[serde(default)]
    pub compact_vhd: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct CleanupResult {
    pub freed_mb: u64,
    pub usage: StorageUsage,
}

#[derive(Debug, Clone, Serialize)]
struct CleanupEvent {
    step: String,
    status: &'static str,
    message: String,
}

fn emit(app: &AppHandle, step: &str, status: &'static str, message: impl Into<String>) {
    let ev = CleanupEvent {
        step: step.to_string(),
        status,
        message: message.into(),
    };
    log::info!("[storage/{}] {} {}", ev.step, ev.status, ev.message);
    let _ = app.emit(PROGRESS_EVENT, ev);
}

/// True while any instance or platform service is pulling, building,
/// starting or running: `compact_vhd` needs the distro to stand still.
fn anything_running(state: &AppState) -> bool {
    let instances = state
        .instances
        .lock()
        .map(|m| {
            m.values().any(|i| {
                matches!(
                    i.status,
                    Status::Pulling | Status::Building | Status::Starting | Status::Running
                )
            })
        })
        .unwrap_or(true);
    if instances {
        return true;
    }
    state
        .services
        .lock()
        .map(|m| {
            m.values().any(|s| {
                matches!(
                    s.state,
                    ServiceState::Pulling | ServiceState::Starting | ServiceState::Running
                )
            })
        })
        .unwrap_or(true)
}

async fn remove_unused_space_images(state: &AppState) -> u32 {
    let Some(keep) = referenced_images(state) else {
        return 0;
    };
    let mut present = list_images("registry.hf.space/*").await;
    present.extend(list_images("aias-local/*").await);
    let mut removed = 0u32;
    for image in images_to_remove(&present, &keep) {
        if wsl::sh(&format!("docker rmi {image} >/dev/null 2>&1"))
            .await
            .map(|o| o.ok())
            .unwrap_or(false)
        {
            removed += 1;
        }
    }
    removed
}

/// Delete top-level HF cache entries (`models--*`, `datasets--*`) older than
/// `days`, returning the space freed in MB. Only the top level is ever
/// touched: subdirectories are the Space's own working files.
async fn trim_hf_cache(days: u32) -> u64 {
    let script = format!(
        "m=$(docker volume inspect -f '{{{{.Mountpoint}}}}' {HF_CACHE_VOLUME} 2>/dev/null); \
         [ -z \"$m\" ] && exit 0; \
         find \"$m/hub\" -maxdepth 1 -mindepth 1 -mtime +{days} 2>/dev/null | while IFS= read -r d; do \
           du -sm \"$d\" 2>/dev/null | cut -f1; rm -rf \"$d\"; \
         done"
    );
    wsl::sh(&script)
        .await
        .map(|o| {
            o.stdout
                .lines()
                .filter_map(|l| l.trim().parse::<u64>().ok())
                .sum()
        })
        .unwrap_or(0)
}

/// `wsl --manage <distro> --set-sparse true` shrinks the vhdx to its content
/// size (WSL 2.x). It needs the distro terminated first, so this stops
/// Docker, Ollama and everything running inside it; callers must check
/// `anything_running` first and tell the user.
async fn compact_vhd(state: &AppState) -> Result<()> {
    if let Ok(mut g) = state.keepalive.lock() {
        if let Some(mut child) = g.take() {
            let _ = child.start_kill();
        }
    }
    wsl::wsl(&["--terminate", DISTRO])
        .await?
        .require("wsl --terminate")?;
    let out = wsl::wsl(&["--manage", DISTRO, "--set-sparse", "true"]).await?;
    let result = out.require("wsl --manage --set-sparse");
    // Bring the distro back up regardless so the app keeps working.
    state.ensure_keepalive();
    result?;
    Ok(())
}

pub async fn cleanup(
    app: &AppHandle,
    state: &AppState,
    opts: CleanupOptions,
) -> Result<CleanupResult> {
    let before = usage().await;

    if opts.dangling_images {
        emit(app, "dangling_images", "start", "Removing dangling images");
        let out = wsl::sh("docker image prune -f 2>&1").await?;
        emit(app, "dangling_images", "ok", out.stdout.trim().to_string());
    }

    if opts.unused_space_images {
        emit(
            app,
            "unused_space_images",
            "start",
            "Removing Space images not in use",
        );
        let removed = remove_unused_space_images(state).await;
        emit(
            app,
            "unused_space_images",
            "ok",
            format!("removed {removed} image(s)"),
        );
    }

    if opts.build_cache {
        emit(
            app,
            "build_cache",
            "start",
            "Pruning the Docker build cache",
        );
        let out = wsl::sh("docker builder prune -f 2>&1").await?;
        emit(app, "build_cache", "ok", out.stdout.trim().to_string());
    }

    if let Some(days) = opts.hf_cache_older_than_days {
        emit(
            app,
            "hf_cache",
            "start",
            format!("Trimming the HF cache (older than {days} days)"),
        );
        let freed = trim_hf_cache(days).await;
        emit(app, "hf_cache", "ok", format!("freed {freed} MB"));
    }

    if opts.compact_vhd {
        if anything_running(state) {
            emit(
                app,
                "compact_vhd",
                "error",
                "skipped: an instance or service is still running",
            );
        } else {
            emit(
                app,
                "compact_vhd",
                "start",
                "Stopping the distro to compact its virtual disk",
            );
            match compact_vhd(state).await {
                Ok(()) => emit(app, "compact_vhd", "ok", "compacted"),
                Err(e) => emit(app, "compact_vhd", "error", e.to_string()),
            }
        }
    }

    let after = usage().await;
    let mut freed_mb = before
        .images_total_mb
        .saturating_add(before.build_cache_mb)
        .saturating_add(before.hf_cache_mb)
        .saturating_sub(after.images_total_mb)
        .saturating_sub(after.build_cache_mb)
        .saturating_sub(after.hf_cache_mb);
    if let (Some(b), Some(a)) = (before.vhdx_mb, after.vhdx_mb) {
        freed_mb = freed_mb.saturating_add(b.saturating_sub(a));
    }
    emit(
        app,
        "done",
        "ok",
        format!("freed {:.1} GB", freed_mb as f64 / 1024.0),
    );

    Ok(CleanupResult {
        freed_mb,
        usage: after,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Numbers from the #63 report: images 267 GB (79 GB reclaimable), build
    /// cache 16 GB, one line per resource type, `Containers` and `Local
    /// Volumes` present but ignored.
    const SAMPLE: &str = r#"{"Type":"Images","TotalCount":42,"Active":3,"Size":"267GB","Reclaimable":"79.1GB (29%)"}
{"Type":"Containers","TotalCount":3,"Active":3,"Size":"120MB","Reclaimable":"0B (0%)"}
{"Type":"Local Volumes","TotalCount":2,"Active":2,"Size":"16GB","Reclaimable":"0B (0%)"}
{"Type":"Build Cache","TotalCount":54,"Active":0,"Size":"16GB","Reclaimable":"16GB (100%)"}
"#;

    #[test]
    fn parses_docker_system_df_json_lines() {
        let usage = parse_system_df(SAMPLE);
        assert_eq!(usage.images_total_mb, 267 * 1024);
        assert_eq!(
            usage.images_reclaimable_mb,
            (79.1f64 * 1024.0).round() as u64
        );
        assert_eq!(usage.build_cache_mb, 16 * 1024);
        assert_eq!(usage.build_cache_reclaimable_mb, 16 * 1024);
    }

    #[test]
    fn parses_blank_and_zero_sizes() {
        let usage = parse_system_df(
            r#"{"Type":"Images","Size":"0B","Reclaimable":"0B"}
{"Type":"Build Cache","Size":"341MB","Reclaimable":""}"#,
        );
        assert_eq!(usage.images_total_mb, 0);
        assert_eq!(usage.build_cache_mb, 341);
        assert_eq!(usage.build_cache_reclaimable_mb, 0);
    }

    #[test]
    fn ignores_unparsable_or_irrelevant_lines() {
        let usage = parse_system_df("not json\n\n{\"Type\":\"Containers\",\"Size\":\"1GB\"}");
        assert_eq!(usage, StorageUsage::default());
    }

    #[test]
    fn space_image_matches_run_space_and_build_naming() {
        assert_eq!(
            space_image("stabilityai/stable-diffusion", false),
            "registry.hf.space/stabilityai-stable-diffusion:latest"
        );
        assert_eq!(
            space_image("stabilityai/stable-diffusion", true),
            "aias-local/stabilityai-stable-diffusion:latest"
        );
    }

    #[test]
    fn unused_image_selection_keeps_only_referenced_images() {
        let present = vec![
            "registry.hf.space/a-b:latest".to_string(),
            "registry.hf.space/c-d:latest".to_string(),
            "aias-local/e-f:latest".to_string(),
        ];
        let mut keep = HashSet::new();
        keep.insert("registry.hf.space/a-b:latest".to_string());
        let removable = images_to_remove(&present, &keep);
        assert_eq!(
            removable,
            vec![
                "registry.hf.space/c-d:latest".to_string(),
                "aias-local/e-f:latest".to_string(),
            ]
        );
    }

    /// A library entry keeps both names, since it does not record whether the
    /// item was pulled from the Hub or built here.
    #[test]
    fn library_entry_reserves_both_image_names() {
        assert_eq!(
            entry_images("owner/name"),
            [
                "registry.hf.space/owner-name:latest".to_string(),
                "aias-local/owner-name:latest".to_string(),
            ]
        );
        let keep: HashSet<String> = entry_images("owner/name").into_iter().collect();
        assert!(images_to_remove(&entry_images("owner/name"), &keep).is_empty());
        assert_eq!(
            images_to_remove(&entry_images("other/repo"), &keep).len(),
            2
        );
    }

    #[test]
    fn unused_image_selection_removes_nothing_when_all_referenced() {
        let present = vec!["registry.hf.space/a-b:latest".to_string()];
        let mut keep = HashSet::new();
        keep.insert("registry.hf.space/a-b:latest".to_string());
        assert!(images_to_remove(&present, &keep).is_empty());
    }
}
