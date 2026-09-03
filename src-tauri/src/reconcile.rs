//! Periodic reconcile loop (#68, #69). `launch_space` / `wait_for_http` and
//! their service equivalents only ever flip status to Running once; nothing
//! else in the app ever looked again. If the container OOM-dies, is stopped
//! outside the app, or the WSL2 distro itself goes away, the UI kept showing
//! a clickable dead thing until the user pressed Stop.
//!
//! Runs on a 10 s tick started from `lib.rs`. Costs nothing while the store
//! is idle: with no tracked instance or service alive there is nothing to
//! check, so the tick returns before spawning `wsl.exe` at all. Once
//! something is live, liveness for every tracked container is read from a
//! single `docker ps` call (not one `docker inspect` per instance/service).

use std::collections::HashSet;
use std::time::Duration;

use tauri::{AppHandle, Manager};

use crate::instances;
use crate::services;
use crate::state::AppState;
use crate::wsl;

pub const TICK: Duration = Duration::from_secs(10);

/// Parse `docker ps --format '{{.Names}}'` output into the set of currently
/// running container names.
pub fn parse_running_names(stdout: &str) -> HashSet<String> {
    stdout
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .map(str::to_string)
        .collect()
}

/// Parse `wsl -l -q --running` output and say whether `distro` is in it.
pub fn distro_is_running(stdout: &str, distro: &str) -> bool {
    stdout.lines().any(|l| l.trim() == distro)
}

/// From a snapshot of currently-running container names, which of
/// `expected` (id, container name) pairs believed to be live are actually
/// dead? Pure decision logic, unit-testable without Docker.
pub fn dead<Id: Clone>(running: &HashSet<String>, expected: &[(Id, String)]) -> Vec<Id> {
    expected
        .iter()
        .filter(|(_, container)| !running.contains(container))
        .map(|(id, _)| id.clone())
        .collect()
}

async fn probe_running_containers() -> Option<HashSet<String>> {
    // The `aias.kind` label (no value) matches both Spaces (`=space`) and
    // platform services (`=service`) in one call.
    let o = wsl::sh("docker ps --filter label=aias.kind --format '{{.Names}}'")
        .await
        .ok()?;
    Some(parse_running_names(&o.stdout))
}

/// Is the distro itself still up? Uses `wsl -l -q --running`, which never
/// starts a stopped distro (unlike `-d <distro> --exec ...`, which would mask
/// exactly the crash this is meant to catch). A failed probe is treated as
/// "unknown, assume up": a transient `wsl.exe` hiccup must not flip a
/// perfectly healthy fleet to Stopped.
async fn probe_distro_running() -> bool {
    match wsl::wsl(&["-l", "-q", "--running"]).await {
        Ok(o) => distro_is_running(&o.stdout, wsl::DISTRO),
        Err(_) => true,
    }
}

/// One tick: re-verify the WSL2 keepalive, then the liveness of every tracked
/// instance and service.
pub async fn tick(app: &AppHandle) {
    let state = app.state::<AppState>();

    let live_instances = instances::running_spaces(&state);
    let live_containers = services::running_containers(&state);
    let live_native = services::running_native(&state);

    if live_instances.is_empty() && live_containers.is_empty() && live_native.is_empty() {
        return;
    }

    if !probe_distro_running().await {
        log::warn!("reconcile: WSL distro is not running; marking the fleet stopped");
        instances::mark_all_stopped_by_distro_loss(app);
        services::mark_all_down_by_distro_loss(app);
        state.ensure_keepalive();
        return;
    }
    state.ensure_keepalive();

    if !live_instances.is_empty() || !live_containers.is_empty() {
        if let Some(running) = probe_running_containers().await {
            for id in dead(&running, &live_instances) {
                instances::mark_container_gone(app, &id);
            }
            for id in dead(&running, &live_containers) {
                services::mark_container_gone(app, id);
            }
        }
    }

    for id in live_native {
        services::recheck_native(app, id).await;
    }
}

/// Start the loop; runs for the lifetime of the app.
pub fn spawn(app: AppHandle) {
    tauri::async_runtime::spawn(async move {
        let mut interval = tokio::time::interval(TICK);
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        loop {
            interval.tick().await;
            tick(&app).await;
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_container_names_skipping_blank_lines() {
        let out = "aias-foo\naias-svc-comfyui\n\n";
        let names = parse_running_names(out);
        assert_eq!(names.len(), 2);
        assert!(names.contains("aias-foo"));
        assert!(names.contains("aias-svc-comfyui"));
    }

    #[test]
    fn distro_running_matches_exact_name_only() {
        assert!(distro_is_running("ai-app-store\n", "ai-app-store"));
        assert!(distro_is_running("Ubuntu\nai-app-store\n", "ai-app-store"));
        assert!(!distro_is_running("Ubuntu\n", "ai-app-store"));
        assert!(!distro_is_running("", "ai-app-store"));
    }

    #[test]
    fn dead_finds_only_what_is_missing_from_the_snapshot() {
        let running: HashSet<String> = ["aias-a", "aias-b"].into_iter().map(String::from).collect();
        let expected: Vec<(String, String)> = vec![
            ("space-a".to_string(), "aias-a".to_string()),
            ("space-b".to_string(), "aias-b".to_string()),
            ("space-c".to_string(), "aias-c".to_string()),
        ];
        assert_eq!(dead(&running, &expected), vec!["space-c".to_string()]);
    }

    #[test]
    fn dead_is_empty_when_everything_expected_is_running() {
        let running: HashSet<String> = ["aias-a"].into_iter().map(String::from).collect();
        let expected: Vec<(String, String)> = vec![("space-a".to_string(), "aias-a".to_string())];
        assert!(dead(&running, &expected).is_empty());
    }

    #[test]
    fn dead_flags_everything_when_the_snapshot_is_empty() {
        let running: HashSet<String> = HashSet::new();
        let expected: Vec<(String, String)> = vec![("space-a".to_string(), "aias-a".to_string())];
        assert_eq!(dead(&running, &expected), vec!["space-a".to_string()]);
    }
}
