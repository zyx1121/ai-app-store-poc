//! The library: what the user added from the Store. Store's Add records an
//! entry here (no download, no GPU lease, no secrets); Running lists the
//! entries, runs, stops and removes them. Entries outlive both the app and the
//! instances they spawn, so they are persisted as `library.json` in the app
//! data dir the same way `token.rs` keeps the HF token, written through a
//! temporary file so a crash mid-write cannot truncate the catalog.
//!
//! An entry's `id` is the instance id the launch paths derive (`space-<slug>`
//! / `model-<slug>`), so a live instance and its entry join by id without a
//! second lookup.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Emitter, Manager};

use crate::instances::{Instance, Kind, Status};
use crate::state::AppState;

const FILE_NAME: &str = "library.json";
pub const UPDATE_EVENT: &str = "library://update";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Entry {
    /// same id scheme as an instance: `space-<slug>` / `model-<slug>`
    pub id: String,
    pub kind: Kind,
    /// HF repo id (`owner/name`), or a bare Ollama library name for models
    pub repo: String,
    /// models only: the quant (`Q4_K_M`) or Ollama library tag (`7b`)
    #[serde(default)]
    pub quant: Option<String>,
    pub display_name: String,
    /// unix seconds, as a string (same shape as `Instance::started_at`)
    pub added_at: String,
}

/// One library entry joined with the live instance of the same id. `status` is
/// `Stopped` when nothing is running: an added item that was never run and a
/// stopped one look the same to the user, both offer Run.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct LibraryItem {
    pub id: String,
    pub kind: Kind,
    pub repo: String,
    pub quant: Option<String>,
    pub display_name: String,
    pub added_at: String,
    pub status: Status,
    pub model_tag: Option<String>,
    pub port: Option<u16>,
    pub url: Option<String>,
    pub error: Option<String>,
    pub log_tail: Vec<String>,
    pub local_build: bool,
    pub gpu: bool,
    pub pull_size_mb: Option<u64>,
    pub progress_pct: Option<u32>,
}

impl LibraryItem {
    /// Anything but `Stopped`: the item holds a container, an Ollama load, or
    /// an in-flight launch, so Running shows it above the rest.
    pub fn live(&self) -> bool {
        self.status != Status::Stopped
    }
}

fn now() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    format!("{secs}")
}

impl Entry {
    pub fn new(
        id: String,
        kind: Kind,
        repo: String,
        quant: Option<String>,
        display_name: String,
    ) -> Self {
        Self {
            id,
            kind,
            repo,
            quant,
            display_name,
            added_at: now(),
        }
    }
}

fn path(app: &AppHandle) -> Option<PathBuf> {
    app.path().app_data_dir().ok().map(|d| d.join(FILE_NAME))
}

/// Read the catalog at startup; a missing or unreadable file just means an
/// empty library (never fails startup).
pub fn load(app: &AppHandle) -> Vec<Entry> {
    let Some(path) = path(app) else {
        return Vec::new();
    };
    let Ok(data) = std::fs::read_to_string(&path) else {
        return Vec::new();
    };
    match serde_json::from_str::<Vec<Entry>>(&data) {
        Ok(entries) => entries,
        Err(e) => {
            log::warn!(
                "library: {} is not readable ({e}); starting empty",
                path.display()
            );
            Vec::new()
        }
    }
}

/// Persist the catalog. Written to `library.json.tmp` and renamed, so an
/// interrupted write leaves the previous catalog intact.
fn save(app: &AppHandle, entries: &[Entry]) -> std::io::Result<()> {
    let path = path(app).ok_or_else(|| std::io::Error::other("no app data directory"))?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let json = serde_json::to_string_pretty(entries).map_err(std::io::Error::other)?;
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, json)?;
    std::fs::rename(&tmp, &path)
}

/// Merge entries with whatever is live, live items first. Pure: the join is
/// the part worth testing, and it needs no app handle.
pub fn merge(entries: &[Entry], live: &[Instance]) -> Vec<LibraryItem> {
    let mut items: Vec<LibraryItem> = entries
        .iter()
        .map(|e| {
            let inst = live.iter().find(|i| i.id == e.id);
            LibraryItem {
                id: e.id.clone(),
                kind: e.kind,
                repo: e.repo.clone(),
                quant: e.quant.clone(),
                // The launch learns the real title from the Hub; prefer it over
                // whatever Add derived from the repo id.
                display_name: inst
                    .map(|i| i.display_name.clone())
                    .unwrap_or_else(|| e.display_name.clone()),
                added_at: e.added_at.clone(),
                status: inst.map(|i| i.status).unwrap_or(Status::Stopped),
                model_tag: inst.and_then(|i| i.model_tag.clone()),
                port: inst.and_then(|i| i.port),
                url: inst.and_then(|i| i.url.clone()),
                error: inst.and_then(|i| i.error.clone()),
                log_tail: inst.map(|i| i.log_tail.clone()).unwrap_or_default(),
                local_build: inst.map(|i| i.local_build).unwrap_or(false),
                gpu: inst.map(|i| i.gpu).unwrap_or(false),
                pull_size_mb: inst.and_then(|i| i.pull_size_mb),
                progress_pct: inst.and_then(|i| i.progress_pct),
            }
        })
        .collect();
    // Live first, then newest addition first; the id breaks ties so the order
    // does not wobble between calls.
    items.sort_by(|a, b| {
        let age = |s: &str| s.parse::<u64>().unwrap_or(0);
        b.live()
            .cmp(&a.live())
            .then(age(&b.added_at).cmp(&age(&a.added_at)))
            .then(a.id.cmp(&b.id))
    });
    items
}

pub fn entries(state: &AppState) -> Vec<Entry> {
    state.library.lock().map(|l| l.clone()).unwrap_or_default()
}

pub fn get(state: &AppState, id: &str) -> Option<Entry> {
    state
        .library
        .lock()
        .ok()
        .and_then(|l| l.iter().find(|e| e.id == id).cloned())
}

/// The whole library joined with the live instances.
pub fn list(app: &AppHandle) -> Vec<LibraryItem> {
    let state = app.state::<AppState>();
    merge(&entries(&state), &crate::instances::list(&state))
}

fn publish(app: &AppHandle) {
    let _ = app.emit(UPDATE_EVENT, list(app));
}

/// Record `entry` unless its id is already in the library, then persist and
/// broadcast. Returns the entry that is now in the library, existing or new,
/// so a second Add of the same item is a no-op instead of an error.
pub fn add(app: &AppHandle, entry: Entry) -> Entry {
    let state = app.state::<AppState>();
    let (stored, changed) = {
        let mut lib = match state.library.lock() {
            Ok(l) => l,
            Err(_) => return entry,
        };
        match lib.iter().find(|e| e.id == entry.id) {
            Some(existing) => (existing.clone(), false),
            None => {
                lib.push(entry.clone());
                (entry, true)
            }
        }
    };
    if changed {
        if let Err(e) = save(app, &entries(&state)) {
            log::warn!("library: could not save ({e})");
        }
        publish(app);
    }
    stored
}

/// Drop the entry, persist and broadcast. Returns what was dropped, `None`
/// when the id was not in the library.
pub fn remove(app: &AppHandle, id: &str) -> Option<Entry> {
    let state = app.state::<AppState>();
    let dropped = {
        let mut lib = state.library.lock().ok()?;
        let idx = lib.iter().position(|e| e.id == id)?;
        lib.remove(idx)
    };
    if let Err(e) = save(app, &entries(&state)) {
        log::warn!("library: could not save ({e})");
    }
    publish(app);
    Some(dropped)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(id: &str, added_at: &str) -> Entry {
        Entry {
            id: id.into(),
            kind: Kind::Space,
            repo: "owner/name".into(),
            quant: None,
            display_name: "Name".into(),
            added_at: added_at.into(),
        }
    }

    fn instance(id: &str, status: Status) -> Instance {
        Instance {
            id: id.into(),
            kind: Kind::Space,
            repo: "owner/name".into(),
            model_tag: None,
            display_name: "Hub Title".into(),
            status,
            port: Some(7860),
            url: Some("http://localhost:7860".into()),
            error: None,
            log_tail: vec!["pulling".into()],
            started_at: "100".into(),
            local_build: false,
            gpu: true,
            pull_size_mb: None,
            progress_pct: None,
        }
    }

    #[test]
    fn entry_survives_a_serde_round_trip() {
        let entries = vec![
            entry("space-owner-name", "10"),
            Entry {
                id: "model-owner-name-q4-k-m".into(),
                kind: Kind::Model,
                repo: "owner/name".into(),
                quant: Some("Q4_K_M".into()),
                display_name: "name (Q4_K_M)".into(),
                added_at: "20".into(),
            },
        ];
        let json = serde_json::to_string(&entries).unwrap();
        assert_eq!(
            serde_json::from_str::<Vec<Entry>>(&json).unwrap(),
            entries,
            "round trip must not lose a field"
        );
        // The persisted shape is the contract with an installed app's file.
        assert!(json.contains(r#""kind":"model""#));
        assert!(json.contains(r#""quant":"Q4_K_M""#));
    }

    /// A catalog written before `quant` existed still loads (`serde(default)`).
    #[test]
    fn entry_loads_without_quant() {
        let json =
            r#"[{"id":"space-a-b","kind":"space","repo":"a/b","display_name":"B","added_at":"1"}]"#;
        let entries: Vec<Entry> = serde_json::from_str(json).unwrap();
        assert_eq!(entries[0].quant, None);
    }

    #[test]
    fn merge_takes_status_from_the_live_instance() {
        let entries = vec![entry("space-a", "10")];
        let items = merge(&entries, &[instance("space-a", Status::Running)]);
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].status, Status::Running);
        assert_eq!(items[0].port, Some(7860));
        assert!(items[0].gpu);
        // The Hub's title wins over the name Add derived from the repo id.
        assert_eq!(items[0].display_name, "Hub Title");
        assert!(items[0].live());
    }

    #[test]
    fn merge_reports_an_entry_with_no_instance_as_stopped() {
        let items = merge(&[entry("space-a", "10")], &[]);
        assert_eq!(items[0].status, Status::Stopped);
        assert_eq!(items[0].url, None);
        assert_eq!(items[0].port, None);
        assert!(items[0].log_tail.is_empty());
        assert!(!items[0].live());
        assert_eq!(items[0].display_name, "Name");
    }

    /// Live items first, then the newest addition; an instance the library
    /// does not know about is not a library item and is left out.
    #[test]
    fn merge_orders_live_first_then_newest() {
        let entries = vec![
            entry("space-old-stopped", "10"),
            entry("space-new-stopped", "30"),
            entry("space-live", "20"),
        ];
        let live = vec![
            instance("space-live", Status::Pulling),
            instance("space-not-in-library", Status::Running),
        ];
        let items = merge(&entries, &live);
        let ids: Vec<&str> = items.iter().map(|i| i.id.as_str()).collect();
        assert_eq!(
            ids,
            ["space-live", "space-new-stopped", "space-old-stopped"]
        );
    }
}
