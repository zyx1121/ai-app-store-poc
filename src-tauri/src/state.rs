use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use tokio::process::Child;

use crate::hardware::Vendor;
use crate::hf::{RawSpace, SpaceSummary};
use crate::instances::Instance;
use crate::library::Entry;
use crate::runtime::RuntimeStatus;
use crate::services::{ServiceId, ServiceStatus};

/// Process-wide state shared by all commands. Everything here is cheap to lock;
/// long work happens in spawned tasks that re-lock briefly to publish results.
pub struct AppState {
    pub http: reqwest::Client,
    pub runtime: Mutex<Option<RuntimeStatus>>,
    pub instances: Mutex<BTreeMap<String, Instance>>,
    /// What the user added from the Store, loaded from `library.json` once at
    /// startup and persisted on every change. Live instances join to it by id.
    pub library: Mutex<Vec<Entry>>,
    pub keepalive: Mutex<Option<Child>>,
    /// `compare_exchange`d, not just locked: `discover()` claims it before the
    /// probe and rolls back on failure, so a transient WSL hiccup right after
    /// launch can be retried instead of leaving the fleet unadopted forever (#38).
    pub discovered: AtomicBool,
    pub services: Mutex<HashMap<ServiceId, ServiceStatus>>,
    /// `ollama serve` we started on Windows (non-NVIDIA machines only).
    pub native_ollama: Mutex<Option<Child>>,
    /// Platform services running as native Windows processes (whisper.cpp, portable ComfyUI).
    pub native_services: Mutex<HashMap<ServiceId, Child>>,
    /// Per-container `--memory` cap for Space containers (75% of the WSL2 VM's
    /// own cap), computed once from `total_ram_mb` at startup (#64).
    pub container_memory_cap_mb: Mutex<Option<u64>>,
    /// Cancellation flag for the in-flight launch of each instance id, if any.
    /// `stop`/`remove` flip it so the background launch task can bail between
    /// its externally visible steps instead of finishing and reviving the
    /// instance the user just stopped (#35).
    pub instance_launches: Mutex<HashMap<String, Arc<AtomicBool>>>,
    /// Same as `instance_launches`, keyed by service id (#35).
    pub service_launches: Mutex<HashMap<ServiceId, Arc<AtomicBool>>>,
    /// Store Space search cache, keyed by repo id, for the life of the session (#67).
    pub space_cache: Mutex<HashMap<String, (SpaceSummary, Instant)>>,
    /// Whole semantic search answers keyed by `query\0category`; the Store
    /// pages through them locally since the Hub does not.
    pub semantic_cache: Mutex<HashMap<String, (Vec<RawSpace>, Instant)>>,
    /// Bumped on every Space search; a spawned detail fetch aborts once this
    /// no longer matches the generation it started with (#67).
    pub search_generation: AtomicU64,
    /// Optional Hugging Face access token; loaded from disk at startup, kept
    /// here only in memory otherwise (#60).
    pub hf_token: Mutex<Option<String>>,
    /// Set after a successful ComfyUI `/free`; cleared once torch reports it
    /// resident again. While set, `gpu.rs` counts only the CUDA context instead
    /// of the launch-estimate floor (#54).
    pub comfyui_freed: Mutex<bool>,
    /// Last time the app itself used a light platform service (Speaches, CV);
    /// drives the idle auto-stop timer (#65). Updated by `services::touch`.
    pub service_last_used: Mutex<HashMap<ServiceId, Instant>>,
    /// GPU plan leases from `gpu_plan`, keyed by lease id, valued by expiry.
    /// Held from the plan through the matching launch so two launches cannot
    /// double-book VRAM (#70).
    pub gpu_leases: Mutex<HashMap<String, Instant>>,
    /// Image sizes from `docker manifest inspect`, in MB, keyed by image
    /// reference. Fetched once per image, right before that Space's pull;
    /// skips the round trip on a re-launch (#55).
    pub image_sizes_mb: Mutex<HashMap<String, u64>>,
}

impl Default for AppState {
    fn default() -> Self {
        let http = reqwest::Client::builder()
            .user_agent(concat!("ai-app-store/", env!("CARGO_PKG_VERSION")))
            .no_proxy()
            .build()
            .expect("reqwest client");
        Self {
            http,
            runtime: Mutex::new(None),
            instances: Mutex::new(BTreeMap::new()),
            library: Mutex::new(Vec::new()),
            keepalive: Mutex::new(None),
            discovered: AtomicBool::new(false),
            services: Mutex::new(HashMap::new()),
            native_ollama: Mutex::new(None),
            native_services: Mutex::new(HashMap::new()),
            container_memory_cap_mb: Mutex::new(None),
            instance_launches: Mutex::new(HashMap::new()),
            service_launches: Mutex::new(HashMap::new()),
            space_cache: Mutex::new(HashMap::new()),
            semantic_cache: Mutex::new(HashMap::new()),
            search_generation: AtomicU64::new(0),
            hf_token: Mutex::new(None),
            comfyui_freed: Mutex::new(false),
            service_last_used: Mutex::new(HashMap::new()),
            gpu_leases: Mutex::new(HashMap::new()),
            image_sizes_mb: Mutex::new(HashMap::new()),
        }
    }
}

impl AppState {
    /// GPU vendor the runtime is built around (CPU until the first probe).
    pub fn vendor(&self) -> Vendor {
        self.runtime
            .lock()
            .ok()
            .and_then(|r| r.as_ref().map(|s| s.vendor))
            .unwrap_or_default()
    }

    /// The host has an NPU (detected in the hardware probe).
    pub fn has_npu(&self) -> bool {
        self.runtime
            .lock()
            .ok()
            .and_then(|r| r.as_ref().map(|s| !s.hardware.npus.is_empty()))
            .unwrap_or(false)
    }

    pub fn has_gpu(&self) -> bool {
        self.runtime
            .lock()
            .map(|r| r.as_ref().map(|s| s.gpu_ok).unwrap_or(false))
            .unwrap_or(false)
    }

    /// Memory a model may occupy on the primary accelerator (dedicated VRAM, or
    /// the shared budget on a unified part). Compatibility verdicts read this,
    /// never the raw `vram_mb`.
    pub fn memory_budget_mb(&self) -> Option<u64> {
        self.runtime
            .lock()
            .ok()
            .and_then(|r| r.as_ref().and_then(|s| s.effective_memory_mb))
    }

    /// `--memory` cap Space containers run under, if the startup probe could
    /// size it (#64). `None` on a fresh state before the first `runtime::refresh`.
    pub fn container_memory_cap_mb(&self) -> Option<u64> {
        self.container_memory_cap_mb.lock().ok().and_then(|g| *g)
    }

    /// Hugging Face token, if the user set one (#60).
    pub fn hf_token(&self) -> Option<String> {
        self.hf_token.lock().ok().and_then(|t| t.clone())
    }

    pub fn is_ready(&self) -> bool {
        self.runtime
            .lock()
            .map(|r| r.as_ref().map(|s| s.ready).unwrap_or(false))
            .unwrap_or(false)
    }

    /// Start (or restart if it died) the process that keeps the distro alive.
    pub fn ensure_keepalive(&self) {
        let mut guard = match self.keepalive.lock() {
            Ok(g) => g,
            Err(_) => return,
        };
        let alive = match guard.as_mut() {
            Some(child) => matches!(child.try_wait(), Ok(None)),
            None => false,
        };
        if !alive {
            match crate::wsl::spawn_keepalive() {
                Ok(child) => *guard = Some(child),
                Err(e) => log::warn!("keepalive: {e}"),
            }
        }
    }

    /// Reserve a fresh cancellation flag for a new launch of `id`, replacing
    /// any stale one left by a finished or already-cancelled launch (#35).
    pub fn begin_instance_launch(&self, id: &str) -> Arc<AtomicBool> {
        let flag = Arc::new(AtomicBool::new(false));
        if let Ok(mut m) = self.instance_launches.lock() {
            m.insert(id.to_string(), flag.clone());
        }
        flag
    }

    /// Tell whatever launch owns `id` to stop at its next checkpoint.
    pub fn cancel_instance_launch(&self, id: &str) {
        if let Ok(m) = self.instance_launches.lock() {
            if let Some(flag) = m.get(id) {
                flag.store(true, Ordering::SeqCst);
            }
        }
    }

    /// Release the launch slot for `id`, but only if nobody started a newer
    /// launch for the same id in the meantime.
    pub fn end_instance_launch(&self, id: &str, flag: &Arc<AtomicBool>) {
        if let Ok(mut m) = self.instance_launches.lock() {
            if m.get(id).is_some_and(|current| Arc::ptr_eq(current, flag)) {
                m.remove(id);
            }
        }
    }

    /// Same as `begin_instance_launch`, keyed by service id (#35).
    pub fn begin_service_launch(&self, id: ServiceId) -> Arc<AtomicBool> {
        let flag = Arc::new(AtomicBool::new(false));
        if let Ok(mut m) = self.service_launches.lock() {
            m.insert(id, flag.clone());
        }
        flag
    }

    pub fn cancel_service_launch(&self, id: ServiceId) {
        if let Ok(m) = self.service_launches.lock() {
            if let Some(flag) = m.get(&id) {
                flag.store(true, Ordering::SeqCst);
            }
        }
    }

    pub fn end_service_launch(&self, id: ServiceId, flag: &Arc<AtomicBool>) {
        if let Ok(mut m) = self.service_launches.lock() {
            if m.get(&id).is_some_and(|current| Arc::ptr_eq(current, flag)) {
                m.remove(&id);
            }
        }
    }
}
