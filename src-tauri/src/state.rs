use std::collections::{BTreeMap, HashMap};
use std::sync::Mutex;

use tokio::process::Child;

use crate::instances::Instance;
use crate::runtime::RuntimeStatus;
use crate::services::{ServiceId, ServiceStatus};

/// Process-wide state shared by all commands. Everything here is cheap to lock;
/// long work happens in spawned tasks that re-lock briefly to publish results.
pub struct AppState {
    pub http: reqwest::Client,
    pub runtime: Mutex<Option<RuntimeStatus>>,
    pub instances: Mutex<BTreeMap<String, Instance>>,
    pub keepalive: Mutex<Option<Child>>,
    pub discovered: Mutex<bool>,
    pub services: Mutex<HashMap<ServiceId, ServiceStatus>>,
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
            keepalive: Mutex::new(None),
            discovered: Mutex::new(false),
            services: Mutex::new(HashMap::new()),
        }
    }
}

impl AppState {
    pub fn has_gpu(&self) -> bool {
        self.runtime
            .lock()
            .map(|r| r.as_ref().map(|s| s.gpu_ok).unwrap_or(false))
            .unwrap_or(false)
    }

    pub fn vram_mb(&self) -> Option<u64> {
        self.runtime
            .lock()
            .ok()
            .and_then(|r| r.as_ref().and_then(|s| s.vram_mb))
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
}
