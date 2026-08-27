use std::collections::{BTreeMap, HashMap};
use std::sync::Mutex;

use tokio::process::Child;

use crate::hardware::Vendor;
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
    /// `ollama serve` we started on Windows (non-NVIDIA machines only).
    pub native_ollama: Mutex<Option<Child>>,
    /// Platform services running as native Windows processes (whisper.cpp, portable ComfyUI).
    pub native_services: Mutex<HashMap<ServiceId, Child>>,
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
            native_ollama: Mutex::new(None),
            native_services: Mutex::new(HashMap::new()),
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
