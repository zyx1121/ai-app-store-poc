//! GPU memory scheduling. A 10 GB card holds one modality's model family at a
//! time: a chat LLM resident in Ollama, ComfyUI's SD weights and a VLM stack in
//! VRAM together and the last one to load hits OOM. Nothing used to free the
//! previous model before the next one loaded (issue #10).
//!
//! The store tracks what is resident (Ollama models from `/api/ps`, GPU-backed
//! platform services, GPU Space containers), estimates a launch against the
//! accelerator's budget (`RuntimeStatus::effective_memory_mb`) and proposes
//! what to unload so it fits. Policy: one GPU-heavy modality at a time, and the
//! UI asks before evicting; the frontend calls `plan`, shows the list, then
//! `release` for each item and proceeds with the launch.

use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Manager};

use crate::error::{Error, Result};
use crate::hf;
use crate::instances;
use crate::ollama;
use crate::services::{self, ServiceId};
use crate::state::AppState;
use crate::wsl;

const MB: u64 = 1024 * 1024;

/// Fraction of the budget a plan may fill; the rest is headroom for the CUDA
/// context, the compositor and allocator fragmentation.
const HEADROOM_PERCENT: u64 = 95;

/// Fraction of the budget that must be free before an unsized request (a Space
/// whose footprint is unknown) is allowed to proceed. Evict heavy residents
/// largest-first only until this much is clear, instead of evicting anything
/// heavy on principle (#65).
const UNSIZED_FREE_PERCENT: u64 = 60;

/// What a running ComfyUI counts as once `/free` has dropped its models: just
/// the CUDA context, until the next job reloads something (#54).
const COMFYUI_FREED_ESTIMATE_MB: u64 = 500;

/// Torch reserved memory above this means a job is resident again; the
/// "freed" flag clears and ComfyUI counts at its launch-estimate floor again.
const COMFYUI_BUSY_THRESHOLD_MB: u64 = 1024;

/// How long a `gpu_plan` lease is honored before a launch must re-plan (#70).
const LEASE_TTL: Duration = Duration::from_secs(120);

/// Estimated footprint of a platform service while it serves, in MB. Measured
/// on the RTX 3080: Speaches CUDA with faster-whisper small plus Kokoro, Triton
/// with its CUDA context plus YOLOv10n, whisper.cpp small-q5_1 on Vulkan.
/// ComfyUI reports its own figure through `/system_stats`; this is the launch
/// estimate for sd-turbo at 512 x 512 under `--lowvram`.
fn service_estimate_mb(id: ServiceId) -> u64 {
    match id {
        ServiceId::Comfyui => 3500,
        ServiceId::Speaches => 1200,
        ServiceId::Cv => 800,
        ServiceId::Whisper => 600,
    }
}

/// Heavy services hold a whole model family and are the eviction candidates;
/// light ones (speech, detection) stay resident next to anything. Also the set
/// of service starts that contend for the shared budget and so require a
/// `gpu_plan` lease (#70).
pub fn service_is_heavy(id: ServiceId) -> bool {
    matches!(id, ServiceId::Comfyui)
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ResidentKind {
    /// A model loaded in Ollama; `id` is its tag.
    Model,
    /// A GPU-backed platform service; `id` is the `ServiceId`.
    Service,
    /// A Space container started with `--gpus all`; `id` is the instance id.
    Space,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Resident {
    pub kind: ResidentKind,
    pub id: String,
    pub name: String,
    /// Memory it holds now; `None` when the runtime cannot report it (Space containers).
    pub vram_mb: Option<u64>,
    /// Holds a whole model family; only heavy residents are ever evicted.
    pub heavy: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct GpuMemory {
    /// The accelerator's budget (dedicated VRAM, or the unified share); `None` on CPU machines.
    pub budget_mb: Option<u64>,
    /// Device-wide memory in use, when a runtime reports it (nvidia-smi, ComfyUI).
    pub used_mb: Option<u64>,
    pub residents: Vec<Resident>,
}

/// What the UI is about to launch.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "lowercase", tag = "kind")]
pub enum Request {
    Model {
        /// Ollama tag (`hf.co/owner/name:QUANT` or `qwen2.5vl:7b`)
        tag: String,
        /// GGUF size when known; the KV-cache margin from `hf::fit` is added
        size_bytes: Option<u64>,
    },
    Service {
        id: ServiceId,
    },
    Space {
        /// the Space's `owner/name`; matched to its instance through the same slug
        id: String,
    },
}

#[derive(Debug, Clone, Serialize)]
pub struct Plan {
    /// Estimated need of the request; `None` when it cannot be sized (Spaces).
    pub need_mb: Option<u64>,
    pub budget_mb: Option<u64>,
    /// Memory held by residents that stay after the plan.
    pub resident_mb: u64,
    /// Residents to unload, largest first, before the launch fits.
    pub evict: Vec<Resident>,
    /// The request fits with nothing unloaded (evict is empty for that reason).
    pub fits_without_eviction: bool,
    /// Holds this plan's answer for `LEASE_TTL`; the matching launch must present
    /// it, or re-plan and refuse (#70). `None` on a machine with no GPU budget,
    /// where nothing is scheduled and no lock is needed.
    pub lease_id: Option<String>,
}

#[derive(Deserialize)]
struct OllamaPs {
    #[serde(default)]
    models: Vec<OllamaLoaded>,
}

#[derive(Deserialize)]
struct OllamaLoaded {
    name: String,
    #[serde(default)]
    size_vram: u64,
}

#[derive(Deserialize)]
struct ComfyStats {
    #[serde(default)]
    devices: Vec<ComfyDevice>,
}

#[derive(Deserialize)]
struct ComfyDevice {
    #[serde(default)]
    vram_total: u64,
    #[serde(default)]
    vram_free: u64,
    #[serde(default)]
    torch_vram_total: u64,
}

fn http(secs: u64) -> Option<reqwest::Client> {
    reqwest::Client::builder()
        .no_proxy()
        .timeout(Duration::from_secs(secs))
        .build()
        .ok()
}

/// Models Ollama holds in GPU memory right now.
async fn ollama_residents() -> Vec<Resident> {
    let (Some(base), Some(http)) = (ollama::reachable_base().await, http(3)) else {
        return vec![];
    };
    let Ok(resp) = http.get(format!("{base}/api/ps")).send().await else {
        return vec![];
    };
    let Ok(ps) = resp.json::<OllamaPs>().await else {
        return vec![];
    };
    ps.models
        .into_iter()
        .filter(|m| m.size_vram > 0)
        .map(|m| Resident {
            kind: ResidentKind::Model,
            id: m.name.clone(),
            name: m.name,
            vram_mb: Some(m.size_vram / MB),
            heavy: true,
        })
        .collect()
}

/// ComfyUI's view of the device: (memory it reserved through torch, device-wide used).
async fn comfyui_memory() -> Option<(u64, Option<u64>)> {
    let http = http(3)?;
    let stats = http
        .get(format!(
            "http://localhost:{}/system_stats",
            services::port(ServiceId::Comfyui)
        ))
        .send()
        .await
        .ok()?
        .json::<ComfyStats>()
        .await
        .ok()?;
    let dev = stats.devices.first()?;
    // On XPU (Intel) ComfyUI reports the device as fully free while models are
    // loaded; a zero is "unknown", not "empty".
    let used = (dev.vram_total > 0 && dev.vram_total > dev.vram_free)
        .then(|| (dev.vram_total - dev.vram_free) / MB)
        .filter(|u| *u > 0);
    Some((dev.torch_vram_total / MB, used))
}

/// Device-wide memory in use on a CUDA card, from nvidia-smi inside the distro.
async fn nvidia_used_mb() -> Option<u64> {
    let out = wsl::sh(
        "/usr/lib/wsl/lib/nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1",
    )
    .await
    .ok()?;
    out.stdout.trim().parse().ok()
}

/// Everything holding GPU memory now, with the budget it is measured against.
pub async fn memory(app: &AppHandle) -> GpuMemory {
    let state = app.state::<AppState>();
    let budget_mb = state.memory_budget_mb();
    let vendor = state.vendor();
    let mut residents = ollama_residents().await;
    let mut used_mb = None;

    for id in [
        ServiceId::Comfyui,
        ServiceId::Speaches,
        ServiceId::Cv,
        ServiceId::Whisper,
    ] {
        if !services::uses_gpu(app, id) {
            continue;
        }
        let Ok(st) = services::status(app, id).await else {
            continue;
        };
        if st.state != services::ServiceState::Running {
            continue;
        }
        let vram_mb = if id == ServiceId::Comfyui {
            match comfyui_memory().await {
                // torch's reserved figure misses the CUDA context and what --lowvram
                // keeps cached: on the RTX 3080 nvidia-smi showed 2.5 GB more while
                // /system_stats reported 32 MB reserved. Never count a running ComfyUI
                // below its launch estimate; over-counting only costs an extra ask.
                // Exception: after a successful `/free` (release() sets the "freed"
                // flag) count only the CUDA context, until torch reports a job
                // resident again (#54).
                Some((torch, used)) => {
                    used_mb = used_mb.or(used);
                    if torch > COMFYUI_BUSY_THRESHOLD_MB {
                        if let Ok(mut freed) = state.comfyui_freed.lock() {
                            *freed = false;
                        }
                        Some(torch.max(service_estimate_mb(id)))
                    } else if state.comfyui_freed.lock().map(|f| *f).unwrap_or(false) {
                        Some(torch.max(COMFYUI_FREED_ESTIMATE_MB))
                    } else {
                        Some(torch.max(service_estimate_mb(id)))
                    }
                }
                None => Some(service_estimate_mb(id)),
            }
        } else {
            Some(service_estimate_mb(id))
        };
        residents.push(Resident {
            kind: ResidentKind::Service,
            id: services::id_str(id).to_string(),
            name: st.display_name,
            vram_mb,
            heavy: service_is_heavy(id),
        });
    }

    if state.has_gpu() {
        for inst in instances::list(&state) {
            if inst.kind == instances::Kind::Space
                && inst.status == instances::Status::Running
                && inst.gpu
            {
                residents.push(Resident {
                    kind: ResidentKind::Space,
                    id: inst.id,
                    name: inst.display_name,
                    vram_mb: None,
                    heavy: true,
                });
            }
        }
    }

    // nvidia-smi is authoritative when it is available: flooring it against our
    // own resident estimate would hide the real number Unload just produced
    // (issue #54/#30). Elsewhere (Vulkan, XPU) there is no device-wide probe, so
    // the residents' known total is the only figure and stays the floor.
    let used_mb = if vendor.reports_device_memory() {
        nvidia_used_mb().await.or(used_mb).filter(|u| *u > 0)
    } else {
        let known: u64 = residents.iter().filter_map(|r| r.vram_mb).sum();
        Some(used_mb.unwrap_or(0).max(known)).filter(|u| *u > 0)
    };
    GpuMemory {
        budget_mb,
        used_mb,
        residents,
    }
}

fn request_need_mb(req: &Request) -> Option<u64> {
    match req {
        Request::Model { size_bytes, .. } => size_bytes.map(|s| hf::model_need_bytes(s) / MB),
        Request::Service { id } => Some(service_estimate_mb(*id)),
        Request::Space { .. } => None,
    }
}

/// The resident the request itself is (already loaded / running), if any.
fn is_self(req: &Request, r: &Resident) -> bool {
    match req {
        Request::Model { tag, .. } => r.kind == ResidentKind::Model && &r.id == tag,
        Request::Service { id } => r.kind == ResidentKind::Service && r.id == services::id_str(*id),
        Request::Space { id } => {
            r.kind == ResidentKind::Space && r.id == format!("space-{}", wsl::slug(id))
        }
    }
}

/// Pure planning step, separated from the probes so it can be tested.
pub fn plan_for(req: &Request, mem: &GpuMemory) -> Plan {
    let need_mb = request_need_mb(req);
    let budget_mb = mem.budget_mb;
    let mut stay: Vec<&Resident> = mem.residents.iter().filter(|r| !is_self(req, r)).collect();
    let already = mem.residents.iter().any(|r| is_self(req, r));
    let known = |rs: &[&Resident]| rs.iter().filter_map(|r| r.vram_mb).sum::<u64>();

    // Already resident: nothing to do, whatever else is loaded.
    if already {
        return Plan {
            need_mb: Some(0),
            budget_mb,
            resident_mb: known(&stay),
            evict: vec![],
            fits_without_eviction: true,
            lease_id: None,
        };
    }

    let fits = |rs: &[&Resident]| match (budget_mb, need_mb) {
        // No budget figure (CPU machine): nothing to schedule.
        (None, _) => true,
        (Some(b), Some(n)) => known(rs) + n <= b * HEADROOM_PERCENT / 100,
        // Unsized request on a GPU (a Space with no known footprint): evict heavy
        // residents largest-first only until enough of the budget is clear, not
        // "any heavy resident blocks" (#65).
        (Some(b), None) => known(rs) <= b * (100 - UNSIZED_FREE_PERCENT) / 100,
    };

    if fits(&stay) {
        return Plan {
            need_mb,
            budget_mb,
            resident_mb: known(&stay),
            evict: vec![],
            fits_without_eviction: true,
            lease_id: None,
        };
    }

    // Evict heavy residents, largest known first, then the unsized ones, until it fits.
    let mut candidates: Vec<&Resident> = stay.iter().copied().filter(|r| r.heavy).collect();
    candidates.sort_by_key(|r| std::cmp::Reverse(r.vram_mb.unwrap_or(0)));
    let mut evict: Vec<Resident> = vec![];
    for c in candidates {
        stay.retain(|r| !(r.kind == c.kind && r.id == c.id));
        evict.push(c.clone());
        if fits(&stay) {
            break;
        }
    }
    Plan {
        need_mb,
        budget_mb,
        resident_mb: known(&stay),
        evict,
        fits_without_eviction: false,
        lease_id: None,
    }
}

/// A lease id unique enough for one process's lifetime: not cryptographic,
/// just distinct from whatever else is live in `AppState::gpu_leases` at once.
fn new_lease_id() -> String {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("{nanos:x}")
}

/// Record a lease for a plan that scheduled against a real budget. Machines
/// with no GPU budget need no lock: `plan_for` never evicts anything there.
fn issue_lease(state: &AppState, budget_mb: Option<u64>) -> Option<String> {
    budget_mb?;
    let id = new_lease_id();
    let expires = Instant::now() + LEASE_TTL;
    if let Ok(mut leases) = state.gpu_leases.lock() {
        leases.retain(|_, exp| *exp > Instant::now());
        leases.insert(id.clone(), expires);
    }
    Some(id)
}

/// Consume a lease from `gpu_plan`. A missing or expired lease means the plan
/// may be stale (someone else's launch or release moved the budget since):
/// refuse instead of guessing (#70). A `None` lease is only valid when this
/// machine has no GPU budget to schedule against.
pub fn require_lease(
    state: &AppState,
    lease_id: Option<&str>,
    budget_mb: Option<u64>,
) -> Result<()> {
    if budget_mb.is_none() {
        return Ok(());
    }
    let Some(id) = lease_id else {
        return Err(Error::Other("GPU plan changed, try again".into()));
    };
    let mut leases = state
        .gpu_leases
        .lock()
        .map_err(|_| Error::Other("GPU plan changed, try again".into()))?;
    leases.retain(|_, exp| *exp > Instant::now());
    match leases.remove(id) {
        Some(_) => Ok(()),
        None => Err(Error::Other("GPU plan changed, try again".into())),
    }
}

/// What must be unloaded before `req` fits on this machine.
pub async fn plan(app: &AppHandle, req: Request) -> Plan {
    let mem = memory(app).await;
    let mut plan = plan_for(&req, &mem);
    plan.lease_id = issue_lease(&app.state::<AppState>(), plan.budget_mb);
    plan
}

/// Unload one resident. Models: `keep_alive: 0` through the Ollama API (works
/// for the WSL and native servers alike) and the matching instance is marked
/// stopped. ComfyUI: `/free` drops its models but keeps the server so the
/// Canvas screen stays usable; other services stop. Spaces stop.
pub async fn release(app: &AppHandle, r: Resident) -> Result<()> {
    match r.kind {
        ResidentKind::Model => {
            let base = ollama::reachable_base()
                .await
                .ok_or_else(|| Error::NotReady("Ollama is not answering".into()))?;
            let http = http(60).ok_or_else(|| Error::Other("http client".into()))?;
            http.post(format!("{base}/api/generate"))
                .json(&serde_json::json!({ "model": r.id, "keep_alive": 0 }))
                .send()
                .await?
                .error_for_status()
                .map_err(|e| Error::Other(format!("ollama unload {}: {e}", r.id)))?;
            instances::mark_model_unloaded(app, &r.id);
            Ok(())
        }
        ResidentKind::Service => {
            let id = services::parse_id(&r.id)
                .ok_or_else(|| Error::Other(format!("`{}` is not a service", r.id)))?;
            if id == ServiceId::Comfyui {
                if let Some(http) = http(60) {
                    let freed = http
                        .post(format!(
                            "http://localhost:{}/free",
                            services::port(ServiceId::Comfyui)
                        ))
                        .json(&serde_json::json!({ "unload_models": true, "free_memory": true }))
                        .send()
                        .await
                        .map(|r| r.status().is_success())
                        .unwrap_or(false);
                    if freed {
                        // /free returns before the allocator has released everything;
                        // verified on the RTX 3080: device use fell from 4.1 GB to 1.6 GB
                        // within 3 s. The server stays up so the Canvas screen keeps working.
                        tokio::time::sleep(Duration::from_secs(3)).await;
                        // Count only the CUDA context until the next job reloads
                        // something (#54); memory() clears this once torch grows again.
                        if let Ok(mut f) = app.state::<AppState>().comfyui_freed.lock() {
                            *f = true;
                        }
                        return Ok(());
                    }
                }
            }
            services::stop(app.clone(), id).await
        }
        ResidentKind::Space => instances::stop(app.clone(), r.id).await,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model(tag: &str, mb: u64) -> Resident {
        Resident {
            kind: ResidentKind::Model,
            id: tag.into(),
            name: tag.into(),
            vram_mb: Some(mb),
            heavy: true,
        }
    }
    fn service(id: ServiceId, mb: u64) -> Resident {
        Resident {
            kind: ResidentKind::Service,
            id: services::id_str(id).into(),
            name: format!("{id:?}"),
            vram_mb: Some(mb),
            heavy: service_is_heavy(id),
        }
    }
    fn space(id: &str) -> Resident {
        Resident {
            kind: ResidentKind::Space,
            id: id.into(),
            name: id.into(),
            vram_mb: None,
            heavy: true,
        }
    }
    fn mem(budget: Option<u64>, residents: Vec<Resident>) -> GpuMemory {
        GpuMemory {
            budget_mb: budget,
            used_mb: None,
            residents,
        }
    }
    const RTX_3080: Option<u64> = Some(10240);
    const CANVAS: Request = Request::Service {
        id: ServiceId::Comfyui,
    };

    #[test]
    fn chat_model_then_canvas_unloads_the_llm_on_a_10gb_card() {
        // The issue's acceptance case: a 7B Q4 LLM (about 5 GB) is resident, Canvas
        // needs 3.5 GB; 5 + 3.5 = 8.5 GB fits 9.7 GB headroom, so nothing is evicted...
        let m = mem(RTX_3080, vec![model("hf.co/x/7b:Q4_K_M", 5000)]);
        assert!(plan_for(&CANVAS, &m).evict.is_empty());
        // ...but with a 16k context the same model sits at 7 GB and must go.
        let m = mem(RTX_3080, vec![model("hf.co/x/7b:Q4_K_M", 7000)]);
        let p = plan_for(&CANVAS, &m);
        assert!(!p.fits_without_eviction);
        assert_eq!(p.evict.len(), 1);
        assert_eq!(p.evict[0].id, "hf.co/x/7b:Q4_K_M");
        assert_eq!(p.resident_mb, 0);
    }

    #[test]
    fn light_services_are_never_evicted_and_still_count() {
        let m = mem(
            RTX_3080,
            vec![
                service(ServiceId::Speaches, 1200),
                service(ServiceId::Cv, 800),
                model("llm", 7000),
            ],
        );
        let p = plan_for(&CANVAS, &m);
        // 2000 (light) + 3500 = 5500 fits once the LLM is gone; light ones stay.
        assert_eq!(
            p.evict.iter().map(|r| r.id.as_str()).collect::<Vec<_>>(),
            ["llm"]
        );
        assert_eq!(p.resident_mb, 2000);
    }

    #[test]
    fn largest_heavy_goes_first_and_eviction_stops_when_it_fits() {
        let m = mem(RTX_3080, vec![model("small", 2000), model("big", 6000)]);
        let req = Request::Model {
            tag: "new".into(),
            size_bytes: Some(4_000_000_000),
        };
        let p = plan_for(&req, &m);
        // need = 4 GB * 1.125 + 1.5 GB = 5.7 GB; 2000 + 5722 fits, so only "big" goes.
        assert_eq!(
            p.evict.iter().map(|r| r.id.as_str()).collect::<Vec<_>>(),
            ["big"]
        );
        assert_eq!(p.resident_mb, 2000);
    }

    #[test]
    fn an_already_resident_model_needs_nothing() {
        let m = mem(
            RTX_3080,
            vec![model("llm", 9000), service(ServiceId::Comfyui, 3000)],
        );
        let req = Request::Model {
            tag: "llm".into(),
            size_bytes: Some(8_000_000_000),
        };
        let p = plan_for(&req, &m);
        assert!(p.fits_without_eviction);
        assert!(p.evict.is_empty());
        assert_eq!(p.need_mb, Some(0));
    }

    #[test]
    fn unsized_space_request_only_needs_a_clear_gpu() {
        let req = Request::Space {
            id: "owner/x".into(),
        };
        assert!(
            plan_for(&req, &mem(RTX_3080, vec![service(ServiceId::Cv, 800)]))
                .evict
                .is_empty()
        );
        let p = plan_for(&req, &mem(RTX_3080, vec![model("llm", 5000)]));
        assert_eq!(p.evict.len(), 1);
    }

    #[test]
    fn unsized_request_fits_when_enough_budget_already_free() {
        // A heavy resident alone no longer blocks an unsized launch (#65): with
        // 60% of the 10 GB budget (6144 MB) free, 2000 MB resident already clears it.
        let req = Request::Space {
            id: "owner/z".into(),
        };
        let p = plan_for(&req, &mem(RTX_3080, vec![model("small", 2000)]));
        assert!(p.fits_without_eviction);
        assert!(p.evict.is_empty());
    }

    #[test]
    fn unsized_request_evicts_only_enough_heavy_residents_to_clear_60_percent() {
        // Two heavy residents (9 GB together) leave nowhere near 60% free; evicting
        // only the larger one (6 GB) already clears it, so the smaller stays.
        let m = mem(RTX_3080, vec![model("big", 6000), model("small", 3000)]);
        let req = Request::Space {
            id: "owner/y".into(),
        };
        let p = plan_for(&req, &m);
        assert_eq!(
            p.evict.iter().map(|r| r.id.as_str()).collect::<Vec<_>>(),
            ["big"]
        );
        assert!(!p.fits_without_eviction);
    }

    #[test]
    fn unsized_residents_go_after_sized_ones() {
        let m = mem(RTX_3080, vec![space("space-hello"), model("llm", 8000)]);
        let p = plan_for(&CANVAS, &m);
        // Dropping the 8 GB LLM already fits; the unsized Space stays.
        assert_eq!(
            p.evict.iter().map(|r| r.id.as_str()).collect::<Vec<_>>(),
            ["llm"]
        );
        // With a smaller LLM the arithmetic never settles, so the Space goes too.
        let m = mem(RTX_3080, vec![space("space-hello"), model("llm", 7000)]);
        let m2 = GpuMemory {
            residents: {
                let mut r = m.residents.clone();
                r.push(service(ServiceId::Speaches, 1200));
                r.push(service(ServiceId::Cv, 800));
                r.push(model("llm2", 1500));
                r
            },
            ..m
        };
        let p = plan_for(&CANVAS, &m2);
        assert!(p.evict.iter().any(|r| r.id == "llm"));
        assert!(!p.evict.iter().any(|r| r.kind == ResidentKind::Service));
    }

    #[test]
    fn no_budget_means_no_scheduling() {
        let m = mem(
            None,
            vec![model("llm", 7000), service(ServiceId::Comfyui, 3000)],
        );
        assert!(plan_for(&CANVAS, &m).fits_without_eviction);
    }

    #[test]
    fn comfyui_resident_is_unloaded_before_a_big_model() {
        let m = mem(RTX_3080, vec![service(ServiceId::Comfyui, 3000)]);
        let req = Request::Model {
            tag: "qwen2.5vl:7b".into(),
            size_bytes: Some(6_000_000_000),
        };
        let p = plan_for(&req, &m);
        assert_eq!(p.evict.len(), 1);
        assert_eq!(p.evict[0].kind, ResidentKind::Service);
    }
}
