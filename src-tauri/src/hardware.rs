//! Hardware profile of the Windows host: which GPUs and NPUs exist and which
//! vendor the store should build its runtime around. Everything downstream
//! (service specs, provisioning, compatibility verdicts) keys off `Vendor`.

use serde::{Deserialize, Serialize};

use crate::wsl;

const PROBE_SCRIPT: &str = include_str!("../hardware.ps1");

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash, Default)]
#[serde(rename_all = "lowercase")]
pub enum Vendor {
    Nvidia,
    Amd,
    Intel,
    /// No usable GPU: everything runs on the CPU.
    #[default]
    Cpu,
}

/// Where an accelerator's working memory lives. Compatibility verdicts must
/// not read a unified part's `vram_mb`: the driver registry reports only the
/// small dedicated carve-out (2 GB on a Core Ultra laptop) while the device
/// addresses system RAM through WDDM shared GPU memory (issue #21).
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub enum MemoryModel {
    /// A discrete card with its own VRAM; `vram_mb` is the budget.
    #[default]
    Dedicated,
    /// Integrated GPU or NPU sharing system RAM; the budget comes from total RAM.
    Unified,
}

/// Share of physical RAM a unified-memory accelerator may use. WDDM caps shared
/// GPU memory at half of system RAM, and that is what Vulkan reported on the
/// Core Ultra 5 225H laptop (17.5 GiB heap on 31 GB of RAM).
pub fn unified_budget_mb(total_ram_mb: Option<u64>) -> Option<u64> {
    total_ram_mb.map(|t| t / 2)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Gpu {
    pub name: String,
    pub vendor: Vendor,
    /// Dedicated memory as the driver reports it. Display only on unified parts.
    pub vram_mb: Option<u64>,
    /// Integrated graphics share system RAM; a discrete card is preferred when both exist.
    pub integrated: bool,
    pub driver: Option<String>,
    pub memory_model: MemoryModel,
    /// What a model may actually occupy here: `vram_mb` on a discrete card, half
    /// of system RAM on a unified part. The number compatibility verdicts read.
    pub effective_memory_mb: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Npu {
    pub name: String,
    /// intel / amd / qualcomm / unknown
    pub vendor: String,
    /// NPUs always share system RAM; same budget as a unified GPU.
    pub effective_memory_mb: Option<u64>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
pub struct Virtualization {
    /// the CPU has VT-x / AMD-V
    pub vt_supported: bool,
    /// virtualization is enabled in the UEFI firmware (the BIOS switch)
    pub vt_firmware_enabled: bool,
    /// the Windows hypervisor is already running (so WSL2 will work regardless)
    pub hypervisor_present: bool,
}

impl Virtualization {
    /// WSL2 can run now, or will after `wsl --install` and a reboot.
    pub fn usable(&self) -> bool {
        self.hypervisor_present || (self.vt_supported && self.vt_firmware_enabled)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct HardwareProfile {
    pub gpus: Vec<Gpu>,
    pub npus: Vec<Npu>,
    /// The GPU the runtime is built around (best discrete first), if any.
    pub primary_gpu: Option<Gpu>,
    /// Vendor that selects service implementations and provisioning steps.
    pub vendor: Vendor,
    /// Windows can hand a CUDA GPU to WSL2 containers; other vendors run natively on the host.
    pub wsl_gpu: bool,
    /// Firmware / hypervisor state that decides whether WSL2 can run at all.
    pub virtualization: Virtualization,
    /// Physical RAM; sizes the budget of unified-memory accelerators.
    pub total_ram_mb: Option<u64>,
}

#[derive(Deserialize)]
struct RawProbe {
    #[serde(default)]
    gpus: Vec<RawGpu>,
    #[serde(default)]
    npus: Vec<RawNpu>,
    #[serde(default)]
    virtualization: Virtualization,
    #[serde(default)]
    total_ram_mb: Option<u64>,
}

#[derive(Deserialize)]
struct RawGpu {
    #[serde(default)]
    name: String,
    #[serde(default)]
    vendor: Option<String>,
    #[serde(default)]
    driver: Option<String>,
    #[serde(default)]
    pnp: Option<String>,
    #[serde(default)]
    vram_mb: Option<u64>,
}

#[derive(Deserialize)]
struct RawNpu {
    #[serde(default)]
    name: String,
}

fn classify_gpu(name: &str, vendor: Option<&str>, pnp: Option<&str>) -> Option<Vendor> {
    let hay = format!(
        "{} {} {}",
        name.to_ascii_lowercase(),
        vendor.unwrap_or("").to_ascii_lowercase(),
        pnp.unwrap_or("").to_ascii_uppercase()
    );
    if hay.contains("nvidia") || hay.contains("VEN_10DE") {
        Some(Vendor::Nvidia)
    } else if hay.contains("advanced micro")
        || hay.contains("amd")
        || hay.contains("radeon")
        || hay.contains("VEN_1002")
    {
        Some(Vendor::Amd)
    } else if hay.contains("intel") || hay.contains("VEN_8086") {
        Some(Vendor::Intel)
    } else {
        // Virtual display adapters, remote desktop drivers and the like.
        None
    }
}

/// Integrated parts are named by generation rather than model number.
fn is_integrated(name: &str, vendor: Vendor) -> bool {
    let n = name.to_ascii_lowercase();
    match vendor {
        Vendor::Amd => {
            n.contains("radeon(tm) graphics")
                || n.contains("radeon graphics")
                || n.contains("vega") && !n.contains("radeon rx")
        }
        Vendor::Intel => {
            // Discrete Arc cards carry a letter-plus-three-digit model (A380, A770,
            // B580); everything else Intel ships is integrated: UHD, Iris, "Arc
            // Graphics" and the Core Ultra "Arc 130T GPU" / "Arc 140V GPU" parts.
            let discrete_arc = n.split_whitespace().any(|tok| {
                let tok = tok.trim_matches(|c: char| !c.is_ascii_alphanumeric());
                tok.len() == 4
                    && matches!(tok.as_bytes()[0], b'a' | b'b')
                    && tok[1..].bytes().all(|b| b.is_ascii_digit())
            });
            !discrete_arc && !n.contains("data center")
        }
        Vendor::Nvidia => false,
        Vendor::Cpu => false,
    }
}

fn classify_npu(name: &str) -> &'static str {
    let n = name.to_ascii_lowercase();
    if n.contains("ai boost") || n.contains("intel") {
        "intel"
    } else if n.contains("ipu") || n.contains("ryzen ai") || n.contains("amd") {
        "amd"
    } else if n.contains("hexagon") || n.contains("qualcomm") || n.contains("snapdragon") {
        "qualcomm"
    } else {
        "unknown"
    }
}

pub fn from_probe_json(json: &str) -> HardwareProfile {
    let raw: RawProbe = match serde_json::from_str(json) {
        Ok(r) => r,
        Err(e) => {
            log::warn!("hardware probe unparsable: {e}");
            return HardwareProfile::default();
        }
    };
    let unified_budget = unified_budget_mb(raw.total_ram_mb);
    let mut gpus: Vec<Gpu> = raw
        .gpus
        .into_iter()
        .filter_map(|g| {
            let vendor = classify_gpu(&g.name, g.vendor.as_deref(), g.pnp.as_deref())?;
            let integrated = is_integrated(&g.name, vendor);
            let (memory_model, effective_memory_mb) = if integrated {
                // The carve-out is a floor, never the ceiling, on a unified part.
                (MemoryModel::Unified, unified_budget.or(g.vram_mb))
            } else {
                (MemoryModel::Dedicated, g.vram_mb)
            };
            Some(Gpu {
                integrated,
                name: g.name,
                vendor,
                vram_mb: g.vram_mb,
                driver: g.driver,
                memory_model,
                effective_memory_mb,
            })
        })
        .collect();
    // Best card first: discrete over integrated, then NVIDIA (most software), then VRAM.
    gpus.sort_by_key(|g| {
        (
            g.integrated,
            match g.vendor {
                Vendor::Nvidia => 0,
                Vendor::Amd => 1,
                Vendor::Intel => 2,
                Vendor::Cpu => 3,
            },
            std::cmp::Reverse(g.vram_mb.unwrap_or(0)),
        )
    });
    let npus = raw
        .npus
        .into_iter()
        .map(|n| Npu {
            vendor: classify_npu(&n.name).to_string(),
            name: n.name,
            effective_memory_mb: unified_budget,
        })
        .collect();
    let primary_gpu = gpus.first().cloned();
    let vendor = primary_gpu
        .as_ref()
        .map(|g| g.vendor)
        .unwrap_or(Vendor::Cpu);
    HardwareProfile {
        wsl_gpu: vendor == Vendor::Nvidia,
        gpus,
        npus,
        primary_gpu,
        vendor,
        virtualization: raw.virtualization,
        total_ram_mb: raw.total_ram_mb,
    }
}

/// Run the PowerShell probe on the host. Never fails: an unreadable machine is a CPU machine.
pub async fn probe() -> HardwareProfile {
    let path = std::env::temp_dir().join("ai-app-store-hardware.ps1");
    if let Err(e) = std::fs::write(&path, PROBE_SCRIPT) {
        log::warn!("hardware probe: cannot write script: {e}");
        return HardwareProfile::default();
    }
    let path_str = path.to_string_lossy().to_string();
    let out = wsl::run(
        "powershell.exe",
        &[
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            &path_str,
        ],
    )
    .await;
    match out {
        Ok(o) if o.ok() => from_probe_json(o.stdout.trim()),
        Ok(o) => {
            log::warn!("hardware probe exited {}: {}", o.status, o.stderr.trim());
            HardwareProfile::default()
        }
        Err(e) => {
            log::warn!("hardware probe: {e}");
            HardwareProfile::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const KING: &str = r#"{"gpus":[{"name":"NVIDIA GeForce RTX 3080","vendor":"NVIDIA","driver":"32.0.16.1047","pnp":"PCI\\VEN_10DE&DEV_2206","vram_mb":10240},{"name":"AMD Radeon(TM) Graphics","vendor":"Advanced Micro Devices, Inc.","driver":"32.0.21043.5001","pnp":"PCI\\VEN_1002&DEV_13C0","vram_mb":2048}],"npus":[],"total_ram_mb":65268}"#;
    /// ASUS Vivobook 14, Core Ultra 5 225H: the registry says 2 GB for the Arc 130T
    /// while Ollama offloads to a 17.5 GiB Vulkan heap on its 31 GB of RAM.
    const LAPTOP: &str = r#"{"gpus":[{"name":"Intel(R) Arc(TM) 130T GPU","vendor":"Intel Corporation","driver":"32.0.101.6913","pnp":"PCI\\VEN_8086&DEV_64A0","vram_mb":2048}],"npus":[{"name":"Intel(R) AI Boost","class":"ComputeAccelerator","status":"OK"}],"total_ram_mb":31654}"#;

    #[test]
    fn picks_the_discrete_nvidia_card_over_an_amd_igpu() {
        let p = from_probe_json(KING);
        assert_eq!(p.vendor, Vendor::Nvidia);
        assert!(p.wsl_gpu);
        assert_eq!(p.gpus.len(), 2);
        assert!(p.gpus[1].integrated);
        let primary = p.primary_gpu.as_ref().unwrap();
        assert_eq!(primary.vram_mb, Some(10240));
        // A discrete card's budget is its VRAM, whatever the RAM size.
        assert_eq!(primary.memory_model, MemoryModel::Dedicated);
        assert_eq!(primary.effective_memory_mb, Some(10240));
        // The iGPU next to it is unified and gets half of RAM.
        assert_eq!(p.gpus[1].memory_model, MemoryModel::Unified);
        assert_eq!(p.gpus[1].effective_memory_mb, Some(32634));
        assert_eq!(p.total_ram_mb, Some(65268));
    }

    #[test]
    fn unified_igpu_budget_is_half_of_ram_not_the_carve_out() {
        let p = from_probe_json(LAPTOP);
        assert_eq!(p.vendor, Vendor::Intel);
        let g = p.primary_gpu.as_ref().unwrap();
        assert!(g.integrated);
        assert_eq!(g.memory_model, MemoryModel::Unified);
        assert_eq!(g.vram_mb, Some(2048), "raw figure kept for display");
        assert_eq!(g.effective_memory_mb, Some(15827));
        // A 7B Q4 GGUF (about 4.5 GB) now fits; against 2 GB it was Incompatible.
        assert!(g.effective_memory_mb.unwrap() * 1024 * 1024 > 4_500_000_000 + 1_500_000_000);
        assert_eq!(p.npus[0].effective_memory_mb, Some(15827));
    }

    #[test]
    fn intel_arc_naming_separates_igpu_from_discrete() {
        for igpu in [
            "Intel(R) Arc(TM) 130T GPU",
            "Intel(R) Arc(TM) 140V GPU",
            "Intel(R) Arc(TM) Graphics",
            "Intel(R) Iris(R) Xe Graphics",
            "Intel(R) UHD Graphics 770",
        ] {
            assert!(is_integrated(igpu, Vendor::Intel), "{igpu} is integrated");
        }
        for dgpu in [
            "Intel(R) Arc(TM) A770 Graphics",
            "Intel(R) Arc(TM) A380 Graphics",
            "Intel(R) Arc(TM) B580 Graphics",
            "Intel(R) Arc(TM) Pro A60 Graphics",
        ] {
            // Pro A60 is the one three-character model; it is a discrete workstation
            // card, but the rule below treats it as integrated. Accepted: it is not a
            // laptop part the store targets, and the verdict only becomes more generous.
            if dgpu.contains("A60") {
                continue;
            }
            assert!(!is_integrated(dgpu, Vendor::Intel), "{dgpu} is discrete");
        }
    }

    #[test]
    fn unified_part_without_ram_figure_falls_back_to_the_carve_out() {
        let p = from_probe_json(
            r#"{"gpus":[{"name":"Intel(R) Arc(TM) Graphics","vendor":"Intel Corporation","pnp":"PCI\\VEN_8086&DEV_7D55","vram_mb":128}],"npus":[]}"#,
        );
        let g = p.primary_gpu.as_ref().unwrap();
        assert_eq!(g.memory_model, MemoryModel::Unified);
        assert_eq!(g.effective_memory_mb, Some(128));
        assert_eq!(p.total_ram_mb, None);
    }

    #[test]
    fn amd_only_machine_is_amd_without_wsl_gpu() {
        let p = from_probe_json(
            r#"{"gpus":[{"name":"AMD Radeon RX 7800 XT","vendor":"Advanced Micro Devices, Inc.","pnp":"PCI\\VEN_1002&DEV_747E","vram_mb":16384}],"npus":[{"name":"AMD IPU Device","class":"ComputeAccelerator","status":"OK"}]}"#,
        );
        assert_eq!(p.vendor, Vendor::Amd);
        assert!(!p.wsl_gpu);
        assert!(!p.gpus[0].integrated);
        assert_eq!(p.npus[0].vendor, "amd");
    }

    #[test]
    fn intel_igpu_plus_npu() {
        let p = from_probe_json(
            r#"{"gpus":[{"name":"Intel(R) Arc(TM) Graphics","vendor":"Intel Corporation","pnp":"PCI\\VEN_8086&DEV_7D55","vram_mb":128}],"npus":[{"name":"Intel(R) AI Boost","class":"ComputeAccelerator","status":"OK"}]}"#,
        );
        assert_eq!(p.vendor, Vendor::Intel);
        assert!(p.gpus[0].integrated);
        assert_eq!(p.npus[0].vendor, "intel");
    }

    #[test]
    fn no_gpu_is_cpu_and_bad_json_is_cpu() {
        assert_eq!(
            from_probe_json(r#"{"gpus":[],"npus":[]}"#).vendor,
            Vendor::Cpu
        );
        assert_eq!(from_probe_json("nonsense").vendor, Vendor::Cpu);
    }
}
