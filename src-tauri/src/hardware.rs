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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Gpu {
    pub name: String,
    pub vendor: Vendor,
    pub vram_mb: Option<u64>,
    /// Integrated graphics share system RAM; a discrete card is preferred when both exist.
    pub integrated: bool,
    pub driver: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Npu {
    pub name: String,
    /// intel / amd / qualcomm / unknown
    pub vendor: String,
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
}

#[derive(Deserialize)]
struct RawProbe {
    #[serde(default)]
    gpus: Vec<RawGpu>,
    #[serde(default)]
    npus: Vec<RawNpu>,
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
            n.contains("uhd")
                || n.contains("iris")
                || n.contains("hd graphics")
                || (n.contains("arc")
                    && n.contains("graphics")
                    && !n.contains("a7")
                    && !n.contains("a5")
                    && !n.contains("b5"))
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
    let mut gpus: Vec<Gpu> = raw
        .gpus
        .into_iter()
        .filter_map(|g| {
            let vendor = classify_gpu(&g.name, g.vendor.as_deref(), g.pnp.as_deref())?;
            Some(Gpu {
                integrated: is_integrated(&g.name, vendor),
                name: g.name,
                vendor,
                vram_mb: g.vram_mb,
                driver: g.driver,
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

    const KING: &str = r#"{"gpus":[{"name":"NVIDIA GeForce RTX 3080","vendor":"NVIDIA","driver":"32.0.16.1047","pnp":"PCI\\VEN_10DE&DEV_2206","vram_mb":10240},{"name":"AMD Radeon(TM) Graphics","vendor":"Advanced Micro Devices, Inc.","driver":"32.0.21043.5001","pnp":"PCI\\VEN_1002&DEV_13C0","vram_mb":2048}],"npus":[]}"#;

    #[test]
    fn picks_the_discrete_nvidia_card_over_an_amd_igpu() {
        let p = from_probe_json(KING);
        assert_eq!(p.vendor, Vendor::Nvidia);
        assert!(p.wsl_gpu);
        assert_eq!(p.gpus.len(), 2);
        assert!(p.gpus[1].integrated);
        assert_eq!(p.primary_gpu.as_ref().unwrap().vram_mb, Some(10240));
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
