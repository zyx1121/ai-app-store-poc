//! Build a Hugging Face Space image on this machine. The Hub only publishes
//! CUDA images built on its own NVIDIA tiers; when there is no image, or the
//! local GPU is not NVIDIA, we clone the Space and build it here. Inside WSL2
//! only NVIDIA GPUs are visible, so every non-NVIDIA build targets the CPU.

use std::sync::atomic::{AtomicBool, Ordering};

use crate::error::{Error, Result};
use crate::hardware::Vendor;
use crate::hf::SpaceSummary;
use crate::wsl;

/// Directory inside the distro where Space sources are cloned.
pub const BUILD_ROOT: &str = "/var/lib/aias/build";

/// Whether the next build may use a Space repo's own Dockerfile verbatim
/// instead of the generated one. A repo's Dockerfile runs arbitrary `RUN`
/// steps with network access on this machine, so it needs the user's consent
/// (Browse.tsx shows a one-time dialog before the first "Build locally").
///
/// This is a flag rather than a `build_space` argument because
/// `instances::build_space` already calls this function with a fixed
/// signature; the `build_space` Tauri command sets it just before starting a
/// build, so it always reflects the choice for the build about to run.
pub static USE_REPO_DOCKERFILE: AtomicBool = AtomicBool::new(false);

/// Packages that only ship CUDA builds; a non-NVIDIA build with these will fail,
/// so say so before spending fifteen minutes on it.
const CUDA_ONLY: &[&str] = &[
    "flash-attn",
    "flash_attn",
    "xformers",
    "bitsandbytes",
    "triton",
    "nvidia-",
    "cupy",
    "apex",
    "deepspeed",
    "tensorrt",
    "onnxruntime-gpu",
    "spconv",
    "pytorch3d",
    "nvdiffrast",
    "kaolin",
    "tiny-cuda-nn",
    "custom_rasterizer",
];

pub fn local_image_name(slug: &str) -> String {
    format!("aias-local/{slug}:latest")
}

/// Requirement lines that cannot build without CUDA, if any.
pub fn cuda_only_requirements(requirements: &str) -> Vec<String> {
    requirements
        .lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty() && !l.starts_with('#') && !l.starts_with('-'))
        .filter(|l| {
            let name = l
                .split(|c: char| !(c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.'))
                .next()
                .unwrap_or("")
                .to_ascii_lowercase();
            CUDA_ONLY.iter().any(|p| name.starts_with(p))
        })
        .map(str::to_string)
        .collect()
}

/// uv index flags that make `torch` resolve to the right wheel for this machine.
fn torch_index(vendor: Vendor) -> &'static str {
    match vendor {
        // PyPI wheels bundle CUDA; nothing to add.
        Vendor::Nvidia => "",
        // WSL2 containers cannot reach AMD or Intel GPUs, so build for the CPU.
        Vendor::Amd | Vendor::Intel | Vendor::Cpu => {
            "--index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match"
        }
    }
}

/// Day the SDK version was published on PyPI, plus a week. Old Spaces only
/// work with the dependency set of their era; `uv --exclude-newer` rebuilds it.
pub async fn sdk_exclude_newer(http: &reqwest::Client, sdk: &str, version: &str) -> Option<String> {
    #[derive(serde::Deserialize)]
    struct Release {
        urls: Vec<Upload>,
    }
    #[derive(serde::Deserialize)]
    struct Upload {
        upload_time: String,
    }
    let pkg = match sdk {
        "gradio" | "streamlit" => sdk,
        _ => return None,
    };
    let r: Release = http
        .get(format!("https://pypi.org/pypi/{pkg}/{version}/json"))
        .send()
        .await
        .ok()?
        .error_for_status()
        .ok()?
        .json()
        .await
        .ok()?;
    let day = r.urls.first()?.upload_time.get(..10)?.to_string();
    plus_days(&day, 7)
}

/// `YYYY-MM-DD` plus `days`, without pulling in a date crate.
pub fn plus_days(day: &str, days: i64) -> Option<String> {
    let mut it = day.split('-');
    let y: i64 = it.next()?.parse().ok()?;
    let m: i64 = it.next()?.parse().ok()?;
    let d: i64 = it.next()?.parse().ok()?;
    // Days from civil (Howard Hinnant's algorithm), then back.
    let yy = if m <= 2 { y - 1 } else { y };
    let era = yy.div_euclid(400);
    let yoe = yy - era * 400;
    let mp = (m + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let z = era * 146_097 + doe - 719_468 + days;
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    Some(format!("{y:04}-{m:02}-{d:02}"))
}

/// Dockerfile mirroring what the Hub does for SDK Spaces: a slim Python base,
/// the SDK pinned to the card's version, the Space's requirements, then the
/// app file as the entrypoint on the SDK's default port.
pub fn dockerfile_for(
    space: &SpaceSummary,
    vendor: Vendor,
    python_version: Option<&str>,
    exclude_newer: Option<&str>,
) -> Option<String> {
    let py = python_version.unwrap_or("3.10");
    let sdk = space.sdk.as_deref()?;
    let (sdk_pkg, cmd, port) = match sdk {
        "gradio" => (
            format!(
                "gradio{}",
                space
                    .sdk_version
                    .as_deref()
                    .map(|v| format!("=={v}"))
                    .unwrap_or_default()
            ),
            format!("python {}", wsl::quote(&space.app_file)),
            7860,
        ),
        "streamlit" => (
            format!(
                "streamlit{}",
                space
                    .sdk_version
                    .as_deref()
                    .map(|v| format!("=={v}"))
                    .unwrap_or_default()
            ),
            format!(
                "streamlit run {} --server.port {} --server.address 0.0.0.0 --server.headless true",
                wsl::quote(&space.app_file),
                space.app_port
            ),
            space.app_port,
        ),
        _ => return None,
    };
    let index = torch_index(vendor);
    let newer = exclude_newer
        .map(|d| format!("--exclude-newer {d}"))
        .unwrap_or_default();
    Some(format!(
        r#"FROM python:{py}-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 GRADIO_SERVER_NAME=0.0.0.0 GRADIO_SERVER_PORT={port} \
    GRADIO_ALLOW_FLAGGING=never GRADIO_FLAGGING_MODE=never HF_HOME=/home/user/.cache/huggingface
RUN apt-get update && apt-get install -y --no-install-recommends git ffmpeg libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/* \
 && useradd -m -u 1000 user && mkdir -p /home/user/app && chown -R user:user /home/user
WORKDIR /home/user/app
COPY --chown=user:user . .
RUN pip install --no-cache-dir uv \
 && uv pip install --system {newer} {index} "{sdk_pkg}" \
 && if [ -f requirements.txt ]; then uv pip install --system {newer} {index} -r requirements.txt; fi \
 && if [ -f packages.txt ]; then echo "packages.txt present: system packages are not installed by the local builder" >&2; fi
USER user
EXPOSE {port}
CMD {cmd}
"#
    ))
}

/// Clone the Space and build `aias-local/<slug>`. Lines go to `on_line`; the
/// caller decides how to surface them. Returns the image name.
pub async fn build_space(
    http: &reqwest::Client,
    space: &SpaceSummary,
    slug: &str,
    vendor: Vendor,
    mut on_line: impl FnMut(String),
) -> Result<String> {
    let dir = format!("{BUILD_ROOT}/{slug}");
    let image = local_image_name(slug);
    let repo = &space.id;

    on_line(format!("git clone https://huggingface.co/spaces/{repo}"));
    let clone = format!(
        "mkdir -p {BUILD_ROOT} && rm -rf '{dir}' && GIT_LFS_SKIP_SMUDGE=0 git clone --depth 1 'https://huggingface.co/spaces/{repo}' '{dir}' 2>&1"
    );
    let code = wsl::stream_lines(wsl::spawn_sh(&clone)?, &mut on_line).await?;
    if code != 0 {
        return Err(Error::Other(format!(
            "clone failed ({code}); the Space may be private or gated"
        )));
    }

    // Inspect what we got: Dockerfile, requirements, python version.
    let probe = wsl::sh(&format!(
        "cd '{dir}' && (test -f Dockerfile && echo HAS_DOCKERFILE); (test -f requirements.txt && echo REQ_BEGIN && cat requirements.txt && echo REQ_END); grep -m1 -E '^python_version:' README.md 2>/dev/null | sed 's/python_version: *//;s/\"//g' | sed 's/^/PYVER /'"
    ))
    .await?;
    let has_dockerfile = probe.stdout.lines().any(|l| l.trim() == "HAS_DOCKERFILE");
    let requirements = probe
        .stdout
        .split("REQ_BEGIN")
        .nth(1)
        .and_then(|s| s.split("REQ_END").next())
        .unwrap_or("")
        .to_string();
    let python_version = probe
        .stdout
        .lines()
        .find_map(|l| l.strip_prefix("PYVER "))
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty());

    if vendor != Vendor::Nvidia {
        let blockers = cuda_only_requirements(&requirements);
        if !blockers.is_empty() {
            return Err(Error::Other(format!(
                "needs CUDA-only packages ({}); it cannot run on this GPU",
                blockers.join(", ")
            )));
        }
    }

    // The repo's own Dockerfile runs arbitrary `RUN` steps with network access on
    // this machine; only use it verbatim when the user consented (Browse.tsx's
    // one-time dialog, threaded through `USE_REPO_DOCKERFILE`). Otherwise prefer
    // the generated Dockerfile even when the Space ships its own.
    let use_repo_dockerfile = has_dockerfile && USE_REPO_DOCKERFILE.load(Ordering::Relaxed);
    if !use_repo_dockerfile {
        let exclude_newer = match (space.sdk.as_deref(), space.sdk_version.as_deref()) {
            (Some(sdk), Some(ver)) => sdk_exclude_newer(http, sdk, ver).await,
            _ => None,
        };
        if let Some(d) = &exclude_newer {
            on_line(format!(
                "resolving dependencies as of {d} (SDK release date + 7 days)"
            ));
        }
        let Some(dockerfile) = dockerfile_for(
            space,
            vendor,
            python_version.as_deref(),
            exclude_newer.as_deref(),
        ) else {
            if has_dockerfile {
                return Err(Error::Other(
                    "this Space has no supported SDK card, only its own Dockerfile; \
                     building it needs consent to run the repo's Dockerfile, which \
                     \"Build locally\" asks for once"
                        .into(),
                ));
            }
            return Err(Error::Other(format!(
                "SDK `{}` has no Dockerfile and no template; cannot build",
                space.sdk.as_deref().unwrap_or("unknown")
            )));
        };
        on_line(format!(
            "generated Dockerfile (python {}, {} target)",
            python_version.as_deref().unwrap_or("3.10"),
            if vendor == Vendor::Nvidia {
                "CUDA"
            } else {
                "CPU"
            }
        ));
        let write = wsl::spawn_script(&format!(
            "cat > '{dir}/Dockerfile' <<'AIAS_EOF'\n{dockerfile}\nAIAS_EOF\n"
        ))
        .await?;
        wsl::stream_lines(write, |_| {}).await?;
    } else {
        on_line("using the Space's own Dockerfile (consent given)".into());
    }

    on_line(format!("docker build -t {image}"));
    let build = format!(
        "cd '{dir}' && DOCKER_BUILDKIT=1 docker build --progress=plain -t '{image}' . 2>&1"
    );
    let code = wsl::stream_lines(wsl::spawn_sh(&build)?, &mut on_line).await?;
    if code != 0 {
        return Err(Error::Other(format!(
            "docker build failed ({code}); see log"
        )));
    }
    on_line(format!("built {image}"));
    Ok(image)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hf::Compat;

    fn space(sdk: &str) -> SpaceSummary {
        SpaceSummary {
            id: "owner/demo".into(),
            author: "owner".into(),
            name: "demo".into(),
            sdk: Some(sdk.into()),
            sdk_version: Some("5.0.0".into()),
            likes: 0,
            hardware: None,
            app_port: if sdk == "streamlit" { 8501 } else { 7860 },
            app_file: "app.py".into(),
            title: None,
            emoji: None,
            compat: Compat::Ready,
            compat_reason: None,
            // `secrets` (#57) is a new SpaceSummary field; this fixture never
            // exercises it, only the local-build flow.
            secrets: vec![],
        }
    }

    #[test]
    fn flags_cuda_only_packages() {
        let req = "gradio\ntorch==2.4.0\nflash-attn>=2.0\n# comment\nxformers\nnumpy";
        assert_eq!(
            cuda_only_requirements(req),
            vec!["flash-attn>=2.0", "xformers"]
        );
        assert!(cuda_only_requirements("gradio\ntorch").is_empty());
    }

    #[test]
    fn gradio_dockerfile_targets_cpu_off_nvidia() {
        let d = dockerfile_for(&space("gradio"), Vendor::Amd, None, Some("2023-09-07")).unwrap();
        assert!(d.contains("FROM python:3.10-slim"));
        assert!(d.contains("download.pytorch.org/whl/cpu"));
        assert!(d.contains("\"gradio==5.0.0\""));
        assert!(d.contains("--exclude-newer 2023-09-07"));
        assert!(d.contains("CMD python 'app.py'"));
        let n = dockerfile_for(&space("gradio"), Vendor::Nvidia, Some("3.11"), None).unwrap();
        assert!(!n.contains("whl/cpu"));
        assert!(!n.contains("--exclude-newer"));
        assert!(n.contains("python:3.11-slim"));
    }

    #[test]
    fn streamlit_and_unknown_sdk() {
        let d = dockerfile_for(&space("streamlit"), Vendor::Cpu, None, None).unwrap();
        assert!(d.contains("streamlit run 'app.py' --server.port 8501"));
        assert!(dockerfile_for(&space("static"), Vendor::Cpu, None, None).is_none());
    }

    #[test]
    fn adds_days_across_month_and_year_ends() {
        assert_eq!(plus_days("2023-08-23", 7).as_deref(), Some("2023-08-30"));
        assert_eq!(plus_days("2023-08-28", 7).as_deref(), Some("2023-09-04"));
        assert_eq!(plus_days("2024-12-28", 7).as_deref(), Some("2025-01-04"));
        assert_eq!(plus_days("2024-02-26", 7).as_deref(), Some("2024-03-04"));
        assert!(plus_days("garbage", 7).is_none());
    }
}
