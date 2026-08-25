//! Build a Hugging Face Space image on this machine. The Hub only publishes
//! CUDA images built on its own NVIDIA tiers; when there is no image, or the
//! local GPU is not NVIDIA, we clone the Space and build it here. Inside WSL2
//! only NVIDIA GPUs are visible, so every non-NVIDIA build targets the CPU.

use crate::error::{Error, Result};
use crate::hardware::Vendor;
use crate::hf::SpaceSummary;
use crate::wsl;

/// Directory inside the distro where Space sources are cloned.
const BUILD_ROOT: &str = "/var/lib/aias/build";

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

/// pip flags that make `torch` resolve to the right wheel for this machine.
fn torch_index(vendor: Vendor) -> &'static str {
    match vendor {
        // PyPI wheels bundle CUDA; nothing to add.
        Vendor::Nvidia => "",
        // WSL2 containers cannot reach AMD or Intel GPUs, so build for the CPU.
        Vendor::Amd | Vendor::Intel | Vendor::Cpu => {
            "--index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple"
        }
    }
}

/// Dockerfile mirroring what the Hub does for SDK Spaces: a slim Python base,
/// the SDK pinned to the card's version, the Space's requirements, then the
/// app file as the entrypoint on the SDK's default port.
pub fn dockerfile_for(
    space: &SpaceSummary,
    vendor: Vendor,
    python_version: Option<&str>,
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
            format!("python {}", space.app_file),
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
                space.app_file, space.app_port
            ),
            space.app_port,
        ),
        _ => return None,
    };
    let index = torch_index(vendor);
    Some(format!(
        r#"FROM python:{py}-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 GRADIO_SERVER_NAME=0.0.0.0 GRADIO_SERVER_PORT={port} HF_HOME=/home/user/.cache/huggingface
RUN apt-get update && apt-get install -y --no-install-recommends git ffmpeg libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/* \
 && useradd -m -u 1000 user
WORKDIR /home/user/app
COPY --chown=user . .
RUN pip install --upgrade pip && pip install {index} "{sdk_pkg}" \
 && if [ -f requirements.txt ]; then pip install {index} -r requirements.txt; fi \
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

    if !has_dockerfile {
        let Some(dockerfile) = dockerfile_for(space, vendor, python_version.as_deref()) else {
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
        on_line("using the Space's own Dockerfile".into());
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
        let d = dockerfile_for(&space("gradio"), Vendor::Amd, None).unwrap();
        assert!(d.contains("FROM python:3.10-slim"));
        assert!(d.contains("download.pytorch.org/whl/cpu"));
        assert!(d.contains("\"gradio==5.0.0\""));
        assert!(d.contains("CMD python app.py"));
        let n = dockerfile_for(&space("gradio"), Vendor::Nvidia, Some("3.11")).unwrap();
        assert!(!n.contains("whl/cpu"));
        assert!(n.contains("python:3.11-slim"));
    }

    #[test]
    fn streamlit_and_unknown_sdk() {
        let d = dockerfile_for(&space("streamlit"), Vendor::Cpu, None).unwrap();
        assert!(d.contains("streamlit run app.py --server.port 8501"));
        assert!(dockerfile_for(&space("static"), Vendor::Cpu, None).is_none());
    }
}
