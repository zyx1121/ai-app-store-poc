mod build;
mod cv;
mod error;
mod fetch;
mod gpu;
mod hardware;
mod hf;
mod instances;
mod ollama;
mod reconcile;
mod runtime;
mod services;
mod state;
mod storage;
mod wsl;

use tauri::{AppHandle, Manager, State};
use tauri_plugin_opener::OpenerExt;

use error::{cmd, CmdResult};
use state::AppState;

#[tauri::command]
async fn runtime_status(state: State<'_, AppState>) -> CmdResult<runtime::RuntimeStatus> {
    Ok(runtime::refresh(&state).await)
}

#[tauri::command]
async fn provision_runtime(
    app: AppHandle,
    state: State<'_, AppState>,
) -> CmdResult<runtime::RuntimeStatus> {
    cmd(runtime::provision(&app, &state).await)
}

#[tauri::command]
async fn search_spaces(
    state: State<'_, AppState>,
    query: String,
    limit: Option<usize>,
) -> CmdResult<Vec<hf::SpaceSummary>> {
    cmd(hf::search_spaces(&state.http, &query, limit.unwrap_or(24), state.has_gpu()).await)
}

#[tauri::command]
async fn search_models(
    state: State<'_, AppState>,
    query: String,
    limit: Option<usize>,
) -> CmdResult<Vec<hf::ModelSummary>> {
    cmd(hf::search_models(&state.http, &query, limit.unwrap_or(24)).await)
}

#[tauri::command]
async fn model_files(state: State<'_, AppState>, repo: String) -> CmdResult<Vec<hf::GgufFile>> {
    cmd(hf::model_files(&state.http, &repo, state.memory_budget_mb()).await)
}

#[tauri::command]
async fn launch_space(app: AppHandle, id: String) -> CmdResult<instances::Instance> {
    cmd(instances::launch_space(app, id).await)
}

#[tauri::command]
async fn build_space(
    app: AppHandle,
    id: String,
    use_repo_dockerfile: bool,
) -> CmdResult<instances::Instance> {
    // Set before starting the build; `build::build_space` (called deep inside
    // `instances::build_space`) reads it instead of taking it as an argument.
    build::USE_REPO_DOCKERFILE.store(use_repo_dockerfile, std::sync::atomic::Ordering::Relaxed);
    cmd(instances::build_space(app, id).await)
}

#[tauri::command]
async fn launch_model(
    app: AppHandle,
    repo: String,
    quant: String,
) -> CmdResult<instances::Instance> {
    cmd(instances::launch_model(app, repo, quant).await)
}

#[tauri::command]
async fn list_instances(app: AppHandle) -> CmdResult<Vec<instances::Instance>> {
    instances::discover(&app).await;
    Ok(instances::list(&app.state::<AppState>()))
}

#[tauri::command]
async fn stop_instance(app: AppHandle, id: String) -> CmdResult<()> {
    cmd(instances::stop(app, id).await)
}

#[tauri::command]
async fn remove_instance(app: AppHandle, id: String) -> CmdResult<()> {
    cmd(instances::remove(app, id).await)
}

#[tauri::command]
async fn service_status(
    app: AppHandle,
    id: services::ServiceId,
) -> CmdResult<services::ServiceStatus> {
    cmd(services::status(&app, id).await)
}

#[tauri::command]
async fn start_service(
    app: AppHandle,
    id: services::ServiceId,
) -> CmdResult<services::ServiceStatus> {
    cmd(services::start(app, id).await)
}

#[tauri::command]
async fn service_models(
    app: AppHandle,
    id: services::ServiceId,
) -> CmdResult<Vec<services::ServiceModel>> {
    cmd(services::models(&app, id).await)
}

#[tauri::command]
async fn install_service_model(
    app: AppHandle,
    id: services::ServiceId,
    name: String,
) -> CmdResult<services::ServiceModel> {
    cmd(services::install_model(app, id, name).await)
}

#[tauri::command]
async fn stop_service(app: AppHandle, id: services::ServiceId) -> CmdResult<()> {
    cmd(services::stop(app, id).await)
}

/// Everything holding GPU memory now and the budget it is measured against.
#[tauri::command]
async fn gpu_memory(app: AppHandle) -> CmdResult<gpu::GpuMemory> {
    Ok(gpu::memory(&app).await)
}

/// What must be unloaded before a launch fits. The UI shows the list and asks.
#[tauri::command]
async fn gpu_plan(app: AppHandle, request: gpu::Request) -> CmdResult<gpu::Plan> {
    Ok(gpu::plan(&app, request).await)
}

/// Unload one resident (a model, a service's weights, a Space container).
#[tauri::command]
async fn gpu_release(app: AppHandle, resident: gpu::Resident) -> CmdResult<()> {
    cmd(gpu::release(&app, resident).await)
}

/// Run a detector on an image through the CV service. The image travels as the raw
/// request body (no JSON encoding of bytes); options come as headers.
#[tauri::command]
async fn cv_detect(
    app: AppHandle,
    request: tauri::ipc::Request<'_>,
) -> CmdResult<cv::DetectResult> {
    let header = |k: &str| {
        request
            .headers()
            .get(k)
            .and_then(|v| v.to_str().ok())
            .map(str::to_string)
    };
    let model = header("x-model").unwrap_or_else(|| "yolov10n".into());
    let min_score: f32 = header("x-min-score")
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.25);
    let tauri::ipc::InvokeBody::Raw(bytes) = request.body() else {
        return Err("cv_detect expects the image as the raw request body".into());
    };
    cmd(cv::detect(&app, &model, bytes, min_score).await)
}

/// Disk usage: Docker images, build cache, the shared HF cache, our local
/// build directory, and the WSL virtual disk on Windows (#63).
#[tauri::command]
async fn storage_usage() -> CmdResult<storage::StorageUsage> {
    Ok(storage::usage().await)
}

/// Free up space per the given options. Progress arrives on `storage://progress`.
#[tauri::command]
async fn storage_cleanup(
    app: AppHandle,
    state: State<'_, AppState>,
    options: storage::CleanupOptions,
) -> CmdResult<storage::CleanupResult> {
    cmd(storage::cleanup(&app, &state, options).await)
}

/// Write a WSL2 memory ceiling to `%USERPROFILE%\.wslconfig` if none exists
/// yet (#64). `provision_runtime` already does this; exposed separately so
/// the Setup screen can offer it to a user who skipped provisioning or wants
/// to retry after deleting a bad file.
#[tauri::command]
fn write_wslconfig(state: State<'_, AppState>) -> CmdResult<String> {
    let total_ram_mb = state
        .runtime
        .lock()
        .ok()
        .and_then(|r| r.as_ref().and_then(|s| s.hardware.total_ram_mb));
    match runtime::write_wslconfig(total_ram_mb) {
        runtime::WslConfigOutcome::Written(path) => Ok(format!("wrote {path}")),
        runtime::WslConfigOutcome::Kept => Ok("existing .wslconfig kept".into()),
        runtime::WslConfigOutcome::Error(e) => Err(e),
    }
}

#[tauri::command]
fn open_url(app: AppHandle, url: String) -> CmdResult<()> {
    if !(url.starts_with("http://") || url.starts_with("https://")) {
        return Err("only http(s) URLs can be opened".into());
    }
    app.opener()
        .open_url(url, None::<&str>)
        .map_err(|e| e.to_string())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_updater::Builder::new().build())
        .plugin(tauri_plugin_process::init())
        .plugin(
            tauri_plugin_log::Builder::new()
                .level(log::LevelFilter::Info)
                .build(),
        )
        .manage(AppState::default())
        .setup(|app| {
            // Warm the runtime status so the first screen is right, and keep
            // the distro alive for the lifetime of the app.
            let handle = app.handle().clone();
            tauri::async_runtime::spawn(async move {
                let state = handle.state::<AppState>();
                let s = runtime::refresh(&state).await;
                log::info!("runtime at startup: {s:?}");
                // Sized once from the hardware probe; Space containers read it
                // on every launch (#64).
                if let Ok(mut g) = state.container_memory_cap_mb.lock() {
                    *g = Some(runtime::container_memory_cap_mb(s.hardware.total_ram_mb));
                }
                instances::discover(&handle).await;
            });
            // Re-verify liveness (a dead container, a stopped distro) every
            // 10 s for as long as the app runs (#68, #69).
            reconcile::spawn(app.handle().clone());
            Ok(())
        })
        .on_window_event(|window, event| {
            if let tauri::WindowEvent::Destroyed = event {
                // Let the distro wind down with us; Docker/Ollama state survives on disk.
                let state = window.app_handle().state::<AppState>();
                let child = state.keepalive.lock().ok().and_then(|mut g| g.take());
                if let Some(mut child) = child {
                    let _ = child.start_kill();
                }
            }
        })
        .invoke_handler(tauri::generate_handler![
            runtime_status,
            provision_runtime,
            search_spaces,
            search_models,
            model_files,
            launch_space,
            build_space,
            launch_model,
            list_instances,
            stop_instance,
            remove_instance,
            service_status,
            start_service,
            stop_service,
            service_models,
            install_service_model,
            cv_detect,
            gpu_memory,
            gpu_plan,
            gpu_release,
            storage_usage,
            storage_cleanup,
            write_wslconfig,
            open_url,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
