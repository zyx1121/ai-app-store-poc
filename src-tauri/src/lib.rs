mod error;
mod hf;
mod instances;
mod runtime;
mod services;
mod state;
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
    cmd(hf::model_files(&state.http, &repo, state.vram_mb()).await)
}

#[tauri::command]
async fn launch_space(app: AppHandle, id: String) -> CmdResult<instances::Instance> {
    cmd(instances::launch_space(app, id).await)
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
async fn service_models(id: services::ServiceId) -> CmdResult<Vec<services::ServiceModel>> {
    cmd(services::models(id).await)
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
                instances::discover(&handle).await;
            });
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
            launch_model,
            list_instances,
            stop_instance,
            remove_instance,
            service_status,
            start_service,
            stop_service,
            service_models,
            install_service_model,
            open_url,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
