//! Optional Hugging Face access token (#60): unlocks gated models and Spaces.
//! `tauri-plugin-store` is not a dependency, so this is a plain file under the
//! app's data dir with owner-only permissions where the platform supports it;
//! the in-memory copy lives in `AppState::hf_token` and this module is only
//! touched on startup (load) and when the user edits it in Setup (save).

use std::path::PathBuf;

use tauri::{AppHandle, Manager};

const FILE_NAME: &str = "hf_token";

fn path(app: &AppHandle) -> Option<PathBuf> {
    app.path().app_data_dir().ok().map(|d| d.join(FILE_NAME))
}

/// Read the token at startup; a missing file or unreadable data just means
/// "no token" (never fails startup).
pub fn load(app: &AppHandle) -> Option<String> {
    let path = path(app)?;
    let data = std::fs::read_to_string(path).ok()?;
    let token = data.trim();
    (!token.is_empty()).then(|| token.to_string())
}

/// Persist (`Some`) or clear (`None`) the token on disk.
pub fn save(app: &AppHandle, token: Option<&str>) -> std::io::Result<()> {
    let path = path(app).ok_or_else(|| std::io::Error::other("no app data directory"))?;
    match token {
        None => {
            if path.exists() {
                std::fs::remove_file(&path)?;
            }
            Ok(())
        }
        Some(t) => {
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(&path, t)?;
            restrict_permissions(&path)
        }
    }
}

/// Owner read/write only; the token is bearer-equivalent to the user's HF account.
#[cfg(unix)]
fn restrict_permissions(path: &std::path::Path) -> std::io::Result<()> {
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))
}

/// Windows has no POSIX mode bits; the app data dir is already scoped to the
/// current user's profile (`%APPDATA%`), which is the best `std::fs` alone
/// gets us here without an ACL crate.
#[cfg(not(unix))]
fn restrict_permissions(_path: &std::path::Path) -> std::io::Result<()> {
    Ok(())
}
