use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("failed to start `{0}`: {1}")]
    Spawn(String, String),
    #[error("`{cmd}` exited with {status}: {stderr}")]
    Command {
        cmd: String,
        status: i32,
        stderr: String,
    },
    #[error("Hugging Face API: {0}")]
    Hf(String),
    /// A local port (a Space, a service, Ollama) did not answer. Kept separate
    /// from `Hf` so a dead container on localhost never reads as an HF API
    /// error in the UI (#73).
    #[error("connection failed: {0}")]
    Network(String),
    #[error("runtime is not ready: {0}")]
    NotReady(String),
    #[error("{0}")]
    Other(String),
}

/// Is `host` the Hugging Face API? Everything else a `reqwest::Error` can
/// carry here is a local port we launched ourselves (a Space, a service,
/// Ollama), never the Hub (#73). Pure so it is unit-testable without a live
/// request.
fn is_hf_host(host: Option<&str>) -> bool {
    host == Some("huggingface.co")
}

impl From<reqwest::Error> for Error {
    fn from(e: reqwest::Error) -> Self {
        // Route by the request's host: only huggingface.co traffic is the HF
        // API; everything else here is a local port we launched ourselves.
        let host = e.url().and_then(|u| u.host_str());
        if is_hf_host(host) {
            Error::Hf(e.to_string())
        } else {
            Error::Network(e.to_string())
        }
    }
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Other(e.to_string())
    }
}

pub type Result<T> = std::result::Result<T, Error>;

/// Tauri commands need a serializable error; a plain string is enough for the UI.
pub type CmdResult<T> = std::result::Result<T, String>;

pub fn cmd<T>(r: Result<T>) -> CmdResult<T> {
    r.map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_the_hub_host_maps_to_hf() {
        assert!(is_hf_host(Some("huggingface.co")));
        assert!(!is_hf_host(Some("localhost")));
        assert!(!is_hf_host(Some("cdn-lfs.huggingface.co")));
        assert!(!is_hf_host(Some("127.0.0.1")));
        assert!(!is_hf_host(None));
    }
}
