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
    #[error("runtime is not ready: {0}")]
    NotReady(String),
    #[error("{0}")]
    Other(String),
}

impl From<reqwest::Error> for Error {
    fn from(e: reqwest::Error) -> Self {
        Error::Hf(e.to_string())
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
