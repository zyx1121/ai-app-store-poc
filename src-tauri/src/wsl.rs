//! Thin wrapper around `wsl.exe`: run commands on the Windows side or inside
//! our distro, without flashing console windows, and with sane text decoding.

use std::process::Stdio;

use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::mpsc;

use crate::error::{Error, Result};

/// Name of the WSL distribution this app owns. Never touch other distros.
pub const DISTRO: &str = "ai-app-store";

#[cfg(windows)]
const CREATE_NO_WINDOW: u32 = 0x0800_0000;

#[derive(Debug)]
pub struct Output {
    pub status: i32,
    pub stdout: String,
    pub stderr: String,
}

impl Output {
    pub fn ok(&self) -> bool {
        self.status == 0
    }

    pub fn require(self, cmd: &str) -> Result<Self> {
        if self.ok() {
            Ok(self)
        } else {
            Err(Error::Command {
                cmd: cmd.to_string(),
                status: self.status,
                stderr: if self.stderr.trim().is_empty() {
                    self.stdout.trim().to_string()
                } else {
                    self.stderr.trim().to_string()
                },
            })
        }
    }
}

fn base(program: &str) -> Command {
    let mut c = Command::new(program);
    // wsl.exe prints UTF-16LE by default; this switches it to UTF-8 (WSL >= 0.64).
    c.env("WSL_UTF8", "1");
    c.stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    #[cfg(windows)]
    c.creation_flags(CREATE_NO_WINDOW);
    c
}

/// Decode process output: UTF-8 normally, UTF-16LE if wsl.exe ignored WSL_UTF8.
fn decode(bytes: &[u8]) -> String {
    let looks_utf16 = bytes.len() >= 4 && bytes.iter().skip(1).step_by(2).take(8).all(|&b| b == 0);
    let s = if looks_utf16 {
        let units: Vec<u16> = bytes
            .chunks(2)
            .map(|c| u16::from_le_bytes([c[0], *c.get(1).unwrap_or(&0)]))
            .collect();
        String::from_utf16_lossy(&units)
    } else {
        String::from_utf8_lossy(bytes).into_owned()
    };
    s.replace(['\r', '\0'], "")
}

pub async fn run(program: &str, args: &[&str]) -> Result<Output> {
    let out = base(program)
        .args(args)
        .output()
        .await
        .map_err(|e| Error::Spawn(program.to_string(), e.to_string()))?;
    Ok(Output {
        status: out.status.code().unwrap_or(-1),
        stdout: decode(&out.stdout),
        stderr: decode(&out.stderr),
    })
}

pub async fn wsl(args: &[&str]) -> Result<Output> {
    run("wsl.exe", args).await
}

/// Run a shell snippet inside our distro as root and wait for it.
pub async fn sh(script: &str) -> Result<Output> {
    wsl(&["-d", DISTRO, "-u", "root", "--", "bash", "-c", script]).await
}

/// Spawn a long-running shell snippet inside our distro; caller streams its output.
pub fn spawn_sh(script: &str) -> Result<Child> {
    let mut c = base("wsl.exe");
    c.args(["-d", DISTRO, "-u", "root", "--", "bash", "-c", script]);
    c.spawn()
        .map_err(|e| Error::Spawn("wsl.exe".into(), e.to_string()))
}

/// Feed a whole script over stdin (`bash -s`) so quoting never bites us.
pub async fn spawn_script(script: &str) -> Result<Child> {
    let mut c = base("wsl.exe");
    c.args(["-d", DISTRO, "-u", "root", "--", "bash", "-s"]);
    c.stdin(Stdio::piped());
    let mut child = c
        .spawn()
        .map_err(|e| Error::Spawn("wsl.exe".into(), e.to_string()))?;
    if let Some(mut stdin) = child.stdin.take() {
        stdin.write_all(script.as_bytes()).await?;
        stdin.shutdown().await?;
    }
    Ok(child)
}

/// Spawn a Windows-side process with streamed output (e.g. `wsl --install`).
pub fn spawn_win(program: &str, args: &[&str]) -> Result<Child> {
    let mut c = base(program);
    c.args(args);
    c.spawn()
        .map_err(|e| Error::Spawn(program.to_string(), e.to_string()))
}

/// Keep a distro alive: WSL shuts a distro down a few seconds after its last
/// client exits, which would kill Docker and Ollama. Holding one idle process
/// open is the standard trick (Docker Desktop does the same).
pub fn spawn_keepalive() -> Result<Child> {
    let mut c = base("wsl.exe");
    c.args(["-d", DISTRO, "-u", "root", "--", "sleep", "infinity"]);
    c.stdout(Stdio::null()).stderr(Stdio::null());
    c.spawn()
        .map_err(|e| Error::Spawn("wsl.exe".into(), e.to_string()))
}

/// Drain stdout and stderr of a child line by line, in arrival order, until it
/// exits. Returns the exit code.
pub async fn stream_lines<F>(mut child: Child, mut on_line: F) -> Result<i32>
where
    F: FnMut(String),
{
    let (tx, mut rx) = mpsc::unbounded_channel::<String>();
    if let Some(out) = child.stdout.take() {
        let tx = tx.clone();
        tokio::spawn(async move {
            let mut lines = BufReader::new(out).lines();
            while let Ok(Some(l)) = lines.next_line().await {
                let _ = tx.send(clean(l));
            }
        });
    }
    if let Some(err) = child.stderr.take() {
        let tx = tx.clone();
        tokio::spawn(async move {
            let mut lines = BufReader::new(err).lines();
            while let Ok(Some(l)) = lines.next_line().await {
                let _ = tx.send(clean(l));
            }
        });
    }
    drop(tx);
    while let Some(l) = rx.recv().await {
        if !l.trim().is_empty() {
            on_line(l);
        }
    }
    let status = child.wait().await?;
    Ok(status.code().unwrap_or(-1))
}

/// Strip carriage returns and ANSI escapes (progress bars) so logs stay readable.
fn clean(l: String) -> String {
    static ANSI: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
    let re = ANSI.get_or_init(|| {
        regex::Regex::new(r"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b[()][A-Za-z0-9]").unwrap()
    });
    let l = l.replace(['\r', '\0'], "");
    re.replace_all(&l, "").trim_end().to_string()
}

/// `owner/name` -> `owner-name`, lowercase, safe for container names and DOM ids.
pub fn slug(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut last_dash = true;
    for ch in s.chars() {
        let c = ch.to_ascii_lowercase();
        if c.is_ascii_alphanumeric() {
            out.push(c);
            last_dash = false;
        } else if !last_dash {
            out.push('-');
            last_dash = true;
        }
    }
    out.trim_end_matches('-').to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slug_is_container_safe() {
        assert_eq!(slug("openai/whisper"), "openai-whisper");
        assert_eq!(
            slug("Qwen/Qwen3-8B-GGUF:Q4_K_M"),
            "qwen-qwen3-8b-gguf-q4-k-m"
        );
    }

    #[test]
    fn decodes_utf16() {
        let s: Vec<u8> = "ab".encode_utf16().flat_map(|u| u.to_le_bytes()).collect();
        let mut long = s.clone();
        long.extend_from_slice(&s);
        long.extend_from_slice(&s);
        assert_eq!(decode(&long), "ababab");
        assert_eq!(decode(b"plain\r\n"), "plain\n");
    }

    #[test]
    fn strips_ansi() {
        assert_eq!(
            clean("\u{1b}[?25lpulling 37%\u{1b}[K".into()),
            "pulling 37%"
        );
    }
}
