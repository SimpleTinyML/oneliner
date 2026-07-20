//! Resolve the LLVM triple used by a Rust compilation target.
//!
//! This is the Rust equivalent of:
//!
//! ```text
//! RUSTC_BOOTSTRAP=1 rustc -Z unstable-options --print target-spec-json --target <target>
//! ```
//!
//! followed by reading the `llvm-target`, `features`, and `cpu` fields from the
//! emitted JSON.

use std::ffi::OsString;
use std::fmt;
use std::process::Command;

/// LLVM-related fields from a Rust target spec.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LlvmTargetInfo {
    pub llvm_triple: String,
    pub features: Option<String>,
    pub cpu: Option<String>,
}

/// Errors that can occur while resolving a target's LLVM fields.
#[derive(Debug)]
pub enum TargetInfoError {
    RustcIo(std::io::Error),
    RustcFailed { status: Option<i32>, stderr: String },
    InvalidJson(serde_json::Error),
    MissingLlvmTarget,
}

impl fmt::Display for TargetInfoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RustcIo(err) => write!(f, "failed to run rustc: {err}"),
            Self::RustcFailed { status, stderr } => {
                write!(f, "rustc failed with status {status:?}: {}", stderr.trim())
            }
            Self::InvalidJson(err) => write!(f, "rustc emitted invalid target spec JSON: {err}"),
            Self::MissingLlvmTarget => write!(f, "target spec JSON did not contain `llvm-target`"),
        }
    }
}

/// Returns the LLVM triple, features, and CPU for a Rust target.
///
/// This uses `RUSTC_BOOTSTRAP=1` because `--print target-spec-json` currently
/// requires `-Z unstable-options`.
pub fn llvm_target_info_from_rust_triple(target: &str) -> Result<LlvmTargetInfo, TargetInfoError> {
    let rustc = std::env::var_os("RUSTC").unwrap_or_else(|| OsString::from("rustc"));
    let output = Command::new(rustc)
        .env("RUSTC_BOOTSTRAP", "1")
        .args([
            "-Z",
            "unstable-options",
            "--print",
            "target-spec-json",
            "--target",
            target,
        ])
        .output()
        .map_err(TargetInfoError::RustcIo)?;

    if !output.status.success() {
        return Err(TargetInfoError::RustcFailed {
            status: output.status.code(),
            stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
        });
    }

    parse_target_spec_json(&output.stdout)
}

/// Extracts `llvm-target`, `features`, and `cpu` from a target spec JSON document.
fn parse_target_spec_json(json: &[u8]) -> Result<LlvmTargetInfo, TargetInfoError> {
    let spec: serde_json::Value =
        serde_json::from_slice(json).map_err(TargetInfoError::InvalidJson)?;

    let llvm_triple = spec
        .get("llvm-target")
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned)
        .ok_or(TargetInfoError::MissingLlvmTarget)?;

    Ok(LlvmTargetInfo {
        llvm_triple,
        features: spec
            .get("features")
            .and_then(serde_json::Value::as_str)
            .map(str::to_owned),
        cpu: spec
            .get("cpu")
            .and_then(serde_json::Value::as_str)
            .map(str::to_owned),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_llvm_target_info() {
        let json = br#"{
            "arch":"xtensa",
            "llvm-target":"xtensa-none-elf",
            "features":"+esp32s3",
            "cpu":"esp32s3"
        }"#;

        assert_eq!(
            parse_target_spec_json(json).unwrap(),
            LlvmTargetInfo {
                llvm_triple: "xtensa-none-elf".to_owned(),
                features: Some("+esp32s3".to_owned()),
                cpu: Some("esp32s3".to_owned()),
            }
        );
    }

    #[test]
    fn errors_when_llvm_target_is_missing() {
        let json = br#"{"arch":"xtensa"}"#;

        assert!(matches!(
            parse_target_spec_json(json),
            Err(TargetInfoError::MissingLlvmTarget)
        ));
    }
}
