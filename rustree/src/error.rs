//! Unified error type for the rustree library.

use std::fmt;

/// Unified error type for all rustree operations.
#[derive(Debug)]
pub enum RustreeError {
    /// Newick or RecPhyloXML parsing errors.
    Parse(String),
    /// Invalid parameters or input values.
    Validation(String),
    /// Node index out of bounds or missing mapping.
    Index(String),
    /// Tree structure errors (missing root, invalid topology).
    Tree(String),
    /// DTL simulation errors (no valid recipients, exceeded retries).
    Simulation(String),
    /// File I/O errors.
    Io(std::io::Error),
    /// XML parsing errors (from quick_xml).
    Xml(String),
    /// External tool errors (ALERax, etc.).
    ExternalTool(String),
}

impl fmt::Display for RustreeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RustreeError::Parse(msg) => write!(f, "Parse error: {msg}"),
            RustreeError::Validation(msg) => write!(f, "Validation error: {msg}"),
            RustreeError::Index(msg) => write!(f, "Index error: {msg}"),
            RustreeError::Tree(msg) => write!(f, "Tree error: {msg}"),
            RustreeError::Simulation(msg) => write!(f, "Simulation error: {msg}"),
            RustreeError::Io(err) => write!(f, "IO error: {err}"),
            RustreeError::Xml(msg) => write!(f, "XML error: {msg}"),
            RustreeError::ExternalTool(msg) => write!(f, "External tool error: {msg}"),
        }
    }
}

impl std::error::Error for RustreeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            RustreeError::Io(err) => Some(err),
            _ => None,
        }
    }
}

// --- From conversions for common error sources ---

impl From<std::io::Error> for RustreeError {
    fn from(err: std::io::Error) -> Self {
        RustreeError::Io(err)
    }
}

impl From<quick_xml::Error> for RustreeError {
    fn from(err: quick_xml::Error) -> Self {
        RustreeError::Xml(err.to_string())
    }
}

impl From<crate::io::recphyloxml::ParseError> for RustreeError {
    fn from(err: crate::io::recphyloxml::ParseError) -> Self {
        match err {
            crate::io::recphyloxml::ParseError::XmlError(e) => RustreeError::Xml(e.to_string()),
            crate::io::recphyloxml::ParseError::IoError(e) => RustreeError::Io(e),
            crate::io::recphyloxml::ParseError::MissingSection(s) => RustreeError::Parse(s),
            crate::io::recphyloxml::ParseError::InvalidFormat(s) => RustreeError::Parse(s),
            crate::io::recphyloxml::ParseError::MissingSpecies(s) => RustreeError::Parse(s),
            crate::io::recphyloxml::ParseError::InvalidEvent(s) => RustreeError::Parse(s),
        }
    }
}

/// Convenience conversion so existing `Result<_, String>` code can be migrated incrementally.
/// Functions returning `Result<_, RustreeError>` can call `.map_err(RustreeError::from_string_error)`
/// on inner calls that still return `Result<_, String>`.
impl RustreeError {
    pub fn from_string_error(s: String) -> Self {
        RustreeError::Tree(s)
    }
}

// --- Binding-layer conversions (feature-gated) ---

#[cfg(feature = "python")]
impl From<RustreeError> for pyo3::PyErr {
    fn from(err: RustreeError) -> pyo3::PyErr {
        pyo3::exceptions::PyRuntimeError::new_err(err.to_string())
    }
}

#[cfg(feature = "r")]
impl From<RustreeError> for extendr_api::Error {
    fn from(err: RustreeError) -> extendr_api::Error {
        extendr_api::Error::Other(err.to_string())
    }
}
