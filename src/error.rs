use thiserror::Error;

#[derive(Debug, Clone, Error)]
pub enum Error {
    #[error("http error: {0}")]
    Http(String),
    #[error("io error: {0}")]
    Io(String),
    #[error("json error: {0}")]
    Json(String),
    #[error("url error: {0}")]
    Url(String),
    #[error("join error: {0}")]
    Join(String),
    #[error("authentication error: {0}")]
    Authentication(String),
    #[error("validation error: {0}")]
    Validation(String),
    #[error("app error: {message}")]
    App {
        message: String,
        raw: serde_json::Value,
    },
    #[error("queue is full")]
    QueueFull,
    #[error("too many requests: {0}")]
    TooManyRequests(String),
    #[error("invalid api endpoint: {0}")]
    InvalidApiEndpoint(String),
    #[error("unsupported protocol: {0}")]
    UnsupportedProtocol(String),
    #[error("api error: {0}")]
    Api(String),
    #[error("cancelled")]
    Cancelled,
    #[error("missing field: {0}")]
    MissingField(String),
    #[error("hugging face error: {0}")]
    HuggingFace(String),
    #[error("invalid source: {0}")]
    InvalidSource(String),
}

impl From<reqwest::Error> for Error {
    fn from(value: reqwest::Error) -> Self {
        Self::Http(value.to_string())
    }
}

impl From<std::io::Error> for Error {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value.to_string())
    }
}

impl From<serde_json::Error> for Error {
    fn from(value: serde_json::Error) -> Self {
        Self::Json(value.to_string())
    }
}

impl From<url::ParseError> for Error {
    fn from(value: url::ParseError) -> Self {
        Self::Url(value.to_string())
    }
}

impl From<tokio::task::JoinError> for Error {
    fn from(value: tokio::task::JoinError) -> Self {
        Self::Join(value.to_string())
    }
}

pub type Result<T> = std::result::Result<T, Error>;
