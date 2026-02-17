use std::time::SystemTime;

use serde::{Deserialize, Serialize};
use serde_json::Value;

pub const API_URL: &str = "api/predict/";
pub const SSE_URL_V0: &str = "queue/join";
pub const SSE_DATA_URL_V0: &str = "queue/data";
pub const SSE_URL: &str = "queue/data";
pub const SSE_DATA_URL: &str = "queue/join";
pub const UPLOAD_URL: &str = "upload";
pub const LOGIN_URL: &str = "login";
pub const CONFIG_URL: &str = "config";
pub const RAW_API_INFO_URL: &str = "info?serialize=False";
pub const SPACE_FETCHER_URL: &str = "https://gradio-space-api-fetcher-v2.hf.space/api";
pub const RESET_URL: &str = "reset";
pub const HEARTBEAT_URL: &str = "heartbeat/{session_hash}";
pub const CANCEL_URL: &str = "cancel";

pub const INVALID_RUNTIME: &[&str] = &[
    "NO_APP_FILE",
    "CONFIG_ERROR",
    "BUILD_ERROR",
    "RUNTIME_ERROR",
    "PAUSED",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Protocol {
    Sse,
    SseV1,
    SseV2,
    SseV21,
    SseV3,
    Ws,
}

impl Protocol {
    pub fn from_str(raw: &str) -> Self {
        match raw {
            "sse" => Self::Sse,
            "sse_v1" => Self::SseV1,
            "sse_v2" => Self::SseV2,
            "sse_v2.1" => Self::SseV21,
            "sse_v3" => Self::SseV3,
            _ => Self::Ws,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum Status {
    Starting,
    JoiningQueue,
    QueueFull,
    InQueue,
    SendingData,
    Processing,
    Iterating,
    Progress,
    Finished,
    Cancelled,
    Log,
}

impl Status {
    pub fn from_server_msg(msg: &str) -> Option<Self> {
        match msg {
            "send_hash" => Some(Self::JoiningQueue),
            "queue_full" => Some(Self::QueueFull),
            "estimation" => Some(Self::InQueue),
            "send_data" => Some(Self::SendingData),
            "process_starts" => Some(Self::Processing),
            "process_generating" => Some(Self::Iterating),
            "process_completed" => Some(Self::Finished),
            "progress" => Some(Self::Progress),
            "log" => Some(Self::Log),
            "Server stopped unexpectedly." => Some(Self::Finished),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ProgressUnit {
    pub index: Option<i64>,
    pub length: Option<i64>,
    pub unit: Option<String>,
    pub progress: Option<f64>,
    pub desc: Option<String>,
}

#[derive(Debug, Clone)]
pub struct StatusUpdate {
    pub code: Status,
    pub rank: Option<i64>,
    pub queue_size: Option<i64>,
    pub eta: Option<f64>,
    pub success: Option<bool>,
    pub time: SystemTime,
    pub progress_data: Option<Vec<ProgressUnit>>,
    pub log: Option<(String, String)>,
}

impl Default for StatusUpdate {
    fn default() -> Self {
        Self {
            code: Status::Starting,
            rank: None,
            queue_size: None,
            eta: None,
            success: None,
            time: SystemTime::now(),
            progress_data: None,
            log: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct OutputUpdate {
    pub outputs: Value,
    pub success: bool,
    pub final_output: bool,
}

#[derive(Debug, Clone)]
pub enum Update {
    Status(StatusUpdate),
    Output(OutputUpdate),
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ApiInfo {
    #[serde(default)]
    pub named_endpoints: serde_json::Map<String, Value>,
    #[serde(default)]
    pub unnamed_endpoints: serde_json::Map<String, Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ParameterInfo {
    pub label: String,
    #[serde(default)]
    pub parameter_name: Option<String>,
    #[serde(default)]
    pub parameter_has_default: Option<bool>,
    #[serde(default)]
    pub parameter_default: Option<Value>,
    #[serde(default)]
    pub python_type: Option<PythonTypeInfo>,
    #[serde(default)]
    pub component: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PythonTypeInfo {
    #[serde(default)]
    pub r#type: Option<String>,
    #[serde(default)]
    pub description: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Config {
    #[serde(default)]
    pub protocol: Option<String>,
    #[serde(default)]
    pub api_prefix: String,
    #[serde(default)]
    pub version: Option<String>,
    #[serde(default)]
    pub dependencies: Vec<Dependency>,
    #[serde(default)]
    pub components: Vec<Component>,
    #[serde(default)]
    pub max_file_size: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Dependency {
    #[serde(default)]
    pub id: Option<i64>,
    #[serde(default)]
    pub api_name: Option<Value>,
    #[serde(default)]
    pub inputs: Vec<i64>,
    #[serde(default)]
    pub outputs: Vec<i64>,
    #[serde(default)]
    pub backend_fn: Option<bool>,
    #[serde(default)]
    pub api_visibility: Option<String>,
    #[serde(default)]
    pub show_api: Option<bool>,
    #[serde(default)]
    pub cancels: Vec<i64>,
}

impl Dependency {
    pub fn api_name_as_string(&self) -> Option<String> {
        self.api_name
            .as_ref()
            .and_then(|v| v.as_str().map(|s| format!("/{s}")))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Component {
    pub id: i64,
    #[serde(rename = "type")]
    pub component_type: String,
    #[serde(default)]
    pub skip_api: Option<bool>,
    #[serde(default)]
    pub api_info: Option<Value>,
    #[serde(default)]
    pub label: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ComponentApiType {
    pub skip: bool,
    pub is_state: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct FileData {
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub data: Option<String>,
    #[serde(default)]
    pub size: Option<u64>,
    #[serde(default)]
    pub is_file: Option<bool>,
    #[serde(default)]
    pub orig_name: Option<String>,
    #[serde(default)]
    pub mime_type: Option<String>,
    #[serde(default)]
    pub is_stream: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ServerMessage {
    pub msg: String,
    #[serde(default)]
    pub output: Value,
    #[serde(default)]
    pub event_id: Option<String>,
    #[serde(default)]
    pub rank: Option<i64>,
    #[serde(default)]
    pub rank_eta: Option<f64>,
    #[serde(default)]
    pub queue_size: Option<i64>,
    #[serde(default)]
    pub success: Option<bool>,
    #[serde(default)]
    pub progress_data: Option<Vec<ProgressUnit>>,
    #[serde(default)]
    pub log: Option<String>,
    #[serde(default)]
    pub level: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SpaceInfoResponse {
    #[serde(default)]
    pub host: Option<String>,
    #[serde(default)]
    pub private: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SpaceRuntimeResponse {
    pub stage: String,
    #[serde(default)]
    pub hardware: Option<SpaceRuntimeHardware>,
    #[serde(default, rename = "gcTimeout")]
    pub sleep_time_seconds: Option<i64>,
}

impl SpaceRuntimeResponse {
    pub fn current_hardware(&self) -> Option<&str> {
        self.hardware.as_ref().and_then(|hw| hw.current.as_deref())
    }

    pub fn requested_hardware(&self) -> Option<&str> {
        self.hardware
            .as_ref()
            .and_then(|hw| hw.requested.as_deref())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SpaceRuntimeHardware {
    #[serde(default)]
    pub current: Option<String>,
    #[serde(default)]
    pub requested: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct WhoAmIResponse {
    pub name: String,
}
