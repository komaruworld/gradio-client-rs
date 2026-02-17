use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use gradio_client_rs::{
    handle_file, ApiView, CallOptions, Client, ClientOptions, Error, Status, ViewApiOptions,
    ViewApiReturnFormat,
};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use tokio::sync::{Mutex, Semaphore};
use tokio::task::JoinSet;

const TOKENS_PATH: &str = "tokens.json";
const SAMPLE_IMAGE_URL: &str =
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/coco_sample.png";

const SPACES: [&str; 12] = [
    "linoyts/Qwen-Image-Edit-2511-Fast",
    "Qwen/Qwen-Image-Edit-2511",
    "dream2589632147/Dream-wan2-2-faster-Pro",
    "Heartsync/NSFW-Uncensored-video2",
    "Lightricks/ltx-2-distilled",
    "multimodalart/wan-2-2-first-last-frame",
    "mrfakename/Z-Image-Turbo",
    "Qwen/Qwen-Image-2512",
    "V0pr0S/ComfyUI-Reactor-Fast-Face-Swap-CPU",
    "not-lain/background-removal",
    "amd/gpt-oss-120b-chatbot",
    "mrfakename/HeartMuLa",
];

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TokenEntry {
    token: String,
    #[serde(default)]
    invalid: bool,
    #[serde(default)]
    invalid_reason: Option<String>,
    #[serde(default)]
    invalidated_at_unix: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct TokenFile {
    tokens: Vec<TokenEntry>,
}

#[derive(Debug, Clone)]
struct TokenManager {
    path: String,
    file: TokenFile,
}

impl TokenManager {
    fn load(path: &str) -> Result<Self, String> {
        let raw =
            std::fs::read_to_string(path).map_err(|err| format!("failed to read {path}: {err}"))?;

        if let Ok(file) = serde_json::from_str::<TokenFile>(&raw) {
            return Ok(Self {
                path: path.to_string(),
                file,
            });
        }

        let legacy = serde_json::from_str::<Vec<String>>(&raw)
            .map_err(|err| format!("failed to parse {path}: {err}"))?;
        let tokens = legacy
            .into_iter()
            .map(|token| TokenEntry {
                token,
                invalid: false,
                invalid_reason: None,
                invalidated_at_unix: None,
            })
            .collect();

        Ok(Self {
            path: path.to_string(),
            file: TokenFile { tokens },
        })
    }

    fn save(&self) -> Result<(), String> {
        let encoded = serde_json::to_string_pretty(&self.file)
            .map_err(|err| format!("failed to encode tokens file: {err}"))?;
        std::fs::write(&self.path, format!("{encoded}\n"))
            .map_err(|err| format!("failed to write {}: {err}", self.path))
    }

    fn next_active(&self) -> Option<TokenCandidate> {
        let index = self.file.tokens.iter().position(|entry| !entry.invalid)?;
        let token = self.file.tokens.get(index)?.token.clone();
        Some(TokenCandidate { index, token })
    }

    fn active_count(&self) -> usize {
        self.file
            .tokens
            .iter()
            .filter(|entry| !entry.invalid)
            .count()
    }

    fn mark_invalid(&mut self, index: usize, reason: &str) {
        if let Some(entry) = self.file.tokens.get_mut(index) {
            entry.invalid = true;
            entry.invalid_reason = Some(reason.to_string());
            entry.invalidated_at_unix = Some(
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .map(|dur| dur.as_secs())
                    .unwrap_or(0),
            );
        }
    }
}

#[derive(Debug, Clone)]
struct TokenCandidate {
    index: usize,
    token: String,
}

#[derive(Debug, Clone)]
struct EndpointJob {
    space: String,
    api_name: String,
    params: Vec<Value>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EndpointStatus {
    Done,
    Skipped,
    Failed,
}

#[derive(Debug, Clone)]
struct EndpointOutcome {
    space: String,
    api_name: String,
    status: EndpointStatus,
    detail: String,
}

impl EndpointOutcome {
    fn done(space: String, api_name: String, detail: String) -> Self {
        Self {
            space,
            api_name,
            status: EndpointStatus::Done,
            detail,
        }
    }

    fn skipped(space: String, api_name: String, detail: String) -> Self {
        Self {
            space,
            api_name,
            status: EndpointStatus::Skipped,
            detail,
        }
    }

    fn failed(space: String, api_name: String, detail: String) -> Self {
        Self {
            space,
            api_name,
            status: EndpointStatus::Failed,
            detail,
        }
    }
}

#[derive(Debug, Clone)]
enum SubmitProbe {
    Done(String),
    RotateToken,
    Failed(String),
}

#[derive(Debug, Clone)]
enum DiscoveryResult {
    Ready(Vec<EndpointJob>),
    Skipped(String),
    Failed(String),
}

#[tokio::main]
async fn main() {
    let token_manager = match TokenManager::load(TOKENS_PATH) {
        Ok(manager) => manager,
        Err(err) => {
            eprintln!("{err}");
            std::process::exit(1);
        }
    };

    let initial_active = token_manager.active_count();
    let tokens = Arc::new(Mutex::new(token_manager));

    let concurrency = std::env::var("CHECK_CONCURRENCY")
        .ok()
        .and_then(|raw| raw.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(6);

    println!(
        "Starting async API check: spaces={}, concurrency={}, active_tokens={}",
        SPACES.len(),
        concurrency,
        initial_active
    );

    let mut jobs = Vec::<EndpointJob>::new();
    let mut skipped_discovery = Vec::<String>::new();
    let mut failed_discovery = Vec::<String>::new();

    for space in SPACES {
        match discover_space_endpoints(space, tokens.clone()).await {
            DiscoveryResult::Ready(mut found) => {
                println!("[DISCOVERY] {space}: {} endpoint(s)", found.len());
                jobs.append(&mut found);
            }
            DiscoveryResult::Skipped(reason) => {
                println!("[DISCOVERY SKIP] {space}: {reason}");
                skipped_discovery.push(format!("{space}: {reason}"));
            }
            DiscoveryResult::Failed(reason) => {
                println!("[DISCOVERY FAIL] {space}: {reason}");
                failed_discovery.push(format!("{space}: {reason}"));
            }
        }
    }

    let semaphore = Arc::new(Semaphore::new(concurrency));
    let mut join_set = JoinSet::new();

    for job in jobs {
        let sem = semaphore.clone();
        let shared_tokens = tokens.clone();
        join_set.spawn(async move {
            match sem.acquire_owned().await {
                Ok(_permit) => probe_endpoint(job, shared_tokens).await,
                Err(_) => EndpointOutcome::failed(
                    job.space,
                    job.api_name,
                    "semaphore closed unexpectedly".to_string(),
                ),
            }
        });
    }

    let mut outcomes = Vec::<EndpointOutcome>::new();
    while let Some(joined) = join_set.join_next().await {
        match joined {
            Ok(outcome) => {
                let tag = match outcome.status {
                    EndpointStatus::Done => "DONE",
                    EndpointStatus::Skipped => "SKIP",
                    EndpointStatus::Failed => "FAIL",
                };
                println!(
                    "[{tag}] {} {} :: {}",
                    outcome.space, outcome.api_name, outcome.detail
                );
                outcomes.push(outcome);
            }
            Err(err) => {
                outcomes.push(EndpointOutcome::failed(
                    "<task>".to_string(),
                    "<unknown>".to_string(),
                    format!("join error: {err}"),
                ));
            }
        }
    }

    let save_result = {
        let guard = tokens.lock().await;
        guard.save()
    };
    if let Err(err) = save_result {
        println!("WARN: failed to save tokens file: {err}");
    }

    let done = outcomes
        .iter()
        .filter(|outcome| outcome.status == EndpointStatus::Done)
        .count();
    let skipped = outcomes
        .iter()
        .filter(|outcome| outcome.status == EndpointStatus::Skipped)
        .count();
    let failed = outcomes
        .iter()
        .filter(|outcome| outcome.status == EndpointStatus::Failed)
        .count();

    println!(
        "\nSummary: endpoints_total={}, done={}, skipped={}, failed={}, discovery_skipped_spaces={}, discovery_failed_spaces={}",
        outcomes.len(),
        done,
        skipped,
        failed,
        skipped_discovery.len(),
        failed_discovery.len()
    );

    if !failed_discovery.is_empty() {
        println!("Discovery failures:");
        for item in &failed_discovery {
            println!("- {item}");
        }
    }

    if failed > 0 {
        println!("Endpoint failures:");
        for item in outcomes
            .iter()
            .filter(|outcome| outcome.status == EndpointStatus::Failed)
        {
            println!("- {} {}: {}", item.space, item.api_name, item.detail);
        }
        std::process::exit(1);
    }
}

async fn discover_space_endpoints(
    space: &str,
    tokens: Arc<Mutex<TokenManager>>,
) -> DiscoveryResult {
    let mut attempts = 0usize;

    loop {
        attempts += 1;
        if attempts > 64 {
            return DiscoveryResult::Skipped("attempt limit reached".to_string());
        }

        let Some(candidate) = next_active_token(tokens.clone()).await else {
            return DiscoveryResult::Skipped("no active tokens left".to_string());
        };

        let client = match init_client(space, Some(candidate.token.as_str())).await {
            Ok(client) => client,
            Err(err) => {
                if is_auth_like_error(&err) {
                    invalidate_token(tokens.clone(), candidate, "auth_error").await;
                    continue;
                }
                return DiscoveryResult::Failed(format!("init failed: {err}"));
            }
        };

        let view = client
            .view_api(ViewApiOptions {
                all_endpoints: Some(true),
                print_info: false,
                return_format: ViewApiReturnFormat::Dict,
            })
            .await;

        let result = match view {
            Ok(ApiView::Dict(value)) => {
                let named = value
                    .get("named_endpoints")
                    .and_then(Value::as_object)
                    .cloned()
                    .unwrap_or_default();
                let mut jobs = Vec::with_capacity(named.len());
                for (api_name, endpoint) in named {
                    let params = endpoint
                        .get("parameters")
                        .and_then(Value::as_array)
                        .cloned()
                        .unwrap_or_default();
                    jobs.push(EndpointJob {
                        space: space.to_string(),
                        api_name,
                        params,
                    });
                }
                if jobs.is_empty() {
                    DiscoveryResult::Failed("named_endpoints is empty".to_string())
                } else {
                    DiscoveryResult::Ready(jobs)
                }
            }
            Ok(_) => DiscoveryResult::Failed("view_api did not return dict".to_string()),
            Err(err) => {
                let message = err.to_string();
                if is_auth_like_error(&message) {
                    invalidate_token(tokens.clone(), candidate, "auth_error").await;
                    client.close().await;
                    continue;
                }
                DiscoveryResult::Failed(format!("view_api failed: {message}"))
            }
        };

        client.close().await;
        return result;
    }
}

async fn probe_endpoint(job: EndpointJob, tokens: Arc<Mutex<TokenManager>>) -> EndpointOutcome {
    let kwargs = match build_required_kwargs(&job.params) {
        Ok(kwargs) => kwargs,
        Err(err) => {
            return EndpointOutcome::failed(
                job.space,
                job.api_name,
                format!("kwargs build failed: {err}"),
            );
        }
    };

    let mut attempts = 0usize;
    loop {
        attempts += 1;
        if attempts > 96 {
            return EndpointOutcome::skipped(
                job.space,
                job.api_name,
                "attempt limit reached".to_string(),
            );
        }

        let Some(candidate) = next_active_token(tokens.clone()).await else {
            return EndpointOutcome::skipped(
                job.space,
                job.api_name,
                "no active tokens left".to_string(),
            );
        };

        let client = match init_client(&job.space, Some(candidate.token.as_str())).await {
            Ok(client) => client,
            Err(err) => {
                if is_auth_like_error(&err) {
                    invalidate_token(tokens.clone(), candidate, "auth_error").await;
                    continue;
                }
                return EndpointOutcome::failed(
                    job.space,
                    job.api_name,
                    format!("init failed: {err}"),
                );
            }
        };

        let probe = run_submit_probe(&client, &job.api_name, kwargs.clone()).await;
        client.close().await;

        match probe {
            SubmitProbe::Done(detail) => {
                return EndpointOutcome::done(job.space, job.api_name, detail);
            }
            SubmitProbe::RotateToken => {
                invalidate_token(tokens.clone(), candidate, "gpu_quota_or_duration").await;
                continue;
            }
            SubmitProbe::Failed(detail) => {
                return EndpointOutcome::failed(job.space, job.api_name, detail);
            }
        }
    }
}

async fn next_active_token(tokens: Arc<Mutex<TokenManager>>) -> Option<TokenCandidate> {
    let guard = tokens.lock().await;
    guard.next_active()
}

async fn invalidate_token(
    tokens: Arc<Mutex<TokenManager>>,
    candidate: TokenCandidate,
    reason: &str,
) {
    let mut guard = tokens.lock().await;
    guard.mark_invalid(candidate.index, reason);
    if let Err(err) = guard.save() {
        println!(
            "WARN: failed to persist token {}: {err}",
            mask_token(&candidate.token)
        );
    }
}

async fn run_submit_probe(
    client: &Client,
    api_name: &str,
    kwargs: Map<String, Value>,
) -> SubmitProbe {
    let mut options = CallOptions::default();
    options.api_name = Some(api_name.to_string());
    options.kwargs = kwargs;
    options.request.timeout = Some(Duration::from_secs(120));

    let job = match client.submit(vec![], options).await {
        Ok(job) => job,
        Err(err) => return SubmitProbe::Failed(format!("submit() returned error: {err}")),
    };

    let started = tokio::time::Instant::now();
    let max_wait = Duration::from_secs(35);

    loop {
        if started.elapsed() >= max_wait {
            let _ = job.cancel().await;
            return SubmitProbe::Failed(
                "no status/result change within 35s after submit".to_string(),
            );
        }

        if let Ok(result) = tokio::time::timeout(Duration::from_millis(300), job.result()).await {
            return match result {
                Ok(value) => SubmitProbe::Done(format!(
                    "job completed (result type: {})",
                    value_type(&value)
                )),
                Err(Error::App { message, .. }) => {
                    if is_quota_or_duration_message(&message) {
                        SubmitProbe::RotateToken
                    } else {
                        SubmitProbe::Done("app responded".to_string())
                    }
                }
                Err(Error::Validation(_)) => SubmitProbe::Done("validation response".to_string()),
                Err(Error::QueueFull) => SubmitProbe::Done("queue full response".to_string()),
                Err(err) => SubmitProbe::Failed(format!("job finished with error: {err}")),
            };
        }

        let status = job.status().await;
        if status.code != Status::Starting {
            let _ = job.cancel().await;
            return SubmitProbe::Done(format!("job accepted; status={:?}", status.code));
        }

        tokio::time::sleep(Duration::from_secs(2)).await;
    }
}

fn build_required_kwargs(params: &[Value]) -> Result<Map<String, Value>, String> {
    let mut kwargs = Map::new();
    for param in params {
        let required = !param
            .get("parameter_has_default")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        if !required {
            continue;
        }

        let name = param
            .get("parameter_name")
            .and_then(Value::as_str)
            .or_else(|| param.get("label").and_then(Value::as_str))
            .ok_or_else(|| "required parameter has no name".to_string())?
            .to_string();

        let value = infer_value(param)?;
        kwargs.insert(name, value);
    }

    Ok(kwargs)
}

fn infer_value(param: &Value) -> Result<Value, String> {
    let component = param
        .get("component")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_ascii_lowercase();
    let label = param
        .get("parameter_name")
        .and_then(Value::as_str)
        .or_else(|| param.get("label").and_then(Value::as_str))
        .unwrap_or("")
        .to_ascii_lowercase();
    let py_type = param
        .get("python_type")
        .and_then(Value::as_object)
        .and_then(|obj| obj.get("type"))
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_ascii_lowercase();

    if component.contains("gallery") && py_type.contains("list[dict(image") {
        let image = handle_file(SAMPLE_IMAGE_URL).map_err(|err| err.to_string())?;
        return Ok(Value::Array(vec![serde_json::json!({
            "image": image,
            "caption": "sample",
        })]));
    }

    if component.contains("image")
        || label.contains("image")
        || py_type.contains("dict(path")
        || py_type.contains("filepath")
    {
        return handle_file(SAMPLE_IMAGE_URL).map_err(|err| err.to_string());
    }

    if let Some(literal) = first_literal(&py_type) {
        return Ok(Value::String(literal));
    }

    if py_type.contains("bool") {
        return Ok(Value::Bool(false));
    }
    if py_type.contains("int") {
        return Ok(Value::Number(1.into()));
    }
    if py_type.contains("float") || py_type.contains("number") {
        return Ok(serde_json::json!(1.0));
    }
    if py_type.contains("list[") {
        return Ok(Value::Array(vec![]));
    }

    Ok(Value::String("test".to_string()))
}

fn first_literal(py_type: &str) -> Option<String> {
    let start = py_type.find("literal[")?;
    let tail = &py_type[start + "literal[".len()..];
    let first_quote = tail.find('\'')?;
    let rest = &tail[first_quote + 1..];
    let second_quote = rest.find('\'')?;
    Some(rest[..second_quote].to_string())
}

fn value_type(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "bool",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

async fn init_client(space: &str, token: Option<&str>) -> Result<Client, String> {
    let mut options = ClientOptions::default();
    options.verbose = false;
    options.token = token.map(ToOwned::to_owned);

    Client::new(space, options)
        .await
        .map_err(|err| err.to_string())
}

fn is_auth_like_error(message: &str) -> bool {
    let normalized = message.to_ascii_lowercase();
    normalized.contains("authentication")
        || normalized.contains("unauthorized")
        || normalized.contains("credentials")
        || normalized.contains("token")
        || normalized.contains("forbidden")
}

fn is_quota_or_duration_message(message: &str) -> bool {
    let normalized = message.to_ascii_lowercase();
    normalized.contains("exceeded your gpu quota")
        || normalized.contains("requested gpu duration")
        || (normalized.contains("quota") && normalized.contains("gpu"))
        || (normalized.contains("duration") && normalized.contains("maximum allowed"))
}

fn mask_token(token: &str) -> String {
    if token.len() <= 10 {
        return "***".to_string();
    }
    let suffix = &token[token.len().saturating_sub(6)..];
    format!("***{suffix}")
}
