use std::collections::{HashMap, HashSet};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use gradio_client_rs::{
    handle_file, ApiView, CallOptions, Client, ClientOptions, Error, Status, ViewApiOptions,
    ViewApiReturnFormat,
};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

const TOKENS_PATH: &str = "tokens.json";
const SAMPLE_IMAGE_URL: &str =
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/coco_sample.png";

#[derive(Clone, Copy)]
struct Case {
    group: &'static str,
    space: &'static str,
    preferred_api: &'static str,
}

const CASES: [Case; 12] = [
    Case {
        group: "image_gen",
        space: "mrfakename/Z-Image-Turbo",
        preferred_api: "/generate_image",
    },
    Case {
        group: "image_gen",
        space: "Qwen/Qwen-Image-2512",
        preferred_api: "/infer",
    },
    Case {
        group: "image_edit",
        space: "linoyts/Qwen-Image-Edit-2511-Fast",
        preferred_api: "/infer",
    },
    Case {
        group: "image_edit",
        space: "Qwen/Qwen-Image-Edit-2511",
        preferred_api: "/infer",
    },
    Case {
        group: "video_gen",
        space: "dream2589632147/Dream-wan2-2-faster-Pro",
        preferred_api: "/generate_video",
    },
    Case {
        group: "video_gen",
        space: "Lightricks/ltx-2-distilled",
        preferred_api: "/generate_video",
    },
    Case {
        group: "video_gen",
        space: "multimodalart/wan-2-2-first-last-frame",
        preferred_api: "/generate_video",
    },
    Case {
        group: "video_gen",
        space: "Heartsync/NSFW-Uncensored-video2",
        preferred_api: "/generate_video",
    },
    Case {
        group: "face_swap",
        space: "V0pr0S/ComfyUI-Reactor-Fast-Face-Swap-CPU",
        preferred_api: "/generate_image",
    },
    Case {
        group: "bg_remove",
        space: "not-lain/background-removal",
        preferred_api: "/image",
    },
    Case {
        group: "chat",
        space: "amd/gpt-oss-120b-chatbot",
        preferred_api: "/chat",
    },
    Case {
        group: "music",
        space: "mrfakename/HeartMuLa",
        preferred_api: "/generate_music",
    },
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

    fn next_active_index(&self) -> Option<usize> {
        self.file.tokens.iter().position(|entry| !entry.invalid)
    }

    fn has_active(&self) -> bool {
        self.next_active_index().is_some()
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

    fn token_at(&self, index: usize) -> Option<&str> {
        self.file
            .tokens
            .get(index)
            .map(|entry| entry.token.as_str())
    }
}

#[derive(Debug, Clone)]
enum ProbeResult {
    Passed(String),
    Quota(String),
    Failed(String),
}

impl ProbeResult {
    fn message(&self) -> &str {
        match self {
            Self::Passed(msg) | Self::Quota(msg) | Self::Failed(msg) => msg,
        }
    }

    fn is_quota(&self) -> bool {
        matches!(self, Self::Quota(_))
    }

    fn is_failed(&self) -> bool {
        matches!(self, Self::Failed(_))
    }
}

#[tokio::main]
async fn main() {
    let mut token_manager = match TokenManager::load(TOKENS_PATH) {
        Ok(manager) => manager,
        Err(err) => {
            eprintln!("{err}");
            std::process::exit(1);
        }
    };

    let mut group_done: HashSet<&'static str> = HashSet::new();
    let mut passed = 0usize;
    let mut failed = 0usize;
    let mut skipped = 0usize;
    let mut failures = Vec::new();
    let mut successes = HashMap::<&'static str, &'static str>::new();

    for case in CASES {
        if group_done.contains(case.group) {
            skipped += 1;
            println!(
                "\n=== {} [{}] ===\nSKIP: group '{}' already has a passing space ({})",
                case.space,
                case.preferred_api,
                case.group,
                successes.get(case.group).copied().unwrap_or("unknown")
            );
            continue;
        }

        println!(
            "\n=== {} [{}] group={} ===",
            case.space, case.preferred_api, case.group
        );

        let mut case_passed = false;
        let mut case_attempt_error = String::from("no attempts");

        loop {
            let token_index = token_manager.next_active_index();
            let token_value =
                token_index.and_then(|index| token_manager.token_at(index).map(ToOwned::to_owned));

            if token_index.is_none() {
                case_attempt_error = "no active tokens left".to_string();
                break;
            }

            let masked = token_value
                .as_deref()
                .map(mask_token)
                .unwrap_or_else(|| "<none>".to_string());
            println!("Attempt with token {masked}");

            let client = match init_client(case.space, token_value.as_deref()).await {
                Ok(client) => client,
                Err(err) => {
                    case_attempt_error = format!("init failed: {err}");
                    if let Some(index) = token_index {
                        if is_auth_like_error(&err) {
                            token_manager.mark_invalid(index, "auth_error");
                            let _ = token_manager.save();
                            println!(
                                "Marked token {} invalid due to auth_error",
                                mask_token(token_manager.token_at(index).unwrap_or(""))
                            );
                            continue;
                        }
                    }
                    break;
                }
            };

            let (api_name, params) = match resolve_endpoint(&client, case.preferred_api).await {
                Ok(ok) => ok,
                Err(err) => {
                    client.close().await;
                    case_attempt_error = format!("endpoint resolve failed: {err}");
                    break;
                }
            };

            println!("Using api_name={api_name}");

            let kwargs = match build_required_kwargs(&params) {
                Ok(kwargs) => kwargs,
                Err(err) => {
                    client.close().await;
                    case_attempt_error = format!("kwargs build failed: {err}");
                    break;
                }
            };

            let submit = run_submit_probe(&client, &api_name, kwargs.clone()).await;
            let quick_predict = if matches!(case.group, "chat" | "bg_remove") {
                Some(run_quick_predict(&client, &api_name, kwargs.clone()).await)
            } else {
                None
            };

            client.close().await;

            let quota_hit =
                submit.is_quota() || quick_predict.as_ref().is_some_and(ProbeResult::is_quota);

            if quota_hit {
                if let Some(index) = token_index {
                    token_manager.mark_invalid(index, "gpu_quota_or_duration");
                    if let Err(err) = token_manager.save() {
                        println!("WARN: failed to persist token invalidation: {err}");
                    }
                    println!(
                        "Token {} marked invalid due to quota/duration",
                        token_value
                            .as_deref()
                            .map(mask_token)
                            .unwrap_or_else(|| "<none>".to_string())
                    );

                    if token_manager.has_active() {
                        continue;
                    }
                    case_attempt_error = "quota hit and no active tokens left".to_string();
                    break;
                }

                case_attempt_error = "quota hit but token was not used".to_string();
                break;
            }

            let predict_failed = quick_predict.as_ref().is_some_and(ProbeResult::is_failed);
            if submit.is_failed() || predict_failed {
                let predict_msg = quick_predict
                    .as_ref()
                    .map(|result| format!("; predict={}", result.message()))
                    .unwrap_or_default();
                case_attempt_error = format!("submit={}{predict_msg}", submit.message());
                break;
            }

            let predict_msg = quick_predict
                .as_ref()
                .map(|result| format!("; predict={}", result.message()))
                .unwrap_or_default();
            println!("PASS submit={}{predict_msg}", submit.message());

            case_passed = true;
            group_done.insert(case.group);
            successes.insert(case.group, case.space);
            passed += 1;
            break;
        }

        if !case_passed {
            failed += 1;
            println!("FAIL: {case_attempt_error}");
            failures.push(format!("{}: {}", case.space, case_attempt_error));
        }
    }

    if let Err(err) = token_manager.save() {
        println!("WARN: failed to save token state: {err}");
    }

    println!("\nSummary: passed={passed}, failed={failed}, skipped={skipped}");
    if !successes.is_empty() {
        println!("Group winners:");
        for (group, space) in successes {
            println!("- {group}: {space}");
        }
    }

    if !failures.is_empty() {
        println!("Failures:");
        for failure in failures {
            println!("- {failure}");
        }
        std::process::exit(1);
    }
}

fn mask_token(token: &str) -> String {
    if token.len() <= 10 {
        return "***".to_string();
    }
    let suffix = &token[token.len().saturating_sub(6)..];
    format!("***{suffix}")
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

async fn resolve_endpoint(
    client: &Client,
    preferred_api: &str,
) -> Result<(String, Vec<Value>), String> {
    let view = client
        .view_api(ViewApiOptions {
            all_endpoints: Some(true),
            print_info: false,
            return_format: ViewApiReturnFormat::Dict,
        })
        .await
        .map_err(|err| err.to_string())?;

    let value = match view {
        ApiView::Dict(value) => value,
        _ => return Err("view_api did not return dict".to_string()),
    };

    let named = value
        .get("named_endpoints")
        .and_then(Value::as_object)
        .ok_or_else(|| "named_endpoints missing".to_string())?;
    if named.is_empty() {
        return Err("named_endpoints is empty".to_string());
    }

    if let Some(endpoint) = named.get(preferred_api) {
        let params = endpoint
            .get("parameters")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        return Ok((preferred_api.to_string(), params));
    }

    let (fallback_name, fallback_endpoint) = named
        .iter()
        .next()
        .ok_or_else(|| "named_endpoints empty".to_string())?;
    let params = fallback_endpoint
        .get("parameters")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    Ok((fallback_name.clone(), params))
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

async fn run_submit_probe(
    client: &Client,
    api_name: &str,
    kwargs: Map<String, Value>,
) -> ProbeResult {
    let mut options = CallOptions::default();
    options.api_name = Some(api_name.to_string());
    options.kwargs = kwargs;
    options.request.timeout = Some(Duration::from_secs(120));

    let job = match client.submit(vec![], options).await {
        Ok(job) => job,
        Err(err) => return ProbeResult::Failed(format!("submit() returned error: {err}")),
    };

    let started = tokio::time::Instant::now();
    let max_wait = Duration::from_secs(30);

    loop {
        if started.elapsed() >= max_wait {
            let _ = job.cancel().await;
            return ProbeResult::Failed(
                "no status/result change within 30s after submit".to_string(),
            );
        }

        if let Ok(result) = tokio::time::timeout(Duration::from_millis(300), job.result()).await {
            return match result {
                Ok(value) => ProbeResult::Passed(format!(
                    "job completed successfully (result type: {})",
                    value_type(&value)
                )),
                Err(Error::App { message, .. }) => {
                    if is_quota_or_duration_message(&message) {
                        ProbeResult::Quota(format!(
                            "job reached app and returned quota/duration error: {message}"
                        ))
                    } else {
                        ProbeResult::Passed(format!(
                            "job reached app and returned app error: {message}"
                        ))
                    }
                }
                Err(Error::Validation(message)) => ProbeResult::Passed(format!(
                    "job reached app and returned validation error: {message}"
                )),
                Err(Error::QueueFull) => {
                    ProbeResult::Passed("queue is full (request reached app queue)".to_string())
                }
                Err(err) => ProbeResult::Failed(format!("job finished with error: {err}")),
            };
        }

        let status = job.status().await;
        if status.code != Status::Starting {
            let _ = job.cancel().await;
            return ProbeResult::Passed(format!("job accepted; status moved to {:?}", status.code));
        }

        tokio::time::sleep(Duration::from_secs(2)).await;
    }
}

async fn run_quick_predict(
    client: &Client,
    api_name: &str,
    kwargs: Map<String, Value>,
) -> ProbeResult {
    let mut options = CallOptions::default();
    options.api_name = Some(api_name.to_string());
    options.kwargs = kwargs;
    options.request.timeout = Some(Duration::from_secs(90));

    let predict =
        tokio::time::timeout(Duration::from_secs(60), client.predict(vec![], options)).await;
    match predict {
        Ok(Ok(value)) => {
            ProbeResult::Passed(format!("predict ok (result type: {})", value_type(&value)))
        }
        Ok(Err(Error::App { message, .. })) => {
            if is_quota_or_duration_message(&message) {
                ProbeResult::Quota(format!(
                    "predict reached app and got quota/duration error: {message}"
                ))
            } else {
                ProbeResult::Passed(format!("predict reached app and got app error: {message}"))
            }
        }
        Ok(Err(Error::Validation(message))) => ProbeResult::Passed(format!(
            "predict reached app and got validation error: {message}"
        )),
        Ok(Err(err)) => ProbeResult::Failed(format!("predict error: {err}")),
        Err(_) => ProbeResult::Failed("predict timeout after 60s".to_string()),
    }
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
