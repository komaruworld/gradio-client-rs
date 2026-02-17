use std::collections::{HashMap, HashSet};
use std::future::Future;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use futures_util::StreamExt;
use reqwest::cookie::Jar;
use reqwest::header;
use semver::Version;
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use tokio::io::AsyncWriteExt;
use tokio::sync::{mpsc, Mutex, Notify, RwLock};
use uuid::Uuid;

use crate::error::{Error, Result};
use crate::hf::{
    choose_target_space_id, pick_hardware, runtime_current_hardware, should_set_sleep_timeout,
    DuplicateOptions, HfApi,
};
use crate::job::{Job, JobWorkerHandle};
use crate::types::{
    ApiInfo, Component, ComponentApiType, Config, Dependency, ParameterInfo, Protocol,
    ServerMessage, Status, StatusUpdate, API_URL, CANCEL_URL, CONFIG_URL, HEARTBEAT_URL,
    INVALID_RUNTIME, LOGIN_URL, RAW_API_INFO_URL, RESET_URL, SPACE_FETCHER_URL, SSE_DATA_URL,
    SSE_DATA_URL_V0, SSE_URL, SSE_URL_V0, UPLOAD_URL,
};
use crate::utils::{
    apply_diff, construct_args, default_temp_dir, ensure_trailing_slash,
    extract_validation_message, is_file_obj, is_file_obj_with_meta, is_http_url_like,
    maybe_prefix_https, parse_gradio_config_from_html, sanitize_parameter_names,
    strip_invalid_filename_characters, traverse,
};

pub type ResultCallback = Arc<dyn Fn(&Value) + Send + Sync + 'static>;

#[derive(Debug, Clone, Default)]
pub struct HttpClientOptions {
    pub request_timeout: Option<Duration>,
    pub connect_timeout: Option<Duration>,
    pub pool_idle_timeout: Option<Duration>,
    pub pool_max_idle_per_host: Option<usize>,
    pub proxy: Option<String>,
    pub no_proxy: bool,
    pub tcp_nodelay: bool,
    pub http2_prior_knowledge: bool,
    pub basic_auth: Option<(String, String)>,
}

#[derive(Debug, Clone, Default)]
pub struct RequestOptions {
    pub timeout: Option<Duration>,
    pub query: Vec<(String, String)>,
    pub basic_auth: Option<(String, String)>,
    pub bearer_auth: Option<String>,
}

#[derive(Debug, Clone)]
pub enum DownloadFiles {
    Directory(PathBuf),
    Disabled,
}

impl Default for DownloadFiles {
    fn default() -> Self {
        Self::Directory(default_temp_dir())
    }
}

#[derive(Debug, Clone)]
pub struct ClientOptions {
    pub token: Option<String>,
    pub verbose: bool,
    pub auth: Option<(String, String)>,
    pub headers: HashMap<String, String>,
    pub download_files: DownloadFiles,
    pub ssl_verify: bool,
    pub skip_components: bool,
    pub analytics_enabled: bool,
    pub zero_gpu_ip_token: Option<String>,
    pub http: HttpClientOptions,
    pub request_timeout: Option<Duration>,
}

impl Default for ClientOptions {
    fn default() -> Self {
        Self {
            token: None,
            verbose: true,
            auth: None,
            headers: HashMap::new(),
            download_files: DownloadFiles::default(),
            ssl_verify: true,
            skip_components: true,
            analytics_enabled: analytics_enabled_default(),
            zero_gpu_ip_token: None,
            http: HttpClientOptions::default(),
            request_timeout: None,
        }
    }
}

#[derive(Clone, Default)]
pub struct CallOptions {
    pub api_name: Option<String>,
    pub fn_index: Option<i64>,
    pub headers: HashMap<String, String>,
    pub kwargs: Map<String, Value>,
    pub request: RequestOptions,
    pub result_callbacks: Vec<ResultCallback>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ViewApiReturnFormat {
    #[default]
    None,
    Dict,
    Text,
}

#[derive(Debug, Clone)]
pub struct ViewApiOptions {
    pub all_endpoints: Option<bool>,
    pub print_info: bool,
    pub return_format: ViewApiReturnFormat,
}

impl Default for ViewApiOptions {
    fn default() -> Self {
        Self {
            all_endpoints: None,
            print_info: true,
            return_format: ViewApiReturnFormat::None,
        }
    }
}

#[derive(Debug, Clone)]
pub enum ApiView {
    None,
    Dict(Value),
    Text(String),
}

#[derive(Debug, Clone)]
struct Endpoint {
    fn_index: i64,
    dependency: Dependency,
    api_name: Option<String>,
    protocol: Protocol,
    input_component_types: Vec<ComponentApiType>,
    output_component_types: Vec<ComponentApiType>,
    parameters_info: Option<Vec<ParameterInfo>>,
    backend_fn: Option<bool>,
    is_valid: bool,
    api_visibility: String,
}

#[derive(Debug)]
struct ClientInner {
    verbose: bool,
    space_id: Option<String>,
    analytics_enabled: bool,
    zero_gpu_ip_token: Option<String>,
    default_basic_auth: Option<(String, String)>,
    skip_components: bool,
    download_files: bool,
    output_dir: Option<PathBuf>,
    headers: HashMap<String, String>,
    src: String,
    src_prefixed: String,
    protocol: Protocol,
    api_url: String,
    sse_url: String,
    sse_data_url: String,
    upload_url: String,
    reset_url: String,
    cancel_url: String,
    heartbeat_url: String,
    app_version: Version,
    config: Config,
    info: ApiInfo,
    endpoints: HashMap<i64, Endpoint>,
    http: reqwest::Client,
    session_hash: RwLock<String>,
    pending_event_ids: Mutex<HashSet<String>>,
    pending_channels: Mutex<HashMap<String, mpsc::UnboundedSender<Option<ServerMessage>>>>,
    stream_running: AtomicBool,
    heartbeat_refresh: AtomicBool,
    heartbeat_kill: AtomicBool,
    heartbeat_notify: Notify,
}

#[derive(Clone, Debug)]
pub struct Client {
    inner: Arc<ClientInner>,
}

impl Client {
    pub async fn new(src: impl AsRef<str>, options: ClientOptions) -> Result<Self> {
        let resolved_token = options
            .token
            .clone()
            .or_else(|| std::env::var("HF_TOKEN").ok())
            .or_else(|| std::env::var("HUGGING_FACE_HUB_TOKEN").ok());

        let user_agent = format!(
            "gradio_client_rs/{} (https://github.com/gradio-app/gradio)",
            env!("CARGO_PKG_VERSION")
        );

        let jar = Arc::new(Jar::default());
        let mut builder = reqwest::Client::builder()
            .cookie_provider(jar)
            .danger_accept_invalid_certs(!options.ssl_verify)
            .user_agent(user_agent.clone());

        let request_timeout = options.http.request_timeout.or(options.request_timeout);
        if let Some(timeout) = request_timeout {
            builder = builder.timeout(timeout);
        }
        if let Some(connect_timeout) = options.http.connect_timeout {
            builder = builder.connect_timeout(connect_timeout);
        }
        if let Some(pool_idle_timeout) = options.http.pool_idle_timeout {
            builder = builder.pool_idle_timeout(pool_idle_timeout);
        }
        if let Some(pool_max_idle_per_host) = options.http.pool_max_idle_per_host {
            builder = builder.pool_max_idle_per_host(pool_max_idle_per_host);
        }
        if let Some(proxy) = &options.http.proxy {
            builder = builder.proxy(
                reqwest::Proxy::all(proxy)
                    .map_err(|err| Error::Api(format!("invalid proxy '{proxy}': {err}")))?,
            );
        }
        if options.http.no_proxy {
            builder = builder.no_proxy();
        }
        builder = builder.tcp_nodelay(options.http.tcp_nodelay);
        if options.http.http2_prior_knowledge {
            // reqwest client-wide HTTP/2 prior knowledge is not available on all backends.
            // We keep the option for API parity and apply HTTP/2 behavior per-request where supported.
        }

        let http = builder.build()?;

        let mut headers = HashMap::new();
        headers.insert("user-agent".to_string(), user_agent.clone());
        if let Some(token) = &resolved_token {
            headers.insert("x-hf-authorization".to_string(), format!("Bearer {token}"));
        }
        headers.extend(options.headers.clone());

        let hf = HfApi::new(http.clone(), resolved_token.clone(), user_agent.clone());

        let mut space_id = None;
        let src = src.as_ref();
        let resolved_src = if is_http_url_like(src) {
            ensure_trailing_slash(src.to_string())
        } else {
            let info = hf.space_info(src).await?;
            let host = info.host.ok_or_else(|| {
                Error::InvalidSource(format!("Could not resolve host for Space: {src}"))
            })?;
            space_id = Some(src.to_string());
            ensure_trailing_slash(maybe_prefix_https(&host))
        };

        if let Some(space) = &space_id {
            let mut state = hf.get_space_runtime(space).await?.stage;
            if state == "BUILDING" {
                if options.verbose {
                    eprintln!("Space is still building. Please wait...");
                }
                loop {
                    tokio::time::sleep(Duration::from_secs(2)).await;
                    state = hf.get_space_runtime(space).await?.stage;
                    if state != "BUILDING" {
                        break;
                    }
                }
            }
            if INVALID_RUNTIME.contains(&state.as_str()) {
                return Err(Error::InvalidSource(format!(
                    "The current space is in the invalid state: {state}."
                )));
            }
        }

        if options.verbose {
            eprintln!("Loaded as API: {resolved_src}");
        }

        let mut client = Self {
            inner: Arc::new(ClientInner {
                verbose: options.verbose,
                space_id,
                analytics_enabled: options.analytics_enabled,
                zero_gpu_ip_token: options.zero_gpu_ip_token,
                default_basic_auth: options.http.basic_auth.clone(),
                skip_components: options.skip_components,
                download_files: !matches!(options.download_files, DownloadFiles::Disabled),
                output_dir: match &options.download_files {
                    DownloadFiles::Directory(path) => Some(path.clone()),
                    DownloadFiles::Disabled => None,
                },
                headers,
                src: resolved_src.clone(),
                src_prefixed: String::new(),
                protocol: Protocol::Ws,
                api_url: String::new(),
                sse_url: String::new(),
                sse_data_url: String::new(),
                upload_url: String::new(),
                reset_url: String::new(),
                cancel_url: String::new(),
                heartbeat_url: String::new(),
                app_version: Version::new(2, 0, 0),
                config: Config::default(),
                info: ApiInfo::default(),
                endpoints: HashMap::new(),
                http,
                session_hash: RwLock::new(Uuid::new_v4().to_string()),
                pending_event_ids: Mutex::new(HashSet::new()),
                pending_channels: Mutex::new(HashMap::new()),
                stream_running: AtomicBool::new(false),
                heartbeat_refresh: AtomicBool::new(false),
                heartbeat_kill: AtomicBool::new(false),
                heartbeat_notify: Notify::new(),
            }),
        };

        if let Some(auth) = &options.auth {
            client.login(auth.clone()).await?;
        }

        client.initialize().await?;
        client.start_heartbeat();
        client.start_telemetry();
        Ok(client)
    }

    pub async fn duplicate(from_id: &str, options: DuplicateOptions) -> Result<Self> {
        let options = options.normalized();
        let user_agent = format!(
            "gradio_client_rs/{} (https://github.com/gradio-app/gradio)",
            env!("CARGO_PKG_VERSION")
        );

        let http = reqwest::Client::builder()
            .user_agent(user_agent.clone())
            .build()?;

        let hf = HfApi::new(http, options.token.clone(), user_agent);
        let original_runtime = hf.get_space_runtime(from_id).await?;
        let username = hf.whoami().await?.name;
        let space_id = choose_target_space_id(&username, from_id, options.to_id.as_deref())?;

        let current_runtime = match hf.get_space_runtime(&space_id).await {
            Ok(runtime) => {
                if options.verbose {
                    eprintln!("Using your existing Space: https://hf.space/{space_id}");
                }
                if options.verbose && options.secrets.is_some() {
                    eprintln!(
                        "Warning: secrets are only applied when a Space is duplicated for the first time."
                    );
                }
                runtime
            }
            Err(Error::HuggingFace(message)) if message.contains("not found") => {
                if options.verbose {
                    eprintln!("Creating a duplicate of {from_id} for your own use...");
                }
                hf.duplicate_space(from_id, &space_id, options.private, true)
                    .await?;
                if let Some(secrets) = &options.secrets {
                    for (key, value) in secrets {
                        hf.add_space_secret(&space_id, key, value).await?;
                    }
                }
                if options.verbose {
                    eprintln!("Created new Space: https://hf.space/{space_id}");
                }
                hf.get_space_runtime(&space_id).await?
            }
            Err(other) => return Err(other),
        };

        let desired_hardware = pick_hardware(&original_runtime, options.hardware.as_deref());
        let current_hardware = runtime_current_hardware(&current_runtime);

        if desired_hardware != current_hardware {
            if let Some(hardware) = &desired_hardware {
                hf.request_space_hardware(&space_id, hardware).await?;
                if options.verbose {
                    eprintln!(
                        "NOTE: this Space uses upgraded hardware: {hardware}. See billing info at https://huggingface.co/settings/billing"
                    );
                }
            }
        }

        if should_set_sleep_timeout(desired_hardware.as_deref()) {
            hf.set_space_sleep_time(&space_id, options.sleep_timeout_minutes * 60)
                .await?;
        }

        let mut client_options = ClientOptions::default();
        client_options.token = options.token;
        client_options.verbose = options.verbose;

        Self::new(space_id, client_options).await
    }

    pub async fn close(&self) {
        self.inner.heartbeat_kill.store(true, Ordering::SeqCst);
        self.inner.heartbeat_notify.notify_waiters();
        self.close_streams().await;
    }

    pub async fn reset_session(&self) {
        *self.inner.session_hash.write().await = Uuid::new_v4().to_string();
        self.inner.heartbeat_refresh.store(true, Ordering::SeqCst);
        self.inner.heartbeat_notify.notify_waiters();
    }

    pub async fn predict(&self, args: Vec<Value>, options: CallOptions) -> Result<Value> {
        let job = self.submit(args, options).await?;
        job.result().await
    }

    pub async fn submit(&self, args: Vec<Value>, options: CallOptions) -> Result<Job> {
        let fn_index = self.infer_fn_index(options.api_name.as_deref(), options.fn_index)?;
        let endpoint = self
            .inner
            .endpoints
            .get(&fn_index)
            .cloned()
            .ok_or_else(|| Error::InvalidApiEndpoint(format!("Missing endpoint: {fn_index}")))?;

        let CallOptions {
            headers,
            kwargs,
            request,
            result_callbacks,
            ..
        } = options;

        let mut request_headers = headers;
        if matches!(
            endpoint.protocol,
            Protocol::Sse | Protocol::SseV1 | Protocol::SseV2 | Protocol::SseV21 | Protocol::SseV3
        ) {
            request_headers.insert("x-gradio-user".to_string(), "api".to_string());
        }

        let args = construct_args(endpoint.parameters_info.as_ref(), args, kwargs)?;
        let (job, mut worker) = Job::new();
        let request_options = request;

        let client = self.clone();
        tokio::spawn(async move {
            let result = client
                .run_job(
                    endpoint,
                    args,
                    request_headers,
                    request_options,
                    &mut worker,
                )
                .await;

            if let Ok(value) = &result {
                for callback in result_callbacks {
                    let callback_value = value.clone();
                    let _ = catch_unwind(AssertUnwindSafe(move || {
                        callback(&callback_value);
                    }));
                }
            }
            worker.finish(result).await;
        });

        Ok(job)
    }

    pub async fn view_api(&self, mut options: ViewApiOptions) -> Result<ApiView> {
        let info = &self.inner.info;
        let num_named_endpoints = info.named_endpoints.len();
        let num_unnamed_endpoints = info.unnamed_endpoints.len();

        if num_named_endpoints == 0 && options.all_endpoints.is_none() {
            options.all_endpoints = Some(true);
        }

        let mut human_info =
            String::from("Client.predict() Usage Info\n---------------------------\n");
        human_info.push_str(&format!("Named API endpoints: {num_named_endpoints}\n"));

        for (api_name, endpoint_info) in &info.named_endpoints {
            human_info.push_str(&self.render_endpoint_info(api_name, endpoint_info)?);
        }

        if options.all_endpoints.unwrap_or(false) {
            human_info.push_str(&format!(
                "\nUnnamed API endpoints: {num_unnamed_endpoints}\n"
            ));
            for (fn_index, endpoint_info) in &info.unnamed_endpoints {
                let parsed = fn_index.parse::<i64>().unwrap_or(0);
                human_info
                    .push_str(&self.render_endpoint_info(&parsed.to_string(), endpoint_info)?);
            }
        } else if num_unnamed_endpoints > 0 {
            human_info.push_str(&format!(
                "\nUnnamed API endpoints: {num_unnamed_endpoints}, to view, run Client.view_api(all_endpoints=True)\n"
            ));
        }

        if options.print_info {
            eprintln!("{human_info}");
        }

        Ok(match options.return_format {
            ViewApiReturnFormat::None => ApiView::None,
            ViewApiReturnFormat::Text => ApiView::Text(human_info),
            ViewApiReturnFormat::Dict => ApiView::Dict(json!({
                "named_endpoints": info.named_endpoints,
                "unnamed_endpoints": info.unnamed_endpoints,
            })),
        })
    }

    fn infer_fn_index(&self, api_name: Option<&str>, fn_index: Option<i64>) -> Result<i64> {
        if let Some(api_name) = api_name {
            for (i, dependency) in self.inner.config.dependencies.iter().enumerate() {
                let Some(config_api_name) = dependency.api_name.as_ref() else {
                    continue;
                };
                if config_api_name == &Value::Bool(false)
                    || dependency.api_visibility.as_deref() == Some("private")
                {
                    continue;
                }
                if let Some(name) = config_api_name.as_str() {
                    if format!("/{name}") == api_name {
                        return Ok(dependency.id.unwrap_or(i as i64));
                    }
                }
            }

            let mut message = format!("Cannot find a function with api_name: {api_name}.");
            if !api_name.starts_with('/') {
                message.push_str(" Did you mean to use a leading slash?");
            }
            return Err(Error::InvalidApiEndpoint(message));
        }

        if let Some(fn_index) = fn_index {
            let endpoint = self.inner.endpoints.get(&fn_index).ok_or_else(|| {
                Error::InvalidApiEndpoint(format!("Invalid function index: {fn_index}"))
            })?;
            if endpoint.is_valid {
                return Ok(fn_index);
            }
            return Err(Error::InvalidApiEndpoint(format!(
                "Invalid function index: {fn_index}"
            )));
        }

        let valid_endpoints: Vec<&Endpoint> = self
            .inner
            .endpoints
            .values()
            .filter(|endpoint| {
                endpoint.is_valid
                    && endpoint.api_name.is_some()
                    && endpoint.backend_fn.is_some()
                    && endpoint.api_visibility == "public"
            })
            .collect();

        if valid_endpoints.len() == 1 {
            Ok(valid_endpoints[0].fn_index)
        } else {
            Err(Error::InvalidApiEndpoint(
                "This Gradio app might have multiple endpoints. Please specify an api_name or fn_index"
                    .to_string(),
            ))
        }
    }

    async fn run_job(
        &self,
        endpoint: Endpoint,
        mut args: Vec<Value>,
        request_headers: HashMap<String, String>,
        request_options: RequestOptions,
        worker: &mut JobWorkerHandle,
    ) -> Result<Value> {
        if !endpoint.is_valid {
            return Err(Error::InvalidApiEndpoint(
                "This API endpoint is disabled by the upstream app".to_string(),
            ));
        }

        if self.inner.skip_components {
            args = self.insert_empty_state(&endpoint, args);
        }
        args = self
            .process_input_files(&endpoint, args, &request_options)
            .await?;

        let final_output_raw = match endpoint.protocol {
            Protocol::Sse => {
                self.run_sse_v0(
                    &endpoint,
                    args.clone(),
                    request_headers.clone(),
                    request_options.clone(),
                    worker,
                )
                .await?
            }
            Protocol::SseV1 | Protocol::SseV2 | Protocol::SseV21 | Protocol::SseV3 => {
                self.run_sse_v1plus(
                    &endpoint,
                    args.clone(),
                    request_headers.clone(),
                    request_options.clone(),
                    worker,
                )
                .await?
            }
            Protocol::Ws => return Err(Error::UnsupportedProtocol("ws".to_string())),
        };

        let final_processed = self
            .process_predictions(
                &endpoint,
                final_output_raw,
                endpoint.protocol,
                &request_options,
            )
            .await?;

        if !worker.has_outputs().await {
            worker
                .push_output(final_processed.clone(), true, true)
                .await;
        }

        Ok(final_processed)
    }

    fn insert_empty_state(&self, endpoint: &Endpoint, mut data: Vec<Value>) -> Vec<Value> {
        for (index, input_type) in endpoint.input_component_types.iter().enumerate() {
            if input_type.is_state {
                data.insert(index, Value::Null);
            }
        }
        data
    }

    async fn process_input_files(
        &self,
        endpoint: &Endpoint,
        data: Vec<Value>,
        request_options: &RequestOptions,
    ) -> Result<Vec<Value>> {
        let mut out = Vec::with_capacity(data.len());
        for (index, value) in data.into_iter().enumerate() {
            out.push(
                self.process_input_value(endpoint, value, index, request_options)
                    .await?,
            );
        }
        Ok(out)
    }

    fn process_input_value<'a>(
        &'a self,
        endpoint: &'a Endpoint,
        value: Value,
        data_index: usize,
        request_options: &'a RequestOptions,
    ) -> Pin<Box<dyn Future<Output = Result<Value>> + Send + 'a>> {
        Box::pin(async move {
            if is_file_obj_with_meta(&value) {
                return self
                    .upload_file(endpoint, &value, data_index, request_options)
                    .await;
            }

            match value {
                Value::Array(items) => {
                    let mut out = Vec::with_capacity(items.len());
                    for item in items {
                        out.push(
                            self.process_input_value(endpoint, item, data_index, request_options)
                                .await?,
                        );
                    }
                    Ok(Value::Array(out))
                }
                Value::Object(mut map) => {
                    let keys: Vec<String> = map.keys().cloned().collect();
                    for key in keys {
                        let item = map.remove(&key).unwrap_or(Value::Null);
                        map.insert(
                            key,
                            self.process_input_value(endpoint, item, data_index, request_options)
                                .await?,
                        );
                    }
                    Ok(Value::Object(map))
                }
                _ => Ok(value),
            }
        })
    }

    async fn upload_file(
        &self,
        endpoint: &Endpoint,
        file_obj: &Value,
        data_index: usize,
        request_options: &RequestOptions,
    ) -> Result<Value> {
        let file_path = file_obj
            .get("path")
            .and_then(Value::as_str)
            .ok_or_else(|| Error::Api("Invalid file object: missing path".to_string()))?;

        let original_name = Path::new(file_path)
            .file_name()
            .and_then(|value| value.to_str())
            .unwrap_or("file")
            .to_string();

        let uploaded_path = if is_http_url_like(file_path) {
            file_path.to_string()
        } else {
            if let Some(max_file_size) = self.inner.config.max_file_size {
                let size = tokio::fs::metadata(file_path).await?.len();
                if size > max_file_size {
                    let component_id = endpoint
                        .dependency
                        .inputs
                        .get(data_index)
                        .copied()
                        .unwrap_or(0);
                    let component = self
                        .inner
                        .config
                        .components
                        .iter()
                        .find(|component| component.id == component_id);
                    let label = component
                        .and_then(|component| component.label.clone())
                        .unwrap_or_default();
                    return Err(Error::Api(format!(
                        "File {file_path} exceeds the maximum file size of {max_file_size} bytes set in {label} component."
                    )));
                }
            }

            let bytes = tokio::fs::read(file_path).await?;
            let part = reqwest::multipart::Part::bytes(bytes).file_name(original_name.clone());
            let form = reqwest::multipart::Form::new().part("files", part);

            let mut request = self.inner.http.post(&self.inner.upload_url);
            request = self.apply_request_options(request, None, Some(request_options))?;
            let response = request.multipart(form).send().await?;

            let status = response.status();
            if !status.is_success() {
                let body = response.text().await.unwrap_or_default();
                return Err(Error::Api(format!(
                    "upload failed with status {}: {body}",
                    status
                )));
            }

            let payload: Value = response.json().await?;
            payload
                .get(0)
                .and_then(Value::as_str)
                .ok_or_else(|| Error::Api("upload response missing path".to_string()))?
                .to_string()
        };

        Ok(json!({
            "path": uploaded_path,
            "orig_name": strip_invalid_filename_characters(&original_name, 200),
            "meta": {"_type": "gradio.FileData"},
        }))
    }

    async fn process_predictions(
        &self,
        endpoint: &Endpoint,
        predictions: Vec<Value>,
        protocol: Protocol,
        request_options: &RequestOptions,
    ) -> Result<Value> {
        let mut output = predictions;

        if self.inner.download_files {
            let mut downloaded = Vec::with_capacity(output.len());
            for value in output {
                downloaded.push(
                    self.download_prediction_value(value, protocol, request_options)
                        .await?,
                );
            }
            output = downloaded;
        }

        if self.inner.skip_components {
            let mut filtered = Vec::new();
            for (value, component_type) in output
                .into_iter()
                .zip(endpoint.output_component_types.iter())
            {
                if !component_type.skip {
                    filtered.push(value);
                }
            }
            output = filtered;
        }

        if output.len() == 1 {
            Ok(output.into_iter().next().unwrap_or(Value::Null))
        } else {
            Ok(Value::Array(output))
        }
    }

    fn download_prediction_value<'a>(
        &'a self,
        value: Value,
        protocol: Protocol,
        request_options: &'a RequestOptions,
    ) -> Pin<Box<dyn Future<Output = Result<Value>> + Send + 'a>> {
        Box::pin(async move {
            let should_download = match protocol {
                Protocol::SseV21 => is_file_obj_with_meta(&value),
                _ => is_file_obj(&value),
            };

            if should_download {
                return self.download_file(value, request_options).await;
            }

            match value {
                Value::Array(items) => {
                    let mut out = Vec::with_capacity(items.len());
                    for item in items {
                        out.push(
                            self.download_prediction_value(item, protocol, request_options)
                                .await?,
                        );
                    }
                    Ok(Value::Array(out))
                }
                Value::Object(mut map) => {
                    let keys: Vec<String> = map.keys().cloned().collect();
                    for key in keys {
                        let value = map.remove(&key).unwrap_or(Value::Null);
                        map.insert(
                            key,
                            self.download_prediction_value(value, protocol, request_options)
                                .await?,
                        );
                    }
                    Ok(Value::Object(map))
                }
                _ => Ok(value),
            }
        })
    }

    async fn download_file(
        &self,
        file_obj: Value,
        request_options: &RequestOptions,
    ) -> Result<Value> {
        let obj = file_obj
            .as_object()
            .ok_or_else(|| Error::Api("invalid file object".to_string()))?;

        let url_path = if obj
            .get("is_stream")
            .and_then(Value::as_bool)
            .unwrap_or(false)
            && obj.get("url").and_then(Value::as_str).is_some()
        {
            let stream_url = obj.get("url").and_then(Value::as_str).unwrap_or_default();
            if is_http_url_like(stream_url) {
                stream_url.to_string()
            } else {
                format!(
                    "{}{}",
                    self.inner.src_prefixed,
                    stream_url.trim_start_matches('/')
                )
            }
        } else {
            let path = obj
                .get("path")
                .and_then(Value::as_str)
                .ok_or_else(|| Error::Api("invalid file object path".to_string()))?;
            format!("{}file={path}", self.inner.src_prefixed)
        };

        let Some(output_root) = &self.inner.output_dir else {
            return Ok(file_obj);
        };
        tokio::fs::create_dir_all(output_root).await?;

        let temp_dir = std::env::temp_dir().join(Uuid::new_v4().to_string());
        tokio::fs::create_dir_all(&temp_dir).await?;

        let filename = Path::new(&url_path)
            .file_name()
            .and_then(|name| name.to_str())
            .filter(|name| !name.is_empty())
            .unwrap_or("file")
            .to_string();

        let temp_path = temp_dir.join(&filename);
        let mut file = tokio::fs::File::create(&temp_path).await?;
        let mut sha = Sha256::new();

        let request = self.apply_request_options(
            self.inner.http.get(&url_path),
            None,
            Some(request_options),
        )?;
        let response = request.send().await?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(Error::Api(format!(
                "download failed with status {}: {body}",
                status
            )));
        }

        let mut stream = response.bytes_stream();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk?;
            sha.update(&chunk);
            file.write_all(&chunk).await?;
        }
        file.flush().await?;

        let digest = format!("{:x}", sha.finalize());
        let final_dir = output_root.join(digest);
        tokio::fs::create_dir_all(&final_dir).await?;
        let final_path = final_dir.join(filename);

        tokio::fs::rename(&temp_path, &final_path).await?;
        Ok(Value::String(final_path.to_string_lossy().to_string()))
    }

    async fn run_sse_v1plus(
        &self,
        endpoint: &Endpoint,
        args: Vec<Value>,
        request_headers: HashMap<String, String>,
        request_options: RequestOptions,
        worker: &mut JobWorkerHandle,
    ) -> Result<Vec<Value>> {
        let data = json!({
            "data": args,
            "fn_index": endpoint.fn_index,
        });

        let session_hash = self.inner.session_hash.read().await.clone();
        let hash_data = json!({
            "fn_index": endpoint.fn_index,
            "session_hash": session_hash,
        });

        let (event_id, mut receiver) = self
            .send_data(
                data,
                hash_data,
                endpoint.protocol,
                Some(request_headers),
                Some(request_options.clone()),
            )
            .await?;

        let mut pending_responses_for_diffs: Option<Vec<Value>> = None;
        loop {
            tokio::select! {
                cancel = worker.recv_cancel() => {
                    if cancel.is_some() {
                        self.cancel_job(endpoint, &event_id).await?;
                        worker.mark_cancelled().await;
                        return Err(Error::Cancelled);
                    }
                }
                message = receiver.recv() => {
                    let Some(message) = message else {
                        return Err(Error::Cancelled);
                    };
                    let Some(message) = message else {
                        return Err(Error::Cancelled);
                    };

                    let status_update = StatusUpdate {
                        code: Status::from_server_msg(&message.msg).unwrap_or(Status::Processing),
                        queue_size: message.queue_size,
                        rank: message.rank,
                        success: message.success,
                        eta: message.rank_eta,
                        progress_data: message.progress_data.clone(),
                        time: std::time::SystemTime::now(),
                        log: match (message.log.clone(), message.level.clone()) {
                            (Some(log), Some(level)) => Some((log, level)),
                            _ => None,
                        },
                    };
                    if self.inner.verbose {
                        if let Some(eta) = status_update.eta {
                            if eta > 30.0 {
                                if let Some(space_id) = &self.inner.space_id {
                                    eprintln!(
                                        "Due to heavy traffic on this app, prediction may take about {} seconds. For faster predictions, consider Client::duplicate({space_id}).",
                                        eta as i64
                                    );
                                }
                            }
                        }
                    }
                    worker.set_status(status_update.clone()).await;

                    let mut output_data: Vec<Value> = message
                        .output
                        .get("data")
                        .and_then(Value::as_array)
                        .cloned()
                        .unwrap_or_default();

                    if message.msg == "process_generating"
                        && matches!(endpoint.protocol, Protocol::SseV2 | Protocol::SseV21 | Protocol::SseV3)
                    {
                        if pending_responses_for_diffs.is_none() {
                            pending_responses_for_diffs = Some(output_data.clone());
                        } else if let Some(previous) = pending_responses_for_diffs.as_mut() {
                            for (index, value) in output_data.clone().into_iter().enumerate() {
                                if let Some(prev) = previous.get(index).cloned() {
                                    let merged = apply_diff(&prev, &value)?;
                                    if let Some(prev_mut) = previous.get_mut(index) {
                                        *prev_mut = merged.clone();
                                    }
                                    if let Some(current_mut) = output_data.get_mut(index) {
                                        *current_mut = merged;
                                    }
                                }
                            }
                        }
                    }

                    if !output_data.is_empty() && status_update.code != Status::Finished {
                        let processed = self
                            .process_predictions(
                                endpoint,
                                output_data,
                                endpoint.protocol,
                                &request_options,
                            )
                            .await?;
                        worker
                            .push_output(processed, message.success.unwrap_or(true), false)
                            .await;
                    }

                    if message.msg == "process_completed" {
                        if !message.success.unwrap_or(true) {
                            let message_text = message
                                .output
                                .get("error")
                                .and_then(Value::as_str)
                                .map(ToOwned::to_owned)
                                .unwrap_or_else(|| {
                                    "The upstream Gradio app has raised an exception but has not enabled verbose error reporting. To enable, set show_error=True in launch().".to_string()
                                });
                            return Err(Error::App {
                                message: message_text,
                                raw: message.output.clone(),
                            });
                        }

                        let final_output = message
                            .output
                            .get("data")
                            .and_then(Value::as_array)
                            .cloned()
                            .unwrap_or_default();
                        return Ok(final_output);
                    }

                    if message.msg == "Server stopped unexpectedly." {
                        return Err(Error::Api("Server stopped.".to_string()));
                    }
                }
            }
        }
    }

    async fn run_sse_v0(
        &self,
        endpoint: &Endpoint,
        args: Vec<Value>,
        request_headers: HashMap<String, String>,
        request_options: RequestOptions,
        worker: &mut JobWorkerHandle,
    ) -> Result<Vec<Value>> {
        let session_hash = self.inner.session_hash.read().await.clone();
        let data = json!({
            "data": args,
            "fn_index": endpoint.fn_index,
        });
        let hash_data = json!({
            "fn_index": endpoint.fn_index,
            "session_hash": session_hash,
        });

        let mut request = self.inner.http.get(&self.inner.sse_url);
        request =
            self.apply_request_options(request, Some(&request_headers), Some(&request_options))?;
        let response = request
            .query(&[(
                "session_hash",
                hash_data["session_hash"].as_str().unwrap_or_default(),
            )])
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(Error::Api(format!(
                "sse stream failed with status {}",
                response.status()
            )));
        }

        let mut stream = response.bytes_stream();
        let mut buffer = Vec::<u8>::new();

        while let Some(chunk) = stream.next().await {
            let chunk = chunk?;
            buffer.extend_from_slice(&chunk);

            while let Some(frame) = pop_sse_frame(&mut buffer) {
                for line in frame.lines() {
                    let line = line.trim();
                    if line.is_empty() {
                        continue;
                    }
                    if !line.starts_with("data:") {
                        return Err(Error::Api(format!("Unexpected message: {line}")));
                    }

                    let payload = line.trim_start_matches("data:").trim();
                    let message: ServerMessage = serde_json::from_str(payload)?;
                    let status_update = StatusUpdate {
                        code: Status::from_server_msg(&message.msg).unwrap_or(Status::Processing),
                        queue_size: message.queue_size,
                        rank: message.rank,
                        success: message.success,
                        eta: message.rank_eta,
                        progress_data: message.progress_data.clone(),
                        time: std::time::SystemTime::now(),
                        log: None,
                    };
                    if self.inner.verbose {
                        if let Some(eta) = status_update.eta {
                            if eta > 30.0 {
                                if let Some(space_id) = &self.inner.space_id {
                                    eprintln!(
                                        "Due to heavy traffic on this app, prediction may take about {} seconds. For faster predictions, consider Client::duplicate({space_id}).",
                                        eta as i64
                                    );
                                }
                            }
                        }
                    }
                    worker.set_status(status_update.clone()).await;

                    if worker.try_recv_cancel() {
                        if let Some(event_id) = &message.event_id {
                            self.reset_event(event_id).await?;
                        }
                        worker.mark_cancelled().await;
                        return Err(Error::Cancelled);
                    }

                    if message.msg == "queue_full" {
                        return Err(Error::QueueFull);
                    }

                    if message.msg == "send_data" {
                        let Some(event_id) = message.event_id else {
                            return Err(Error::MissingField("event_id".to_string()));
                        };
                        let payload = json!({
                            "event_id": event_id,
                            "data": data["data"],
                            "fn_index": endpoint.fn_index,
                            "session_hash": hash_data["session_hash"],
                        });
                        let response = self
                            .apply_request_options(
                                self.inner.http.post(&self.inner.sse_data_url),
                                Some(&request_headers),
                                Some(&request_options),
                            )?
                            .json(&payload)
                            .send()
                            .await?;
                        if !response.status().is_success() {
                            return Err(Error::Api(format!(
                                "failed to send queued data: {}",
                                response.status()
                            )));
                        }
                    }

                    let output_data: Vec<Value> = message
                        .output
                        .get("data")
                        .and_then(Value::as_array)
                        .cloned()
                        .unwrap_or_default();

                    if !output_data.is_empty() && status_update.code != Status::Finished {
                        let processed = self
                            .process_predictions(
                                endpoint,
                                output_data,
                                endpoint.protocol,
                                &request_options,
                            )
                            .await?;
                        worker
                            .push_output(processed, message.success.unwrap_or(true), false)
                            .await;
                    }

                    if message.msg == "process_completed" {
                        if !message.success.unwrap_or(true) {
                            let message_text = message
                                .output
                                .get("error")
                                .and_then(Value::as_str)
                                .map(ToOwned::to_owned)
                                .unwrap_or_else(|| {
                                    "The upstream Gradio app has raised an exception but has not enabled verbose error reporting."
                                        .to_string()
                                });
                            return Err(Error::App {
                                message: message_text,
                                raw: message.output.clone(),
                            });
                        }
                        return Ok(message
                            .output
                            .get("data")
                            .and_then(Value::as_array)
                            .cloned()
                            .unwrap_or_default());
                    }
                }
            }
        }

        Err(Error::Api(
            "Did not receive process_completed message.".to_string(),
        ))
    }

    async fn send_data(
        &self,
        data: Value,
        hash_data: Value,
        protocol: Protocol,
        request_headers: Option<HashMap<String, String>>,
        request_options: Option<RequestOptions>,
    ) -> Result<(String, mpsc::UnboundedReceiver<Option<ServerMessage>>)> {
        let mut payload = data;
        if let Some(payload_obj) = payload.as_object_mut() {
            let hash_obj = hash_data
                .as_object()
                .ok_or_else(|| Error::Api("hash data must be object".to_string()))?;
            for (key, value) in hash_obj {
                payload_obj.insert(key.clone(), value.clone());
            }
        }

        let response = self
            .apply_request_options(
                self.inner.http.post(&self.inner.sse_data_url),
                request_headers.as_ref(),
                request_options.as_ref(),
            )?
            .json(&payload)
            .send()
            .await?;

        if response.status().as_u16() == 503 {
            return Err(Error::QueueFull);
        }

        let status = response.status();
        let body: Value = response.json().await.unwrap_or(Value::Null);
        if let Some(validation_message) = extract_validation_message(status.as_u16(), &body) {
            return Err(Error::Validation(validation_message));
        }

        if !status.is_success() {
            return Err(Error::Api(format!(
                "send_data failed with status {status}: {body}"
            )));
        }

        let event_id = body
            .get("event_id")
            .and_then(Value::as_str)
            .ok_or_else(|| Error::MissingField("event_id".to_string()))?
            .to_string();

        let (tx, rx) = mpsc::unbounded_channel();
        self.inner
            .pending_channels
            .lock()
            .await
            .insert(event_id.clone(), tx);
        self.inner
            .pending_event_ids
            .lock()
            .await
            .insert(event_id.clone());

        let session_hash = hash_data
            .get("session_hash")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        self.ensure_stream_task(protocol, session_hash, request_options.unwrap_or_default())
            .await;

        Ok((event_id, rx))
    }

    async fn ensure_stream_task(
        &self,
        protocol: Protocol,
        session_hash: String,
        request_options: RequestOptions,
    ) {
        if self
            .inner
            .stream_running
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_ok()
        {
            let client = self.clone();
            tokio::spawn(async move {
                let _ = client
                    .stream_messages(protocol, session_hash, request_options)
                    .await;
                client.inner.stream_running.store(false, Ordering::SeqCst);
                client.close_streams().await;
            });
        }
    }

    async fn stream_messages(
        &self,
        protocol: Protocol,
        session_hash: String,
        request_options: RequestOptions,
    ) -> Result<()> {
        let response = self
            .apply_request_options(
                self.inner.http.get(&self.inner.sse_url),
                None,
                Some(&request_options),
            )?
            .query(&[("session_hash", session_hash)])
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(Error::Api(format!(
                "stream_messages failed with status {}",
                response.status()
            )));
        }

        let mut stream = response.bytes_stream();
        let mut buffer = Vec::<u8>::new();

        while let Some(chunk) = stream.next().await {
            let chunk = chunk?;
            buffer.extend_from_slice(&chunk);

            while let Some(frame) = pop_sse_frame(&mut buffer) {
                for line in frame.lines() {
                    let line = line.trim();
                    if line.is_empty() {
                        continue;
                    }
                    if !line.starts_with("data:") {
                        return Err(Error::Api(format!("Unexpected SSE line: {line}")));
                    }
                    let payload = line.trim_start_matches("data:").trim();
                    let message: ServerMessage = serde_json::from_str(payload)?;

                    if message.msg == "heartbeat" {
                        continue;
                    }
                    if message.msg == "close_stream" {
                        return Ok(());
                    }

                    if message.msg == "Server stopped unexpectedly." {
                        let channels = self.inner.pending_channels.lock().await.clone();
                        for (_, sender) in channels {
                            let _ = sender.send(Some(message.clone()));
                            let _ = sender.send(None);
                        }
                        return Ok(());
                    }

                    let Some(event_id) = message.event_id.clone() else {
                        continue;
                    };

                    if let Some(sender) = self.inner.pending_channels.lock().await.get(&event_id) {
                        let _ = sender.send(Some(message.clone()));
                    }

                    if message.msg == "process_completed" {
                        self.inner.pending_event_ids.lock().await.remove(&event_id);
                        self.inner.pending_channels.lock().await.remove(&event_id);
                    }

                    if protocol != Protocol::SseV3
                        && self.inner.pending_event_ids.lock().await.is_empty()
                    {
                        return Ok(());
                    }
                }
            }
        }

        Ok(())
    }

    async fn close_streams(&self) {
        let mut channels = self.inner.pending_channels.lock().await;
        for (_, sender) in channels.iter() {
            let _ = sender.send(None);
        }
        channels.clear();
        self.inner.pending_event_ids.lock().await.clear();
    }

    async fn cancel_job(&self, endpoint: &Endpoint, event_id: &str) -> Result<()> {
        self.reset_event(event_id).await?;

        if self.inner.app_version > Version::new(4, 29, 0) {
            let payload = json!({
                "fn_index": endpoint.fn_index,
                "session_hash": self.inner.session_hash.read().await.clone(),
                "event_id": event_id,
            });
            let _ = self
                .apply_request_options(self.inner.http.post(&self.inner.cancel_url), None, None)?
                .json(&payload)
                .send()
                .await?;
            return Ok(());
        }

        let mut candidate: Option<i64> = None;
        let mut smallest_other = usize::MAX;
        for (index, dependency) in self.inner.config.dependencies.iter().enumerate() {
            if dependency.cancels.contains(&endpoint.fn_index) {
                let other = dependency
                    .cancels
                    .iter()
                    .filter(|item| **item != endpoint.fn_index)
                    .count();
                if other < smallest_other {
                    smallest_other = other;
                    candidate = Some(index as i64);
                }
            }
        }

        if let Some(cancel_fn_index) = candidate {
            let payload = json!({
                "data": [],
                "fn_index": cancel_fn_index,
                "session_hash": self.inner.session_hash.read().await.clone(),
            });
            let _ = self
                .apply_request_options(self.inner.http.post(&self.inner.api_url), None, None)?
                .json(&payload)
                .send()
                .await?;
        }

        Ok(())
    }

    async fn reset_event(&self, event_id: &str) -> Result<()> {
        let payload = json!({ "event_id": event_id });
        let _ = self
            .apply_request_options(self.inner.http.post(&self.inner.reset_url), None, None)?
            .json(&payload)
            .send()
            .await?;
        Ok(())
    }

    fn apply_request_options(
        &self,
        mut request: reqwest::RequestBuilder,
        request_headers: Option<&HashMap<String, String>>,
        request_options: Option<&RequestOptions>,
    ) -> Result<reqwest::RequestBuilder> {
        let headers = self.headers_for_request(request_headers)?;
        request = request.headers(headers);

        if let Some(options) = request_options {
            if let Some(timeout) = options.timeout {
                request = request.timeout(timeout);
            }
            if !options.query.is_empty() {
                request = request.query(&options.query);
            }
            if let Some((username, password)) = &options.basic_auth {
                request = request.basic_auth(username, Some(password));
            } else if let Some((username, password)) = &self.inner.default_basic_auth {
                request = request.basic_auth(username, Some(password));
            }
            if let Some(token) = &options.bearer_auth {
                request = request.bearer_auth(token);
            }
        } else if let Some((username, password)) = &self.inner.default_basic_auth {
            request = request.basic_auth(username, Some(password));
        }

        Ok(request)
    }

    fn headers_for_request(
        &self,
        request_headers: Option<&HashMap<String, String>>,
    ) -> Result<header::HeaderMap> {
        let mut headers = header::HeaderMap::new();
        for (key, value) in &self.inner.headers {
            headers.insert(
                header::HeaderName::from_bytes(key.as_bytes())
                    .map_err(|err| Error::Api(format!("invalid header key {key}: {err}")))?,
                header::HeaderValue::from_str(value)
                    .map_err(|err| Error::Api(format!("invalid header value for {key}: {err}")))?,
            );
        }

        if let Some(request_headers) = request_headers {
            for (key, value) in request_headers {
                headers.insert(
                    header::HeaderName::from_bytes(key.as_bytes())
                        .map_err(|err| Error::Api(format!("invalid header key {key}: {err}")))?,
                    header::HeaderValue::from_str(value).map_err(|err| {
                        Error::Api(format!("invalid request header value for {key}: {err}"))
                    })?,
                );
            }
        }

        self.add_zero_gpu_headers(&mut headers)?;
        Ok(headers)
    }

    fn add_zero_gpu_headers(&self, headers: &mut header::HeaderMap) -> Result<()> {
        if self.inner.space_id.is_none() {
            return Ok(());
        }

        if headers.contains_key("x-ip-token") {
            return Ok(());
        }

        let ip_token = self
            .inner
            .zero_gpu_ip_token
            .clone()
            .or_else(|| std::env::var("GRADIO_X_IP_TOKEN").ok());
        if let Some(ip_token) = ip_token {
            headers.insert(
                header::HeaderName::from_static("x-ip-token"),
                header::HeaderValue::from_str(&ip_token)
                    .map_err(|err| Error::Api(format!("invalid x-ip-token header value: {err}")))?,
            );
        }

        Ok(())
    }

    async fn initialize(&mut self) -> Result<()> {
        let config = self.get_config().await?;
        let protocol = Protocol::from_str(config.protocol.as_deref().unwrap_or("ws"));

        let api_prefix = config.api_prefix.trim_start_matches('/');
        let src_prefixed = ensure_trailing_slash(format!("{}{}", self.inner.src, api_prefix));

        let sse_url = if protocol == Protocol::Sse {
            format!("{}{}", src_prefixed, SSE_URL_V0)
        } else {
            format!("{}{}", src_prefixed, SSE_URL)
        };
        let sse_data_url = if protocol == Protocol::Sse {
            format!("{}{}", src_prefixed, SSE_DATA_URL_V0)
        } else {
            format!("{}{}", src_prefixed, SSE_DATA_URL)
        };

        let app_version = parse_version(config.version.as_deref().unwrap_or("2.0"));
        let info = self
            .get_api_info(&config, app_version.clone(), &src_prefixed)
            .await?;

        let endpoints = self.build_endpoints(&config, &info, protocol);

        if let Some(output_dir) = &self.inner.output_dir {
            tokio::fs::create_dir_all(output_dir).await?;
            if !output_dir.is_dir() {
                return Err(Error::Api(format!(
                    "Path {} is not a directory",
                    output_dir.display()
                )));
            }
        }

        let inner = Arc::get_mut(&mut self.inner)
            .ok_or_else(|| Error::Api("Client already shared".to_string()))?;
        inner.config = config;
        inner.protocol = protocol;
        inner.src_prefixed = src_prefixed.clone();
        inner.api_url = format!("{}{}", src_prefixed, API_URL);
        inner.sse_url = sse_url;
        inner.sse_data_url = sse_data_url;
        inner.upload_url = format!("{}{}", src_prefixed, UPLOAD_URL);
        inner.reset_url = format!("{}{}", src_prefixed, RESET_URL);
        inner.cancel_url = format!("{}{}", src_prefixed, CANCEL_URL);
        inner.heartbeat_url = format!("{}{}", src_prefixed, HEARTBEAT_URL);
        inner.app_version = app_version;
        inner.info = info;
        inner.endpoints = endpoints;

        Ok(())
    }

    fn start_heartbeat(&self) {
        let client = self.clone();
        tokio::spawn(async move {
            loop {
                if client.inner.heartbeat_kill.load(Ordering::SeqCst) {
                    return;
                }

                let session_hash = client.inner.session_hash.read().await.clone();
                let url = client
                    .inner
                    .heartbeat_url
                    .replace("{session_hash}", &session_hash);

                let request =
                    match client.apply_request_options(client.inner.http.get(url), None, None) {
                        Ok(request) => request.timeout(Duration::from_secs(20)).send().await,
                        Err(_) => return,
                    };

                let Ok(response) = request else {
                    return;
                };

                let mut stream = response.bytes_stream();
                loop {
                    tokio::select! {
                        _ = client.inner.heartbeat_notify.notified() => {
                            if client.inner.heartbeat_kill.load(Ordering::SeqCst) {
                                return;
                            }
                            if client.inner.heartbeat_refresh.swap(false, Ordering::SeqCst) {
                                break;
                            }
                        }
                        next = stream.next() => {
                            match next {
                                Some(Ok(_)) => {}
                                _ => break,
                            }
                        }
                    }
                }
            }
        });
    }

    fn start_telemetry(&self) {
        if !self.inner.analytics_enabled {
            return;
        }
        let disabled = std::env::var("HF_HUB_DISABLE_TELEMETRY")
            .map(|value| value == "1" || value.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        let offline = std::env::var("HF_HUB_OFFLINE")
            .map(|value| value == "1" || value.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        if disabled || offline {
            return;
        }

        let client = self.clone();
        tokio::spawn(async move {
            let _ = client.send_telemetry("py_client/initiated").await;
        });
    }

    async fn send_telemetry(&self, topic: &str) -> Result<()> {
        let topic = topic
            .split('/')
            .filter(|part| !part.is_empty())
            .map(|part| part.replace(' ', "%20"))
            .collect::<Vec<_>>()
            .join("/");
        let url = format!("https://huggingface.co/api/telemetry/{topic}");

        let mut headers = self.headers_for_request(None)?;
        headers.remove(header::AUTHORIZATION);
        headers.remove("x-hf-authorization");
        headers.insert(
            header::HeaderName::from_static("x-gradio-client-src"),
            header::HeaderValue::from_str(&self.inner.src)
                .map_err(|err| Error::Api(format!("invalid telemetry header: {err}")))?,
        );

        let _ = self.inner.http.head(url).headers(headers).send().await;
        Ok(())
    }

    async fn login(&self, auth: (String, String)) -> Result<()> {
        let response = self
            .apply_request_options(
                self.inner
                    .http
                    .post(format!("{}{}", self.inner.src, LOGIN_URL)),
                None,
                None,
            )?
            .form(&[("username", auth.0), ("password", auth.1)])
            .send()
            .await?;

        if response.status() == reqwest::StatusCode::UNAUTHORIZED {
            return Err(Error::Authentication(format!(
                "Could not login to {}. Invalid credentials.",
                self.inner.src
            )));
        }

        if !response.status().is_success() {
            return Err(Error::Authentication(format!(
                "Could not login to {}.",
                self.inner.src
            )));
        }

        Ok(())
    }

    async fn get_config(&self) -> Result<Config> {
        let response = self
            .apply_request_options(
                self.inner
                    .http
                    .get(format!("{}{}", self.inner.src, CONFIG_URL)),
                None,
                None,
            )?
            .send()
            .await?;

        if response.status().is_success() {
            let value: Value = response.json().await?;
            return Ok(serde_json::from_value(value)?);
        }

        if response.status() == reqwest::StatusCode::UNAUTHORIZED {
            return Err(Error::Authentication(format!(
                "Could not load {} as credentials were not provided. Please login.",
                self.inner.src
            )));
        }

        if response.status() == reqwest::StatusCode::TOO_MANY_REQUESTS {
            return Err(Error::TooManyRequests(
                "Too many requests to the API, please try again later.".to_string(),
            ));
        }

        let fallback = self
            .apply_request_options(self.inner.http.get(&self.inner.src), None, None)?
            .send()
            .await?;

        if !fallback.status().is_success() {
            return Err(Error::Api(format!(
                "Could not fetch config for {}",
                self.inner.src
            )));
        }

        let html = fallback.text().await?;
        let Some(config_value) = parse_gradio_config_from_html(&html) else {
            return Err(Error::Api(format!(
                "Could not get Gradio config from {}",
                self.inner.src
            )));
        };

        if config_value.get("allow_flagging").is_some() {
            return Err(Error::Api(
                "Gradio 2.x is not supported by this client.".to_string(),
            ));
        }

        Ok(serde_json::from_value(config_value)?)
    }

    async fn get_api_info(
        &self,
        config: &Config,
        app_version: Version,
        src_prefixed: &str,
    ) -> Result<ApiInfo> {
        let raw_info = if app_version > Version::new(3, 36, 1) {
            let response = self
                .apply_request_options(
                    self.inner
                        .http
                        .get(format!("{}{}", src_prefixed, RAW_API_INFO_URL)),
                    None,
                    None,
                )?
                .send()
                .await?;

            if !response.status().is_success() {
                return Err(Error::Api(format!(
                    "Could not fetch api info for {}",
                    self.inner.src
                )));
            }

            response.json::<Value>().await?
        } else {
            let payload = json!({
                "config": serde_json::to_string(config)?,
                "serialize": false,
            });
            let response = self
                .apply_request_options(self.inner.http.post(SPACE_FETCHER_URL), None, None)?
                .json(&payload)
                .send()
                .await?;

            if !response.status().is_success() {
                return Err(Error::Api(format!(
                    "Could not fetch api info for {}",
                    self.inner.src
                )));
            }

            response
                .json::<Value>()
                .await?
                .get("api")
                .cloned()
                .unwrap_or(Value::Null)
        };

        let mut named_endpoints = Map::new();
        let mut unnamed_endpoints = Map::new();

        if let Some(named) = raw_info.get("named_endpoints").and_then(Value::as_object) {
            for (api_name, endpoint) in named {
                let include = if endpoint.get("api_visibility").is_some() {
                    endpoint
                        .get("api_visibility")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        != "private"
                } else {
                    endpoint
                        .get("show_api")
                        .and_then(Value::as_bool)
                        .unwrap_or(true)
                };
                if include {
                    named_endpoints.insert(api_name.clone(), endpoint.clone());
                }
            }
        }

        if let Some(unnamed) = raw_info.get("unnamed_endpoints").and_then(Value::as_object) {
            for (fn_index, endpoint) in unnamed {
                let include = if endpoint.get("api_visibility").is_some() {
                    endpoint
                        .get("api_visibility")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        != "private"
                } else {
                    endpoint
                        .get("show_api")
                        .and_then(Value::as_bool)
                        .unwrap_or(true)
                };
                if include {
                    unnamed_endpoints.insert(fn_index.clone(), endpoint.clone());
                }
            }
        }

        Ok(ApiInfo {
            named_endpoints,
            unnamed_endpoints,
        })
    }

    fn build_endpoints(
        &self,
        config: &Config,
        info: &ApiInfo,
        protocol: Protocol,
    ) -> HashMap<i64, Endpoint> {
        let mut endpoints = HashMap::new();
        for (index, dependency) in config.dependencies.iter().enumerate() {
            let fn_index = dependency.id.unwrap_or(index as i64);
            let api_name = dependency.api_name_as_string();

            let input_component_types = dependency
                .inputs
                .iter()
                .map(|id| self.get_component_type(config, *id))
                .collect::<Vec<_>>();
            let output_component_types = dependency
                .outputs
                .iter()
                .map(|id| self.get_component_type(config, *id))
                .collect::<Vec<_>>();

            let parameters_info = api_name
                .as_ref()
                .and_then(|api_name| info.named_endpoints.get(api_name))
                .and_then(|endpoint| endpoint.get("parameters"))
                .cloned()
                .and_then(|value| serde_json::from_value::<Vec<ParameterInfo>>(value).ok());

            let (is_valid, api_visibility) = if let Some(visibility) = &dependency.api_visibility {
                (visibility != "private", visibility.clone())
            } else {
                let is_valid = !matches!(dependency.api_name, Some(Value::Bool(false)));
                let visibility = if !is_valid {
                    "private".to_string()
                } else if dependency.show_api == Some(false) {
                    "undocumented".to_string()
                } else {
                    "public".to_string()
                };
                (is_valid, visibility)
            };

            endpoints.insert(
                fn_index,
                Endpoint {
                    fn_index,
                    dependency: dependency.clone(),
                    api_name,
                    protocol,
                    input_component_types,
                    output_component_types,
                    parameters_info,
                    backend_fn: dependency.backend_fn,
                    is_valid,
                    api_visibility,
                },
            );
        }
        endpoints
    }

    fn get_component_type(&self, config: &Config, component_id: i64) -> ComponentApiType {
        let component = config
            .components
            .iter()
            .find(|component| component.id == component_id)
            .cloned()
            .unwrap_or_else(Component::default);

        let skip = component
            .skip_api
            .unwrap_or_else(|| is_skipped_component(&component.component_type));
        let is_state = component.component_type == "state";

        ComponentApiType { skip, is_state }
    }

    fn render_endpoint_info(&self, name_or_index: &str, endpoint_info: &Value) -> Result<String> {
        let parameter_info = endpoint_info
            .get("parameters")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        let return_info = endpoint_info
            .get("returns")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();

        let parameter_names = parameter_info
            .iter()
            .map(|param| {
                param
                    .get("parameter_name")
                    .and_then(Value::as_str)
                    .or_else(|| param.get("label").and_then(Value::as_str))
                    .map(sanitize_parameter_names)
                    .unwrap_or_else(|| "arg".to_string())
            })
            .collect::<Vec<_>>();

        let rendered_parameters = if parameter_names.is_empty() {
            String::new()
        } else {
            format!("{}, ", parameter_names.join(", "))
        };

        let return_values = return_info
            .iter()
            .map(|param| {
                param
                    .get("label")
                    .and_then(Value::as_str)
                    .map(sanitize_parameter_names)
                    .unwrap_or_else(|| "output".to_string())
            })
            .collect::<Vec<_>>();

        let rendered_return_values = if return_values.len() > 1 {
            format!("({})", return_values.join(", "))
        } else {
            return_values.join(", ")
        };

        let final_param = if name_or_index.starts_with('/') {
            format!("api_name=\"{name_or_index}\"")
        } else {
            format!("fn_index={name_or_index}")
        };

        let mut out = format!(
            "\n - predict({rendered_parameters}{final_param}) -> {rendered_return_values}\n"
        );
        out.push_str("    Parameters:\n");

        if parameter_info.is_empty() {
            out.push_str("     - None\n");
        } else {
            for info in &parameter_info {
                let component = info
                    .get("component")
                    .and_then(Value::as_str)
                    .unwrap_or("Component");
                let label = info
                    .get("parameter_name")
                    .and_then(Value::as_str)
                    .or_else(|| info.get("label").and_then(Value::as_str))
                    .unwrap_or("arg");
                let mut type_str = info
                    .get("python_type")
                    .and_then(Value::as_object)
                    .and_then(|v| v.get("type"))
                    .and_then(Value::as_str)
                    .unwrap_or("Any")
                    .to_string();
                let has_default = info
                    .get("parameter_has_default")
                    .and_then(Value::as_bool)
                    .unwrap_or(false);
                let required = if has_default {
                    let default_value = info
                        .get("parameter_default")
                        .cloned()
                        .unwrap_or(Value::Null);
                    let default_value = traverse(
                        &default_value,
                        &|value| {
                            let url = value.get("url").and_then(Value::as_str).unwrap_or_default();
                            Value::String(format!("handle_file(\"{url}\")"))
                        },
                        &|value| is_file_obj_with_meta(value),
                    );
                    if default_value.is_null() {
                        type_str.push_str(" | None");
                    }
                    format!(
                        "(not required, defaults to: {})",
                        render_default_value(&default_value)
                    )
                } else {
                    "(required)".to_string()
                };
                let desc = info
                    .get("python_type")
                    .and_then(Value::as_object)
                    .and_then(|v| v.get("description"))
                    .and_then(Value::as_str)
                    .filter(|text| !text.is_empty())
                    .map(|text| format!(" ({text})"))
                    .unwrap_or_default();
                out.push_str(&format!(
                    "     - [{component}] {}: {type_str} {required}{desc}\n",
                    sanitize_parameter_names(label),
                ));
            }
        }

        out.push_str("    Returns:\n");
        if return_info.is_empty() {
            out.push_str("     - None\n");
        } else {
            for info in &return_info {
                let component = info
                    .get("component")
                    .and_then(Value::as_str)
                    .unwrap_or("Component");
                let label = info
                    .get("label")
                    .and_then(Value::as_str)
                    .unwrap_or("output");
                let type_str = info
                    .get("python_type")
                    .and_then(Value::as_object)
                    .and_then(|v| v.get("type"))
                    .and_then(Value::as_str)
                    .unwrap_or("Any");
                let desc = info
                    .get("python_type")
                    .and_then(Value::as_object)
                    .and_then(|v| v.get("description"))
                    .and_then(Value::as_str)
                    .filter(|text| !text.is_empty())
                    .map(|text| format!(" ({text})"))
                    .unwrap_or_default();
                out.push_str(&format!(
                    "     - [{component}] {}: {type_str}{desc}\n",
                    sanitize_parameter_names(label),
                ));
            }
        }

        Ok(out)
    }
}

fn analytics_enabled_default() -> bool {
    match std::env::var("GRADIO_ANALYTICS_ENABLED") {
        Ok(value) => {
            let normalized = value.trim().to_ascii_lowercase();
            !matches!(normalized.as_str(), "0" | "false" | "off" | "no")
        }
        Err(_) => true,
    }
}

fn render_default_value(value: &Value) -> String {
    match value {
        Value::String(text) if text.starts_with("handle_file(") => text.clone(),
        _ => value.to_string(),
    }
}

fn parse_version(raw: &str) -> Version {
    Version::parse(raw)
        .or_else(|_| Version::parse(&format!("{raw}.0")))
        .or_else(|_| Version::parse(&format!("{raw}.0.0")))
        .unwrap_or_else(|_| Version::new(2, 0, 0))
}

fn pop_sse_frame(buffer: &mut Vec<u8>) -> Option<String> {
    let mut index = None;
    for i in 0..buffer.len().saturating_sub(1) {
        if buffer[i] == b'\n' && buffer[i + 1] == b'\n' {
            index = Some(i);
            break;
        }
    }

    let index = index?;
    let frame = buffer[..index].to_vec();
    buffer.drain(..index + 2);
    Some(String::from_utf8_lossy(&frame).replace('\r', ""))
}

fn is_skipped_component(component_type: &str) -> bool {
    matches!(
        component_type,
        "state"
            | "row"
            | "column"
            | "tabs"
            | "tab"
            | "tabitem"
            | "box"
            | "form"
            | "accordion"
            | "group"
            | "interpretation"
            | "dataset"
            | "sidebar"
    )
}

impl Drop for Client {
    fn drop(&mut self) {
        self.inner.heartbeat_kill.store(true, Ordering::SeqCst);
        self.inner.heartbeat_notify.notify_waiters();
    }
}

#[cfg(test)]
mod tests {
    use super::{is_skipped_component, parse_version, pop_sse_frame};

    #[test]
    fn parse_version_with_short_input() {
        assert_eq!(parse_version("5.0"), semver::Version::new(5, 0, 0));
    }

    #[test]
    fn skip_component_set_contains_state() {
        assert!(is_skipped_component("state"));
        assert!(!is_skipped_component("textbox"));
    }

    #[test]
    fn pop_sse_frame_parses_until_double_newline() {
        let mut buffer = b"data: {\"msg\":\"heartbeat\"}\n\nmore".to_vec();
        let frame = pop_sse_frame(&mut buffer).unwrap();
        assert!(frame.contains("heartbeat"));
        assert_eq!(buffer, b"more");
    }
}
