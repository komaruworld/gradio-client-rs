use std::collections::HashMap;
use std::future::Future;
use std::path::{Path, PathBuf};
use std::pin::Pin;

use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine;
use futures_util::StreamExt;
use serde_json::{Map, Value};
use tokio::io::AsyncWriteExt;

use crate::error::{Error, Result};
use crate::types::ParameterInfo;

pub fn default_temp_dir() -> PathBuf {
    std::env::var("GRADIO_TEMP_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| std::env::temp_dir().join("gradio"))
}

pub fn is_http_url_like(value: &str) -> bool {
    value.starts_with("http://") || value.starts_with("https://")
}

pub async fn probe_url(possible_url: &str) -> bool {
    let client = reqwest::Client::new();
    let user_agent = "gradio (https://gradio.app/; gradio-team@huggingface.co)";

    match client
        .head(possible_url)
        .header(reqwest::header::USER_AGENT, user_agent)
        .send()
        .await
    {
        Ok(response) if response.status() == reqwest::StatusCode::METHOD_NOT_ALLOWED => client
            .get(possible_url)
            .header(reqwest::header::USER_AGENT, user_agent)
            .send()
            .await
            .map(|response| response.status().is_success())
            .unwrap_or(false),
        Ok(response) => response.status().is_success(),
        Err(_) => false,
    }
}

pub async fn is_valid_url(possible_url: &str) -> bool {
    is_http_url_like(possible_url) && probe_url(possible_url).await
}

pub fn is_filepath(value: &str) -> bool {
    let path = Path::new(value);
    path.exists() && path.is_file()
}

pub fn is_file_obj(value: &Value) -> bool {
    value
        .as_object()
        .and_then(|obj| obj.get("path"))
        .and_then(Value::as_str)
        .is_some()
}

pub fn is_file_obj_with_meta(value: &Value) -> bool {
    let Some(obj) = value.as_object() else {
        return false;
    };
    let Some(path) = obj.get("path").and_then(Value::as_str) else {
        return false;
    };
    if path.is_empty() {
        return false;
    }
    obj.get("meta")
        .and_then(Value::as_object)
        .and_then(|meta| meta.get("_type"))
        .and_then(Value::as_str)
        == Some("gradio.FileData")
}

pub fn is_file_obj_with_url(value: &Value) -> bool {
    is_file_obj_with_meta(value)
        && value
            .as_object()
            .and_then(|obj| obj.get("url"))
            .and_then(Value::as_str)
            .is_some()
}

pub fn get_mimetype(filename: &str) -> Option<String> {
    let lower = filename.to_ascii_lowercase();
    if lower.ends_with(".vtt") {
        return Some("text/vtt".to_string());
    }
    if lower.ends_with(".webp") {
        return Some("image/webp".to_string());
    }

    mime_guess::from_path(filename)
        .first_raw()
        .map(|mime| mime.replace("x-wav", "wav").replace("x-flac", "flac"))
}

pub fn get_extension(encoding: &str) -> Option<String> {
    let mime = if encoding.starts_with("data:") {
        encoding
            .trim_start_matches("data:")
            .split(';')
            .next()
            .unwrap_or_default()
            .replace("audio/wav", "audio/x-wav")
    } else {
        encoding
            .replace("audio/wav", "audio/x-wav")
            .split(';')
            .next()
            .unwrap_or_default()
            .to_string()
    };

    if mime == "audio/flac" {
        return Some("flac".to_string());
    }

    if mime.is_empty() {
        return None;
    }

    mime_guess::get_mime_extensions_str(&mime)
        .and_then(|extensions| extensions.first().map(|ext| (*ext).to_string()))
}

pub fn is_valid_file(file_path: &str, file_types: &[String]) -> bool {
    let mime_type = get_mimetype(file_path);

    for file_type in file_types {
        if file_type == "file" {
            return true;
        }

        if let Some(extension) = file_type.strip_prefix('.') {
            let file_ext = Path::new(file_path)
                .extension()
                .and_then(|ext| ext.to_str())
                .unwrap_or_default()
                .to_ascii_lowercase();
            if extension.to_ascii_lowercase() == file_ext {
                return true;
            }
        } else if let Some(mime_type) = &mime_type {
            if mime_type.starts_with(&format!("{file_type}/")) {
                return true;
            }
        }
    }

    false
}

pub async fn create_tmp_copy_of_file(file_path: &str, dir: Option<&Path>) -> Result<PathBuf> {
    let directory = dir
        .map(Path::to_path_buf)
        .unwrap_or_else(std::env::temp_dir)
        .join(uuid::Uuid::new_v4().to_string());
    tokio::fs::create_dir_all(&directory).await?;

    let destination = directory.join(
        Path::new(file_path)
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("file"),
    );
    tokio::fs::copy(file_path, &destination).await?;
    Ok(destination)
}

pub async fn download_tmp_copy_of_file(
    url_path: &str,
    token: Option<&str>,
    dir: Option<&Path>,
) -> Result<PathBuf> {
    let directory = dir
        .map(Path::to_path_buf)
        .unwrap_or_else(std::env::temp_dir)
        .join(uuid::Uuid::new_v4().to_string());
    tokio::fs::create_dir_all(&directory).await?;

    let destination = directory.join(
        Path::new(url_path)
            .file_name()
            .and_then(|name| name.to_str())
            .filter(|name| !name.is_empty())
            .unwrap_or("file"),
    );

    let mut request = reqwest::Client::new().get(url_path);
    if let Some(token) = token {
        request = request.bearer_auth(token);
    }

    let response = request.send().await?;
    if !response.status().is_success() {
        return Err(Error::Api(format!(
            "failed to download temporary copy from {url_path}: {}",
            response.status()
        )));
    }

    let bytes = response.bytes().await?;
    tokio::fs::write(&destination, &bytes).await?;
    Ok(destination)
}

pub async fn encode_file_to_base64(path: impl AsRef<Path>) -> Result<String> {
    let path = path.as_ref();
    let bytes = tokio::fs::read(path).await?;
    let base64 = BASE64.encode(bytes);
    let mime = get_mimetype(&path.to_string_lossy()).unwrap_or_default();
    Ok(format!("data:{mime};base64,{base64}"))
}

pub async fn encode_url_to_base64(url: &str) -> Result<String> {
    let response = reqwest::Client::new().get(url).send().await?;
    if !response.status().is_success() {
        return Err(Error::Api(format!(
            "failed to download content from {url}: {}",
            response.status()
        )));
    }
    let bytes = response.bytes().await?;
    let base64 = BASE64.encode(bytes);
    let mime = get_mimetype(url).unwrap_or_default();
    Ok(format!("data:{mime};base64,{base64}"))
}

pub async fn encode_url_or_file_to_base64(path_or_url: &str) -> Result<String> {
    if is_http_url_like(path_or_url) {
        encode_url_to_base64(path_or_url).await
    } else {
        encode_file_to_base64(path_or_url).await
    }
}

pub async fn download_byte_stream(url: &str, token: Option<&str>) -> Result<Vec<u8>> {
    let mut request = reqwest::Client::new().get(url);
    if let Some(token) = token {
        request = request.bearer_auth(token);
    }

    let response = request.send().await?;
    if !response.status().is_success() {
        return Err(Error::Api(format!(
            "failed to stream bytes from {url}: {}",
            response.status()
        )));
    }

    let mut stream = response.bytes_stream();
    let mut bytes = Vec::new();
    while let Some(chunk) = stream.next().await {
        bytes.extend_from_slice(&chunk?);
    }
    Ok(bytes)
}

pub async fn download_byte_stream_chunks(url: &str, token: Option<&str>) -> Result<Vec<Vec<u8>>> {
    let mut request = reqwest::Client::new().get(url);
    if let Some(token) = token {
        request = request.bearer_auth(token);
    }

    let response = request.send().await?;
    if !response.status().is_success() {
        return Err(Error::Api(format!(
            "failed to stream byte chunks from {url}: {}",
            response.status()
        )));
    }

    let mut stream = response.bytes_stream();
    let mut chunks = Vec::new();
    while let Some(chunk) = stream.next().await {
        chunks.push(chunk?.to_vec());
    }
    Ok(chunks)
}

pub fn decode_base64_to_binary(encoding: &str) -> Result<(Vec<u8>, Option<String>)> {
    let extension = get_extension(encoding);
    let data = encoding.rsplit(',').next().unwrap_or_default();
    BASE64
        .decode(data)
        .map(|decoded| (decoded, extension))
        .map_err(|err| Error::Api(format!("invalid base64 payload: {err}")))
}

pub async fn decode_base64_to_file(
    encoding: &str,
    file_path: Option<&Path>,
    dir: Option<&Path>,
) -> Result<PathBuf> {
    let (data, extension) = decode_base64_to_binary(encoding)?;
    let directory = dir
        .map(Path::to_path_buf)
        .unwrap_or_else(std::env::temp_dir)
        .join(uuid::Uuid::new_v4().to_string());
    tokio::fs::create_dir_all(&directory).await?;

    let output = if let Some(file_path) = file_path {
        file_path.to_path_buf()
    } else {
        let ext = extension.unwrap_or_else(|| "bin".to_string());
        directory.join(format!("file.{ext}"))
    };

    let mut file = tokio::fs::File::create(&output).await?;
    file.write_all(&data).await?;
    file.flush().await?;
    Ok(output)
}

pub fn strip_invalid_filename_characters(filename: &str, max_bytes: usize) -> String {
    let mut name = String::new();
    let mut ext = String::new();
    if let Some((left, right)) = filename.rsplit_once('.') {
        name.push_str(left);
        ext.push('.');
        ext.push_str(right);
    } else {
        name.push_str(filename);
    }

    let filtered: String = name
        .chars()
        .filter(|ch| ch.is_alphanumeric() || matches!(ch, '.' | '_' | '-' | ',' | ' '))
        .collect();

    let mut result = format!("{filtered}{ext}");
    while result.as_bytes().len() > max_bytes {
        if filtered.is_empty() {
            break;
        }
        let mut tmp = filtered.clone();
        tmp.pop();
        result = format!("{tmp}{ext}");
        if tmp.is_empty() {
            break;
        }
    }
    result
}

pub fn sanitize_parameter_names(original: &str) -> String {
    original
        .chars()
        .filter(|ch| ch.is_alphanumeric() || *ch == ' ' || *ch == '_')
        .collect::<String>()
        .replace(' ', "_")
        .to_lowercase()
}

pub fn handle_file(path_or_url: impl AsRef<str>) -> Result<Value> {
    let raw = path_or_url.as_ref().to_string();
    let mut out = Map::new();
    out.insert("path".to_string(), Value::String(raw.clone()));
    out.insert(
        "meta".to_string(),
        serde_json::json!({"_type": "gradio.FileData"}),
    );

    if is_http_url_like(&raw) {
        let orig_name = raw
            .split('/')
            .next_back()
            .filter(|s| !s.is_empty())
            .unwrap_or("file")
            .to_string();
        out.insert("orig_name".to_string(), Value::String(orig_name));
        out.insert("url".to_string(), Value::String(raw));
        return Ok(Value::Object(out));
    }

    if is_filepath(&raw) {
        let name = Path::new(&raw)
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("file")
            .to_string();
        out.insert("orig_name".to_string(), Value::String(name));
        return Ok(Value::Object(out));
    }

    Err(Error::Api(format!(
        "File {raw} does not exist on local filesystem and is not a valid URL."
    )))
}

pub fn file(path_or_url: impl AsRef<str>) -> Result<Value> {
    handle_file(path_or_url)
}

pub fn construct_args(
    parameters_info: Option<&Vec<ParameterInfo>>,
    args: Vec<Value>,
    kwargs: Map<String, Value>,
) -> Result<Vec<Value>> {
    if parameters_info.is_none() {
        if !kwargs.is_empty() {
            return Err(Error::Api(
                "This endpoint does not support key-word arguments.".to_string(),
            ));
        }
        return Ok(args);
    }

    let parameters_info = parameters_info.expect("checked is_some above");
    let num_args = args.len();
    let mut out: Vec<Option<Value>> = args.into_iter().map(Some).collect();
    if out.len() < parameters_info.len() {
        out.resize(parameters_info.len(), None);
    }

    let mut kwarg_arg_mapping = HashMap::<String, usize>::new();
    let mut kwarg_names = Vec::<String>::new();

    for (index, param_info) in parameters_info.iter().enumerate() {
        if let Some(parameter_name) = &param_info.parameter_name {
            kwarg_arg_mapping.insert(parameter_name.clone(), index);
            kwarg_names.push(parameter_name.clone());
        } else {
            kwarg_names.push(format!("argument {index}"));
        }

        if param_info.parameter_has_default.unwrap_or(false) && out[index].is_none() {
            out[index] = Some(param_info.parameter_default.clone().unwrap_or(Value::Null));
        }
    }

    for (key, value) in kwargs {
        let Some(index) = kwarg_arg_mapping.get(&key).copied() else {
            return Err(Error::Api(format!(
                "Parameter `{key}` is not a valid key-word argument."
            )));
        };

        if index < num_args {
            return Err(Error::Api(format!(
                "Parameter `{key}` is already set as a positional argument."
            )));
        }

        out[index] = Some(value);
    }

    let Some(missing_index) = out.iter().position(Option::is_none) else {
        return Ok(out.into_iter().flatten().collect());
    };

    let name = kwarg_names
        .get(missing_index)
        .cloned()
        .unwrap_or_else(|| format!("argument {missing_index}"));
    Err(Error::Api(format!(
        "No value provided for required argument: {name}"
    )))
}

pub fn extract_validation_message(status_code: u16, body: &Value) -> Option<String> {
    if status_code != 422 {
        return None;
    }

    let mut messages: Vec<String> = Vec::new();
    if let Some(detail) = body.get("detail").and_then(Value::as_array) {
        for (index, error_info) in detail.iter().enumerate() {
            if error_info
                .get("__type__")
                .and_then(Value::as_str)
                .unwrap_or_default()
                == "validate"
                && error_info
                    .get("is_valid")
                    .and_then(Value::as_bool)
                    .is_some_and(|valid| !valid)
            {
                let param_name = error_info
                    .get("parameter_name")
                    .and_then(Value::as_str)
                    .map(ToOwned::to_owned)
                    .unwrap_or_else(|| format!("parameter_{index}"));
                let message = error_info
                    .get("message")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                messages.push(format!("- {param_name}: {message}"));
            }
        }
    }

    if messages.is_empty() {
        None
    } else {
        let mut with_header = vec![format!(
            "{} parameter(s) failed validation:",
            messages.len()
        )];
        with_header.extend(messages);
        Some(with_header.join("\n"))
    }
}

pub async fn set_space_timeout(
    space_id: &str,
    token: Option<&str>,
    timeout_in_seconds: i64,
) -> Result<()> {
    let mut request = reqwest::Client::new()
        .post(format!(
            "https://huggingface.co/api/spaces/{space_id}/sleeptime"
        ))
        .json(&serde_json::json!({ "seconds": timeout_in_seconds }));

    if let Some(token) = token {
        request = request.bearer_auth(token);
    }

    let response = request.send().await?;
    if !response.status().is_success() {
        return Err(Error::HuggingFace(format!(
            "Could not set sleep timeout on duplicated Space {}: {}",
            space_id,
            response.status()
        )));
    }

    Ok(())
}

#[derive(Debug, Clone)]
pub enum JsonInput {
    Str(String),
    Value(Value),
}

impl From<&str> for JsonInput {
    fn from(value: &str) -> Self {
        Self::Str(value.to_string())
    }
}

impl From<String> for JsonInput {
    fn from(value: String) -> Self {
        Self::Str(value)
    }
}

impl From<Value> for JsonInput {
    fn from(value: Value) -> Self {
        Self::Value(value)
    }
}

pub async fn dict_or_str_to_json_file(
    jsn: impl Into<JsonInput>,
    dir: Option<&Path>,
) -> Result<PathBuf> {
    let value = match jsn.into() {
        JsonInput::Str(raw) => serde_json::from_str::<Value>(&raw)
            .map_err(|err| Error::Api(format!("invalid JSON string: {err}")))?,
        JsonInput::Value(value) => value,
    };

    let directory = dir
        .map(Path::to_path_buf)
        .unwrap_or_else(std::env::temp_dir)
        .join(uuid::Uuid::new_v4().to_string());
    tokio::fs::create_dir_all(&directory).await?;

    let file_path = directory.join("payload.json");
    let pretty = serde_json::to_vec_pretty(&value)?;
    tokio::fs::write(&file_path, pretty).await?;
    Ok(file_path)
}

pub async fn file_to_json(file_path: impl AsRef<Path>) -> Result<Value> {
    let bytes = tokio::fs::read(file_path).await?;
    serde_json::from_slice::<Value>(&bytes)
        .map_err(|err| Error::Api(format!("invalid JSON file: {err}")))
}

pub fn traverse(
    json_obj: &Value,
    func: &dyn Fn(&Value) -> Value,
    is_root: &dyn Fn(&Value) -> bool,
) -> Value {
    if is_root(json_obj) {
        return func(json_obj);
    }

    match json_obj {
        Value::Object(map) => {
            let mut out = Map::new();
            for (key, value) in map {
                out.insert(key.clone(), traverse(value, func, is_root));
            }
            Value::Object(out)
        }
        Value::Array(items) => Value::Array(
            items
                .iter()
                .map(|item| traverse(item, func, is_root))
                .collect(),
        ),
        _ => json_obj.clone(),
    }
}

pub fn async_traverse<'a, F, Fut, G>(
    json_obj: &'a Value,
    func: &'a F,
    is_root: &'a G,
) -> Pin<Box<dyn Future<Output = Value> + Send + 'a>>
where
    F: Fn(&Value) -> Fut + Send + Sync + 'a,
    Fut: Future<Output = Value> + Send + 'a,
    G: Fn(&Value) -> bool + Send + Sync + 'a,
{
    Box::pin(async move {
        if is_root(json_obj) {
            return func(json_obj).await;
        }

        match json_obj {
            Value::Object(map) => {
                let mut out = Map::new();
                for (key, value) in map {
                    out.insert(key.clone(), async_traverse(value, func, is_root).await);
                }
                Value::Object(out)
            }
            Value::Array(items) => {
                let mut out = Vec::with_capacity(items.len());
                for item in items {
                    out.push(async_traverse(item, func, is_root).await);
                }
                Value::Array(out)
            }
            _ => json_obj.clone(),
        }
    })
}

pub fn get_type(schema: &Value) -> Option<String> {
    if let Some(value) = schema.get("const") {
        return Some(match value {
            Value::String(_) => "string".to_string(),
            Value::Number(number) if number.is_i64() => "integer".to_string(),
            Value::Number(_) => "number".to_string(),
            Value::Bool(_) => "boolean".to_string(),
            Value::Null => "null".to_string(),
            Value::Array(_) => "array".to_string(),
            Value::Object(_) => "object".to_string(),
        });
    }

    if let Some(schema_type) = schema.get("type") {
        match schema_type {
            Value::String(value) => Some(value.to_string()),
            Value::Array(values) => values
                .first()
                .and_then(Value::as_str)
                .map(ToOwned::to_owned),
            _ => None,
        }
    } else if schema.get("enum").is_some() {
        Some("enum".to_string())
    } else {
        None
    }
}

pub fn json_schema_to_python_type(schema: &Value) -> String {
    json_schema_to_python_type_internal(schema, schema.get("$defs").and_then(Value::as_object))
}

fn json_schema_to_python_type_internal(
    schema: &Value,
    defs: Option<&serde_json::Map<String, Value>>,
) -> String {
    if let Some(reference) = schema.get("$ref").and_then(Value::as_str) {
        let key = reference
            .trim_start_matches("#/$defs/")
            .trim_start_matches("#/definitions/");
        if let Some(defs) = defs {
            if let Some(definition) = defs.get(key) {
                return json_schema_to_python_type_internal(definition, Some(defs));
            }
        }
        return key.to_string();
    }

    if let Some(any_of) = schema.get("anyOf").and_then(Value::as_array) {
        return any_of
            .iter()
            .map(|item| json_schema_to_python_type_internal(item, defs))
            .collect::<Vec<_>>()
            .join(" | ");
    }

    if let Some(one_of) = schema.get("oneOf").and_then(Value::as_array) {
        return one_of
            .iter()
            .map(|item| json_schema_to_python_type_internal(item, defs))
            .collect::<Vec<_>>()
            .join(" | ");
    }

    if let Some(values) = schema.get("enum").and_then(Value::as_array) {
        let literals = values
            .iter()
            .map(|value| match value {
                Value::String(value) => format!("\"{value}\""),
                _ => value.to_string(),
            })
            .collect::<Vec<_>>()
            .join(", ");
        return format!("Literal[{literals}]");
    }

    match get_type(schema).as_deref() {
        Some("string") => "str".to_string(),
        Some("integer") => "int".to_string(),
        Some("number") => "float".to_string(),
        Some("boolean") => "bool".to_string(),
        Some("null") => "None".to_string(),
        Some("array") => {
            let item_type = schema
                .get("items")
                .map(|items| json_schema_to_python_type_internal(items, defs))
                .unwrap_or_else(|| "Any".to_string());
            format!("list[{item_type}]")
        }
        Some("object") => {
            if schema
                .get("properties")
                .and_then(Value::as_object)
                .is_some()
            {
                "dict[str, Any]".to_string()
            } else {
                "dict".to_string()
            }
        }
        Some(other) => other.to_string(),
        None => "Any".to_string(),
    }
}

pub fn python_type_to_json_schema(type_hint: &str) -> Value {
    let type_hint = type_hint.trim();
    if type_hint.contains('|') {
        let options = type_hint
            .split('|')
            .map(|part| python_type_to_json_schema(part.trim()))
            .collect::<Vec<_>>();
        return serde_json::json!({"anyOf": options});
    }

    match type_hint {
        "str" | "String" => serde_json::json!({"type": "string"}),
        "int" | "i64" | "u64" | "usize" => serde_json::json!({"type": "integer"}),
        "float" | "f64" | "f32" | "number" => serde_json::json!({"type": "number"}),
        "bool" => serde_json::json!({"type": "boolean"}),
        "None" | "null" => serde_json::json!({"type": "null"}),
        hint if hint.starts_with("list[") && hint.ends_with(']') => {
            let inner = &hint[5..hint.len() - 1];
            serde_json::json!({"type": "array", "items": python_type_to_json_schema(inner)})
        }
        hint if hint.starts_with("dict[") || hint == "dict" => {
            serde_json::json!({"type": "object"})
        }
        hint if hint.starts_with("Literal[") && hint.ends_with(']') => {
            let inner = &hint[8..hint.len() - 1];
            let values = inner
                .split(',')
                .map(|part| {
                    let part = part.trim();
                    if let Some(stripped) = part.strip_prefix('"').and_then(|p| p.strip_suffix('"'))
                    {
                        Value::String(stripped.to_string())
                    } else if let Ok(value) = part.parse::<i64>() {
                        Value::Number(value.into())
                    } else if part.eq_ignore_ascii_case("true") {
                        Value::Bool(true)
                    } else if part.eq_ignore_ascii_case("false") {
                        Value::Bool(false)
                    } else {
                        Value::String(part.to_string())
                    }
                })
                .collect::<Vec<_>>();
            serde_json::json!({"enum": values})
        }
        _ => serde_json::json!({}),
    }
}

pub fn value_is_file(api_info: &Value) -> bool {
    let rendered = json_schema_to_python_type(api_info);
    rendered.contains("FileData")
        || rendered.contains("filepath")
        || rendered.contains("list[FileData]")
}

#[derive(Debug, Clone)]
enum PathToken {
    Key(String),
    Index(usize),
}

fn parse_path(path: &Value) -> Result<Vec<PathToken>> {
    let Some(segments) = path.as_array() else {
        return Err(Error::Api("diff path must be an array".to_string()));
    };

    let mut out = Vec::with_capacity(segments.len());
    for segment in segments {
        if let Some(index) = segment.as_u64() {
            out.push(PathToken::Index(index as usize));
        } else if let Some(index) = segment.as_i64() {
            out.push(PathToken::Index(index.max(0) as usize));
        } else if let Some(key) = segment.as_str() {
            out.push(PathToken::Key(key.to_string()));
        } else {
            return Err(Error::Api("unsupported diff path segment".to_string()));
        }
    }

    Ok(out)
}

fn apply_append(target: &mut Value, value: &Value) -> Result<()> {
    match (target, value) {
        (Value::String(dst), Value::String(src)) => {
            dst.push_str(src);
            Ok(())
        }
        (Value::Array(dst), Value::Array(src)) => {
            dst.extend(src.iter().cloned());
            Ok(())
        }
        _ => Err(Error::Api(
            "append is only supported for strings and arrays".to_string(),
        )),
    }
}

fn apply_action(target: &mut Value, action: &str, value: &Value) -> Result<()> {
    match action {
        "replace" => {
            *target = value.clone();
            Ok(())
        }
        "append" => apply_append(target, value),
        _ => Err(Error::Api(format!("unsupported root action: {action}"))),
    }
}

fn apply_edit(target: &mut Value, path: &[PathToken], action: &str, value: &Value) -> Result<()> {
    if path.is_empty() {
        return apply_action(target, action, value);
    }

    match (&path[0], target) {
        (PathToken::Index(index), Value::Array(items)) => {
            if path.len() == 1 {
                match action {
                    "replace" => {
                        if *index >= items.len() {
                            return Err(Error::Api("replace index out of bounds".to_string()));
                        }
                        items[*index] = value.clone();
                        Ok(())
                    }
                    "append" => {
                        if *index >= items.len() {
                            return Err(Error::Api("append index out of bounds".to_string()));
                        }
                        apply_append(&mut items[*index], value)
                    }
                    "add" => {
                        let idx = (*index).min(items.len());
                        items.insert(idx, value.clone());
                        Ok(())
                    }
                    "delete" => {
                        if *index >= items.len() {
                            return Err(Error::Api("delete index out of bounds".to_string()));
                        }
                        items.remove(*index);
                        Ok(())
                    }
                    _ => Err(Error::Api(format!("unknown action: {action}"))),
                }
            } else {
                if *index >= items.len() {
                    return Err(Error::Api("path index out of bounds".to_string()));
                }
                apply_edit(&mut items[*index], &path[1..], action, value)
            }
        }
        (PathToken::Key(key), Value::Object(map)) => {
            if path.len() == 1 {
                match action {
                    "replace" => {
                        map.insert(key.clone(), value.clone());
                        Ok(())
                    }
                    "append" => {
                        let Some(entry) = map.get_mut(key) else {
                            return Err(Error::Api("append key missing".to_string()));
                        };
                        apply_append(entry, value)
                    }
                    "add" => {
                        map.insert(key.clone(), value.clone());
                        Ok(())
                    }
                    "delete" => {
                        map.remove(key);
                        Ok(())
                    }
                    _ => Err(Error::Api(format!("unknown action: {action}"))),
                }
            } else {
                let Some(next) = map.get_mut(key) else {
                    return Err(Error::Api("path key missing".to_string()));
                };
                apply_edit(next, &path[1..], action, value)
            }
        }
        _ => Err(Error::Api(
            "path does not match target structure".to_string(),
        )),
    }
}

pub fn apply_diff(obj: &Value, diff: &Value) -> Result<Value> {
    let mut current = obj.clone();
    let edits = diff
        .as_array()
        .ok_or_else(|| Error::Api("diff must be an array".to_string()))?;

    for edit in edits {
        let parts = edit
            .as_array()
            .ok_or_else(|| Error::Api("diff operation must be a 3-item array".to_string()))?;
        if parts.len() != 3 {
            return Err(Error::Api(
                "diff operation must contain 3 items".to_string(),
            ));
        }
        let action = parts[0]
            .as_str()
            .ok_or_else(|| Error::Api("diff action must be string".to_string()))?;
        let path = parse_path(&parts[1])?;
        apply_edit(&mut current, &path, action, &parts[2])?;
    }

    Ok(current)
}

pub fn parse_gradio_config_from_html(html: &str) -> Option<Value> {
    let marker = "window.gradio_config = ";
    let start = html.find(marker)? + marker.len();
    let rest = &html[start..];
    let end = rest.find(";</script>")?;
    let json_str = rest[..end].trim();
    serde_json::from_str(json_str).ok()
}

pub fn ensure_trailing_slash(mut src: String) -> String {
    if !src.ends_with('/') {
        src.push('/');
    }
    src
}

pub fn maybe_prefix_https(host_or_url: &str) -> String {
    if host_or_url.starts_with("http://") || host_or_url.starts_with("https://") {
        host_or_url.to_string()
    } else {
        format!("https://{host_or_url}")
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::{
        apply_diff, construct_args, get_extension, get_mimetype, handle_file,
        json_schema_to_python_type, python_type_to_json_schema,
    };
    use crate::types::{ParameterInfo, PythonTypeInfo};

    #[test]
    fn apply_diff_works() {
        let obj = json!(["a", {"text": "hello"}]);
        let diff = json!(["append", [1, "text"], " world"]);
        let diff = json!([diff, ["replace", [0], "b"]]);
        let out = apply_diff(&obj, &diff).unwrap();
        assert_eq!(out, json!(["b", {"text": "hello world"}]));
    }

    #[test]
    fn construct_args_supports_defaults_and_kwargs() {
        let params = vec![
            ParameterInfo {
                label: "x".to_string(),
                parameter_name: Some("x".to_string()),
                parameter_has_default: Some(false),
                parameter_default: None,
                python_type: Some(PythonTypeInfo {
                    r#type: Some("int".to_string()),
                    description: None,
                }),
                component: Some("Number".to_string()),
            },
            ParameterInfo {
                label: "y".to_string(),
                parameter_name: Some("y".to_string()),
                parameter_has_default: Some(true),
                parameter_default: Some(json!(5)),
                python_type: Some(PythonTypeInfo {
                    r#type: Some("int".to_string()),
                    description: None,
                }),
                component: Some("Number".to_string()),
            },
        ];

        let out = construct_args(Some(&params), vec![json!(3)], serde_json::Map::new()).unwrap();
        assert_eq!(out, vec![json!(3), json!(5)]);

        let mut kwargs = serde_json::Map::new();
        kwargs.insert("y".to_string(), json!(9));
        let out = construct_args(Some(&params), vec![json!(3)], kwargs).unwrap();
        assert_eq!(out, vec![json!(3), json!(9)]);
    }

    #[test]
    fn handle_file_url() {
        let out = handle_file("https://example.com/a.png").unwrap();
        assert_eq!(out["meta"]["_type"], "gradio.FileData");
        assert_eq!(out["orig_name"], "a.png");
    }

    #[test]
    fn mimetype_and_extension_helpers() {
        assert_eq!(get_mimetype("demo.webp").as_deref(), Some("image/webp"));
        assert_eq!(
            get_extension("data:image/png;base64,aaa").as_deref(),
            Some("png")
        );
    }

    #[test]
    fn schema_conversion_helpers() {
        let schema = json!({"type": "array", "items": {"type": "string"}});
        assert_eq!(json_schema_to_python_type(&schema), "list[str]");

        let back = python_type_to_json_schema("list[str]");
        assert_eq!(back["type"], "array");
    }
}
