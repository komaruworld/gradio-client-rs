use gradio_client_rs::{ApiView, Client, ClientOptions, ViewApiOptions, ViewApiReturnFormat};
use serde::Deserialize;
use serde_json::Value;

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

#[derive(Debug, Deserialize)]
struct TokenEntry {
    token: String,
    #[serde(default)]
    invalid: bool,
}

#[derive(Debug, Deserialize)]
struct TokenFile {
    tokens: Vec<TokenEntry>,
}

#[tokio::main]
async fn main() {
    let tokens = load_tokens();

    for space in SPACES {
        println!("\n=== {space} ===");
        let client = match init_client(space, &tokens).await {
            Ok(client) => client,
            Err(err) => {
                println!("init failed: {err}");
                continue;
            }
        };

        let view = client
            .view_api(ViewApiOptions {
                all_endpoints: Some(true),
                print_info: false,
                return_format: ViewApiReturnFormat::Dict,
            })
            .await;

        match view {
            Ok(ApiView::Dict(value)) => print_named_endpoints(&value),
            Ok(_) => println!("unexpected view_api format"),
            Err(err) => println!("view_api failed: {err}"),
        }

        client.close().await;
    }
}

fn load_tokens() -> Vec<String> {
    let Ok(raw) = std::fs::read_to_string("tokens.json") else {
        return Vec::new();
    };

    if let Ok(file) = serde_json::from_str::<TokenFile>(&raw) {
        return file
            .tokens
            .into_iter()
            .filter(|entry| !entry.invalid)
            .map(|entry| entry.token)
            .collect();
    }

    serde_json::from_str::<Vec<String>>(&raw).unwrap_or_default()
}

fn print_named_endpoints(value: &Value) {
    let Some(named) = value.get("named_endpoints").and_then(Value::as_object) else {
        println!("named_endpoints: none");
        return;
    };

    if named.is_empty() {
        println!("named_endpoints: empty");
        return;
    }

    for (api_name, endpoint) in named {
        println!("api_name: {api_name}");
        let params = endpoint
            .get("parameters")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        if params.is_empty() {
            println!("  parameters: none");
            continue;
        }

        for param in params {
            let label = param
                .get("parameter_name")
                .and_then(Value::as_str)
                .or_else(|| param.get("label").and_then(Value::as_str))
                .unwrap_or("arg");
            let component = param
                .get("component")
                .and_then(Value::as_str)
                .unwrap_or("Component");
            let py_type = param
                .get("python_type")
                .and_then(Value::as_object)
                .and_then(|obj| obj.get("type"))
                .and_then(Value::as_str)
                .unwrap_or("Any");
            let has_default = param
                .get("parameter_has_default")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            println!(
                "  - {} [{}] type={} required={}",
                label, component, py_type, !has_default
            );
        }
    }
}

async fn init_client(space: &str, tokens: &[String]) -> Result<Client, String> {
    let mut options = ClientOptions::default();
    options.verbose = false;

    if let Ok(client) = Client::new(space, options.clone()).await {
        return Ok(client);
    }

    for token in tokens {
        let mut with_token = options.clone();
        with_token.token = Some(token.clone());
        if let Ok(client) = Client::new(space, with_token).await {
            return Ok(client);
        }
    }

    Err("no working token found".to_string())
}
