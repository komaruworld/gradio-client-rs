use gradio_client_rs::{ApiView, Client, ClientOptions, ViewApiOptions, ViewApiReturnFormat};
use serde::Deserialize;
use serde_json::Value;

const CASES: [(&str, &str); 12] = [
    ("linoyts/Qwen-Image-Edit-2511-Fast", "/edit"),
    ("Qwen/Qwen-Image-Edit-2511", "/edit4"),
    ("dream2589632147/Dream-wan2-2-faster-Pro", "/vid"),
    ("Heartsync/NSFW-Uncensored-video2", "/vid2"),
    ("Lightricks/ltx-2-distilled", "/vid3"),
    ("multimodalart/wan-2-2-first-last-frame", "/vid4"),
    ("mrfakename/Z-Image-Turbo", "/z"),
    ("Qwen/Qwen-Image-2512", "/qwen"),
    ("V0pr0S/ComfyUI-Reactor-Fast-Face-Swap-CPU", "/swap"),
    ("not-lain/background-removal", "/rmbg"),
    ("amd/gpt-oss-120b-chatbot", "/ask"),
    ("mrfakename/HeartMuLa", "/music"),
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

    let mut passed = 0usize;
    let mut failed = 0usize;
    let mut failures = Vec::new();

    for (space, api_name) in CASES {
        println!("\n=== {space} ({api_name}) ===");
        let client = match init_client(space, &tokens).await {
            Ok(client) => client,
            Err(err) => {
                failed += 1;
                failures.push(format!("{space}: init failed: {err}"));
                println!("FAIL init: {err}");
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

        let (has_api, named_apis) = match view {
            Ok(ApiView::Dict(value)) => {
                let named = named_api_names(&value);
                (named.iter().any(|name| name == api_name), named)
            }
            Ok(_) => (false, Vec::new()),
            Err(err) => {
                failed += 1;
                failures.push(format!("{space}: view_api failed: {err}"));
                println!("FAIL view_api: {err}");
                let _ = client.close().await;
                continue;
            }
        };

        client.close().await;

        if has_api {
            passed += 1;
            println!("OK api_name found");
        } else {
            failed += 1;
            failures.push(format!(
                "{space}: api_name '{api_name}' not found in named_endpoints"
            ));
            if named_apis.is_empty() {
                println!("FAIL api_name not found (named_endpoints is empty)");
            } else {
                println!(
                    "FAIL api_name not found; named_endpoints={}",
                    named_apis.join(", ")
                );
            }
        }
    }

    println!("\nSummary: passed={passed}, failed={failed}");
    if !failures.is_empty() {
        println!("Failures:");
        for item in failures {
            println!("- {item}");
        }
        std::process::exit(1);
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

fn named_api_names(value: &Value) -> Vec<String> {
    value
        .get("named_endpoints")
        .and_then(Value::as_object)
        .map(|named| named.keys().cloned().collect())
        .unwrap_or_default()
}

async fn init_client(space: &str, tokens: &[String]) -> Result<Client, String> {
    let mut opts = ClientOptions::default();
    opts.verbose = false;

    match Client::new(space, opts.clone()).await {
        Ok(client) => return Ok(client),
        Err(err) => {
            let message = err.to_string();
            if !is_auth_like_error(&message) {
                return Err(message);
            }
        }
    }

    for token in tokens {
        let mut token_opts = opts.clone();
        token_opts.token = Some(token.clone());
        match Client::new(space, token_opts).await {
            Ok(client) => return Ok(client),
            Err(err) => {
                let message = err.to_string();
                if !is_auth_like_error(&message) {
                    return Err(message);
                }
            }
        }
    }

    Err("no working token found".to_string())
}

fn is_auth_like_error(message: &str) -> bool {
    let normalized = message.to_ascii_lowercase();
    normalized.contains("authentication")
        || normalized.contains("unauthorized")
        || normalized.contains("credentials")
        || normalized.contains("token")
        || normalized.contains("forbidden")
        || normalized.contains("private space")
}
