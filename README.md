# gradio_client_rs

Async Rust client for Gradio apps and Hugging Face Spaces.

## Install

```toml
[dependencies]
gradio_client_rs = "0.1.0"
```

## Quick Start

```rust
use gradio_client_rs::{CallOptions, Client, ClientOptions};
use serde_json::json;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Client::new("gradio/calculator", ClientOptions::default()).await?;
    let result = client
        .predict(
            vec![json!(5), json!("add"), json!(4)],
            CallOptions {
                api_name: Some("/predict".to_string()),
                ..CallOptions::default()
            },
        )
        .await?;
    println!("{result}");
    Ok(())
}
```

## License

Apache-2.0
