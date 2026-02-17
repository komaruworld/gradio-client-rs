use gradio_client_rs::{CallOptions, Client, ClientOptions};
use serde_json::json;

#[tokio::test]
#[ignore = "requires network access and a live Gradio endpoint"]
async fn parity_predict_named_endpoint() {
    let client = Client::new("gradio/calculator", ClientOptions::default())
        .await
        .expect("client should initialize");

    let out = client
        .predict(
            vec![json!(5), json!("add"), json!(4)],
            CallOptions {
                api_name: Some("/predict".to_string()),
                ..CallOptions::default()
            },
        )
        .await
        .expect("predict should work");

    assert_eq!(out, json!(9.0));
}

#[tokio::test]
#[ignore = "requires network access and a live Gradio endpoint"]
async fn parity_submit_job_lifecycle() {
    let client = Client::new("gradio/calculator", ClientOptions::default())
        .await
        .expect("client should initialize");

    let job = client
        .submit(
            vec![json!(5), json!("add"), json!(4)],
            CallOptions {
                api_name: Some("/predict".to_string()),
                ..CallOptions::default()
            },
        )
        .await
        .expect("submit should work");

    let result = job.result().await.expect("result should resolve");
    assert_eq!(result, json!(9.0));
}

#[tokio::test]
#[ignore = "requires network access and HF auth token"]
async fn parity_duplicate_flow() {
    let token = std::env::var("HF_TOKEN").expect("HF_TOKEN must be set for duplicate parity test");
    let options = gradio_client_rs::DuplicateOptions {
        token: Some(token),
        private: true,
        verbose: false,
        ..Default::default()
    };

    let _client = Client::duplicate("gradio/calculator", options)
        .await
        .expect("duplicate should initialize a client");
}
