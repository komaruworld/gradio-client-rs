use std::collections::HashMap;

use reqwest::{header, StatusCode};
use serde_json::json;

use crate::error::{Error, Result};
use crate::types::{SpaceInfoResponse, SpaceRuntimeResponse, WhoAmIResponse};

const HF_ENDPOINT: &str = "https://huggingface.co";

#[derive(Clone, Debug)]
pub struct HfApi {
    http: reqwest::Client,
    token: Option<String>,
    user_agent: String,
}

impl HfApi {
    pub fn new(http: reqwest::Client, token: Option<String>, user_agent: String) -> Self {
        Self {
            http,
            token,
            user_agent,
        }
    }

    fn headers(&self) -> Result<header::HeaderMap> {
        let mut headers = header::HeaderMap::new();
        headers.insert(
            header::USER_AGENT,
            header::HeaderValue::from_str(&self.user_agent)
                .map_err(|err| Error::Api(format!("invalid user-agent header: {err}")))?,
        );
        if let Some(token) = &self.token {
            let value = format!("Bearer {token}");
            headers.insert(
                header::AUTHORIZATION,
                header::HeaderValue::from_str(&value)
                    .map_err(|err| Error::Api(format!("invalid authorization header: {err}")))?,
            );
        }
        Ok(headers)
    }

    pub async fn whoami(&self) -> Result<WhoAmIResponse> {
        let response = self
            .http
            .get(format!("{HF_ENDPOINT}/api/whoami-v2"))
            .headers(self.headers()?)
            .send()
            .await?;

        if response.status() == StatusCode::UNAUTHORIZED {
            return Err(Error::HuggingFace("Invalid user token.".to_string()));
        }

        let status = response.status();
        if !status.is_success() {
            return Err(Error::HuggingFace(format!(
                "whoami failed with status {}",
                status
            )));
        }

        Ok(response.json().await?)
    }

    pub async fn space_info(&self, repo_id: &str) -> Result<SpaceInfoResponse> {
        let response = self
            .http
            .get(format!("{HF_ENDPOINT}/api/spaces/{repo_id}"))
            .headers(self.headers()?)
            .send()
            .await?;

        if response.status() == StatusCode::NOT_FOUND {
            return Err(Error::HuggingFace(format!("Space not found: {repo_id}")));
        }

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(Error::HuggingFace(format!(
                "space_info failed with status {}: {body}",
                status
            )));
        }

        Ok(response.json().await?)
    }

    pub async fn get_space_runtime(&self, repo_id: &str) -> Result<SpaceRuntimeResponse> {
        let response = self
            .http
            .get(format!("{HF_ENDPOINT}/api/spaces/{repo_id}/runtime"))
            .headers(self.headers()?)
            .send()
            .await?;

        if response.status() == StatusCode::NOT_FOUND {
            return Err(Error::HuggingFace(format!("Space not found: {repo_id}")));
        }

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(Error::HuggingFace(format!(
                "get_space_runtime failed with status {}: {body}",
                status
            )));
        }

        Ok(response.json().await?)
    }

    pub async fn duplicate_space(
        &self,
        from_id: &str,
        to_id: &str,
        private: bool,
        exist_ok: bool,
    ) -> Result<()> {
        let payload = json!({
            "repository": to_id,
            "private": private,
        });

        let response = self
            .http
            .post(format!("{HF_ENDPOINT}/api/spaces/{from_id}/duplicate"))
            .headers(self.headers()?)
            .json(&payload)
            .send()
            .await?;

        let status = response.status();
        if status == StatusCode::CONFLICT && exist_ok {
            return Ok(());
        }

        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(Error::HuggingFace(format!(
                "duplicate_space failed with status {}: {body}",
                status
            )));
        }

        Ok(())
    }

    pub async fn request_space_hardware(&self, repo_id: &str, hardware: &str) -> Result<()> {
        let payload = json!({
            "flavor": hardware,
        });

        let response = self
            .http
            .post(format!("{HF_ENDPOINT}/api/spaces/{repo_id}/hardware"))
            .headers(self.headers()?)
            .json(&payload)
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(Error::HuggingFace(format!(
                "request_space_hardware failed with status {}: {body}",
                status
            )));
        }

        Ok(())
    }

    pub async fn set_space_sleep_time(&self, repo_id: &str, seconds: i64) -> Result<()> {
        let payload = json!({
            "seconds": seconds,
        });

        let response = self
            .http
            .post(format!("{HF_ENDPOINT}/api/spaces/{repo_id}/sleeptime"))
            .headers(self.headers()?)
            .json(&payload)
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(Error::HuggingFace(format!(
                "set_space_sleep_time failed with status {}: {body}",
                status
            )));
        }

        Ok(())
    }

    pub async fn add_space_secret(&self, repo_id: &str, key: &str, value: &str) -> Result<()> {
        let payload = json!({
            "key": key,
            "value": value,
        });

        let response = self
            .http
            .post(format!("{HF_ENDPOINT}/api/spaces/{repo_id}/secrets"))
            .headers(self.headers()?)
            .json(&payload)
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(Error::HuggingFace(format!(
                "add_space_secret failed with status {}: {body}",
                status
            )));
        }

        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct DuplicateOptions {
    pub to_id: Option<String>,
    pub token: Option<String>,
    pub private: bool,
    pub hardware: Option<String>,
    pub secrets: Option<HashMap<String, String>>,
    pub sleep_timeout_minutes: i64,
    pub verbose: bool,
}

impl Default for DuplicateOptions {
    fn default() -> Self {
        Self {
            to_id: None,
            token: None,
            private: true,
            hardware: None,
            secrets: None,
            sleep_timeout_minutes: 5,
            verbose: true,
        }
    }
}

impl DuplicateOptions {
    pub fn normalized(mut self) -> Self {
        if self.sleep_timeout_minutes <= 0 {
            self.sleep_timeout_minutes = 5;
        }
        self
    }
}

pub fn split_space_repo_name(space_id: &str) -> Result<&str> {
    space_id
        .split('/')
        .nth(1)
        .ok_or_else(|| Error::HuggingFace(format!("Invalid space id: {space_id}")))
}

pub fn normalize_to_id_for_user(to_id: &str) -> &str {
    to_id.split('/').next_back().unwrap_or(to_id)
}

pub fn choose_target_space_id(
    username: &str,
    from_id: &str,
    to_id: Option<&str>,
) -> Result<String> {
    let repo_name = match to_id {
        Some(to_id) => normalize_to_id_for_user(to_id),
        None => split_space_repo_name(from_id)?,
    };
    Ok(format!("{username}/{repo_name}"))
}

pub fn pick_hardware(original: &SpaceRuntimeResponse, requested: Option<&str>) -> Option<String> {
    requested
        .map(ToOwned::to_owned)
        .or_else(|| original.current_hardware().map(ToOwned::to_owned))
}

pub fn should_set_sleep_timeout(hardware: Option<&str>) -> bool {
    hardware.map(|value| value != "cpu-basic").unwrap_or(false)
}

pub fn runtime_current_hardware(runtime: &SpaceRuntimeResponse) -> Option<String> {
    runtime
        .requested_hardware()
        .map(ToOwned::to_owned)
        .or_else(|| runtime.current_hardware().map(ToOwned::to_owned))
}
