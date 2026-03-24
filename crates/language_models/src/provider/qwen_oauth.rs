use anyhow::{Result, anyhow};
use base64::Engine as _;
use credentials_provider::CredentialsProvider;
use futures::future::BoxFuture;
use futures::{AsyncBufReadExt, AsyncReadExt as _, FutureExt, StreamExt, io::BufReader};
use gpui::{AnyView, App, AsyncApp, Context, Entity, Task, Window};
use http_client::{AsyncBody, HttpClient, Method, Request};
use language_model::{
    AuthenticateError, IconOrSvg, LanguageModel, LanguageModelCompletionError,
    LanguageModelCompletionEvent, LanguageModelId, LanguageModelName, LanguageModelProvider,
    LanguageModelProviderId, LanguageModelProviderName, LanguageModelProviderState,
    LanguageModelRequest, RateLimiter,
};
use rand::RngCore as _;
use sha2::Digest as _;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use ui::{ConfiguredApiCard, prelude::*};
use util::ResultExt;

use super::open_ai::{OpenAiEventMapper, count_open_ai_tokens, into_open_ai};

const PROVIDER_ID: LanguageModelProviderId = LanguageModelProviderId::new("qwen-oauth");
const PROVIDER_NAME: LanguageModelProviderName = LanguageModelProviderName::new("Qwen");

const DEVICE_CODE_URL: &str = "https://chat.qwen.ai/api/v1/oauth2/device/code";
const TOKEN_URL: &str = "https://chat.qwen.ai/api/v1/oauth2/token";
const CLIENT_ID: &str = "f0304373b74a44d2b584a3fb70ca9e56";
const SCOPES: &str = "openid profile email model.completion";
const KEYCHAIN_URL: &str = "https://chat.qwen.ai/oauth";
const FALLBACK_API_URL: &str = "https://dashscope.aliyuncs.com/compatible-mode/v1";
const POLL_TIMEOUT_SECS: u64 = 300;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum QwenModel {
    Coder,
    Vision,
}

impl QwenModel {
    fn all() -> &'static [QwenModel] {
        &[QwenModel::Coder, QwenModel::Vision]
    }

    fn id(&self) -> &'static str {
        match self {
            Self::Coder => "coder-model",
            Self::Vision => "vision-model",
        }
    }

    fn display_name(&self) -> &'static str {
        match self {
            Self::Coder => "Qwen Coder (Plus)",
            Self::Vision => "Qwen Vision",
        }
    }

    fn max_token_count(&self) -> u64 {
        128_000
    }

    fn max_output_tokens(&self) -> Option<u64> {
        Some(65_536)
    }

    fn supports_images(&self) -> bool {
        matches!(self, Self::Vision)
    }
}

#[derive(serde::Serialize, serde::Deserialize, Clone)]
struct StoredTokens {
    access_token: String,
    refresh_token: Option<String>,
    expires_at: u64,
    resource_url: Option<String>,
}

impl StoredTokens {
    fn is_expired(&self) -> bool {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        self.expires_at <= now + 30
    }

    fn api_url(&self) -> String {
        match &self.resource_url {
            Some(url) => {
                let url = if url.starts_with("http") {
                    url.clone()
                } else {
                    format!("https://{url}")
                };
                if url.ends_with("/v1") {
                    url
                } else {
                    format!("{url}/v1")
                }
            }
            None => FALLBACK_API_URL.to_string(),
        }
    }
}

struct DeviceCodeResponse {
    device_code: String,
    verification_uri_complete: String,
    interval: u64,
}

pub struct QwenOAuthLanguageModelProvider {
    http_client: Arc<dyn HttpClient>,
    state: Entity<State>,
}

pub struct State {
    stored_tokens: Option<StoredTokens>,
    http_client: Arc<dyn HttpClient>,
    auth_task: Option<Task<()>>,
}

impl State {
    fn is_authenticated(&self) -> bool {
        self.stored_tokens.is_some()
    }

    fn current_token_and_url(&self) -> Option<(String, String)> {
        self.stored_tokens
            .as_ref()
            .map(|t| (t.access_token.clone(), t.api_url()))
    }

    fn authenticate(&mut self, cx: &mut Context<Self>) -> Task<Result<(), AuthenticateError>> {
        if self.is_authenticated() {
            return Task::ready(Ok(()));
        }
        let load_task = self.load(cx);
        cx.spawn(async move |_, _| load_task.await.map_err(AuthenticateError::Other))
    }

    fn load(&mut self, cx: &mut Context<Self>) -> Task<Result<()>> {
        let credentials_provider = <dyn CredentialsProvider>::global(cx);
        let http_client = self.http_client.clone();
        cx.spawn(async move |this, cx| {
            let creds = credentials_provider
                .read_credentials(KEYCHAIN_URL, cx)
                .await?;

            let Some((_, bytes)) = creds else {
                return Ok(());
            };

            let tokens: StoredTokens = serde_json::from_slice(&bytes)?;

            let tokens = if tokens.is_expired() {
                match tokens.refresh_token.clone() {
                    Some(refresh_token) => {
                        match refresh_access_token(http_client, refresh_token, cx).await {
                            Ok(new_tokens) => {
                                let token_bytes = serde_json::to_vec(&new_tokens)?;
                                credentials_provider
                                    .write_credentials(KEYCHAIN_URL, "Bearer", &token_bytes, cx)
                                    .await
                                    .log_err();
                                new_tokens
                            }
                            Err(err) => {
                                log::error!("Qwen token refresh failed: {err}");
                                credentials_provider
                                    .delete_credentials(KEYCHAIN_URL, cx)
                                    .await
                                    .log_err();
                                return Ok(());
                            }
                        }
                    }
                    None => {
                        credentials_provider
                            .delete_credentials(KEYCHAIN_URL, cx)
                            .await
                            .log_err();
                        return Ok(());
                    }
                }
            } else {
                tokens
            };

            this.update(cx, |state, cx| {
                state.stored_tokens = Some(tokens);
                cx.notify();
            })
            .ok();

            Ok(())
        })
    }

    fn start_oauth_flow(&mut self, cx: &mut Context<Self>) {
        if self.auth_task.is_some() {
            return;
        }

        let http_client = self.http_client.clone();
        let credentials_provider = <dyn CredentialsProvider>::global(cx);

        let task = cx.spawn(async move |this, cx| {
            let (verifier, challenge) = generate_pkce_pair();

            let device_response = match cx
                .background_executor()
                .spawn(request_device_code(http_client.clone(), challenge))
                .await
            {
                Ok(resp) => resp,
                Err(err) => {
                    log::error!("Qwen device code request failed: {err}");
                    this.update(cx, |state, cx| {
                        state.auth_task = None;
                        cx.notify();
                    })
                    .log_err();
                    return;
                }
            };

            this.update(cx, |_, cx| {
                cx.open_url(&device_response.verification_uri_complete);
            })
            .log_err();

            let tokens = match cx
                .background_executor()
                .spawn(poll_for_token(
                    http_client.clone(),
                    device_response.device_code,
                    verifier,
                    device_response.interval,
                ))
                .await
            {
                Ok(tokens) => tokens,
                Err(err) => {
                    log::error!("Qwen token polling failed: {err}");
                    this.update(cx, |state, cx| {
                        state.auth_task = None;
                        cx.notify();
                    })
                    .log_err();
                    return;
                }
            };

            let token_bytes = match serde_json::to_vec(&tokens) {
                Ok(bytes) => bytes,
                Err(err) => {
                    log::error!("Qwen token serialisation error: {err}");
                    this.update(cx, |state, cx| {
                        state.auth_task = None;
                        cx.notify();
                    })
                    .log_err();
                    return;
                }
            };

            credentials_provider
                .write_credentials(KEYCHAIN_URL, "Bearer", &token_bytes, cx)
                .await
                .log_err();

            this.update(cx, |state, cx| {
                state.stored_tokens = Some(tokens);
                state.auth_task = None;
                cx.notify();
            })
            .log_err();
        });

        cx.notify();
        self.auth_task = Some(task);
    }

    fn reset(&mut self, cx: &mut Context<Self>) -> Task<Result<()>> {
        self.stored_tokens = None;
        self.auth_task = None;
        cx.notify();
        let credentials_provider = <dyn CredentialsProvider>::global(cx);
        cx.spawn(async move |_, cx| {
            credentials_provider
                .delete_credentials(KEYCHAIN_URL, cx)
                .await
        })
    }
}

impl QwenOAuthLanguageModelProvider {
    pub fn new(http_client: Arc<dyn HttpClient>, cx: &mut App) -> Self {
        let state = cx.new(|_cx| State {
            stored_tokens: None,
            http_client: http_client.clone(),
            auth_task: None,
        });

        Self { http_client, state }
    }

    fn create_language_model(&self, model: QwenModel) -> Arc<dyn LanguageModel> {
        Arc::new(QwenOAuthLanguageModel {
            id: LanguageModelId::from(model.id().to_string()),
            model,
            state: self.state.clone(),
            http_client: self.http_client.clone(),
            request_limiter: RateLimiter::new(4),
        })
    }
}

impl LanguageModelProviderState for QwenOAuthLanguageModelProvider {
    type ObservableEntity = State;

    fn observable_entity(&self) -> Option<Entity<Self::ObservableEntity>> {
        Some(self.state.clone())
    }
}

impl LanguageModelProvider for QwenOAuthLanguageModelProvider {
    fn id(&self) -> LanguageModelProviderId {
        PROVIDER_ID
    }

    fn name(&self) -> LanguageModelProviderName {
        PROVIDER_NAME
    }

    fn icon(&self) -> IconOrSvg {
        IconOrSvg::Icon(IconName::AiOpenAiCompat)
    }

    fn default_model(&self, _cx: &App) -> Option<Arc<dyn LanguageModel>> {
        Some(self.create_language_model(QwenModel::Coder))
    }

    fn default_fast_model(&self, _cx: &App) -> Option<Arc<dyn LanguageModel>> {
        Some(self.create_language_model(QwenModel::Coder))
    }

    fn provided_models(&self, _cx: &App) -> Vec<Arc<dyn LanguageModel>> {
        QwenModel::all()
            .iter()
            .map(|&model| self.create_language_model(model))
            .collect()
    }

    fn is_authenticated(&self, cx: &App) -> bool {
        self.state.read(cx).is_authenticated()
    }

    fn authenticate(&self, cx: &mut App) -> Task<Result<(), AuthenticateError>> {
        self.state.update(cx, |state, cx| state.authenticate(cx))
    }

    fn configuration_view(
        &self,
        _target_agent: language_model::ConfigurationViewTargetAgent,
        _window: &mut Window,
        cx: &mut App,
    ) -> AnyView {
        cx.new(|cx| ConfigurationView::new(self.state.clone(), cx))
            .into()
    }

    fn reset_credentials(&self, cx: &mut App) -> Task<Result<()>> {
        self.state.update(cx, |state, cx| state.reset(cx))
    }
}

struct QwenOAuthLanguageModel {
    id: LanguageModelId,
    model: QwenModel,
    state: Entity<State>,
    http_client: Arc<dyn HttpClient>,
    request_limiter: RateLimiter,
}

impl QwenOAuthLanguageModel {
    fn stream_completion_inner(
        &self,
        request: open_ai::Request,
        cx: &AsyncApp,
    ) -> BoxFuture<
        'static,
        Result<futures::stream::BoxStream<'static, Result<open_ai::ResponseStreamEvent>>>,
    > {
        let http_client = self.http_client.clone();
        let token_and_url = self
            .state
            .read_with(cx, |state, _| state.current_token_and_url());
        let future = self.request_limiter.stream(async move {
            let Some((token, api_url)) = token_and_url else {
                return Err(language_model::LanguageModelCompletionError::NoApiKey {
                    provider: PROVIDER_NAME,
                });
            };
            let response =
                qwen_stream_completion(http_client.as_ref(), &api_url, &token, request).await?;
            Ok(response)
        });
        async move { Ok(future.await?.boxed()) }.boxed()
    }
}

impl LanguageModel for QwenOAuthLanguageModel {
    fn id(&self) -> LanguageModelId {
        self.id.clone()
    }

    fn name(&self) -> LanguageModelName {
        LanguageModelName::from(self.model.display_name().to_string())
    }

    fn provider_id(&self) -> LanguageModelProviderId {
        PROVIDER_ID
    }

    fn provider_name(&self) -> LanguageModelProviderName {
        PROVIDER_NAME
    }

    fn supports_tools(&self) -> bool {
        true
    }

    fn supports_images(&self) -> bool {
        self.model.supports_images()
    }

    fn supports_tool_choice(&self, choice: language_model::LanguageModelToolChoice) -> bool {
        use language_model::LanguageModelToolChoice;
        match choice {
            LanguageModelToolChoice::Auto => true,
            LanguageModelToolChoice::Any => true,
            LanguageModelToolChoice::None => true,
        }
    }

    fn telemetry_id(&self) -> String {
        format!("qwen-oauth/{}", self.model.id())
    }

    fn max_token_count(&self) -> u64 {
        self.model.max_token_count()
    }

    fn max_output_tokens(&self) -> Option<u64> {
        self.model.max_output_tokens()
    }

    fn count_tokens(
        &self,
        request: LanguageModelRequest,
        cx: &App,
    ) -> BoxFuture<'static, Result<u64>> {
        let model = open_ai::Model::Custom {
            name: self.model.id().to_owned(),
            display_name: None,
            max_tokens: self.model.max_token_count(),
            max_output_tokens: self.model.max_output_tokens(),
            max_completion_tokens: None,
            reasoning_effort: None,
            supports_chat_completions: true,
        };
        count_open_ai_tokens(request, model, cx)
    }

    fn stream_completion(
        &self,
        request: LanguageModelRequest,
        cx: &AsyncApp,
    ) -> BoxFuture<
        'static,
        Result<
            futures::stream::BoxStream<
                'static,
                Result<LanguageModelCompletionEvent, LanguageModelCompletionError>,
            >,
            LanguageModelCompletionError,
        >,
    > {
        let open_ai_request = into_open_ai(
            request,
            self.model.id(),
            false,
            false,
            self.model.max_output_tokens(),
            None,
        );
        let completions = self.stream_completion_inner(open_ai_request, cx);
        async move {
            let mapper = OpenAiEventMapper::new();
            Ok(mapper.map_stream(completions.await?).boxed())
        }
        .boxed()
    }
}

struct ConfigurationView {
    state: Entity<State>,
    load_credentials_task: Option<Task<()>>,
}

impl ConfigurationView {
    fn new(state: Entity<State>, cx: &mut Context<Self>) -> Self {
        cx.observe(&state, |_, _, cx| cx.notify()).detach();

        let state_clone = state.clone();
        let load_credentials_task = Some(cx.spawn(async move |this, cx| {
            let task = state_clone.update(cx, |state, cx| state.authenticate(cx));
            let _ = task.await;
            this.update(cx, |this, cx| {
                this.load_credentials_task = None;
                cx.notify();
            })
            .log_err();
        }));

        Self {
            state,
            load_credentials_task,
        }
    }

    fn sign_in(&mut self, _: &gpui::ClickEvent, _window: &mut Window, cx: &mut Context<Self>) {
        self.state
            .update(cx, |state, cx| state.start_oauth_flow(cx));
    }

    fn sign_out(&mut self, _: &gpui::ClickEvent, _window: &mut Window, cx: &mut Context<Self>) {
        self.state
            .update(cx, |state, cx| state.reset(cx))
            .detach_and_log_err(cx);
    }
}

impl Render for ConfigurationView {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        if self.load_credentials_task.is_some() {
            return div().child(Label::new("Loading…")).into_any();
        }

        let is_authenticated = self.state.read(cx).is_authenticated();
        let is_signing_in = self.state.read(cx).auth_task.is_some();

        if is_authenticated {
            ConfiguredApiCard::new("Signed in to Qwen")
                .on_click(cx.listener(Self::sign_out))
                .into_any_element()
        } else {
            v_flex()
                .gap_2()
                .child(Label::new(
                    "Sign in with your Qwen account to use Qwen models.",
                ))
                .child(
                    Button::new(
                        "sign-in",
                        if is_signing_in {
                            "Waiting for browser authorization…"
                        } else {
                            "Sign in with Qwen"
                        },
                    )
                    .disabled(is_signing_in)
                    .on_click(cx.listener(Self::sign_in)),
                )
                .into_any_element()
        }
    }
}

async fn qwen_stream_completion(
    http_client: &dyn HttpClient,
    api_url: &str,
    api_key: &str,
    request: open_ai::Request,
) -> Result<
    futures::stream::BoxStream<'static, anyhow::Result<open_ai::ResponseStreamEvent>>,
    open_ai::RequestError,
> {
    let uri = format!("{api_url}/chat/completions");

    let mut body_value =
        serde_json::to_value(&request).map_err(|e| open_ai::RequestError::Other(e.into()))?;
    if let Some(obj) = body_value.as_object_mut() {
        obj.insert("enable_thinking".to_owned(), serde_json::Value::Bool(false));
    }

    let http_request = Request::builder()
        .method(Method::POST)
        .uri(uri)
        .header("Content-Type", "application/json")
        .header("Authorization", format!("Bearer {}", api_key.trim()))
        .body(AsyncBody::from(
            serde_json::to_string(&body_value)
                .map_err(|e| open_ai::RequestError::Other(e.into()))?,
        ))
        .map_err(|e| open_ai::RequestError::Other(e.into()))?;

    let mut response = http_client.send(http_request).await?;
    if response.status().is_success() {
        let reader = BufReader::new(response.into_body());
        Ok(reader
            .lines()
            .filter_map(|line| async move {
                match line {
                    Ok(line) => {
                        let line = line
                            .strip_prefix("data: ")
                            .or_else(|| line.strip_prefix("data:"))?;
                        if line == "[DONE]" {
                            None
                        } else {
                            match serde_json::from_str::<open_ai::ResponseStreamResult>(line) {
                                Ok(open_ai::ResponseStreamResult::Ok(event)) => Some(Ok(event)),
                                Ok(open_ai::ResponseStreamResult::Err { error }) => {
                                    let message = serde_json::to_value(&error)
                                        .ok()
                                        .and_then(|v| {
                                            v.get("message")?.as_str().map(str::to_owned)
                                        })
                                        .unwrap_or_else(|| format!("{error:?}"));
                                    Some(Err(anyhow!(message)))
                                }
                                Err(error) => {
                                    log::error!(
                                        "Failed to parse Qwen response: `{error}`\nResponse: `{line}`"
                                    );
                                    Some(Err(anyhow!(error)))
                                }
                            }
                        }
                    }
                    Err(error) => Some(Err(anyhow!(error))),
                }
            })
            .boxed())
    } else {
        let mut body = String::new();
        response
            .body_mut()
            .read_to_string(&mut body)
            .await
            .map_err(|e| open_ai::RequestError::Other(e.into()))?;
        Err(open_ai::RequestError::HttpResponseError {
            provider: "Qwen".to_owned(),
            status_code: response.status(),
            body,
            headers: response.headers().clone(),
        })
    }
}

async fn request_device_code(
    http_client: Arc<dyn HttpClient>,
    challenge: String,
) -> Result<DeviceCodeResponse> {
    let body = format!(
        "client_id={client_id}&scope={scope}&code_challenge={challenge}&code_challenge_method=S256",
        client_id = CLIENT_ID,
        scope = percent_encode(SCOPES),
        challenge = percent_encode(&challenge),
    );

    let request = Request::builder()
        .method(Method::POST)
        .uri(DEVICE_CODE_URL)
        .header("Content-Type", "application/x-www-form-urlencoded")
        .header("x-request-id", generate_uuid())
        .body(AsyncBody::from(body))?;

    let mut response = http_client.send(request).await?;
    let mut body = String::new();
    response.body_mut().read_to_string(&mut body).await?;

    if !response.status().is_success() {
        return Err(anyhow!(
            "Qwen device code request failed: {} — {}",
            response.status(),
            body
        ));
    }

    let json: serde_json::Value = serde_json::from_str(&body)
        .map_err(|e| anyhow!("Failed to parse device code response: {e}"))?;

    let device_code = json["device_code"]
        .as_str()
        .ok_or_else(|| anyhow!("Missing device_code in response"))?
        .to_owned();

    let verification_uri_complete = json["verification_uri_complete"]
        .as_str()
        .or_else(|| json["verification_uri"].as_str())
        .ok_or_else(|| anyhow!("Missing verification_uri in response"))?
        .to_owned();

    let interval = json["interval"].as_u64().unwrap_or(5);

    Ok(DeviceCodeResponse {
        device_code,
        verification_uri_complete,
        interval,
    })
}

async fn poll_for_token(
    http_client: Arc<dyn HttpClient>,
    device_code: String,
    verifier: String,
    interval: u64,
) -> Result<StoredTokens> {
    let deadline = std::time::Instant::now() + Duration::from_secs(POLL_TIMEOUT_SECS);
    let mut current_interval = interval.max(5);

    loop {
        if std::time::Instant::now() >= deadline {
            return Err(anyhow!(
                "Qwen device authorization timed out after 5 minutes"
            ));
        }

        smol::Timer::after(Duration::from_secs(current_interval)).await;

        let body = format!(
            "grant_type={grant_type}&client_id={client_id}&device_code={device_code}&code_verifier={verifier}",
            grant_type = percent_encode("urn:ietf:params:oauth:grant-type:device_code"),
            client_id = CLIENT_ID,
            device_code = percent_encode(&device_code),
            verifier = percent_encode(&verifier),
        );

        let request = Request::builder()
            .method(Method::POST)
            .uri(TOKEN_URL)
            .header("Content-Type", "application/x-www-form-urlencoded")
            .body(AsyncBody::from(body))?;

        let mut response = http_client.send(request).await?;
        let mut body = String::new();
        response.body_mut().read_to_string(&mut body).await?;

        let json: serde_json::Value =
            serde_json::from_str(&body).unwrap_or(serde_json::Value::Null);

        if let Some(error) = json["error"].as_str() {
            match error {
                "authorization_pending" => continue,
                "slow_down" => {
                    current_interval += 5;
                    continue;
                }
                "expired_token" => return Err(anyhow!("Qwen device code expired")),
                "access_denied" => return Err(anyhow!("Qwen authorization denied by user")),
                other => return Err(anyhow!("Qwen token error: {other}")),
            }
        }

        if let Some(access_token) = json["access_token"].as_str() {
            let refresh_token = json["refresh_token"].as_str().map(str::to_owned);
            let expires_in = json["expires_in"].as_u64().unwrap_or(3600);
            let expires_at = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs()
                + expires_in;
            let resource_url = json["resource_url"].as_str().map(str::to_owned);

            return Ok(StoredTokens {
                access_token: access_token.to_owned(),
                refresh_token,
                expires_at,
                resource_url,
            });
        }

        return Err(anyhow!("Unexpected Qwen token response: {body}"));
    }
}

async fn refresh_access_token(
    http_client: Arc<dyn HttpClient>,
    refresh_token: String,
    _cx: &AsyncApp,
) -> Result<StoredTokens> {
    let body = format!(
        "grant_type=refresh_token&refresh_token={refresh_token}&client_id={client_id}",
        refresh_token = percent_encode(&refresh_token),
        client_id = CLIENT_ID,
    );

    let request = Request::builder()
        .method(Method::POST)
        .uri(TOKEN_URL)
        .header("Content-Type", "application/x-www-form-urlencoded")
        .body(AsyncBody::from(body))?;

    let mut response = http_client.send(request).await?;
    let mut body = String::new();
    response.body_mut().read_to_string(&mut body).await?;

    if !response.status().is_success() {
        return Err(anyhow!(
            "Qwen token refresh failed: {} — {}",
            response.status(),
            body
        ));
    }

    let json: serde_json::Value = serde_json::from_str(&body)
        .map_err(|e| anyhow!("Failed to parse refresh response: {e}"))?;

    let access_token = json["access_token"]
        .as_str()
        .ok_or_else(|| anyhow!("Missing access_token in refresh response"))?
        .to_owned();

    let refresh_token = json["refresh_token"].as_str().map(str::to_owned);
    let expires_in = json["expires_in"].as_u64().unwrap_or(3600);
    let expires_at = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
        + expires_in;
    let resource_url = json["resource_url"].as_str().map(str::to_owned);

    Ok(StoredTokens {
        access_token,
        refresh_token,
        expires_at,
        resource_url,
    })
}

fn generate_pkce_pair() -> (String, String) {
    let mut bytes = [0u8; 32];
    rand::rng().fill_bytes(&mut bytes);
    let verifier = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(bytes);
    let digest = sha2::Sha256::digest(verifier.as_bytes());
    let challenge = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(digest);
    (verifier, challenge)
}

fn generate_uuid() -> String {
    let mut bytes = [0u8; 16];
    rand::rng().fill_bytes(&mut bytes);
    format!(
        "{:02x}{:02x}{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
        bytes[0],
        bytes[1],
        bytes[2],
        bytes[3],
        bytes[4],
        bytes[5],
        bytes[6],
        bytes[7],
        bytes[8],
        bytes[9],
        bytes[10],
        bytes[11],
        bytes[12],
        bytes[13],
        bytes[14],
        bytes[15],
    )
}

fn percent_encode(s: &str) -> String {
    let mut encoded = String::with_capacity(s.len() * 3);
    for byte in s.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                encoded.push(byte as char);
            }
            b => {
                encoded.push('%');
                encoded.push(char::from_digit((b >> 4) as u32, 16).unwrap_or('0'));
                encoded.push(char::from_digit((b & 0xf) as u32, 16).unwrap_or('0'));
            }
        }
    }
    encoded
}
