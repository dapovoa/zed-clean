use anyhow::{Result, anyhow};
use base64::Engine as _;
use collections::HashSet;
use credentials_provider::CredentialsProvider;
use futures::future::BoxFuture;
use futures::{AsyncReadExt as _, FutureExt, StreamExt};
use gpui::{AnyView, App, AsyncApp, Context, Entity, Task, Window};
use http_client::{AsyncBody, HttpClient, Method, Request, Url};
use language_model::{
    AuthenticateError, IconOrSvg, LanguageModel, LanguageModelCompletionError,
    LanguageModelCompletionEvent, LanguageModelId, LanguageModelName, LanguageModelProvider,
    LanguageModelProviderId, LanguageModelProviderName, LanguageModelProviderState,
    LanguageModelRequest, RateLimiter,
};
use rand::RngCore as _;
use release_channel::AppVersion;
use sha2::Digest as _;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use ui::{ConfiguredApiCard, prelude::*};
use util::ResultExt;

use super::open_ai::{OpenAiResponseEventMapper, count_open_ai_tokens, into_open_ai_response};

const PROVIDER_ID: LanguageModelProviderId = LanguageModelProviderId::new("openai-oauth");
const PROVIDER_NAME: LanguageModelProviderName = LanguageModelProviderName::new("ChatGPT");

const AUTH_URL: &str = "https://auth.openai.com/oauth/authorize";
const TOKEN_URL: &str = "https://auth.openai.com/oauth/token";
const CLIENT_ID: &str = "app_EMoamEEZ73f0CkXaXp7hrann";
const REDIRECT_URI: &str = "http://localhost:1455/auth/callback";
const SCOPES: &str = "openid profile email offline_access";
const KEYCHAIN_URL: &str = "https://auth.openai.com/oauth";
const CHATGPT_CODEX_API_URL: &str = "https://chatgpt.com/backend-api/codex";
const CHATGPT_CODEX_MODELS_URL: &str = "https://chatgpt.com/backend-api/codex/models";

#[derive(Clone, Debug, PartialEq)]
struct OAuthModelCatalog {
    models: Vec<OAuthModel>,
    default_model: Option<String>,
    default_fast_model: Option<String>,
    recommended_models: Vec<String>,
}

#[derive(Clone, Debug, PartialEq)]
struct OAuthModel {
    id: Arc<str>,
    display_name: Arc<str>,
    max_token_count: u64,
    max_output_tokens: Option<u64>,
    supports_tools: bool,
    supports_images: bool,
    supports_thinking: bool,
    supports_parallel_tool_calls: bool,
    supports_chat_completions: bool,
    reasoning_effort: Option<open_ai::ReasoningEffort>,
    is_latest: bool,
}

impl OAuthModel {
    fn as_open_ai_model(&self) -> open_ai::Model {
        open_ai::Model::Custom {
            name: self.id.to_string(),
            display_name: Some(self.display_name.to_string()),
            max_tokens: self.max_token_count,
            max_output_tokens: self.max_output_tokens,
            max_completion_tokens: self.max_output_tokens,
            reasoning_effort: self.reasoning_effort.clone(),
            supports_chat_completions: self.supports_chat_completions,
        }
    }
}

fn fallback_catalog() -> OAuthModelCatalog {
    OAuthModelCatalog {
        models: vec![
            OAuthModel {
                id: Arc::from("gpt-5"),
                display_name: Arc::from("GPT-5"),
                max_token_count: 272_000,
                max_output_tokens: Some(128_000),
                supports_tools: true,
                supports_images: true,
                supports_thinking: false,
                supports_parallel_tool_calls: true,
                supports_chat_completions: true,
                reasoning_effort: None,
                is_latest: true,
            },
            OAuthModel {
                id: Arc::from("gpt-5-mini"),
                display_name: Arc::from("GPT-5 Mini"),
                max_token_count: 272_000,
                max_output_tokens: Some(128_000),
                supports_tools: true,
                supports_images: true,
                supports_thinking: false,
                supports_parallel_tool_calls: true,
                supports_chat_completions: true,
                reasoning_effort: None,
                is_latest: false,
            },
            OAuthModel {
                id: Arc::from("gpt-5-codex"),
                display_name: Arc::from("GPT-5 Codex"),
                max_token_count: 272_000,
                max_output_tokens: Some(128_000),
                supports_tools: true,
                supports_images: true,
                supports_thinking: false,
                supports_parallel_tool_calls: true,
                supports_chat_completions: false,
                reasoning_effort: None,
                is_latest: false,
            },
        ],
        default_model: Some("gpt-5".to_string()),
        default_fast_model: Some("gpt-5-mini".to_string()),
        recommended_models: vec![
            "gpt-5".to_string(),
            "gpt-5-mini".to_string(),
            "gpt-5-codex".to_string(),
        ],
    }
}

#[derive(serde::Serialize, serde::Deserialize, Clone)]
struct StoredTokens {
    access_token: String,
    refresh_token: Option<String>,
    expires_at: u64,
}

impl StoredTokens {
    fn is_expired(&self) -> bool {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        self.expires_at <= now + 60
    }
}

pub struct OpenAiOAuthLanguageModelProvider {
    http_client: Arc<dyn HttpClient>,
    state: Entity<State>,
}

pub struct State {
    stored_tokens: Option<StoredTokens>,
    http_client: Arc<dyn HttpClient>,
    auth_task: Option<Task<()>>,
    fetch_models_task: Option<Task<()>>,
    catalog: OAuthModelCatalog,
}

impl State {
    fn is_authenticated(&self) -> bool {
        self.stored_tokens.is_some()
    }

    fn current_token(&self) -> Option<String> {
        self.stored_tokens.as_ref().map(|t| t.access_token.clone())
    }

    fn start_fetch_models(&mut self, cx: &mut Context<Self>) {
        let Some(access_token) = self.current_token() else {
            self.catalog = fallback_catalog();
            self.fetch_models_task = None;
            cx.notify();
            return;
        };

        let http_client = self.http_client.clone();
        let version = AppVersion::global(cx);
        let client_version = format!("{}.{}.{}", version.major, version.minor, version.patch);
        self.fetch_models_task = Some(cx.spawn(async move |this, cx| {
            match fetch_model_catalog(http_client, access_token, client_version).await {
                Ok(catalog) => {
                    this.update(cx, |state, cx| {
                        state.catalog = catalog;
                        state.fetch_models_task = None;
                        cx.notify();
                    })
                    .log_err();
                }
                Err(err) => {
                    log::warn!(
                        "Failed to fetch ChatGPT OAuth models, using fallback catalog: {err}"
                    );
                    this.update(cx, |state, cx| {
                        state.catalog = fallback_catalog();
                        state.fetch_models_task = None;
                        cx.notify();
                    })
                    .log_err();
                }
            }
        }));
        cx.notify();
    }

    fn authenticate(&mut self, cx: &mut Context<Self>) -> Task<Result<(), AuthenticateError>> {
        if self.is_authenticated() {
            if self.fetch_models_task.is_none() {
                self.start_fetch_models(cx);
            }
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
                                log::error!("OpenAI token refresh failed: {err}");
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
                state.start_fetch_models(cx);
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

        let (verifier, challenge) = generate_pkce_pair();
        let nonce = generate_nonce();

        let mut auth_url = Url::parse(AUTH_URL).expect("AUTH_URL is a valid URL");
        auth_url
            .query_pairs_mut()
            .append_pair("response_type", "code")
            .append_pair("client_id", CLIENT_ID)
            .append_pair("redirect_uri", REDIRECT_URI)
            .append_pair("scope", SCOPES)
            .append_pair("state", &nonce)
            .append_pair("code_challenge", &challenge)
            .append_pair("code_challenge_method", "S256");

        cx.open_url(auth_url.as_str());

        let http_client = self.http_client.clone();
        let credentials_provider = <dyn CredentialsProvider>::global(cx);

        let task = cx.spawn(async move |this, cx| {
            let code = match cx.background_executor().spawn(listen_for_callback()).await {
                Ok(code) => code,
                Err(err) => {
                    log::error!("OpenAI OAuth callback error: {err}");
                    this.update(cx, |state, cx| {
                        state.auth_task = None;
                        cx.notify();
                    })
                    .log_err();
                    return;
                }
            };

            let tokens = match exchange_code_for_token(http_client, code, verifier, cx).await {
                Ok(tokens) => tokens,
                Err(err) => {
                    log::error!("OpenAI token exchange error: {err}");
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
                    log::error!("OpenAI token serialisation error: {err}");
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
                state.start_fetch_models(cx);
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
        self.fetch_models_task = None;
        self.catalog = fallback_catalog();
        cx.notify();
        let credentials_provider = <dyn CredentialsProvider>::global(cx);
        cx.spawn(async move |_, cx| {
            credentials_provider
                .delete_credentials(KEYCHAIN_URL, cx)
                .await
        })
    }
}

impl OpenAiOAuthLanguageModelProvider {
    pub fn new(http_client: Arc<dyn HttpClient>, cx: &mut App) -> Self {
        let state = cx.new(|_cx| State {
            stored_tokens: None,
            http_client: http_client.clone(),
            auth_task: None,
            fetch_models_task: None,
            catalog: fallback_catalog(),
        });

        Self { http_client, state }
    }

    fn create_language_model(&self, model: OAuthModel) -> Arc<dyn LanguageModel> {
        Arc::new(OpenAiOAuthLanguageModel {
            id: LanguageModelId::from(model.id.to_string()),
            model,
            state: self.state.clone(),
            http_client: self.http_client.clone(),
            request_limiter: RateLimiter::new(4),
        })
    }

    fn find_model(&self, model_id: &str, cx: &App) -> Option<OAuthModel> {
        self.state
            .read(cx)
            .catalog
            .models
            .iter()
            .find(|model| model.id.as_ref() == model_id)
            .cloned()
    }
}

impl LanguageModelProviderState for OpenAiOAuthLanguageModelProvider {
    type ObservableEntity = State;

    fn observable_entity(&self) -> Option<Entity<Self::ObservableEntity>> {
        Some(self.state.clone())
    }
}

impl LanguageModelProvider for OpenAiOAuthLanguageModelProvider {
    fn id(&self) -> LanguageModelProviderId {
        PROVIDER_ID
    }

    fn name(&self) -> LanguageModelProviderName {
        PROVIDER_NAME
    }

    fn icon(&self) -> IconOrSvg {
        IconOrSvg::Icon(IconName::AiOpenAi)
    }

    fn default_model(&self, cx: &App) -> Option<Arc<dyn LanguageModel>> {
        let model_id = self.state.read(cx).catalog.default_model.clone()?;
        self.find_model(&model_id, cx)
            .map(|model| self.create_language_model(model))
    }

    fn default_fast_model(&self, cx: &App) -> Option<Arc<dyn LanguageModel>> {
        let model_id = self.state.read(cx).catalog.default_fast_model.clone()?;
        self.find_model(&model_id, cx)
            .map(|model| self.create_language_model(model))
    }

    fn recommended_models(&self, cx: &App) -> Vec<Arc<dyn LanguageModel>> {
        self.state
            .read(cx)
            .catalog
            .recommended_models
            .iter()
            .filter_map(|model_id| self.find_model(model_id, cx))
            .map(|model| self.create_language_model(model))
            .collect()
    }

    fn provided_models(&self, cx: &App) -> Vec<Arc<dyn LanguageModel>> {
        self.state
            .read(cx)
            .catalog
            .models
            .iter()
            .cloned()
            .map(|model| self.create_language_model(model))
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

struct OpenAiOAuthLanguageModel {
    id: LanguageModelId,
    model: OAuthModel,
    state: Entity<State>,
    http_client: Arc<dyn HttpClient>,
    request_limiter: RateLimiter,
}

impl OpenAiOAuthLanguageModel {
    fn stream_response_inner(
        &self,
        request: open_ai::responses::Request,
        cx: &AsyncApp,
    ) -> BoxFuture<
        'static,
        Result<futures::stream::BoxStream<'static, Result<open_ai::responses::StreamEvent>>>,
    > {
        let http_client = self.http_client.clone();
        let token = self.state.read_with(cx, |state, _| state.current_token());
        let provider = PROVIDER_NAME;
        let future = self.request_limiter.stream(async move {
            let Some(token) = token else {
                return Err(language_model::LanguageModelCompletionError::NoApiKey { provider });
            };
            let response = open_ai::responses::stream_response(
                http_client.as_ref(),
                provider.0.as_str(),
                CHATGPT_CODEX_API_URL,
                &token,
                request,
            )
            .await?;
            Ok(response)
        });
        async move { Ok(future.await?.boxed()) }.boxed()
    }
}

impl LanguageModel for OpenAiOAuthLanguageModel {
    fn id(&self) -> LanguageModelId {
        self.id.clone()
    }

    fn name(&self) -> LanguageModelName {
        LanguageModelName::from(self.model.display_name.to_string())
    }

    fn provider_id(&self) -> LanguageModelProviderId {
        PROVIDER_ID
    }

    fn provider_name(&self) -> LanguageModelProviderName {
        PROVIDER_NAME
    }

    fn supports_tools(&self) -> bool {
        self.model.supports_tools
    }

    fn supports_images(&self) -> bool {
        self.model.supports_images
    }

    fn supports_tool_choice(&self, choice: language_model::LanguageModelToolChoice) -> bool {
        use language_model::LanguageModelToolChoice;
        match choice {
            LanguageModelToolChoice::Auto => true,
            LanguageModelToolChoice::Any => true,
            LanguageModelToolChoice::None => true,
        }
    }

    fn supports_thinking(&self) -> bool {
        self.model.supports_thinking
    }

    fn supports_split_token_display(&self) -> bool {
        true
    }

    fn telemetry_id(&self) -> String {
        format!("openai-oauth/{}", self.model.id)
    }

    fn is_latest(&self) -> bool {
        self.model.is_latest
    }

    fn max_token_count(&self) -> u64 {
        self.model.max_token_count
    }

    fn max_output_tokens(&self) -> Option<u64> {
        self.model.max_output_tokens
    }

    fn count_tokens(
        &self,
        request: LanguageModelRequest,
        cx: &App,
    ) -> BoxFuture<'static, Result<u64>> {
        count_open_ai_tokens(request, self.model.as_open_ai_model(), cx)
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
        let model = self.model.as_open_ai_model();
        let open_ai_request = into_open_ai_response(
            request,
            model.id(),
            self.model.supports_parallel_tool_calls,
            false,
            self.max_output_tokens(),
            model.reasoning_effort(),
        );
        let completions = self.stream_response_inner(open_ai_request, cx);
        async move {
            let mapper = OpenAiResponseEventMapper::new();
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
            // Not logging the error — "not signed in" is a normal state here.
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
            ConfiguredApiCard::new("Signed in to ChatGPT")
                .on_click(cx.listener(Self::sign_out))
                .into_any_element()
        } else {
            v_flex()
                .gap_2()
                .child(Label::new(
                    "Sign in with your ChatGPT Plus or Pro account to use OpenAI models.",
                ))
                .child(
                    Button::new(
                        "sign-in",
                        if is_signing_in {
                            "Signing in…"
                        } else {
                            "Sign in with OpenAI"
                        },
                    )
                    .disabled(is_signing_in)
                    .on_click(cx.listener(Self::sign_in)),
                )
                .into_any_element()
        }
    }
}

fn generate_pkce_pair() -> (String, String) {
    let mut bytes = [0u8; 32];
    rand::rng().fill_bytes(&mut bytes);
    let verifier = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(bytes);
    let digest = sha2::Sha256::digest(verifier.as_bytes());
    let challenge = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(digest);
    (verifier, challenge)
}

fn generate_nonce() -> String {
    let mut bytes = [0u8; 16];
    rand::rng().fill_bytes(&mut bytes);
    base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(bytes)
}

async fn listen_for_callback() -> Result<String> {
    use smol::io::AsyncWriteExt as _;
    use smol::net::TcpListener;

    let listener = TcpListener::bind("127.0.0.1:1455").await?;
    let (mut stream, _) = listener.accept().await?;

    let mut buf = vec![0u8; 4096];
    let n = smol::io::AsyncReadExt::read(&mut stream, &mut buf).await?;
    let request = std::str::from_utf8(&buf[..n])
        .map_err(|_| anyhow!("OAuth callback request is not valid UTF-8"))?;

    let code = parse_code_from_http_request(request)?;

    let response = concat!(
        "HTTP/1.1 200 OK\r\n",
        "Content-Type: text/html; charset=utf-8\r\n",
        "\r\n",
        "<!DOCTYPE html>",
        "<html lang=\"en\">",
        "<head><meta charset=\"utf-8\"><title>Zed &mdash; Signed in</title>",
        "<style>",
        "body{font-family:system-ui,sans-serif;display:flex;align-items:center;",
        "justify-content:center;height:100vh;margin:0;background:#0d0d0d;color:#fff;}",
        ".card{text-align:center;padding:2rem 3rem;border-radius:12px;",
        "background:#1a1a1a;border:1px solid #333;}",
        "h1{font-size:1.4rem;margin:0 0 .5rem;}",
        "p{color:#888;margin:0;font-size:.9rem;}",
        "</style></head>",
        "<body><div class=\"card\">",
        "<h1>&#10003;&nbsp; Signed in to ChatGPT</h1>",
        "<p>You can close this tab and return to Zed.</p>",
        "</div></body></html>",
    );
    stream.write_all(response.as_bytes()).await?;

    Ok(code)
}

fn parse_code_from_http_request(request: &str) -> Result<String> {
    let first_line = request
        .lines()
        .next()
        .ok_or_else(|| anyhow!("Empty OAuth callback request"))?;

    let path = first_line
        .split_whitespace()
        .nth(1)
        .ok_or_else(|| anyhow!("Malformed OAuth callback request line"))?;

    let query = path.split('?').nth(1).unwrap_or("");
    for param in query.split('&') {
        let mut parts = param.splitn(2, '=');
        let key = parts.next().unwrap_or("");
        let value = parts.next().unwrap_or("");
        if key == "code" {
            return Ok(url_decode(value));
        }
    }

    Err(anyhow!("No authorization code found in OAuth callback"))
}

fn url_decode(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    let mut bytes = s.bytes().peekable();
    while let Some(b) = bytes.next() {
        if b == b'%' {
            let h1 = bytes.next().and_then(|c| (c as char).to_digit(16));
            let h2 = bytes.next().and_then(|c| (c as char).to_digit(16));
            if let (Some(h1), Some(h2)) = (h1, h2) {
                result.push(((h1 * 16 + h2) as u8) as char);
            }
        } else if b == b'+' {
            result.push(' ');
        } else {
            result.push(b as char);
        }
    }
    result
}

async fn exchange_code_for_token(
    http_client: Arc<dyn HttpClient>,
    code: String,
    verifier: String,
    _cx: &AsyncApp,
) -> Result<StoredTokens> {
    let body = format!(
        "grant_type=authorization_code&code={code}&redirect_uri={redirect_uri}&client_id={client_id}&code_verifier={verifier}",
        code = percent_encode(&code),
        redirect_uri = percent_encode(REDIRECT_URI),
        client_id = CLIENT_ID,
        verifier = percent_encode(&verifier),
    );
    do_token_request(http_client, body).await
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
    do_token_request(http_client, body).await
}

async fn do_token_request(
    http_client: Arc<dyn HttpClient>,
    form_body: String,
) -> Result<StoredTokens> {
    let request = Request::builder()
        .method(Method::POST)
        .uri(TOKEN_URL)
        .header("Content-Type", "application/x-www-form-urlencoded")
        .body(AsyncBody::from(form_body))?;

    let mut response = http_client.send(request).await?;

    let mut body = String::new();
    response.body_mut().read_to_string(&mut body).await?;

    if !response.status().is_success() {
        return Err(anyhow!(
            "OpenAI token endpoint returned {}: {}",
            response.status(),
            body
        ));
    }

    let json: serde_json::Value =
        serde_json::from_str(&body).map_err(|e| anyhow!("Failed to parse token response: {e}"))?;

    let access_token = json["access_token"]
        .as_str()
        .ok_or_else(|| anyhow!("Token response missing access_token"))?
        .to_owned();

    let refresh_token = json["refresh_token"].as_str().map(str::to_owned);

    let expires_in = json["expires_in"].as_u64().unwrap_or(3600);
    let expires_at = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
        + expires_in;

    Ok(StoredTokens {
        access_token,
        refresh_token,
        expires_at,
    })
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

async fn fetch_model_catalog(
    http_client: Arc<dyn HttpClient>,
    access_token: String,
    client_version: String,
) -> Result<OAuthModelCatalog> {
    let mut url = Url::parse(CHATGPT_CODEX_MODELS_URL)?;
    url.query_pairs_mut()
        .append_pair("client_version", &client_version);

    let request = Request::builder()
        .method(Method::GET)
        .uri(url.as_str())
        .header("Authorization", format!("Bearer {access_token}"))
        .header("Accept", "application/json")
        .body(AsyncBody::empty())?;

    let mut response = http_client.send(request).await?;
    let mut body = String::new();
    response.body_mut().read_to_string(&mut body).await?;

    if !response.status().is_success() {
        return Err(anyhow!(
            "ChatGPT/Codex models endpoint returned {}: {}",
            response.status(),
            body
        ));
    }

    parse_model_catalog_response(&body)
}

fn parse_model_catalog_response(body: &str) -> Result<OAuthModelCatalog> {
    let json: serde_json::Value = serde_json::from_str(body)
        .map_err(|err| anyhow!("Failed to parse ChatGPT/Codex models response: {err}"))?;

    let root = json
        .as_object()
        .ok_or_else(|| anyhow!("ChatGPT/Codex models response must be a JSON object"))?;

    let models_value = root
        .get("models")
        .and_then(|value| value.as_array())
        .ok_or_else(|| anyhow!("ChatGPT/Codex models response missing `models` array"))?;

    let mut models = Vec::new();
    let mut seen_model_ids = HashSet::default();
    for value in models_value {
        let Some(model) = parse_oauth_model(value) else {
            continue;
        };
        if seen_model_ids.insert(model.id.to_string()) {
            models.push(model);
        }
    }

    if models.is_empty() {
        return Err(anyhow!(
            "ChatGPT/Codex models response did not contain any valid models"
        ));
    }

    let default_model = root
        .get("default_model")
        .and_then(model_id_from_value)
        .or_else(|| models.first().map(|model| model.id.to_string()));
    let default_fast_model = root
        .get("default_fast_model")
        .and_then(model_id_from_value)
        .or_else(|| {
            models
                .iter()
                .map(|model| model.id.to_string())
                .find(|id| id.contains("mini") || id.contains("fast"))
        })
        .or_else(|| default_model.clone());
    let recommended_models = root
        .get("recommended_models")
        .and_then(|value| value.as_array())
        .map(|recommended| {
            let allowed_ids = models
                .iter()
                .map(|model| model.id.as_ref())
                .collect::<HashSet<_>>();
            let mut seen_ids = HashSet::default();

            recommended
                .iter()
                .filter_map(model_id_from_value)
                .filter(|id| allowed_ids.contains(id.as_str()) && seen_ids.insert(id.clone()))
                .collect()
        })
        .unwrap_or_else(|| {
            default_model
                .iter()
                .cloned()
                .chain(default_fast_model.iter().cloned())
                .collect()
        });

    models.sort_by(oauth_model_sort_key);

    Ok(OAuthModelCatalog {
        models,
        default_model,
        default_fast_model,
        recommended_models,
    })
}

fn parse_oauth_model(value: &serde_json::Value) -> Option<OAuthModel> {
    let object = value.as_object()?;
    let id = object
        .get("id")
        .or_else(|| object.get("slug"))
        .and_then(|value| value.as_str())?;
    let display_name = normalize_model_display_name(id);

    Some(OAuthModel {
        id: Arc::from(id),
        display_name: Arc::from(display_name),
        max_token_count: object
            .get("max_token_count")
            .or_else(|| object.get("max_tokens"))
            .or_else(|| object.get("context_window"))
            .and_then(value_as_u64)
            .unwrap_or(272_000),
        max_output_tokens: object
            .get("max_output_tokens")
            .or_else(|| object.get("max_completion_tokens"))
            .and_then(value_as_u64),
        supports_tools: object
            .get("supports_tools")
            .and_then(|value| value.as_bool())
            .unwrap_or(true),
        supports_images: object
            .get("supports_images")
            .and_then(|value| value.as_bool())
            .unwrap_or(false),
        supports_thinking: object
            .get("supports_thinking")
            .and_then(|value| value.as_bool())
            .unwrap_or(false),
        supports_parallel_tool_calls: object
            .get("supports_parallel_tool_calls")
            .and_then(|value| value.as_bool())
            .unwrap_or(false),
        supports_chat_completions: object
            .get("supports_chat_completions")
            .and_then(|value| value.as_bool())
            .unwrap_or(true),
        reasoning_effort: None,
        is_latest: object
            .get("is_latest")
            .and_then(|value| value.as_bool())
            .unwrap_or(false),
    })
}

fn model_id_from_value(value: &serde_json::Value) -> Option<String> {
    match value {
        serde_json::Value::String(id) => Some(id.clone()),
        serde_json::Value::Object(object) => object
            .get("id")
            .or_else(|| object.get("slug"))
            .and_then(|value| value.as_str())
            .map(ToOwned::to_owned),
        _ => None,
    }
}

fn value_as_u64(value: &serde_json::Value) -> Option<u64> {
    value
        .as_u64()
        .or_else(|| value.as_str().and_then(|value| value.parse().ok()))
}

fn normalize_model_display_name(id: &str) -> String {
    id.split('-')
        .map(|part| match part {
            "gpt" => "GPT".to_string(),
            "codex" => "Codex".to_string(),
            "mini" => "Mini".to_string(),
            "max" => "Max".to_string(),
            "nano" => "Nano".to_string(),
            "fast" => "Fast".to_string(),
            other => other.to_uppercase(),
        })
        .collect::<Vec<_>>()
        .join("-")
}

fn oauth_model_sort_key(a: &OAuthModel, b: &OAuthModel) -> std::cmp::Ordering {
    parse_model_sort_key(&a.id)
        .cmp(&parse_model_sort_key(&b.id))
        .reverse()
        .then_with(|| a.id.cmp(&b.id))
}

fn parse_model_sort_key(id: &str) -> (u32, u32, u32, i32) {
    let without_prefix = id.strip_prefix("gpt-").unwrap_or(id);
    let mut parts = without_prefix.split('-');
    let version = parts.next().unwrap_or_default();
    let variant = parts.collect::<Vec<_>>();

    let mut version_parts = version.split('.');
    let major = version_parts
        .next()
        .and_then(|part| part.parse::<u32>().ok())
        .unwrap_or(0);
    let minor = version_parts
        .next()
        .and_then(|part| part.parse::<u32>().ok())
        .unwrap_or(0);
    let patch = version_parts
        .next()
        .and_then(|part| part.parse::<u32>().ok())
        .unwrap_or(0);

    let variant_rank = match variant.as_slice() {
        [] => 50,
        ["mini"] => 40,
        ["codex"] => 30,
        ["codex", "max"] => 20,
        ["codex", "mini"] => 10,
        _ => 0,
    };

    (major, minor, patch, variant_rank)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_chatgpt_codex_catalog() {
        let catalog = parse_model_catalog_response(
            r#"{
                "models": [
                    {
                        "id": "gpt-5",
                        "display_name": "GPT-5",
                        "max_token_count": 272000,
                        "max_output_tokens": 128000,
                        "supports_tools": true,
                        "supports_images": true,
                        "supports_parallel_tool_calls": true,
                        "supports_chat_completions": true,
                        "is_latest": true
                    },
                    {
                        "id": "gpt-5-codex",
                        "display_name": "GPT-5 Codex",
                        "max_token_count": 272000,
                        "max_output_tokens": 128000,
                        "supports_tools": true,
                        "supports_chat_completions": false
                    }
                ],
                "default_model": "gpt-5",
                "default_fast_model": "gpt-5",
                "recommended_models": ["gpt-5-codex"]
            }"#,
        )
        .unwrap();

        assert_eq!(catalog.models.len(), 2);
        assert_eq!(catalog.default_model.as_deref(), Some("gpt-5"));
        assert_eq!(catalog.default_fast_model.as_deref(), Some("gpt-5"));
        assert_eq!(catalog.recommended_models, vec!["gpt-5-codex"]);
        assert_eq!(catalog.models[0].id.as_ref(), "gpt-5");
        assert_eq!(catalog.models[1].id.as_ref(), "gpt-5-codex");
        assert_eq!(catalog.models[0].display_name.as_ref(), "GPT-5");
        assert_eq!(catalog.models[1].display_name.as_ref(), "GPT-5-Codex");
        assert!(catalog
            .models
            .iter()
            .any(|model| model.id.as_ref() == "gpt-5-codex" && !model.supports_chat_completions));
    }

    #[test]
    fn falls_back_to_local_defaults_when_backend_omits_them() {
        let catalog = parse_model_catalog_response(
            r#"{
                "models": [
                    { "id": "gpt-5", "display_name": "GPT-5" },
                    { "id": "gpt-5-mini", "display_name": "GPT-5 Mini" }
                ]
            }"#,
        )
        .unwrap();

        assert_eq!(catalog.default_model.as_deref(), Some("gpt-5"));
        assert_eq!(catalog.default_fast_model.as_deref(), Some("gpt-5-mini"));
        assert_eq!(catalog.recommended_models, vec!["gpt-5", "gpt-5-mini"]);
    }

    #[test]
    fn preserves_backend_order_while_deduplicating_models() {
        let catalog = parse_model_catalog_response(
            r#"{
                "models": [
                    { "id": "gpt-5.4", "display_name": "GPT-5.4" },
                    { "id": "gpt-5", "display_name": "GPT-5" },
                    { "id": "gpt-5.4", "display_name": "GPT-5.4 duplicate" }
                ]
            }"#,
        )
        .unwrap();

        assert_eq!(catalog.models.len(), 2);
        assert_eq!(catalog.models[0].id.as_ref(), "gpt-5.4");
        assert_eq!(catalog.models[1].id.as_ref(), "gpt-5");
    }

    #[test]
    fn sorts_models_by_version_and_variant_for_display() {
        let catalog = parse_model_catalog_response(
            r#"{
                "models": [
                    { "id": "gpt-5.1-codex-mini" },
                    { "id": "gpt-5.4-mini" },
                    { "id": "gpt-5.4" },
                    { "id": "gpt-5.3-codex" }
                ]
            }"#,
        )
        .unwrap();

        let ids = catalog
            .models
            .iter()
            .map(|model| model.id.as_ref())
            .collect::<Vec<_>>();

        assert_eq!(
            ids,
            vec![
                "gpt-5.4",
                "gpt-5.4-mini",
                "gpt-5.3-codex",
                "gpt-5.1-codex-mini"
            ]
        );
    }
}
