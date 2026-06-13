use axum::{
    body::Body,
    extract::State,
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use reqwest::Client;
use serde::Deserialize;
use std::collections::HashMap;
use std::fs;
use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;
use tower_http::cors::CorsLayer;
use tracing::{info, warn};
use uuid::Uuid;

const MODELS: &[&str] = &[
    "deepseek-v4-flash-free",
    "deepseek-v4-flash",
    "mimo-v2.5-free",
    "minimax-m3-free",
    "nemotron-3-super-free",
];

#[derive(Debug, Clone, Deserialize)]
struct ApiKeys {
    #[serde(flatten)]
    keys: HashMap<String, String>,
}

struct UpstreamTokens {
    tokens: Vec<String>,
    counter: AtomicUsize,
}

impl UpstreamTokens {
    fn from_env() -> Self {
        let raw = std::env::var("UPSTREAM_TOKENS").unwrap_or_else(|_| "public".into());
        let tokens: Vec<String> = raw
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();
        if tokens.is_empty() {
            panic!("UPSTREAM_TOKENS is empty or not set");
        }
        info!("Loaded {} upstream token(s) for rotation", tokens.len());
        Self { tokens, counter: AtomicUsize::new(0) }
    }

    fn next(&self) -> &str {
        let idx = self.counter.fetch_add(1, Ordering::Relaxed) % self.tokens.len();
        &self.tokens[idx]
    }
}

struct AppState {
    client: Client,
    api_keys: ApiKeys,
    upstream_tokens: UpstreamTokens,
}

fn check_auth(state: &AppState, token: &str) -> bool {
    state.api_keys.keys.values().any(|k| k == token)
}

fn has_429(body: &serde_json::Value) -> bool {
    body.get("error")
        .and_then(|e| e.get("type").and_then(|t| t.as_str()))
        .map(|t| t.contains("RateLimit") || t.contains("FreeUsageLimit") || t == "rate_limit_error")
        .unwrap_or(false)
}

async fn handle_health() -> Response {
    Json(serde_json::json!({
        "status": "ok",
        "version": "0.3.0",
        "models": MODELS.len(),
    }))
    .into_response()
}

async fn handle_models() -> Response {
    Json(serde_json::json!({
        "object": "list",
        "data": MODELS.iter().map(|id| {
            serde_json::json!({
                "id": id,
                "object": "model",
                "created": 1779000000,
                "owned_by": "opencode-free",
            })
        }).collect::<Vec<_>>(),
    }))
    .into_response()
}

async fn do_upstream(
    state: &AppState,
    body: &serde_json::Value,
    is_stream: bool,
    retry_count: usize,
) -> Response {
    let zen_body = serde_json::json!({
        "model": body.get("model"),
        "messages": body.get("messages"),
        "stream": is_stream,
        "tools": body.get("tools"),
        "tool_choice": body.get("tool_choice"),
    });

    let request_id = Uuid::new_v4().to_string();
    let bear_token = state.upstream_tokens.next();

    let mut zen_headers = reqwest::header::HeaderMap::new();
    zen_headers.insert("content-type", "application/json".parse().unwrap());
    zen_headers.insert(
        "authorization",
        format!("Bearer {}", bear_token).parse().unwrap(),
    );
    zen_headers.insert(
        "user-agent",
        format!("opencode-proxy-rust/0.3").parse().unwrap(),
    );
    zen_headers.insert("x-opencode-client", "proxy".parse().unwrap());
    zen_headers.insert("x-opencode-request", request_id.parse().unwrap());

    let upstream_req = state
        .client
        .post("https://opencode.ai/zen/v1/chat/completions")
        .headers(zen_headers)
        .json(&zen_body);

    match upstream_req.send().await {
        Ok(upstream_resp) => {
            let status = upstream_resp.status();

            if is_stream {
                use futures_util::StreamExt;
                let stream = upstream_resp.bytes_stream()
                    .map(|r| r.map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e)))
                    .filter_map(|r| async move {
                        match r {
                            Ok(b) if !b.is_empty() => Some(Ok(b)),
                            Ok(_) => None,
                            Err(e) => Some(Err(e)),
                        }
                    });

                Response::builder()
                    .status(StatusCode::OK)
                    .header("content-type", "text/event-stream")
                    .header("cache-control", "no-cache")
                    .header("x-accel-buffering", "no")
                    .body(Body::from_stream(stream))
                    .unwrap_or_else(|_| {
                        (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({
                            "error": {"message": "failed to build stream response"}
                        }))).into_response()
                    })
            } else {
                let bytes = match tokio::time::timeout(
                    Duration::from_secs(55),
                    upstream_resp.bytes(),
                )
                .await
                {
                    Ok(Ok(b)) => b,
                    Ok(Err(e)) => {
                        warn!("upstream body read error: {e}");
                        return (
                            StatusCode::BAD_GATEWAY,
                            Json(serde_json::json!({"error": {
                                "message": format!("upstream body error: {e}")
                            }})),
                        ).into_response();
                    }
                    Err(_) => {
                        warn!("upstream body read timed out");
                        return (
                            StatusCode::GATEWAY_TIMEOUT,
                            Json(serde_json::json!({"error": {
                                "message": "upstream body timed out"
                            }})),
                        ).into_response();
                    }
                };

                let resp_body: serde_json::Value =
                    serde_json::from_slice(&bytes).unwrap_or(serde_json::json!({"error": "parse error"}));

                if status == StatusCode::TOO_MANY_REQUESTS && has_429(&resp_body) && retry_count > 0 {
                    warn!("upstream 429 with token {}, retrying ({} left)", bear_token, retry_count - 1);
                    return Box::pin(do_upstream(state, body, false, retry_count - 1)).await;
                }

                (StatusCode::from_u16(status.as_u16()).unwrap_or(StatusCode::BAD_GATEWAY), Json(resp_body))
                    .into_response()
            }
        }
        Err(e) => {
            warn!("upstream connection error: {e}");
            if retry_count > 0 {
                return Box::pin(do_upstream(state, body, is_stream, retry_count - 1)).await;
            }
            (
                StatusCode::BAD_GATEWAY,
                Json(serde_json::json!({"error": {
                    "message": format!("upstream error: {e}"),
                    "type": "upstream_error"
                }})),
            )
                .into_response()
        }
    }
}

async fn handle_chat(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(body): Json<serde_json::Value>,
) -> Response {
    let req_body_str = serde_json::to_string(&body).unwrap_or_default();
    info!("REQ: model={} stream={} body_len={}",
        body.get("model").and_then(|m| m.as_str()).unwrap_or("?"),
        body.get("stream").and_then(|s| s.as_bool()).unwrap_or(false),
        req_body_str.len(),
    );

    let tok = headers
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
        .or_else(|| headers.get("x-api-key").and_then(|v| v.to_str().ok()));

    let authed = tok.is_some_and(|t| check_auth(&state, t));
    if !authed {
        return (
            StatusCode::UNAUTHORIZED,
            Json(serde_json::json!({"error": {"message": "Invalid API key", "type": "auth_error"}})),
        ).into_response();
    }

    let is_stream = body
        .get("stream")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    do_upstream(&state, &body, is_stream, 3).await
}

fn load_or_generate_keys(path: &str) -> ApiKeys {
    match fs::read_to_string(path) {
        Ok(data) => serde_json::from_str(&data).unwrap_or_else(|e| {
            warn!("Failed to parse {path}: {e}, generating new keys");
            generate_keys(path)
        }),
        Err(_) => {
            info!("{path} not found, generating new keys");
            generate_keys(path)
        }
    }
}

fn generate_keys(path: &str) -> ApiKeys {
    let admin = format!("oc-{}", Uuid::new_v4().to_string().replace('-', ""));
    let user = format!("oc-{}", Uuid::new_v4().to_string().replace('-', ""));
    let keys = serde_json::json!({"admin": admin, "user-default": user});
    if let Ok(json) = serde_json::to_string_pretty(&keys) {
        let _ = fs::write(path, &json);
    }
    let mut map = HashMap::new();
    map.insert("admin".into(), admin);
    map.insert("user-default".into(), user);
    ApiKeys { keys: map }
}

#[tokio::main(flavor = "multi_thread", worker_threads = 4)]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .init();

    let port: u16 = std::env::var("PROXY_PORT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(6446);

    let keys_path = std::env::var("KEYS_FILE").unwrap_or_else(|_| "./api-keys.json".into());
    let api_keys = load_or_generate_keys(&keys_path);
    let upstream_tokens = UpstreamTokens::from_env();
    let upstream_token_count = upstream_tokens.tokens.len();

    let state = Arc::new(AppState {
        client: Client::builder()
            .timeout(Duration::from_secs(60))
            .connect_timeout(Duration::from_secs(10))
            .http1_only()
            .pool_max_idle_per_host(32)
            .pool_idle_timeout(Duration::from_secs(30))
            .tcp_keepalive(Duration::from_secs(15))
            .build()
            .expect("Failed to build HTTP client"),
        api_keys,
        upstream_tokens,
    });

    let app = Router::new()
        .route("/v1/chat/completions", post(handle_chat))
        .route("/zen/v1/chat/completions", post(handle_chat))
        .route("/health", get(handle_health))
        .route("/v1/models", get(handle_models))
        .layer(CorsLayer::permissive())
        .with_state(state);

    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    info!("opencode-proxy-rust v0.3 listening on {addr} (tokens={})", upstream_token_count);
    println!("opencode-proxy-rust v0.3 listening on {addr} (tokens={})", upstream_token_count);

    let listener = tokio::net::TcpListener::bind(addr).await.expect("bind failed");

    axum::serve(listener, app)
        .with_graceful_shutdown(async {
            tokio::signal::ctrl_c().await.ok();
        })
        .await
        .unwrap_or_else(|e| warn!("Server error: {e}"));
}
