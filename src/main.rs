use axum::{
    body::Body,
    extract::{State},
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

struct AppState {
    client: Client,
    api_keys: ApiKeys,
}

fn check_auth(state: &AppState, token: &str) -> bool {
    state.api_keys.keys.values().any(|k| k == token)
}

async fn handle_health() -> Response {
    Json(serde_json::json!({
        "status": "ok",
        "version": "0.2.0",
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

async fn handle_chat(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(body): Json<serde_json::Value>,
) -> Response {
    // Log incoming request summary
    let req_body_str = serde_json::to_string(&body).unwrap_or_default();
    info!("REQ: model={} stream={} body_len={} body_preview={}",
        body.get("model").and_then(|m| m.as_str()).unwrap_or("?"),
        body.get("stream").and_then(|s| s.as_bool()).unwrap_or(false),
        req_body_str.len(),
        &req_body_str[..std::cmp::min(200, req_body_str.len())]
    );

    // Auth
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
        )
            .into_response();
    }

    let is_stream = body
        .get("stream")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    // Build upstream request — strip null fields to avoid upstream rejection
    let mut zen_body = serde_json::Map::new();
    if let Some(model) = body.get("model") {
        zen_body.insert("model".into(), model.clone());
    }
    if let Some(messages) = body.get("messages") {
        zen_body.insert("messages".into(), messages.clone());
    }
    zen_body.insert("stream".into(), serde_json::Value::Bool(is_stream));
    if let Some(tools) = body.get("tools") {
        if tools.as_array().map_or(false, |a| !a.is_empty()) {
            zen_body.insert("tools".into(), tools.clone());
        }
    }
    if let Some(tc) = body.get("tool_choice") {
        zen_body.insert("tool_choice".into(), tc.clone());
    }

    let request_id = Uuid::new_v4().to_string();

    let mut zen_headers = reqwest::header::HeaderMap::new();
    zen_headers.insert("content-type", "application/json".parse().unwrap());
    zen_headers.insert("authorization", "Bearer public".parse().unwrap());
    zen_headers.insert(
        "user-agent",
        "opencode-proxy-rust/0.2".parse().unwrap(),
    );
    zen_headers.insert("x-opencode-client", "proxy".parse().unwrap());
    zen_headers.insert("x-opencode-request", request_id.parse().unwrap());
    zen_headers.insert(
        "x-opencode-session",
        format!("ses_{}", &request_id[..12]).parse().unwrap(),
    );

    // Timeout for the entire upstream request
    let upstream_fut = state
        .client
        .post("https://opencode.ai/zen/v1/chat/completions")
        .headers(zen_headers)
        .json(&zen_body)
        .send();

    let upstream_resp = match tokio::time::timeout(Duration::from_secs(55), upstream_fut).await {
        Ok(Ok(r)) => r,
        Ok(Err(e)) => {
            warn!("upstream request error: {e}");
            return (
                StatusCode::BAD_GATEWAY,
                Json(serde_json::json!({"error": {
                    "message": format!("upstream error: {e}"),
                    "type": "upstream_error"
                }})),
            )
                .into_response();
        }
        Err(_) => {
            warn!("upstream request timed out after 55s");
            return (
                StatusCode::GATEWAY_TIMEOUT,
                Json(serde_json::json!({"error": {
                    "message": "upstream request timed out",
                    "type": "timeout_error"
                }})),
            )
                .into_response();
        }
    };

    let status = upstream_resp.status();

    // If upstream returned an error, return it as-is regardless of stream mode
    if !status.is_success() {
        let bytes = match tokio::time::timeout(Duration::from_secs(10), upstream_resp.bytes()).await
        {
            Ok(Ok(b)) => b,
            _ => {
                warn!("failed to read upstream error body");
                return (
                    StatusCode::BAD_GATEWAY,
                    Json(serde_json::json!({"error": {"message": "upstream error"}})),
                )
                    .into_response();
            }
        };
        let resp_body: serde_json::Value =
            serde_json::from_slice(&bytes).unwrap_or(serde_json::json!({"error": "upstream error"}));
        return (StatusCode::from_u16(status.as_u16()).unwrap_or(StatusCode::BAD_GATEWAY), Json(resp_body))
            .into_response();
    }

    if is_stream {
        use futures_util::StreamExt;

        let stream = upstream_resp
            .bytes_stream()
            .map(|r| match r {
                Ok(b) => Ok(b),
                Err(e) => {
                    warn!("upstream stream error: {e}");
                    Err(std::io::Error::new(std::io::ErrorKind::Other, e))
                }
            })
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
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({"error": {"message": "failed to build stream response"}})),
                )
                    .into_response()
            })
    } else {
        let bytes = match tokio::time::timeout(Duration::from_secs(55), upstream_resp.bytes()).await
        {
            Ok(Ok(b)) => b,
            Ok(Err(e)) => {
                warn!("upstream body read error: {e}");
                return (
                    StatusCode::BAD_GATEWAY,
                    Json(serde_json::json!({"error": {"message": format!("upstream body error: {e}")}})),
                )
                    .into_response();
            }
            Err(_) => {
                warn!("upstream body read timed out");
                return (
                    StatusCode::GATEWAY_TIMEOUT,
                    Json(serde_json::json!({"error": {"message": "upstream body timed out"}})),
                )
                    .into_response();
            }
        };

        let resp_body: serde_json::Value =
            serde_json::from_slice(&bytes).unwrap_or(serde_json::json!({"error": "parse error"}));

        (StatusCode::from_u16(status.as_u16()).unwrap_or(StatusCode::BAD_GATEWAY), Json(resp_body))
            .into_response()
    }
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

    let state = Arc::new(AppState {
        client: Client::builder()
            .timeout(Duration::from_secs(60))
            .connect_timeout(Duration::from_secs(10))
            .http1_only()
            .build()
            .expect("Failed to build HTTP client"),
        api_keys,
    });

    let app = Router::new()
        .route("/v1/chat/completions", post(handle_chat))
        .route("/zen/v1/chat/completions", post(handle_chat))
        .route("/health", get(handle_health))
        .route("/v1/models", get(handle_models))
        .layer(CorsLayer::permissive())
        .with_state(state);

    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    info!("opencode-proxy-rust listening on {addr}");
    println!("opencode-proxy-rust listening on {addr}");

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
