use axum::{
    body::Body,
    extract::{Request, State},
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    routing::post,
    Json, Router,
};
use reqwest::Client;
use serde::Deserialize;
use std::collections::HashMap;
use std::fs;
use std::net::SocketAddr;
use std::sync::Arc;
use tower_http::cors::CorsLayer;
use tracing::{info, warn};
use uuid::Uuid;

#[derive(Debug, Clone, Deserialize)]
struct ApiKeys {
    #[serde(flatten)]
    keys: HashMap<String, String>,
}

struct AppState {
    client: Client,
    api_keys: ApiKeys,
}

fn auth(req: &Request) -> Option<String> {
    let hdr = req
        .headers()
        .get("authorization")
        .or_else(|| req.headers().get("x-api-key"))?;
    let tok = hdr
        .to_str()
        .ok()?
        .strip_prefix("Bearer ")
        .unwrap_or(hdr.to_str().ok()?);
    Some(tok.to_string())
}

fn check_auth(state: &AppState, token: &str) -> bool {
    state.api_keys.keys.values().any(|k| k == token)
}

async fn handle_chat(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(body): Json<serde_json::Value>,
) -> Response {
    // Auth
    let tok = headers
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
        .or_else(|| {
            headers
                .get("x-api-key")
                .and_then(|v| v.to_str().ok())
        });

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

    // Build upstream request
    let zen_body = serde_json::json!({
        "model": body.get("model"),
        "messages": body.get("messages"),
        "stream": is_stream,
        "tools": body.get("tools"),
        "tool_choice": body.get("tool_choice"),
    });

    let request_id = Uuid::new_v4().to_string();

    let mut zen_headers = reqwest::header::HeaderMap::new();
    zen_headers.insert("content-type", "application/json".parse().unwrap());
    zen_headers.insert("authorization", "Bearer public".parse().unwrap());
    zen_headers.insert(
        "user-agent",
        format!("opencode-proxy-rust/0.1").parse().unwrap(),
    );
    zen_headers.insert("x-opencode-client", "proxy".parse().unwrap());
    zen_headers.insert("x-opencode-request", request_id.parse().unwrap());
    zen_headers.insert(
        "x-opencode-session",
        format!("ses_{}", &request_id[..12]).parse().unwrap(),
    );

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
                // reqwest bytes_stream -> map to io::Result
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
                    .unwrap_or_default()
            } else {
                let bytes = upstream_resp.bytes().await.unwrap_or_default();
                let resp_body: serde_json::Value =
                    serde_json::from_slice(&bytes).unwrap_or(serde_json::json!({"error": "parse error"}));

                (StatusCode::from_u16(status.as_u16()).unwrap(), Json(resp_body)).into_response()
            }
        }
        Err(e) => (
            StatusCode::BAD_GATEWAY,
            Json(serde_json::json!({"error": {"message": format!("upstream error: {e}"), "type": "upstream_error"}})),
        )
            .into_response(),
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

#[tokio::main]
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
            .timeout(std::time::Duration::from_secs(120))
            .http1_only()
            .build()
            .expect("Failed to build HTTP client"),
        api_keys,
    });

    let app = Router::new()
        .route("/v1/chat/completions", post(handle_chat))
        .route("/zen/v1/chat/completions", post(handle_chat))
        .layer(CorsLayer::permissive())
        .with_state(state);

    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    info!("opencode-proxy-rust listening on {addr}");
    println!("opencode-proxy-rust listening on {addr}");

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
