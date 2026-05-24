use tracing_subscriber::{fmt, EnvFilter};

/// Initialize tracing from `RUST_LOG` (default: `clip_tag=info`).
pub fn init() {
    let filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("clip_tag=info"));

    fmt()
        .with_env_filter(filter)
        .with_target(true)
        .try_init()
        .ok();
}
