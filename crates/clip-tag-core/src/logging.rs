use tracing_subscriber::{fmt, EnvFilter};

/// Initialize tracing from `RUST_LOG` (default: `clip_tag=info`).
pub fn init() {
    init_with_default("clip_tag=info");
}

/// Quieter setup used by `--dry-run`: silences engine progress lines so the
/// dry-run report at the end isn't interleaved with model-load info. An
/// explicit `RUST_LOG` still wins.
pub fn init_quiet() {
    init_with_default("clip_tag=warn");
}

fn init_with_default(default: &str) {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(default));
    fmt()
        .with_env_filter(filter)
        .with_target(true)
        .try_init()
        .ok();
}
