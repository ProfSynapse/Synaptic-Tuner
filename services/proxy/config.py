# services/proxy/config.py
# Proxy-specific configuration dataclass.
# Used by services/proxy/app.py to configure the logging proxy.
# Can be constructed from shared.flywheel.config.FlywheelConfig or from
# environment variables directly (for standalone proxy operation).

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ProxyConfig:
    """Configuration for the FastAPI logging proxy.

    Fields mirror the proxy-relevant subset of FlywheelConfig.
    Construct via from_env() for standalone use or from_flywheel_config()
    when running as part of the full flywheel pipeline.
    """

    vllm_base_url: str = "http://localhost:8000"
    catalog_backend: str = "sqlite"
    catalog_path: str = ".tracking/flywheel.db"
    catalog_url: str | None = None
    tenant_id: str | None = None
    log_dir: str = "inference_logs"
    log_level: str = "INFO"
    proxy_port: int = 8080
    proxy_timeout_seconds: float = 120.0
    flush_interval_seconds: float = 1.0
    # When true, inject logprobs + return_tokens_as_token_ids into forwarded
    # chat-completion requests so the logger can capture token-faithful rollouts.
    inject_logprobs: bool = False

    @classmethod
    def from_env(cls) -> ProxyConfig:
        """Build config from environment variables with sensible defaults."""
        vllm_host = os.environ.get("FLYWHEEL_VLLM_HOST", "localhost")
        vllm_port = os.environ.get("FLYWHEEL_VLLM_PORT", "8000")
        return cls(
            vllm_base_url=f"http://{vllm_host}:{vllm_port}",
            catalog_backend=os.environ.get("FLYWHEEL_CATALOG_BACKEND", "sqlite"),
            catalog_path=os.environ.get(
                "FLYWHEEL_CATALOG_PATH", ".tracking/flywheel.db"
            ),
            catalog_url=os.environ.get("FLYWHEEL_CATALOG_URL"),
            tenant_id=os.environ.get("FLYWHEEL_TENANT_ID"),
            log_dir=os.environ.get("FLYWHEEL_LOG_DIR", "inference_logs"),
            log_level=os.environ.get("FLYWHEEL_LOG_LEVEL", "INFO"),
            proxy_port=int(os.environ.get("FLYWHEEL_PROXY_PORT", "8080")),
            proxy_timeout_seconds=float(
                os.environ.get("FLYWHEEL_PROXY_TIMEOUT", "120.0")
            ),
            flush_interval_seconds=float(
                os.environ.get("FLYWHEEL_FLUSH_INTERVAL", "1.0")
            ),
            inject_logprobs=os.environ.get("FLYWHEEL_INJECT_LOGPROBS", "0")
            in ("1", "true", "True"),
        )

    @classmethod
    def from_flywheel_config(cls, config) -> ProxyConfig:
        """Build from a shared.flywheel.config.FlywheelConfig instance.

        Args:
            config: FlywheelConfig instance (imported by caller to avoid
                    hard dependency on shared.flywheel at import time).
        """
        return cls(
            vllm_base_url=f"http://{config.vllm_host}:{config.vllm_port}",
            catalog_backend=config.catalog_backend,
            catalog_path=config.catalog_path,
            catalog_url=config.catalog_url,
            tenant_id=config.tenant_id,
            log_dir=config.log_dir,
            log_level="INFO",
            proxy_port=config.proxy_port,
            proxy_timeout_seconds=config.proxy_timeout_seconds,
            flush_interval_seconds=config.flush_interval_seconds,
            inject_logprobs=getattr(config, "proxy_inject_logprobs", False),
        )
