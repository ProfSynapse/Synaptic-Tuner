# services/proxy/app.py
# FastAPI logging proxy that sits in front of vLLM.
# Receives OpenAI-compatible API calls, logs chat completions asynchronously,
# and forwards all requests to the vLLM backend transparently.
# Used by: standalone via `uvicorn services.proxy.app:app` or from the
# flywheel CLI via `tuner flywheel serve`.

from __future__ import annotations

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response

from services.proxy.config import ProxyConfig

logger = logging.getLogger(__name__)

# Path that triggers inference logging (only chat completions are logged)
_CHAT_COMPLETIONS_PATH = "v1/chat/completions"


# ---------------------------------------------------------------------------
# Lifespan: startup / shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize proxy resources on startup, clean up on shutdown.

    Attempts to load FlywheelConfig from shared.flywheel.config if available.
    Falls back to ProxyConfig.from_env() for standalone operation.
    """
    config = _load_config()

    logging.basicConfig(
        level=getattr(logging, config.log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Initialize catalog and inference logger from shared.flywheel
    catalog = None
    inference_logger = None
    flywheel_config = None
    try:
        from shared.flywheel.catalog import create_catalog
        from shared.flywheel.config import FlywheelConfig, load_flywheel_config
        from shared.flywheel.inference_logger import InferenceLogger

        flywheel_config = load_flywheel_config()
        catalog = await create_catalog(
            backend=config.catalog_backend,
            path=config.catalog_path,
            url=config.catalog_url,
            tenant_id=config.tenant_id,
        )
        inference_logger = InferenceLogger(
            log_dir=Path(config.log_dir),
            catalog=catalog,
            config=flywheel_config,
        )
        await inference_logger.start()
        logger.info(
            "Inference logger started (catalog=%s, log_dir=%s)",
            config.catalog_backend,
            config.log_dir,
        )
    except ImportError:
        logger.warning(
            "shared.flywheel not available -- running in passthrough mode "
            "(no inference logging)"
        )
    except Exception:
        logger.exception("Failed to initialize inference logger -- continuing without logging")

    http_client = httpx.AsyncClient(
        base_url=config.vllm_base_url,
        timeout=httpx.Timeout(config.proxy_timeout_seconds),
    )

    app.state.config = config
    app.state.catalog = catalog
    app.state.inference_logger = inference_logger
    app.state.http_client = http_client
    app.state.stats = {"total_proxied": 0, "total_logged": 0, "log_errors": 0}

    logger.info(
        "Flywheel proxy ready (vllm=%s, port=%d)",
        config.vllm_base_url,
        config.proxy_port,
    )

    yield

    # Shutdown
    if inference_logger is not None:
        await inference_logger.stop()
    await http_client.aclose()
    if catalog is not None:
        try:
            await catalog.close()
        except Exception:
            logger.exception("Error closing catalog")

    logger.info("Flywheel proxy shut down")


app = FastAPI(title="Flywheel Logging Proxy", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Proxy-specific endpoints (prefixed with /flywheel to avoid vLLM collisions)
# ---------------------------------------------------------------------------

@app.get("/flywheel/health")
async def health(request: Request) -> dict:
    """Health check endpoint for the proxy itself."""
    config: ProxyConfig = request.app.state.config
    return {
        "status": "ok",
        "catalog": config.catalog_backend,
        "logging_enabled": request.app.state.inference_logger is not None,
    }


_STATS_TOKEN = os.environ.get("FLYWHEEL_STATS_TOKEN", "")


async def _check_stats_auth(
    authorization: str | None = Header(default=None),
) -> None:
    """Verify Bearer token for /flywheel/stats when FLYWHEEL_STATS_TOKEN is set.

    If the env var is empty or unset, the endpoint is open (localhost dev mode).
    """
    if not _STATS_TOKEN:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer credential")
    import hmac
    if not hmac.compare_digest(authorization[7:], _STATS_TOKEN):
        raise HTTPException(status_code=403, detail="Invalid credential")


@app.get("/flywheel/stats", dependencies=[Depends(_check_stats_auth)])
async def stats(request: Request) -> dict:
    """Return logging statistics (total logged, today's count, queue depth)."""
    return dict(request.app.state.stats)


# ---------------------------------------------------------------------------
# Catch-all proxy route
# ---------------------------------------------------------------------------

@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy(request: Request, path: str) -> Response:
    """Forward all requests to vLLM, logging chat completions.

    Only POST /v1/chat/completions requests are logged.
    All other requests are forwarded transparently.
    Auth headers are passed through without inspection.
    """
    http_client: httpx.AsyncClient = request.app.state.http_client

    # Read request body (needed for forwarding and potentially logging)
    body = await request.body()

    # Build headers to forward (pass auth through, set content type)
    forward_headers = _build_forward_headers(request)

    # Determine if this is a loggable request
    should_log = (
        request.method == "POST"
        and path.strip("/") == _CHAT_COMPLETIONS_PATH
        and request.app.state.inference_logger is not None
    )

    # Time the request for latency tracking
    start_time = time.monotonic()

    try:
        vllm_response = await http_client.request(
            method=request.method,
            url=f"/{path}",
            content=body,
            headers=forward_headers,
        )
    except httpx.ConnectError:
        logger.error("Cannot connect to vLLM at %s", request.app.state.config.vllm_base_url)
        return Response(
            content='{"error": "vLLM backend unavailable"}',
            status_code=503,
            media_type="application/json",
        )
    except httpx.TimeoutException:
        logger.error("Request to vLLM timed out (path=%s)", path)
        return Response(
            content='{"error": "vLLM request timed out"}',
            status_code=504,
            media_type="application/json",
        )

    latency_ms = (time.monotonic() - start_time) * 1000
    request.app.state.stats["total_proxied"] += 1

    # Fire-and-forget logging for chat completions
    if should_log and vllm_response.status_code == 200:
        asyncio.create_task(
            _log_inference(
                app=request.app,
                body=body,
                response_body=vllm_response.content,
                latency_ms=latency_ms,
            )
        )

    return Response(
        content=vllm_response.content,
        status_code=vllm_response.status_code,
        headers=dict(vllm_response.headers),
        media_type=vllm_response.headers.get("content-type"),
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_config() -> ProxyConfig:
    """Load proxy config, preferring FlywheelConfig if available."""
    try:
        from shared.flywheel.config import load_flywheel_config

        flywheel_config = load_flywheel_config()
        return ProxyConfig.from_flywheel_config(flywheel_config)
    except ImportError:
        logger.info("FlywheelConfig not available, loading from environment")
        return ProxyConfig.from_env()


def _build_forward_headers(request: Request) -> dict[str, str]:
    """Extract headers to forward to vLLM.

    Passes through Authorization and Content-Type.
    Strips hop-by-hop headers that should not be forwarded.
    """
    headers: dict[str, str] = {}

    if "content-type" in request.headers:
        headers["content-type"] = request.headers["content-type"]

    if "authorization" in request.headers:
        headers["authorization"] = request.headers["authorization"]

    if "accept" in request.headers:
        headers["accept"] = request.headers["accept"]

    return headers


async def _log_inference(
    app: FastAPI,
    body: bytes,
    response_body: bytes,
    latency_ms: float,
) -> None:
    """Parse request/response and log via InferenceLogger.

    This runs as a fire-and-forget task -- failures are caught and logged,
    never propagated to the caller.
    """
    import json

    try:
        request_data: dict[str, Any] = json.loads(body)
        response_data: dict[str, Any] = json.loads(response_body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        logger.warning("Failed to parse request/response JSON for logging")
        app.state.stats["log_errors"] += 1
        return

    model_id = request_data.get("model", "unknown")

    try:
        inference_logger = app.state.inference_logger
        await inference_logger.log_inference(
            request=request_data,
            response=response_data,
            latency_ms=latency_ms,
            model_id=model_id,
        )
        app.state.stats["total_logged"] += 1
    except Exception:
        logger.warning("Inference logging failed (model=%s)", model_id, exc_info=True)
        app.state.stats["log_errors"] += 1
