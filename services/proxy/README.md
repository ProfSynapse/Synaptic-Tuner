# Flywheel Logging Proxy

FastAPI reverse proxy that sits between OpenAI-compatible clients and vLLM. It forwards all requests transparently and asynchronously logs `POST /v1/chat/completions` responses for the data flywheel pipeline.

## Quick Start

```bash
# Ensure vLLM is running on port 8000, then:
uvicorn services.proxy.app:app --host 0.0.0.0 --port 8080
```

Point your OpenAI client at `http://localhost:8080` instead of the vLLM port.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `FLYWHEEL_VLLM_HOST` | `localhost` | vLLM server hostname |
| `FLYWHEEL_VLLM_PORT` | `8000` | vLLM server port |
| `FLYWHEEL_CATALOG_BACKEND` | `sqlite` | Log catalog backend (`sqlite` or `postgres`) |
| `FLYWHEEL_CATALOG_PATH` | `.tracking/flywheel.db` | SQLite database path |
| `FLYWHEEL_CATALOG_URL` | (none) | Postgres connection URL (when backend=postgres) |
| `FLYWHEEL_TENANT_ID` | (none) | Postgres tenant ID (schema isolation) |
| `FLYWHEEL_LOG_DIR` | `inference_logs` | Directory for JSONL log files |
| `FLYWHEEL_LOG_LEVEL` | `INFO` | Python log level |
| `FLYWHEEL_PROXY_PORT` | `8080` | Proxy listen port (informational for config) |
| `FLYWHEEL_PROXY_TIMEOUT` | `120.0` | Timeout in seconds for vLLM requests |
| `FLYWHEEL_FLUSH_INTERVAL` | `1.0` | Seconds between JSONL flush batches |

## Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/{path}` | ALL | Transparent proxy to vLLM |
| `/flywheel/health` | GET | Proxy health check |
| `/flywheel/stats` | GET | Logging statistics |

## How Logging Works

1. Client sends request to proxy on `:8080`
2. Proxy forwards to vLLM on `:8000`
3. vLLM response is returned to client immediately
4. If the request was `POST /v1/chat/completions` with a 200 response, the request/response pair is enqueued for async logging (fire-and-forget)
5. Background writer persists to date-partitioned JSONL files and indexes in the log catalog

Logging never blocks or slows inference responses. If logging fails, a warning is emitted and the response is still returned.

## Integration with FlywheelConfig

When `shared.flywheel.config` is importable, the proxy loads its configuration from the standard `FlywheelConfig` (YAML-based). When running standalone (without the full flywheel package), it reads environment variables directly.
