"""OpenRouter provider implementation."""

import json
import re
import requests
from typing import Dict, Any, List

from ..base import BaseLLMClient
from ..exceptions import LLMConnectionError, LLMResponseError
from ..usage import LLMCompletionV1, LLMStructuredV1, usage_from_openai_block


class OpenRouterBatchRejectedError(LLMResponseError):
    """A provider-confirmed 4xx rejection proving no batch was accepted."""

    def __init__(self, status_code: int, code: str | None = None):
        safe_code = _safe_error_code(code)
        detail = f", code={safe_code}" if safe_code else ""
        super().__init__(f"OpenRouter batch submission rejected: HTTP {status_code}{detail}")
        self.status_code = status_code
        self.code = safe_code


class OpenRouterBatchSubmissionAmbiguousError(LLMResponseError):
    """A POST outcome that cannot safely be retried without reconciliation."""


class OpenRouterClient(BaseLLMClient):
    """OpenRouter API client with structured output and provider routing support."""

    def __init__(
        self,
        api_key: str,
        model: str,
        provider: Dict[str, Any] = None,
        timeout_seconds: float = 60.0,
        thinking_effort: str | None = None,
    ):
        """
        Initialize OpenRouter client.

        Args:
            api_key: OpenRouter API key
            model: Model name (e.g., 'openai/gpt-5-mini')
            provider: Optional provider routing config:
                - order: List of provider names to prioritize (e.g., ["Groq", "Together"])
                - allow_fallbacks: Whether to fall back to other providers (default: True)
                - require_parameters: Only use providers supporting all params (default: False)
                - data_collection: "allow" or "deny" to filter by data policy
        """
        self.api_key = api_key
        self.model = model
        self.provider = provider
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        self.batch_api_url = "https://openrouter.ai/api/beta/batches"
        self.timeout_seconds = float(timeout_seconds)
        self.thinking_effort = _normalize_thinking_effort(thinking_effort)

    @property
    def provider_name(self) -> str:
        return "openrouter"

    @property
    def model_name(self) -> str:
        return self.model

    def list_models(self) -> List[str]:
        """List models available via OpenRouter."""
        url = "https://openrouter.ai/api/v1/models"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            models = []
            for item in data.get("data", []):
                mid = item.get("id") or item.get("name")
                if mid:
                    models.append(str(mid))
            return models
        except Exception as e:
            raise LLMResponseError(f"Failed to list OpenRouter models: {e}")

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs
    ) -> LLMCompletionV1:
        """Send chat completion request to OpenRouter.

        Usage is measured from the response ``usage`` block
        (``prompt_tokens`` / ``completion_tokens``); ``None`` when absent.
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        # Add provider routing if configured
        if self.provider:
            payload["provider"] = self.provider
        if self.thinking_effort:
            payload["reasoning"] = {"effort": self.thinking_effort}

        try:
            data = self._make_request(payload)
            usage = usage_from_openai_block(data.get("usage"))
            message = data["choices"][0]["message"]
            content = message.get("content")

            if content is None:
                tool_calls = message.get("tool_calls")
                if tool_calls:
                    return LLMCompletionV1(json.dumps({"content": None, "tool_calls": tool_calls}), usage)
                raise LLMResponseError("Empty response from OpenRouter")

            if not isinstance(content, str):
                content = str(content)
            if not content.strip():
                tool_calls = message.get("tool_calls")
                if tool_calls:
                    return LLMCompletionV1(json.dumps({"content": None, "tool_calls": tool_calls}), usage)
                raise LLMResponseError("Empty response from OpenRouter")

            return LLMCompletionV1(content, usage)

        except Exception as e:
            raise LLMResponseError(f"OpenRouter chat request failed: {e}")

    def structured_output(
        self,
        messages: List[Dict[str, str]],
        schema: Dict[str, Any],
        temperature: float = 0.3,
        max_tokens: int | None = None,
        **kwargs
    ) -> LLMStructuredV1:
        """Send request with JSON schema for structured output."""
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": schema.get("name", "response"),
                    "strict": True,
                    "schema": schema
                }
            }
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        # Add provider routing if configured
        if self.provider:
            payload["provider"] = self.provider
        if self.thinking_effort:
            payload["reasoning"] = {"effort": self.thinking_effort}

        content = ""
        try:
            data = self._make_request(payload)
            usage = usage_from_openai_block(data.get("usage"))
            content = data["choices"][0]["message"]["content"]

            # Parse JSON response
            if not content or not content.strip():
                raise LLMResponseError("Empty response from OpenRouter")

            value = json.loads(content)
            if not isinstance(value, dict):
                raise LLMResponseError("Structured output from OpenRouter is not a JSON object")
            return LLMStructuredV1(value, usage)

        except json.JSONDecodeError as e:
            raise LLMResponseError(
                f"Failed to parse structured output: {e}\nResponse excerpt: {_truncate_response(content)}",
                raw_response=content,
            )
        except Exception as e:
            raise LLMResponseError(f"OpenRouter structured output failed: {e}")

    def _make_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Make HTTP request to OpenRouter API."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/ProfSynapse/Toolset-Training",
            "X-Title": "Shared LLM Client"
        }

        try:
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            return response.json()

        except requests.exceptions.ConnectionError as e:
            raise LLMConnectionError(f"Cannot connect to OpenRouter: {e}")
        except requests.exceptions.Timeout as e:
            raise LLMConnectionError(f"OpenRouter request timed out: {e}")
        except requests.exceptions.HTTPError as e:
            # Try to get error details from response
            try:
                error_detail = response.json()
                raise LLMResponseError(f"OpenRouter HTTP error: {e}\nDetails: {error_detail}")
            except:
                raise LLMResponseError(f"OpenRouter HTTP error: {e}")
        except Exception as e:
            raise LLMConnectionError(f"OpenRouter request failed: {e}")

    def submit_batch(
        self,
        items: List[Dict[str, Any]],
        *,
        endpoint: str = "/v1/chat/completions",
    ) -> Dict[str, Any]:
        """Submit an asynchronous OpenRouter batch without polling it.

        This is intentionally OpenRouter-specific rather than part of the
        provider-neutral ``BaseLLMClient`` contract.  Callers own durable
        idempotency state around this effectful operation.
        """
        if endpoint != "/v1/chat/completions":
            raise ValueError("OpenRouter batch endpoint must be /v1/chat/completions")
        if not isinstance(items, list) or not items:
            raise ValueError("OpenRouter batch items must be a non-empty list")

        seen: set[str] = set()
        normalized: list[dict[str, Any]] = []
        for index, item in enumerate(items):
            if not isinstance(item, dict):
                raise ValueError(f"OpenRouter batch item {index} must be a mapping")
            custom_id = item.get("custom_id")
            body = item.get("body")
            if not isinstance(custom_id, str) or not custom_id.strip():
                raise ValueError(f"OpenRouter batch item {index} requires a non-empty custom_id")
            if custom_id in seen:
                raise ValueError(f"duplicate OpenRouter batch custom_id: {custom_id}")
            if not isinstance(body, dict):
                raise ValueError(f"OpenRouter batch item {custom_id} body must be a mapping")
            if body.get("model") != self.model:
                raise ValueError(f"OpenRouter batch item {custom_id} model must equal outer model")
            seen.add(custom_id)
            normalized.append({"custom_id": custom_id, "body": dict(body)})

        payload = {
            "endpoint": endpoint,
            "model": self.model,
            "requests": normalized,
        }
        result = self._make_batch_request("post", self.batch_api_url, payload=payload)
        try:
            _validate_batch_object(result, operation="submit")
        except LLMResponseError as exc:
            raise OpenRouterBatchSubmissionAmbiguousError(
                "OpenRouter batch submission returned an invalid success object"
            ) from exc
        return result

    def observe_batch(self, batch_id: str) -> Dict[str, Any]:
        """Retrieve one OpenRouter batch object, including inline results."""
        if not isinstance(batch_id, str) or not re.fullmatch(r"[A-Za-z0-9._-]+", batch_id):
            raise ValueError("OpenRouter batch_id contains unsupported characters")
        result = self._make_batch_request("get", f"{self.batch_api_url}/{batch_id}")
        _validate_batch_object(result, operation="observe")
        if result["id"] != batch_id:
            raise LLMResponseError("OpenRouter batch observation returned a different batch id")
        return result

    def _make_batch_request(
        self,
        method: str,
        url: str,
        *,
        payload: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Make one non-retrying Batch API request and return a JSON object."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/ProfSynapse/Toolset-Training",
            "X-Title": "Shared LLM Client",
        }
        try:
            if method == "post":
                response = requests.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout_seconds,
                )
            elif method == "get":
                response = requests.get(url, headers=headers, timeout=self.timeout_seconds)
            else:  # pragma: no cover - private invariant
                raise ValueError(f"unsupported Batch API method: {method}")
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as exc:
            # A transport failure can be ambiguous after POST.  Deliberately do
            # not retry here; the durable caller must stop and reconcile.
            if method == "post":
                raise OpenRouterBatchSubmissionAmbiguousError(
                    f"OpenRouter batch submission transport failed: {type(exc).__name__}"
                ) from None
            raise LLMConnectionError(f"OpenRouter batch {method} transport failed: {type(exc).__name__}") from None
        except requests.exceptions.RequestException as exc:
            if method == "post":
                raise OpenRouterBatchSubmissionAmbiguousError(
                    f"OpenRouter batch submission request failed: {type(exc).__name__}"
                ) from None
            raise LLMConnectionError(f"OpenRouter batch {method} request failed: {type(exc).__name__}") from None

        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError:
            code = _closed_error_code(response)
            if method == "post" and 400 <= response.status_code < 500:
                raise OpenRouterBatchRejectedError(response.status_code, code) from None
            message = f"OpenRouter batch {method} HTTP {response.status_code}"
            if code:
                message += f" (code={code})"
            if method == "post":
                raise OpenRouterBatchSubmissionAmbiguousError(message) from None
            raise LLMResponseError(message) from None

        try:
            decoded = response.json()
        except (ValueError, json.JSONDecodeError):
            if method == "post":
                raise OpenRouterBatchSubmissionAmbiguousError(
                    "OpenRouter batch submission returned invalid JSON"
                ) from None
            raise LLMResponseError(f"OpenRouter batch {method} returned invalid JSON") from None
        if not isinstance(decoded, dict):
            if method == "post":
                raise OpenRouterBatchSubmissionAmbiguousError(
                    "OpenRouter batch submission returned a non-object response"
                )
            raise LLMResponseError(f"OpenRouter batch {method} returned a non-object response")
        try:
            json.dumps(decoded, allow_nan=False)
        except (TypeError, ValueError):
            if method == "post":
                raise OpenRouterBatchSubmissionAmbiguousError(
                    "OpenRouter batch submission returned non-finite or non-JSON data"
                ) from None
            raise LLMResponseError(
                f"OpenRouter batch {method} returned non-finite or non-JSON data"
            ) from None
        return decoded

    def test_connection(self) -> bool:
        """Test OpenRouter API connection."""
        try:
            test_payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": "test"}],
                "max_tokens": 10  # Higher for reasoning models
            }
            self._make_request(test_payload)
            return True
        except Exception as e:
            # Print error for debugging
            print(f"    Connection test error: {e}")
            return False


def _truncate_response(content: Any, limit: int = 1200) -> str:
    """Return a compact single-line excerpt for debugging malformed structured output."""
    text = str(content or "").replace("\r", "\\r").replace("\n", "\\n")
    if len(text) <= limit:
        return text
    return f"{text[:limit]}...<truncated>"


def _normalize_thinking_effort(value: str | None) -> str | None:
    if value is None:
        return None
    value = str(value).strip().lower()
    return value or None


def _validate_batch_object(value: Dict[str, Any], *, operation: str) -> None:
    batch_id = value.get("id")
    status = value.get("status")
    if not isinstance(batch_id, str) or not batch_id:
        raise LLMResponseError(f"OpenRouter batch {operation} response omitted id")
    if not isinstance(status, str) or not status:
        raise LLMResponseError(f"OpenRouter batch {operation} response omitted status")


def _closed_error_code(response: Any) -> str | None:
    """Return only a bounded provider error code; messages may echo prompts."""
    try:
        payload = response.json()
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    error = payload.get("error")
    if not isinstance(error, dict):
        return None
    code = error.get("code")
    return _safe_error_code(code)


def _safe_error_code(value: object) -> str | None:
    if not isinstance(value, (str, int)):
        return None
    normalized = str(value)
    if len(normalized) > 64 or re.fullmatch(r"[A-Za-z0-9._-]+", normalized) is None:
        return None
    return normalized
