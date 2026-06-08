"""SynthChat LLM Client Pool - Caching and stage-specific client selection.

Location: SynthChat/llm/client_pool.py
Purpose: Manage a cache of LLM clients keyed by (provider, model, routing,
         timeout) so that repeated calls with the same spec reuse a single
         client instance.
Usage: Created by SynthChatGenerator.__init__ and used throughout generation
       to resolve per-stage LLM overrides.
"""

import json
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

from shared.llm.factory import create_client as _default_create_client


class LLMClientPool:
    """Cache-backed pool of LLM clients with stage-specific resolution."""

    def __init__(self, default_client: Any, client_factory: Any = None) -> None:
        self.default_client = default_client
        self._cache: Dict[Tuple[Any, ...], Any] = {}
        self._create_client = client_factory or _default_create_client

    def get_stage_clients(self, stage_config: Optional[Dict[str, Any]]) -> List[Any]:
        """Resolve the client chain for a given stage config dict."""
        if not isinstance(stage_config, dict):
            return [self.default_client]

        client_chain: List[Any] = []
        primary_spec = self.normalize_stage_spec(stage_config)
        if primary_spec is None:
            client_chain.append(self.default_client)
        else:
            client_chain.append(self.get_or_create(primary_spec))

        fallback_specs = stage_config.get("fallback_models")
        if isinstance(fallback_specs, list):
            for raw_spec in fallback_specs:
                spec = self.normalize_stage_spec(raw_spec)
                if spec is None:
                    continue
                client = self.get_or_create(spec)
                if all(client is not existing for existing in client_chain):
                    client_chain.append(client)
        return client_chain

    @staticmethod
    def normalize_stage_spec(value: Any) -> Optional[Dict[str, Any]]:
        """Normalize a stage LLM spec into a canonical dict or None."""
        if isinstance(value, str):
            model = value.strip()
            return {"model": model} if model else None
        if not isinstance(value, dict):
            return None

        model = str(value.get("model") or "").strip()
        provider = str(value.get("provider") or "").strip().lower()
        provider_routing = value.get("provider_routing")
        timeout_seconds = value.get("timeout_seconds")
        thinking_effort = value.get("thinking_effort", value.get("reasoning_effort"))

        has_override = bool(
            model
            or provider
            or provider_routing is not None
            or timeout_seconds is not None
            or thinking_effort is not None
        )
        if not has_override:
            return None

        spec: Dict[str, Any] = {}
        if model:
            spec["model"] = model
        if provider:
            spec["provider"] = provider
        if provider_routing is not None:
            spec["provider_routing"] = provider_routing
        if timeout_seconds is not None:
            spec["timeout_seconds"] = timeout_seconds
        if thinking_effort is not None:
            spec["thinking_effort"] = thinking_effort
        return spec

    def get_or_create(self, spec: Dict[str, Any]) -> Any:
        """Return a cached client matching spec, or create and cache a new one."""
        base_provider = str(getattr(self.default_client, "provider_name", "openrouter") or "openrouter").strip().lower()
        base_model = str(getattr(self.default_client, "model_name", "") or "").strip()
        base_provider_routing = deepcopy(getattr(self.default_client, "provider", None))
        base_timeout_seconds = getattr(self.default_client, "timeout_seconds", None)
        base_thinking_effort = getattr(self.default_client, "thinking_effort", None)

        provider = str(spec.get("provider") or base_provider).strip().lower()
        model = str(spec.get("model") or base_model).strip()
        provider_routing = deepcopy(spec.get("provider_routing", base_provider_routing))
        timeout_seconds = spec.get("timeout_seconds", base_timeout_seconds)
        thinking_effort = spec.get("thinking_effort", base_thinking_effort)

        cache_key = (
            provider,
            model,
            json.dumps(provider_routing, sort_keys=True) if provider_routing is not None else "",
            str(timeout_seconds) if timeout_seconds is not None else "",
            str(thinking_effort) if thinking_effort is not None else "",
        )
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        if provider == base_provider and model == base_model:
            same_routing = provider_routing == base_provider_routing
            same_timeout = (
                timeout_seconds == base_timeout_seconds
                or (timeout_seconds is None and base_timeout_seconds is None)
            )
            same_thinking_effort = thinking_effort == base_thinking_effort
            if same_routing and same_timeout and same_thinking_effort:
                self._cache[cache_key] = self.default_client
                return self.default_client

        config_defaults = {
            "provider": provider,
            "model": model,
        }
        if provider_routing is not None:
            config_defaults["provider_routing"] = provider_routing
        if timeout_seconds is not None:
            config_defaults["timeout_seconds"] = timeout_seconds
        if thinking_effort is not None:
            config_defaults["thinking_effort"] = thinking_effort

        client = self._create_client(
            provider=provider,
            model=model,
            config_defaults=config_defaults,
        )
        self._cache[cache_key] = client
        return client
