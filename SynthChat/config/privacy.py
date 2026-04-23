"""Privacy preprocess configuration helpers."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional

from ..utils.yaml_loader import load_yaml

_CONFIG_DIR = Path(__file__).parent

_DEFAULT_PRIVACY_SETTINGS: Dict[str, Any] = {
    "enabled": False,
    "profile": None,
    "apply_to": {
        "docs": True,
        "input_jsonl": False,
        "generated_output": False,
    },
    "on_error": "fail",
}

_DEFAULT_PRIVACY_PROFILES: Dict[str, Any] = {
    "profiles": {
        "mask_only": {
            "detector": {
                "provider": "opf",
                "checkpoint": None,
                "device": "cpu",
                "output_mode": "typed",
            },
            "transform": {
                "mode": "mask",
                "keep_typed_placeholders": True,
                "consistency_scope": "document",
            },
        },
        "realistic_pseudonyms": {
            "detector": {
                "provider": "opf",
                "checkpoint": None,
                "device": "cpu",
                "output_mode": "typed",
            },
            "transform": {
                "mode": "pseudonymize",
                "provider": "programmatic",
                "consistency_scope": "document",
                "faker_locale": "en_US",
                "fake_email_domain": "example.com",
                "preserve_date_shape": True,
                "preserve_account_number_shape": True,
                "secret_strategy": "mask",
                "llm_polish": {
                    "enabled": False,
                },
            },
        },
    }
}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep-merge override into a copy of base."""
    result = deepcopy(base)
    for key, value in (override or {}).items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def get_default_privacy_settings() -> Dict[str, Any]:
    """Return default privacy preprocess settings."""
    return deepcopy(_DEFAULT_PRIVACY_SETTINGS)


def load_privacy_profiles(config_path: Optional[str | Path] = None) -> Dict[str, Any]:
    """Load named privacy preprocess profiles."""
    path = Path(config_path) if config_path else _CONFIG_DIR / "privacy_profiles.yaml"
    if path.is_file():
        data = load_yaml(path)
        if isinstance(data, dict) and isinstance(data.get("profiles"), dict):
            return {
                "profiles": {
                    **deepcopy(_DEFAULT_PRIVACY_PROFILES["profiles"]),
                    **deepcopy(data["profiles"]),
                }
            }
    return deepcopy(_DEFAULT_PRIVACY_PROFILES)


def resolve_privacy_settings(settings: Dict[str, Any], overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Resolve global privacy settings from defaults, settings, and optional overrides."""
    repo_settings = settings.get("privacy_preprocess") if isinstance(settings, dict) else None
    resolved = _deep_merge(_DEFAULT_PRIVACY_SETTINGS, repo_settings or {})
    if overrides:
        resolved = _deep_merge(resolved, overrides)
    return resolved


def resolve_privacy_profile(
    *,
    profile_name: Optional[str],
    profiles_registry: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Resolve a named privacy profile."""
    if not profile_name:
        return None
    profiles = (profiles_registry or {}).get("profiles") or {}
    profile = profiles.get(profile_name)
    if not isinstance(profile, dict):
        raise KeyError(f"Privacy profile not found: {profile_name}")
    return deepcopy(profile)

