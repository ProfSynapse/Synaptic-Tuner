"""Privacy preprocess orchestration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from shared.llm.providers.lmstudio import LMStudioClient

from ..config.privacy import (
    load_privacy_profiles,
    resolve_privacy_profile,
    resolve_privacy_settings,
)
from .privacy_filter import PrivacyFilterService
from .pseudonymizer import Pseudonymizer


@dataclass(frozen=True)
class PrivacyPreprocessResult:
    """Combined sanitize result for one text input."""

    original_text: str
    masked_text: str
    sanitized_text: str
    changed: bool
    profile_name: str
    detection: Dict[str, Any]
    transform: Dict[str, Any]
    polish: Optional[Dict[str, Any]] = None

    def to_metadata(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "profile": self.profile_name,
            "changed": self.changed,
            "detection": dict(self.detection),
            "transform": dict(self.transform),
        }
        if self.polish is not None:
            payload["polish"] = dict(self.polish)
        return payload


class OpenAICompatiblePolisher:
    """Optional post-sanitize polishing step using an OpenAI-compatible endpoint."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = dict(config or {})
        host_env = str(self.config.get("host_env", "VLLM_HOST")).strip() or "VLLM_HOST"
        port_env = str(self.config.get("port_env", "VLLM_PORT")).strip() or "VLLM_PORT"
        host = os.getenv(host_env, str(self.config.get("default_host", "127.0.0.1")))
        port = int(os.getenv(port_env, str(self.config.get("default_port", 8000))))
        model = str(self.config.get("model", "local-model")).strip() or "local-model"
        self.temperature = float(self.config.get("temperature", 0.1))
        self.max_tokens = int(self.config.get("max_tokens", 2048))
        self.client = LMStudioClient(host=host, port=port, model=model)

    def polish(self, text: str) -> str:
        """Polish already-sanitized text without changing entities."""
        messages = [
            {
                "role": "system",
                "content": (
                    "Rewrite the user-provided text for fluency and naturalness. "
                    "Do not add or remove facts. Preserve all placeholder tokens and all synthetic identities exactly. "
                    "Do not invent, recover, or infer any original private information. "
                    "Return plain text only."
                ),
            },
            {"role": "user", "content": text},
        ]
        return self.client.chat(
            messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )


class PrivacyPreprocessor:
    """Config-driven text privacy preprocess pipeline."""

    def __init__(
        self,
        *,
        profile_name: str,
        profile: Dict[str, Any],
        detector: Optional[PrivacyFilterService] = None,
        pseudonymizer: Optional[Pseudonymizer] = None,
        polisher: Optional[OpenAICompatiblePolisher] = None,
    ):
        self.profile_name = profile_name
        self.profile = dict(profile or {})
        self.detector = detector or PrivacyFilterService(self.profile.get("detector"))
        self.transform_config = dict(self.profile.get("transform") or {})
        self.pseudonymizer = pseudonymizer or Pseudonymizer(self.transform_config)
        polish_cfg = dict(self.transform_config.get("llm_polish") or {})
        self.polisher = polisher or (OpenAICompatiblePolisher(polish_cfg) if polish_cfg.get("enabled") else None)

    @classmethod
    def from_registry(
        cls,
        *,
        profile_name: str,
        profiles_registry: Dict[str, Any],
    ) -> "PrivacyPreprocessor":
        profile = resolve_privacy_profile(profile_name=profile_name, profiles_registry=profiles_registry)
        if profile is None:
            raise KeyError(f"Privacy profile not found: {profile_name}")
        return cls(profile_name=profile_name, profile=profile)

    def process_text(
        self,
        text: str,
        *,
        scope_key: Optional[str] = None,
    ) -> PrivacyPreprocessResult:
        """Run detect -> transform -> optional polish on one text string."""
        detection = self.detector.detect(text)
        mode = str(self.transform_config.get("mode", "mask")).strip().lower() or "mask"
        sanitized_text = detection.masked_text
        transform_meta: Dict[str, Any] = {
            "mode": mode,
            "changed": sanitized_text != text,
        }

        if mode == "pseudonymize":
            pseudonymized = self.pseudonymizer.pseudonymize(detection, scope_key=scope_key)
            sanitized_text = pseudonymized.replaced_text
            transform_meta.update(pseudonymized.to_metadata())

        polish_meta: Optional[Dict[str, Any]] = None
        if self.polisher is not None and sanitized_text.strip():
            before = sanitized_text
            after = self.polisher.polish(sanitized_text)
            sanitized_text = after.strip() or before
            polish_meta = {
                "enabled": True,
                "changed": sanitized_text != before,
            }

        return PrivacyPreprocessResult(
            original_text=text,
            masked_text=detection.masked_text,
            sanitized_text=sanitized_text,
            changed=sanitized_text != text,
            profile_name=self.profile_name,
            detection=detection.to_metadata(),
            transform=transform_meta,
            polish=polish_meta,
        )

    def sanitize_payload(
        self,
        payload: Any,
        *,
        scope_key: Optional[str] = None,
    ) -> Tuple[Any, List[Dict[str, Any]]]:
        """Recursively sanitize string values inside a payload."""
        reports: List[Dict[str, Any]] = []

        def walk(value: Any, path: str) -> Any:
            if isinstance(value, str):
                result = self.process_text(value, scope_key=scope_key or path)
                if result.changed:
                    reports.append({"path": path, **result.to_metadata()})
                return result.sanitized_text
            if isinstance(value, list):
                return [walk(item, f"{path}[{idx}]") for idx, item in enumerate(value)]
            if isinstance(value, dict):
                return {key: walk(item, f"{path}.{key}" if path else str(key)) for key, item in value.items()}
            return value

        return walk(payload, ""), reports


def summarize_privacy_reports(
    *,
    profile_name: str,
    reports: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Summarize recursive sanitize reports into lightweight metadata."""
    labels = set()
    span_count = 0
    for report in reports:
        detection = report.get("detection") if isinstance(report, dict) else None
        if not isinstance(detection, dict):
            continue
        for label in detection.get("labels") or []:
            if label:
                labels.add(str(label))
        try:
            span_count += int(detection.get("span_count", 0) or 0)
        except (TypeError, ValueError):
            continue

    return {
        "profile": profile_name,
        "changed": bool(reports),
        "report_count": len(reports),
        "reports": reports,
        "detection": {
            "labels": sorted(labels),
            "span_count": span_count,
        },
    }


def sanitize_payload_with_metadata(
    payload: Any,
    *,
    preprocessor: PrivacyPreprocessor,
    scope_key: Optional[str] = None,
    metadata_field: str = "privacy_preprocess",
) -> Tuple[Any, Dict[str, Any]]:
    """Sanitize a payload and attach summary metadata when the payload is a dict."""
    sanitized_payload, reports = preprocessor.sanitize_payload(payload, scope_key=scope_key)
    summary = summarize_privacy_reports(profile_name=preprocessor.profile_name, reports=reports)

    if isinstance(sanitized_payload, dict):
        metadata = sanitized_payload.setdefault("metadata", {})
        if isinstance(metadata, dict):
            metadata[metadata_field] = summary

    return sanitized_payload, summary


def resolve_privacy_preprocessor(
    *,
    config_dir: str | Path,
    settings: Dict[str, Any],
    apply_target: str,
    profile_override: Optional[str] = None,
) -> Optional[PrivacyPreprocessor]:
    """Resolve an enabled privacy preprocessor for one CLI mode/target."""
    overrides: Dict[str, Any] = {}
    if profile_override:
        overrides = {"enabled": True, "profile": profile_override}
    resolved = resolve_privacy_settings(settings, overrides)
    apply_to = resolved.get("apply_to") if isinstance(resolved, dict) else {}
    if not resolved.get("enabled"):
        return None
    if not bool((apply_to or {}).get(apply_target, False)):
        return None

    profile_name = str(resolved.get("profile") or "").strip()
    if not profile_name:
        return None

    profiles_registry = load_privacy_profiles(Path(config_dir) / "privacy_profiles.yaml")
    return PrivacyPreprocessor.from_registry(
        profile_name=profile_name,
        profiles_registry=profiles_registry,
    )
