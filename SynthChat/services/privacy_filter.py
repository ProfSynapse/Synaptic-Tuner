"""Privacy detection wrappers for local sanitization."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional


_PLACEHOLDER_BY_LABEL = {
    "private_person": "[PRIVATE_PERSON]",
    "private_address": "[PRIVATE_ADDRESS]",
    "private_email": "[PRIVATE_EMAIL]",
    "private_phone": "[PRIVATE_PHONE]",
    "private_url": "[PRIVATE_URL]",
    "private_date": "[PRIVATE_DATE]",
    "account_number": "[ACCOUNT_NUMBER]",
    "secret": "[SECRET]",
}


@dataclass(frozen=True)
class PrivacySpan:
    """One detected privacy span."""

    label: str
    start: int
    end: int
    text: str
    placeholder: str


@dataclass(frozen=True)
class PrivacyDetectionResult:
    """Detection output for one input string."""

    text: str
    masked_text: str
    spans: tuple[PrivacySpan, ...]
    provider: str

    def to_metadata(self) -> Dict[str, Any]:
        counts: Dict[str, int] = {}
        for span in self.spans:
            counts[span.label] = counts.get(span.label, 0) + 1
        return {
            "provider": self.provider,
            "span_count": len(self.spans),
            "labels": sorted(counts.keys()),
            "by_label": counts,
            "masked_text_changed": self.masked_text != self.text,
            "spans": [
                {
                    "label": span.label,
                    "start": int(span.start),
                    "end": int(span.end),
                    "placeholder": span.placeholder,
                }
                for span in self.spans
            ],
        }


class PrivacyFilterService:
    """Local privacy detector service."""

    _runtime_local = threading.local()

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = dict(config or {})
        self.provider = str(self.config.get("provider", "opf")).strip().lower() or "opf"
        self.checkpoint = self.config.get("checkpoint")
        self.device = str(self.config.get("device", "cpu")).strip().lower() or "cpu"
        self.output_mode = str(self.config.get("output_mode", "typed")).strip().lower() or "typed"

    def detect(self, text: str) -> PrivacyDetectionResult:
        """Detect privacy spans and return a typed masked rendering."""
        if self.provider != "opf":
            raise ValueError(f"Unsupported privacy detector provider: {self.provider}")

        redactor = self._get_opf_runtime()
        result = redactor.redact(text)
        spans = self._normalize_spans(getattr(result, "detected_spans", ()) or ())
        masked_text = self._apply_placeholders(text, spans)
        return PrivacyDetectionResult(
            text=text,
            masked_text=masked_text,
            spans=tuple(spans),
            provider="opf",
        )

    def _get_opf_runtime(self):
        """Load or reuse a cached OPF runtime in this thread."""
        cache = getattr(self._runtime_local, "cache", None)
        if cache is None:
            cache = {}
            self._runtime_local.cache = cache

        cache_key = (self.checkpoint or "", self.device, self.output_mode)
        if cache_key in cache:
            return cache[cache_key]

        try:
            from opf import OPF  # type: ignore
        except ImportError as exc:  # pragma: no cover - import depends on local runtime
            raise RuntimeError(
                "Privacy profile requires the OPF runtime, but `opf` is not installed. "
                "Install the OpenAI Privacy Filter helper runtime before enabling this profile."
            ) from exc

        redactor = OPF(
            model=self.checkpoint,
            device="cuda" if self.device == "cuda" else "cpu",
            output_mode="typed",
            output_text_only=False,
        )
        cache[cache_key] = redactor
        return redactor

    def _normalize_spans(self, raw_spans: Iterable[Any]) -> List[PrivacySpan]:
        spans: List[PrivacySpan] = []
        for raw in raw_spans:
            label = str(getattr(raw, "label", "") or "").strip().lower()
            start = int(getattr(raw, "start", 0) or 0)
            end = int(getattr(raw, "end", 0) or 0)
            text = str(getattr(raw, "text", "") or "")
            if not label or end <= start:
                continue
            spans.append(
                PrivacySpan(
                    label=label,
                    start=start,
                    end=end,
                    text=text,
                    placeholder=_PLACEHOLDER_BY_LABEL.get(label, "[REDACTED]"),
                )
            )
        spans.sort(key=lambda item: (item.start, item.end))
        return spans

    @staticmethod
    def _apply_placeholders(text: str, spans: List[PrivacySpan]) -> str:
        if not spans:
            return text
        pieces: List[str] = []
        cursor = 0
        for span in spans:
            if span.start < cursor:
                continue
            pieces.append(text[cursor:span.start])
            pieces.append(span.placeholder)
            cursor = span.end
        pieces.append(text[cursor:])
        return "".join(pieces)
