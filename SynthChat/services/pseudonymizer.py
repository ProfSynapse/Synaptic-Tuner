"""Programmatic pseudonymization helpers."""

from __future__ import annotations

import hashlib
import os
import random
import re
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Dict, List, Optional

try:  # pragma: no cover - optional dependency
    from faker import Faker  # type: ignore
except ImportError:  # pragma: no cover - fallback covered indirectly
    Faker = None

from .privacy_filter import PrivacyDetectionResult, PrivacySpan


class _SimpleFaker:
    """Minimal fallback generator when Faker is unavailable."""

    FIRST_NAMES = ["Alex", "Jordan", "Taylor", "Morgan", "Casey", "Riley"]
    LAST_NAMES = ["Hart", "Rowe", "Quinn", "Blake", "Parker", "Reed"]
    STREETS = ["Maple Ave", "Oak Street", "Pine Road", "Cedar Lane"]
    CITIES = ["Springfield", "Fairview", "Riverton", "Ashford"]
    STATES = ["CA", "NY", "TX", "WA"]

    def __init__(self, locale: str = "en_US"):
        self.locale = locale
        self.random = random.Random(0)

    def seed_instance(self, seed: int) -> None:
        self.random.seed(seed)

    def name(self) -> str:
        return f"{self.random.choice(self.FIRST_NAMES)} {self.random.choice(self.LAST_NAMES)}"

    def first_name(self) -> str:
        return self.random.choice(self.FIRST_NAMES)

    def last_name(self) -> str:
        return self.random.choice(self.LAST_NAMES)

    def phone_number(self) -> str:
        return f"({self.random.randint(200, 999)}) {self.random.randint(200, 999)}-{self.random.randint(1000, 9999)}"

    def street_address(self) -> str:
        return f"{self.random.randint(100, 9999)} {self.random.choice(self.STREETS)}"

    def city(self) -> str:
        return self.random.choice(self.CITIES)

    def state_abbr(self) -> str:
        return self.random.choice(self.STATES)

    def postcode(self) -> str:
        return f"{self.random.randint(10000, 99999)}"

    def url(self) -> str:
        return f"https://www.{self.random.choice(['example', 'sample', 'demo'])}.test"


def _make_faker(locale: str):
    if Faker is not None:  # pragma: no branch
        return Faker(locale)
    return _SimpleFaker(locale)


@dataclass(frozen=True)
class PseudonymizationResult:
    """Result from pseudonymizing already-detected spans."""

    text: str
    replaced_text: str
    replacements: Dict[str, str]
    strategy: str

    def to_metadata(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy,
            "replacement_count": len(self.replacements),
            "changed": self.text != self.replaced_text,
        }


class Pseudonymizer:
    """Replace typed privacy spans with synthetic realistic values."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = dict(config or {})
        self.locale = str(self.config.get("faker_locale", "en_US")).strip() or "en_US"
        self.fake_email_domain = str(self.config.get("fake_email_domain", "example.com")).strip() or "example.com"
        self.consistency_scope = str(self.config.get("consistency_scope", "document")).strip().lower() or "document"
        self.preserve_date_shape = bool(self.config.get("preserve_date_shape", True))
        self.preserve_account_number_shape = bool(self.config.get("preserve_account_number_shape", True))
        self.secret_strategy = str(self.config.get("secret_strategy", "mask")).strip().lower() or "mask"
        self.seed_salt = str(self.config.get("seed_salt") or os.getenv("SYNTHCHAT_PRIVACY_SEED", "synthchat-privacy"))
        self._scope_maps: Dict[str, Dict[tuple[str, str], str]] = {}

    def pseudonymize(
        self,
        detection: PrivacyDetectionResult,
        *,
        scope_key: Optional[str],
    ) -> PseudonymizationResult:
        """Replace detected spans with synthetic values."""
        if not detection.spans:
            return PseudonymizationResult(
                text=detection.text,
                replaced_text=detection.text,
                replacements={},
                strategy="programmatic",
            )

        effective_scope = self._resolve_scope_key(scope_key)
        mapping = self._scope_maps.setdefault(effective_scope, {})
        replacements: Dict[str, str] = {}
        pieces: List[str] = []
        cursor = 0

        for span in detection.spans:
            if span.start < cursor:
                continue
            pieces.append(detection.text[cursor:span.start])
            replacement = mapping.get((span.label, span.text))
            if replacement is None:
                replacement = self._replacement_for_span(span, effective_scope)
                mapping[(span.label, span.text)] = replacement
            replacements[span.text] = replacement
            pieces.append(replacement)
            cursor = span.end

        pieces.append(detection.text[cursor:])
        return PseudonymizationResult(
            text=detection.text,
            replaced_text="".join(pieces),
            replacements=replacements,
            strategy="programmatic",
        )

    def _resolve_scope_key(self, scope_key: Optional[str]) -> str:
        if self.consistency_scope == "run":
            return "__run__"
        if self.consistency_scope == "global":
            return "__global__"
        return str(scope_key or "__document__")

    def _replacement_for_span(self, span: PrivacySpan, scope_key: str) -> str:
        rng = random.Random(self._seed_for(scope_key, span.label, span.text))
        fake = _make_faker(self.locale)
        if hasattr(fake, "seed_instance"):
            fake.seed_instance(rng.randint(0, 2**31 - 1))

        if span.label == "private_person":
            return str(fake.name())
        if span.label == "private_email":
            return self._fake_email(fake, rng)
        if span.label == "private_phone":
            return self._fake_phone(fake)
        if span.label == "private_address":
            return self._fake_address(fake)
        if span.label == "private_url":
            return self._fake_url(fake)
        if span.label == "private_date":
            return self._fake_date(span.text, rng)
        if span.label == "account_number":
            if self.preserve_account_number_shape:
                return self._shape_preserving_digits(span.text, rng)
            return "".join(str(rng.randint(0, 9)) for _ in range(max(8, len(re.sub(r"\D", "", span.text)))))
        if span.label == "secret":
            if self.secret_strategy == "mask":
                return span.placeholder
            return self._fake_secret(span.text, rng)
        return span.placeholder

    def _seed_for(self, scope_key: str, label: str, text: str) -> int:
        digest = hashlib.sha256(f"{self.seed_salt}|{scope_key}|{label}|{text}".encode("utf-8")).hexdigest()
        return int(digest[:16], 16)

    def _fake_email(self, fake: Any, rng: random.Random) -> str:
        first = str(getattr(fake, "first_name")())
        last = str(getattr(fake, "last_name")())
        separator = rng.choice([".", "_", ""])
        local = f"{first}{separator}{last}".lower()
        return f"{local}@{self.fake_email_domain}"

    def _fake_phone(self, fake: Any) -> str:
        return str(getattr(fake, "phone_number")())

    def _fake_address(self, fake: Any) -> str:
        street = str(getattr(fake, "street_address")())
        city = str(getattr(fake, "city")())
        state = str(getattr(fake, "state_abbr")())
        postcode = str(getattr(fake, "postcode")())
        return f"{street}, {city}, {state} {postcode}"

    def _fake_url(self, fake: Any) -> str:
        return str(getattr(fake, "url")())

    def _fake_date(self, original_text: str, rng: random.Random) -> str:
        if self.preserve_date_shape:
            preserved = self._shape_preserving_date(original_text, rng)
            if preserved:
                return preserved
        base = date(2024, 1, 1) + timedelta(days=rng.randint(0, 365))
        return base.isoformat()

    def _shape_preserving_date(self, original_text: str, rng: random.Random) -> Optional[str]:
        text = original_text.strip()
        iso_match = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", text)
        if iso_match:
            shifted = date(int(iso_match.group(1)), int(iso_match.group(2)), int(iso_match.group(3))) + timedelta(days=rng.randint(30, 400))
            return shifted.isoformat()
        slash_match = re.fullmatch(r"(\d{1,2})/(\d{1,2})/(\d{4})", text)
        if slash_match:
            month = rng.randint(1, 12)
            day = rng.randint(1, 28)
            year = max(2000, min(2035, int(slash_match.group(3)) + rng.randint(-2, 2)))
            return f"{month:0{len(slash_match.group(1))}d}/{day:0{len(slash_match.group(2))}d}/{year}"
        month_name_match = re.fullmatch(r"([A-Za-z]+) (\d{1,2}), (\d{4})", text)
        if month_name_match:
            month_names = [
                "January", "February", "March", "April", "May", "June",
                "July", "August", "September", "October", "November", "December",
            ]
            month = rng.choice(month_names)
            day = rng.randint(1, 28)
            year = max(2000, min(2035, int(month_name_match.group(3)) + rng.randint(-2, 2)))
            return f"{month} {day}, {year}"
        return None

    def _shape_preserving_digits(self, original_text: str, rng: random.Random) -> str:
        pieces: List[str] = []
        for char in original_text:
            if char.isdigit():
                pieces.append(str(rng.randint(0, 9)))
            else:
                pieces.append(char)
        return "".join(pieces)

    def _fake_secret(self, original_text: str, rng: random.Random) -> str:
        alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        if original_text.startswith("sk-"):
            return "sk-" + "".join(rng.choice(alphabet) for _ in range(max(20, len(original_text) - 3)))
        if original_text.startswith("hf_"):
            return "hf_" + "".join(rng.choice(alphabet) for _ in range(max(20, len(original_text) - 3)))
        length = max(16, len(original_text))
        return "".join(rng.choice(alphabet) for _ in range(length))
