from __future__ import annotations

import json
import shutil
from pathlib import Path
from uuid import uuid4

from SynthChat.generator import SynthChatGenerator
from SynthChat.modes.sanitize import _sanitize_jsonl
from SynthChat.services.privacy_filter import PrivacyDetectionResult, PrivacySpan
from SynthChat.services.privacy_preprocess import (
    PrivacyPreprocessResult,
    PrivacyPreprocessor,
    resolve_privacy_preprocessor,
    sanitize_payload_with_metadata,
)
from SynthChat.services.pseudonymizer import Pseudonymizer
from SynthChat.utils.docs_loader import DocFile


class _FakeLLMClient:
    def __init__(self, responses=None):
        self._responses = list(responses or [])
        self.messages = []
        self.structured_messages = []
        self.default_max_tokens = None
        self.provider = None
        self.timeout_seconds = 60.0

    @property
    def provider_name(self):
        return "openrouter"

    @property
    def model_name(self):
        return "fake/model"

    def chat(self, messages, temperature=0.7, max_tokens=2048):
        self.messages.append({"messages": messages, "temperature": temperature, "max_tokens": max_tokens})
        if not self._responses:
            raise AssertionError("No more fake responses available")
        return self._responses.pop(0)

    def structured_output(self, messages, schema, temperature=0.3, max_tokens=2048):
        self.structured_messages.append(
            {
                "messages": messages,
                "schema": schema,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        raise AssertionError("Structured output was not expected in this test")


class _FakeDetector:
    def __init__(self, detection: PrivacyDetectionResult):
        self.detection = detection
        self.calls = []

    def detect(self, text: str) -> PrivacyDetectionResult:
        self.calls.append(text)
        return self.detection


class _FakePreprocessor:
    def __init__(self, sanitized_text: str):
        self.sanitized_text = sanitized_text
        self.calls = []

    def process_text(self, text: str, *, scope_key=None):
        self.calls.append({"text": text, "scope_key": scope_key})
        return PrivacyPreprocessResult(
            original_text=text,
            masked_text=self.sanitized_text,
            sanitized_text=self.sanitized_text,
            changed=self.sanitized_text != text,
            profile_name="mask_only",
            detection={"labels": ["private_person"], "span_count": 1, "by_label": {"private_person": 1}, "spans": []},
            transform={"mode": "mask", "changed": self.sanitized_text != text},
            polish=None,
        )


class _FakeSanitizePreprocessor:
    profile_name = "mask_only"

    def sanitize_payload(self, payload, *, scope_key=None):
        data = json.loads(json.dumps(payload))
        text = data["conversations"][0]["content"]
        data["conversations"][0]["content"] = text.replace("Jane Roe", "[PRIVATE_PERSON]").replace(
            "jane.roe@example.com", "[PRIVATE_EMAIL]"
        )
        return data, [{"path": "conversations[0].content", "profile": self.profile_name, "changed": True}]


class _FakeOutputPreprocessor:
    profile_name = "mask_only"

    def process_text(self, text: str, *, scope_key=None):
        return PrivacyPreprocessResult(
            original_text=text,
            masked_text=text,
            sanitized_text=text,
            changed=False,
            profile_name=self.profile_name,
            detection={"labels": [], "span_count": 0, "by_label": {}, "spans": []},
            transform={"mode": "mask", "changed": False},
            polish=None,
        )

    def sanitize_payload(self, payload, *, scope_key=None):
        data = json.loads(json.dumps(payload))
        reports = []
        for index, message in enumerate(data.get("conversations", [])):
            content = message.get("content")
            if isinstance(content, str) and "Jane Roe" in content:
                message["content"] = content.replace("Jane Roe", "[PRIVATE_PERSON]")
                reports.append(
                    {
                        "path": f"conversations[{index}].content",
                        "profile": self.profile_name,
                        "changed": True,
                        "detection": {
                            "labels": ["private_person"],
                            "span_count": 1,
                        },
                    }
                )
        return data, reports


def test_pseudonymizer_preserves_account_number_shape():
    detection = PrivacyDetectionResult(
        text="Account 4829-1037-5581 is on file.",
        masked_text="Account [ACCOUNT_NUMBER] is on file.",
        spans=(
            PrivacySpan(
                label="account_number",
                start=8,
                end=22,
                text="4829-1037-5581",
                placeholder="[ACCOUNT_NUMBER]",
            ),
        ),
        provider="opf",
    )
    result = Pseudonymizer({"preserve_account_number_shape": True}).pseudonymize(
        detection,
        scope_key="doc-1",
    )
    replaced = result.replaced_text.split("Account ", 1)[1].split(" is on file.", 1)[0]
    assert replaced != "4829-1037-5581"
    assert len(replaced) == len("4829-1037-5581")
    assert replaced.count("-") == 2


def test_privacy_preprocessor_pseudonymizes_detected_spans():
    text = "Contact Jane Roe at jane.roe@example.com."
    detection = PrivacyDetectionResult(
        text=text,
        masked_text="Contact [PRIVATE_PERSON] at [PRIVATE_EMAIL].",
        spans=(
            PrivacySpan(
                label="private_person",
                start=8,
                end=16,
                text="Jane Roe",
                placeholder="[PRIVATE_PERSON]",
            ),
            PrivacySpan(
                label="private_email",
                start=20,
                end=40,
                text="jane.roe@example.com",
                placeholder="[PRIVATE_EMAIL]",
            ),
        ),
        provider="opf",
    )
    preprocessor = PrivacyPreprocessor(
        profile_name="realistic_pseudonyms",
        profile={
            "detector": {"provider": "opf"},
            "transform": {
                "mode": "pseudonymize",
                "provider": "programmatic",
                "fake_email_domain": "example.com",
            },
        },
        detector=_FakeDetector(detection),
    )
    result = preprocessor.process_text(text, scope_key="doc-1")
    assert result.changed is True
    assert "Jane Roe" not in result.sanitized_text
    assert "jane.roe@example.com" not in result.sanitized_text
    assert "@example.com" in result.sanitized_text


def test_generator_uses_sanitized_doc_seed_in_prompt():
    repo_root = Path(__file__).resolve().parents[1]
    config_dir = repo_root / "SynthChat" / "config"
    scenarios_dir = repo_root / "SynthChat" / "scenarios"
    rubrics_dir = repo_root / "SynthChat" / "rubrics"

    fake_preprocessor = _FakePreprocessor("Customer record for [PRIVATE_PERSON].")

    def _factory(*, profile_name, profiles_registry):
        return fake_preprocessor

    client = _FakeLLMClient(
        responses=[
            "Write a request about the customer record.",
            "I can help with that.",
        ]
    )
    generator = SynthChatGenerator(
        config_dir=config_dir,
        scenarios_dir=scenarios_dir,
        rubrics_dir=rubrics_dir,
        llm_client=client,
        engine=None,
        enable_stage_validation=False,
        privacy_settings={"enabled": False, "apply_to": {"docs": True}},
        privacy_preprocessor_factory=_factory,
    )

    scenario = {
        "type": "behavioral",
        "system": False,
        "seed_data": {"preprocess_profile": "mask_only"},
        "prompts": {
            "user": "Create a user request about this source document:\n{doc_content}",
            "assistant": "Reply briefly to the user.",
        },
    }
    doc_context = DocFile(path="tests/fixtures/privacy/raw_seed_docs/customer_support_email.txt", content="Customer record for Jane Roe.")

    result = generator.generate_single(
        scenario_key="privacy_doc_test",
        scenario=scenario,
        max_iterations=1,
        randomize_params=False,
        doc_context=doc_context,
    )

    prompt_dump = json.dumps(client.messages[0]["messages"])
    assert "[PRIVATE_PERSON]" in prompt_dump
    assert "Jane Roe" not in prompt_dump
    assert result.example["metadata"]["source_doc_privacy"]["profile"] == "mask_only"


def test_sanitize_jsonl_adds_privacy_metadata():
    tmp_path = Path("tmp/test_privacy_preprocess") / uuid4().hex
    tmp_path.mkdir(parents=True, exist_ok=True)
    try:
        input_path = tmp_path / "input.jsonl"
        output_path = tmp_path / "output.jsonl"
        input_path.write_text(
            '\n'.join(
                [
                    json.dumps({"_meta": {"version": 1}}),
                    json.dumps(
                        {
                            "conversations": [
                                {"role": "user", "content": "Contact Jane Roe at jane.roe@example.com"}
                            ]
                        }
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        summary = _sanitize_jsonl(input_path, output_path, _FakeSanitizePreprocessor())
        lines = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip()]

        assert summary["records_processed"] == 1
        assert lines[0]["_meta"]["privacy_preprocess"]["profile"] == "mask_only"
        assert lines[1]["metadata"]["privacy_preprocess"]["changed"] is True
        assert "[PRIVATE_PERSON]" in lines[1]["conversations"][0]["content"]
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_sanitize_payload_with_metadata_adds_summary():
    payload = {
        "conversations": [
            {"role": "user", "content": "Contact Jane Roe at jane.roe@example.com"}
        ]
    }
    sanitized_payload, summary = sanitize_payload_with_metadata(
        payload,
        preprocessor=_FakeSanitizePreprocessor(),
        scope_key="example-1",
        metadata_field="privacy_preprocess_input",
    )

    assert summary["profile"] == "mask_only"
    assert summary["changed"] is True
    assert sanitized_payload["metadata"]["privacy_preprocess_input"]["report_count"] == 1
    assert "[PRIVATE_PERSON]" in sanitized_payload["conversations"][0]["content"]


def test_resolve_privacy_preprocessor_for_input_jsonl():
    repo_root = Path(__file__).resolve().parents[1]
    preprocessor = resolve_privacy_preprocessor(
        config_dir=repo_root / "SynthChat" / "config",
        settings={
            "privacy_preprocess": {
                "enabled": True,
                "profile": "mask_only",
                "apply_to": {
                    "docs": False,
                    "input_jsonl": True,
                    "generated_output": False,
                },
            }
        },
        apply_target="input_jsonl",
    )

    assert preprocessor is not None
    assert preprocessor.profile_name == "mask_only"


def test_generator_sanitizes_generated_output_when_enabled():
    repo_root = Path(__file__).resolve().parents[1]
    config_dir = repo_root / "SynthChat" / "config"
    scenarios_dir = repo_root / "SynthChat" / "scenarios"
    rubrics_dir = repo_root / "SynthChat" / "rubrics"

    fake_preprocessor = _FakeOutputPreprocessor()

    def _factory(*, profile_name, profiles_registry):
        return fake_preprocessor

    client = _FakeLLMClient(
        responses=[
            "Ask Jane Roe to confirm the account update.",
            "Jane Roe has already confirmed the update.",
        ]
    )
    generator = SynthChatGenerator(
        config_dir=config_dir,
        scenarios_dir=scenarios_dir,
        rubrics_dir=rubrics_dir,
        llm_client=client,
        engine=None,
        enable_stage_validation=False,
        privacy_settings={"enabled": False, "apply_to": {"generated_output": True}},
        privacy_preprocessor_factory=_factory,
    )

    scenario = {
        "type": "behavioral",
        "system": False,
        "seed_data": {"preprocess_profile": "mask_only"},
        "prompts": {
            "user": "Create a user request about an account update.",
            "assistant": "Reply briefly to the user.",
        },
    }

    result = generator.generate_single(
        scenario_key="privacy_output_test",
        scenario=scenario,
        max_iterations=1,
        randomize_params=False,
    )

    rendered = json.dumps(result.example["conversations"])
    assert "Jane Roe" not in rendered
    assert "[PRIVATE_PERSON]" in rendered
    assert result.example["metadata"]["generated_output_privacy"]["changed"] is True
    assert "privacy_label:private_person" in result.example["metadata"]["labels"]["flat"]
