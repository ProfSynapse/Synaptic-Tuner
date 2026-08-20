from __future__ import annotations

import sys
import subprocess
from types import ModuleType, SimpleNamespace

import pytest

from tuner.cloud.hf_provider_adapter import (
    HFProviderAdapter,
    PINNED_HF_HUB_VERSION,
    load_hf_jp_provider,
)
from tuner.core.exceptions import CloudProviderError


class FakeClient:
    def __init__(self, *, private: bool = True, created_id: str = "owner/bucket") -> None:
        self.private = private
        self.created_id = created_id
        self.calls: list[str] = []

    def create_bucket(self, bucket_id, *, private=None, resource_group_id=None, region=None, exist_ok=False, token=None):
        self.calls.append("create")
        return SimpleNamespace(bucket_id=self.created_id)

    def bucket_info(self, bucket_id, *, token=None):
        self.calls.append("info")
        return SimpleNamespace(id=bucket_id, private=self.private)

    def list_bucket_tree(self, bucket_id, prefix=None, *, recursive=None, token=None):
        self.calls.append("list")
        return []

    def batch_bucket_files(self, bucket_id, *, add=None, copy=None, delete=None, token=None):
        self.calls.append("upload")

    def download_bucket_files(self, bucket_id, files, *, raise_on_missing_files=False, token=None):
        self.calls.append("download")


def test_probe_precedes_effects_and_private_identity_is_exact() -> None:
    client = FakeClient()
    adapter = HFProviderAdapter(client, token="not-a-real-token", client_version=PINNED_HF_HUB_VERSION)
    with pytest.raises(CloudProviderError, match="signatures"):
        adapter.ensure_private_bucket("owner/bucket")
    assert client.calls == []
    adapter.probe_signatures()
    assert adapter.ensure_private_bucket("owner/bucket") == "owner/bucket"
    assert client.calls == ["create", "info"]
    assert "not-a-real-token" not in repr(adapter)


@pytest.mark.parametrize(
    ("client", "message"),
    [
        (FakeClient(private=False), "not proven private"),
        (FakeClient(created_id="other/bucket"), "canonical identity"),
    ],
)
def test_bucket_privacy_and_canonical_identity_fail_closed(client, message) -> None:
    adapter = HFProviderAdapter(client, token="value", client_version=PINNED_HF_HUB_VERSION)
    adapter.probe_signatures()
    with pytest.raises(CloudProviderError, match=message):
        adapter.ensure_private_bucket("owner/bucket")
    assert "upload" not in client.calls


def test_incompatible_signature_fails_before_create() -> None:
    client = FakeClient()
    client.batch_bucket_files = lambda bucket_id, **kwargs: None
    adapter = HFProviderAdapter(client, token="value", client_version=PINNED_HF_HUB_VERSION)
    with pytest.raises(CloudProviderError, match="batch_bucket_files"):
        adapter.probe_signatures()
    assert client.calls == []


def test_unexpected_required_parameter_fails_signature_probe_before_create() -> None:
    class EvilClient(FakeClient):
        def create_bucket(self, bucket_id, evil, *, private=None, resource_group_id=None, region=None, exist_ok=False, token=None):
            raise AssertionError("must not run")

    client = EvilClient()
    adapter = HFProviderAdapter(client, token="value", client_version=PINNED_HF_HUB_VERSION)
    with pytest.raises(CloudProviderError, match="unexpected parameters"):
        adapter.probe_signatures()
    assert client.calls == []


def test_changed_default_fails_signature_probe() -> None:
    class DriftedClient(FakeClient):
        def download_bucket_files(self, bucket_id, files, *, raise_on_missing_files=True, token=None):
            raise AssertionError("must not run")

    adapter = HFProviderAdapter(
        DriftedClient(), token="value", client_version=PINNED_HF_HUB_VERSION
    )
    with pytest.raises(CloudProviderError, match="raise_on_missing_files"):
        adapter.probe_signatures()


def test_provider_errors_are_bounded_and_do_not_echo_tokens() -> None:
    class Exploding(FakeClient):
        def create_bucket(self, bucket_id, *, private=None, resource_group_id=None, region=None, exist_ok=False, token=None):
            raise RuntimeError(f"Bearer {token}")

    adapter = HFProviderAdapter(Exploding(), token="sensitive-value", client_version=PINNED_HF_HUB_VERSION)
    adapter.probe_signatures()
    with pytest.raises(CloudProviderError) as caught:
        adapter.ensure_private_bucket("owner/bucket")
    assert "sensitive-value" not in str(caught.value)


def test_factory_imports_only_when_called_and_requires_exact_version(monkeypatch) -> None:
    module = ModuleType("huggingface_hub")
    module.__version__ = PINNED_HF_HUB_VERSION
    module.HfApi = lambda token=False: FakeClient()
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)
    adapter = load_hf_jp_provider(token="value")
    assert adapter.client_version == PINNED_HF_HUB_VERSION

    module.__version__ = "1.26.0"
    with pytest.raises(CloudProviderError, match="requires huggingface_hub"):
        load_hf_jp_provider(token="value")


def test_importing_jp_modules_does_not_import_huggingface_hub() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys;"
                "import tuner.cloud.hf_provider_adapter;"
                "import tuner.cloud.hf_provisioning_operator;"
                "import tuner.handlers.hf_source_handler;"
                "assert 'huggingface_hub' not in sys.modules"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
