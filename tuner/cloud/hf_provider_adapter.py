"""Narrow Hugging Face Buckets adapter for the isolated JP launcher.

This module intentionally has no import-time dependency on ``huggingface_hub``.
The pinned client is imported only by :func:`load_hf_jp_provider` after the
operator has explicitly selected the isolated launcher environment.
"""

from __future__ import annotations

import importlib
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from tuner.core.exceptions import CloudProviderError


PINNED_HF_HUB_VERSION = "1.27.0"


@dataclass(frozen=True)
class HFRemoteMember:
    path: str
    entry_type: str
    provider_object: object


class HFProviderAdapter:
    """Effect-aware wrapper around the exact v1.27.0 Buckets surface."""

    def __init__(self, client: Any, *, token: str, client_version: str) -> None:
        if not isinstance(token, str) or not token.strip():
            raise CloudProviderError("HF JP requires a non-empty explicitly selected token.")
        if client_version != PINNED_HF_HUB_VERSION:
            raise CloudProviderError(
                f"HF JP requires huggingface_hub=={PINNED_HF_HUB_VERSION}."
            )
        self._client = client
        self._token = token.strip()
        self.client_version = client_version
        self._probed = False

    def __repr__(self) -> str:
        return f"HFProviderAdapter(client_version={self.client_version!r}, token=<redacted>)"

    def probe_signatures(self) -> None:
        """Prove every read and mutation signature before the first mutation."""

        empty = inspect.Parameter.empty
        requirements = {
            "create_bucket": (
                ("bucket_id", inspect.Parameter.POSITIONAL_OR_KEYWORD, empty),
                ("private", inspect.Parameter.KEYWORD_ONLY, None),
                ("resource_group_id", inspect.Parameter.KEYWORD_ONLY, None),
                ("region", inspect.Parameter.KEYWORD_ONLY, None),
                ("exist_ok", inspect.Parameter.KEYWORD_ONLY, False),
                ("token", inspect.Parameter.KEYWORD_ONLY, None),
            ),
            "bucket_info": (
                ("bucket_id", inspect.Parameter.POSITIONAL_OR_KEYWORD, empty),
                ("token", inspect.Parameter.KEYWORD_ONLY, None),
            ),
            "list_bucket_tree": (
                ("bucket_id", inspect.Parameter.POSITIONAL_OR_KEYWORD, empty),
                ("prefix", inspect.Parameter.POSITIONAL_OR_KEYWORD, None),
                ("recursive", inspect.Parameter.KEYWORD_ONLY, None),
                ("token", inspect.Parameter.KEYWORD_ONLY, None),
            ),
            "batch_bucket_files": (
                ("bucket_id", inspect.Parameter.POSITIONAL_OR_KEYWORD, empty),
                ("add", inspect.Parameter.KEYWORD_ONLY, None),
                ("copy", inspect.Parameter.KEYWORD_ONLY, None),
                ("delete", inspect.Parameter.KEYWORD_ONLY, None),
                ("token", inspect.Parameter.KEYWORD_ONLY, None),
            ),
            "download_bucket_files": (
                ("bucket_id", inspect.Parameter.POSITIONAL_OR_KEYWORD, empty),
                ("files", inspect.Parameter.POSITIONAL_OR_KEYWORD, empty),
                ("raise_on_missing_files", inspect.Parameter.KEYWORD_ONLY, False),
                ("token", inspect.Parameter.KEYWORD_ONLY, None),
            ),
        }
        for name, expected in requirements.items():
            operation = getattr(self._client, name, None)
            if not callable(operation):
                raise CloudProviderError(f"HF JP client is missing required {name} API.")
            try:
                parameters = tuple(inspect.signature(operation).parameters.values())
            except (TypeError, ValueError) as exc:
                raise CloudProviderError(
                    f"HF JP could not prove the required {name} signature."
                ) from exc
            if tuple(parameter.name for parameter in parameters) != tuple(
                item[0] for item in expected
            ):
                raise CloudProviderError(
                    f"HF JP client {name} signature has unexpected parameters."
                )
            for parameter, (parameter_name, required_kind, required_default) in zip(
                parameters, expected
            ):
                if (
                    parameter.name != parameter_name
                    or parameter.kind != required_kind
                    or parameter.default != required_default
                ):
                    raise CloudProviderError(
                        f"HF JP client {name} signature is incompatible at {parameter_name}."
                    )
        self._probed = True

    def ensure_private_bucket(self, bucket_id: str) -> str:
        self._require_probed()
        requested = _canonical_bucket_id(bucket_id)
        try:
            created = self._client.create_bucket(
                requested,
                private=True,
                resource_group_id=None,
                region=None,
                exist_ok=True,
                token=self._token,
            )
            created_id = _canonical_bucket_id(getattr(created, "bucket_id", ""))
            info = self._client.bucket_info(requested, token=self._token)
            info_id = _canonical_bucket_id(getattr(info, "id", ""))
            private = getattr(info, "private", None)
        except Exception:
            raise CloudProviderError("HF JP could not resolve the required private bucket.") from None
        if created_id != requested or info_id != requested:
            raise CloudProviderError("HF JP bucket canonical identity does not match the descriptor.")
        if private is not True:
            raise CloudProviderError("HF JP descriptor bucket is not proven private.")
        return requested

    def list_members(self, bucket_id: str, *, prefix: str) -> tuple[HFRemoteMember, ...]:
        self._require_probed()
        try:
            entries = self._client.list_bucket_tree(
                bucket_id,
                prefix=prefix,
                recursive=True,
                token=self._token,
            )
            result: list[HFRemoteMember] = []
            for entry in entries:
                entry_type = getattr(entry, "type", None)
                if entry_type not in {"file", "directory"}:
                    raise ValueError("invalid tree entry type")
                path = getattr(entry, "path", None)
                if not isinstance(path, str) or not path:
                    raise ValueError("invalid file entry")
                result.append(
                    HFRemoteMember(
                        path=path,
                        entry_type=entry_type,
                        provider_object=entry,
                    )
                )
            return tuple(sorted(result, key=lambda item: (item.path, item.entry_type)))
        except Exception:
            raise CloudProviderError("HF JP could not inspect the immutable bucket prefix.") from None

    def upload_once(
        self,
        bucket_id: str,
        *,
        additions: Sequence[tuple[str | Path | bytes, str]],
    ) -> None:
        self._require_probed()
        self._client.batch_bucket_files(
            bucket_id,
            add=list(additions),
            copy=None,
            delete=None,
            token=self._token,
        )

    def download_members(
        self,
        bucket_id: str,
        *,
        files: Sequence[tuple[object, Path]],
    ) -> None:
        self._require_probed()
        try:
            self._client.download_bucket_files(
                bucket_id,
                list(files),
                raise_on_missing_files=True,
                token=self._token,
            )
        except Exception:
            raise CloudProviderError("HF JP could not read back the immutable bucket prefix.") from None

    def _require_probed(self) -> None:
        if not self._probed:
            raise CloudProviderError("HF JP provider signatures must be proven before use.")


def load_hf_jp_provider(*, token: str) -> HFProviderAdapter:
    """Import and construct the pinned provider client inside the JP runtime."""

    try:
        hub = importlib.import_module("huggingface_hub")
        version = str(getattr(hub, "__version__", ""))
        client_type = getattr(hub, "HfApi")
        client = client_type(token=False)
    except Exception:
        raise CloudProviderError(
            "The isolated HF JP launcher is unavailable or incompatible."
        ) from None
    return HFProviderAdapter(client, token=token, client_version=version)


def _canonical_bucket_id(value: object) -> str:
    if not isinstance(value, str):
        raise CloudProviderError("HF JP bucket identity is invalid.")
    normalized = value.strip().strip("/")
    parts = normalized.split("/")
    if len(parts) != 2 or any(not part or part in {".", ".."} for part in parts):
        raise CloudProviderError("HF JP bucket identity must be namespace/name.")
    if any(any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for character in part) for part in parts):
        raise CloudProviderError("HF JP bucket identity contains unsupported characters.")
    return normalized


__all__ = [
    "HFProviderAdapter",
    "HFRemoteMember",
    "PINNED_HF_HUB_VERSION",
    "load_hf_jp_provider",
]
