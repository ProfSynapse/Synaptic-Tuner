"""Owned exact-file download child for the protected HF training smoke."""

from __future__ import annotations

import argparse
import inspect
import json
import os
import re
import sys
from pathlib import Path, PurePosixPath


HF_ENDPOINT = "https://huggingface.co"
HF_HUB_VERSION = "1.27.0"
SCHEMA = "synaptic-hf-training-download-child/v1"
_SUCCESS = b'{"schema_version":"synaptic-hf-training-download-child/v1","status":"PASS"}\n'
_FAILURE = b'{"schema_version":"synaptic-hf-training-download-child/v1","status":"FAILED"}\n'
_BUCKET = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")
_HEX = re.compile(r"^[0-9a-f]+$")
_PREFIX = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*(?:/[A-Za-z0-9][A-Za-z0-9._-]*)*$")
_EXPECTED_PATHS = tuple(sorted((
    "source-lock.json", "exclusive-sentinel.json",
    "checkpoint-1/adapter_model.safetensors", "checkpoint-1/adapter_config.json",
    "checkpoint-1/trainer_state.json", "checkpoint-1/optimizer.pt",
    "checkpoint-1/scheduler.pt", "final_model/adapter_model.safetensors",
    "final_model/adapter_config.json", "final_model/tokenizer_config.json",
    "training_lineage.json", "step-evidence.json", "result.json", "manifest.json",
    "inventory.json",
)))


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise RuntimeError("invalid arguments")


def _parser() -> argparse.ArgumentParser:
    parser = _Parser(add_help=False)
    parser.add_argument("--bucket-id", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--destination", required=True)
    return parser


def _pairs(values: list[object]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in values:
        if key in result:
            raise RuntimeError("invalid inventory")
        result[key] = value
    return result


def _constant(_value: str) -> object:
    raise RuntimeError("invalid inventory")


def _inventory(path: Path, destination: Path, prefix: str) -> list[tuple[object, Path]]:
    raw = path.read_bytes()
    if not raw or len(raw) > 64 * 1024:
        raise RuntimeError("invalid inventory")
    value = json.loads(raw.decode("ascii"), object_pairs_hook=_pairs, parse_constant=_constant)
    if not isinstance(value, list) or len(value) != 15:
        raise RuntimeError("invalid inventory")
    canonical = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("ascii")
    if raw != canonical:
        raise RuntimeError("invalid inventory")
    from huggingface_hub.hf_api import BucketFile
    fields = getattr(BucketFile, "__dataclass_fields__", None)
    if not isinstance(fields, dict) or set(fields) != {
        "type", "path", "size", "xet_hash", "mtime", "uploaded_at",
    }:
        raise RuntimeError("invalid provider")

    pairs: list[tuple[object, Path]] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, dict) or set(item) != {"path", "bytes", "provider_xet_hash"}:
            raise RuntimeError("invalid inventory")
        relative = item["path"]
        size = item["bytes"]
        xet_hash = item["provider_xet_hash"]
        if (
            not isinstance(relative, str)
            or relative in seen
            or PurePosixPath(relative).as_posix() != relative
            or PurePosixPath(relative).is_absolute()
            or any(part in {"", ".", ".."} for part in PurePosixPath(relative).parts)
            or type(size) is not int
            or size < 0
            or (xet_hash is not None and (not isinstance(xet_hash, str) or _HEX.fullmatch(xet_hash) is None))
        ):
            raise RuntimeError("invalid inventory")
        seen.add(relative)
        remote = BucketFile(
            type="file", path=f"{prefix}/{relative}", size=size, xetHash=xet_hash,
        )
        local = destination.joinpath(*PurePosixPath(relative).parts)
        local.parent.mkdir(parents=True, exist_ok=True)
        pairs.append((remote, local))
    if tuple(item["path"] for item in value) != _EXPECTED_PATHS:
        raise RuntimeError("invalid inventory")
    return pairs


def run(argv: list[str] | None = None) -> int:
    saved_out = saved_err = null_fd = None
    outcome = _FAILURE
    code = 125
    report = False
    token = bytearray()
    clients: list[object] = []
    try:
        saved_out = os.dup(1)
        saved_err = os.dup(2)
        null_fd = os.open(os.devnull, os.O_WRONLY)
        os.dup2(null_fd, 1)
        os.dup2(null_fd, 2)
        args = _parser().parse_args(argv)
        if _BUCKET.fullmatch(args.bucket_id) is None or _PREFIX.fullmatch(args.prefix) is None:
            raise RuntimeError("invalid bucket")
        destination = Path(args.destination).resolve(strict=True)
        if not destination.is_dir() or any(destination.iterdir()):
            raise RuntimeError("invalid destination")
        token = bytearray(sys.stdin.buffer.read(4097))
        if not token or len(token) > 4096 or b"\n" in token or b"\r" in token:
            raise RuntimeError("invalid token")
        import httpx
        import huggingface_hub

        if huggingface_hub.__version__ != HF_HUB_VERSION:
            raise RuntimeError("invalid provider")
        def factory() -> httpx.Client:
            client = httpx.Client(
                base_url=HF_ENDPOINT,
                timeout=httpx.Timeout(connect=10.0, read=30.0, write=30.0, pool=10.0),
                follow_redirects=True, max_redirects=5, trust_env=False,
            )
            clients.append(client)
            return client

        huggingface_hub.set_client_factory(factory)
        api = huggingface_hub.HfApi(endpoint=HF_ENDPOINT, token=False)
        try:
            parameters = inspect.signature(api.download_bucket_files).parameters
        except (TypeError, ValueError) as exc:
            raise RuntimeError("invalid provider") from exc
        if not {"bucket_id", "files", "raise_on_missing_files", "token"}.issubset(parameters):
            raise RuntimeError("invalid provider")
        if any(parameters[name].kind is inspect.Parameter.VAR_KEYWORD for name in (
            "bucket_id", "files", "raise_on_missing_files", "token",
        )):
            raise RuntimeError("invalid provider")
        pairs = _inventory(Path(args.inventory), destination, args.prefix)
        decoded = token.decode("ascii")
        api.download_bucket_files(
            args.bucket_id, pairs, raise_on_missing_files=True, token=decoded,
        )
        decoded = ""
        outcome = _SUCCESS
        code = 0
        report = True
    except Exception:
        outcome = _FAILURE
        code = 125
        report = True
    finally:
        for index in range(len(token)):
            token[index] = 0
        for client in clients:
            try:
                client.close()
            except Exception:
                outcome = _FAILURE
                code = 125
        if saved_out is not None:
            try:
                os.dup2(saved_out, 1)
            except OSError:
                pass
        if saved_err is not None:
            try:
                os.dup2(saved_err, 2)
            except OSError:
                pass
        for descriptor in (null_fd, saved_out, saved_err):
            if descriptor is not None:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
        if report:
            try:
                os.write(1, outcome)
            except OSError:
                pass
    return code


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
