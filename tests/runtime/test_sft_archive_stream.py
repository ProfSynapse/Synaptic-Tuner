from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import pytest

from tuner.runtime import verification


def _safetensors(payload: bytes = b"\x01\x00\x00\x00") -> bytes:
    header = json.dumps(
        {"weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}},
        separators=(",", ":"),
    ).encode()
    header += b" " * ((8 - len(header) % 8) % 8)
    return len(header).to_bytes(8, "little") + header + payload


def _archive(members: tuple[tuple[str, bytes], ...]) -> bytes:
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w") as archive:
        for name, content in members:
            member = tarfile.TarInfo(name)
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))
    return output.getvalue()


def _model_archive() -> bytes:
    return _archive((
        (
            "adapter_config.json",
            b'{"base_model_name_or_path":"example/model","peft_type":"LORA"}',
        ),
        ("adapter_model.safetensors", _safetensors()),
    ))


def test_archive_stream_has_identical_bytesio_and_file_semantics(tmp_path: Path) -> None:
    content = _model_archive()
    expected = verification._validate_sft_archive_stream(
        io.BytesIO(content), "model", locked_model_ref="example/model"
    )
    path = tmp_path / "model.tar"
    path.write_bytes(content)
    with path.open("rb") as stream:
        observed = verification._validate_sft_archive_stream(
            stream, "model", locked_model_ref="example/model"
        )
    assert expected == observed == (
        frozenset({"adapter_config.json", "adapter_model.safetensors"}),
        True,
    )


class _ObservedStream(io.BytesIO):
    def __init__(self, content: bytes) -> None:
        super().__init__(content)
        self.requests: list[int] = []

    def read(self, size: int = -1) -> bytes:
        self.requests.append(size)
        return super().read(size)


def test_archive_stream_reads_weight_payload_in_bounded_chunks() -> None:
    payload = b"\x01\x00\x00\x00" * 300_000
    header = json.dumps(
        {
            "weight": {
                "dtype": "F32",
                "shape": [300_000],
                "data_offsets": [0, len(payload)],
            }
        },
        separators=(",", ":"),
    ).encode()
    header += b" " * ((8 - len(header) % 8) % 8)
    content = _archive((
        (
            "adapter_config.json",
            b'{"base_model_name_or_path":"example/model","peft_type":"LORA"}',
        ),
        ("adapter_model.safetensors", len(header).to_bytes(8, "little") + header + payload),
    ))
    stream = _ObservedStream(content)
    assert verification._validate_sft_archive_stream(
        stream, "model", locked_model_ref="example/model"
    )[1]
    assert -1 not in stream.requests
    assert max(stream.requests) <= 1024 * 1024


@pytest.mark.parametrize(
    "content",
    (
        b"",
        b"not a tar archive",
        _model_archive()[:1550],
    ),
)
def test_archive_stream_rejects_empty_malformed_or_truncated(content: bytes) -> None:
    stream = io.BytesIO(content)
    try:
        result = verification._validate_sft_archive_stream(
            stream, "model", locked_model_ref="example/model"
        )
    except tarfile.TarError:
        pass
    else:
        assert result == (frozenset(), False)
    assert not stream.closed


def test_archive_stream_rejects_member_over_bound(monkeypatch: pytest.MonkeyPatch) -> None:
    content = _archive((("tokenizer.json", b"x" * 33),))
    monkeypatch.setattr(verification, "MAX_ARCHIVE_MEMBER_BYTES", 32)
    assert verification._validate_sft_archive_stream(
        io.BytesIO(content), "tokenizer"
    ) == (frozenset(), False)


@pytest.mark.parametrize(
    "members",
    (
        (("../adapter_config.json", b"{}"), ("adapter_model.safetensors", _safetensors())),
        (("adapter_config.json", b"{}"), ("adapter_model.safetensors", b"payload")),
        (("adapter_config.json", b"{}"), ("weights.bin", b"payload")),
    ),
)
def test_archive_stream_preserves_existing_semantic_denials(
    members: tuple[tuple[str, bytes], ...]
) -> None:
    assert verification._validate_sft_archive_stream(
        io.BytesIO(_archive(members)), "model", locked_model_ref="example/model"
    ) == (frozenset(), False)


def test_archive_stream_requires_seekable_stream_at_origin() -> None:
    stream = io.BytesIO(_model_archive())
    stream.read(1)
    with pytest.raises(ValueError, match="positioned at zero"):
        verification._validate_sft_archive_stream(
            stream, "model", locked_model_ref="example/model"
        )


class _NonseekableStream(io.BytesIO):
    def seekable(self) -> bool:
        return False


def test_archive_stream_rejects_nonseekable_stream_without_closing_it() -> None:
    stream = _NonseekableStream(_model_archive())
    with pytest.raises(ValueError, match="seekable"):
        verification._validate_sft_archive_stream(
            stream, "model", locked_model_ref="example/model"
        )
    assert not stream.closed


class _FailingReadStream(io.BytesIO):
    def read(self, size: int = -1) -> bytes:
        raise OSError("read failed")


class _FailingSeekStream(io.BytesIO):
    def seek(self, offset: int, whence: int = 0) -> int:
        raise OSError("seek failed")


@pytest.mark.parametrize("stream_type", (_FailingReadStream, _FailingSeekStream))
def test_archive_stream_propagates_io_failure_without_closing_caller_stream(
    stream_type: type[io.BytesIO],
) -> None:
    stream = stream_type(_model_archive())
    with pytest.raises(OSError):
        verification._validate_sft_archive_stream(
            stream, "model", locked_model_ref="example/model"
        )
    assert not stream.closed


def test_archive_stream_does_not_close_successful_caller_stream() -> None:
    stream = io.BytesIO(_model_archive())
    assert verification._validate_sft_archive_stream(
        stream, "model", locked_model_ref="example/model"
    )[1]
    assert not stream.closed
