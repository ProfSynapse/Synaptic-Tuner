"""Finite dataset reads through the existing DataAPI, with unchanged cursors."""
import io
import json

import pytest

from synaptic_tuner.api.v1.data_facade import DatasetValidateRequest, DataOperationError, DataOperationCode
from synaptic_tuner.api.v1.reference import data
from tests.contract.test_reference_data_v1 import Harness, FakeLocalClient, PROJECT


def row(content="x"):
    return json.dumps({"conversations": [{"role": "user", "content": content}]}).encode()


def api(tmp_path):
    return Harness(tmp_path, FakeLocalClient([])).api


@pytest.mark.parametrize("ending", [b"\n", b"\r\n", b"\r", b""])
def test_exact_line_and_total_limit_eof(tmp_path, monkeypatch, ending):
    raw = row() + ending
    monkeypatch.setattr(data, "_MAX_VALIDATION_LINE_BYTES", len(raw))
    monkeypatch.setattr(data, "_MAX_VALIDATION_READ_BYTES", len(raw))
    path = tmp_path / "data.jsonl"
    path.write_bytes(raw)
    result = api(tmp_path).validate(DatasetValidateRequest(PROJECT, str(path)))
    assert (result.rows_checked, result.rows_valid, result.truncated) == (1, 1, False)


@pytest.mark.parametrize("kind", ["line", "total", "prefix", "lookahead", "physical"])
def test_exceeded_limits_reject_before_unbounded_work(tmp_path, monkeypatch, kind):
    raw = row() + b"\n"
    cursor = None
    limit = 1
    if kind == "line":
        monkeypatch.setattr(data, "_MAX_VALIDATION_LINE_BYTES", len(raw) - 1)
    elif kind == "total":
        monkeypatch.setattr(data, "_MAX_VALIDATION_READ_BYTES", len(raw) - 1)
    elif kind == "prefix":
        raw = b" \n" * 20 + raw
        cursor = data._encode_cursor("20")
        monkeypatch.setattr(data, "_MAX_VALIDATION_READ_BYTES", 39)
    elif kind == "lookahead":
        raw += row("next") + b"\n"
        monkeypatch.setattr(data, "_MAX_VALIDATION_READ_BYTES", len(raw) - 1)
    else:
        raw = b"\n" * 4 + raw
        monkeypatch.setattr(data, "_MAX_VALIDATION_PHYSICAL_LINES", 3)
    path = tmp_path / "data.jsonl"
    path.write_bytes(raw)
    with pytest.raises(DataOperationError) as error:
        api(tmp_path).validate(DatasetValidateRequest(PROJECT, str(path), cursor, limit))
    assert error.value.code is DataOperationCode.MANIFEST_INVALID


def test_reads_clamp_to_remaining_byte_budget_before_parsing(monkeypatch):
    monkeypatch.setattr(data, "_MAX_VALIDATION_READ_BYTES", 5)
    monkeypatch.setattr(data, "_MAX_VALIDATION_LINE_BYTES", 4)
    class Observed(io.BytesIO):
        def read(self, count=-1):
            assert 0 < count <= min(65536, 6 - self.tell())
            return super().read(count)
    with pytest.raises(DataOperationError):
        list(data._validation_lines(Observed(b"a\nb\nc\n")))


@pytest.mark.parametrize("cursor_text", ["9" * 20, "1000001", "\u0661"])
def test_large_or_nonascii_cursor_rejects_before_file_read(tmp_path, monkeypatch, cursor_text):
    path = tmp_path / "data.jsonl"
    path.write_bytes(row())
    monkeypatch.setattr(data, "_validation_lines", lambda _: pytest.fail("cursor read input"))
    with pytest.raises(DataOperationError) as error:
        api(tmp_path).validate(DatasetValidateRequest(PROJECT, str(path), data._encode_cursor(cursor_text)))
    assert error.value.code is DataOperationCode.CURSOR_INVALID


def test_blank_lines_and_universal_newlines_continue_without_skipping_rows(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_bytes(b"\r\n" + row() + b"\r\n \r" + row("two") + b"\n\r" + row("three"))
    public = api(tmp_path)
    cursor = None
    counts = []
    for _ in range(3):
        page = public.validate(DatasetValidateRequest(PROJECT, str(path), cursor, 1))
        counts.append((page.rows_checked, page.rows_valid))
        cursor = page.next_cursor
    assert counts == [(1, 1)] * 3
    assert cursor is None


def test_malformed_utf8_preserves_replacement_decoding(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_bytes(row().replace(b'"x"', b'"\xff"') + b"\r")
    result = api(tmp_path).validate(DatasetValidateRequest(PROJECT, str(path)))
    assert (result.rows_checked, result.rows_valid) == (1, 1)


@pytest.mark.parametrize("raw", [b"", b"\r\n \n\r", b"\n" * 4])
def test_empty_and_blank_only_pages_finish(tmp_path, raw):
    path = tmp_path / "data.jsonl"
    path.write_bytes(raw)
    result = api(tmp_path).validate(DatasetValidateRequest(PROJECT, str(path)))
    assert (result.rows_checked, result.rows_valid, result.next_cursor, result.truncated) == (0, 0, None, False)


def test_crlf_split_between_bounded_reads_is_one_physical_line(tmp_path):
    # First 64 KiB read ends with CR; next read must consume LF as one ending.
    content = "x" * (65535 - len(row("")))
    raw = row(content) + b"\r\n" + row("next") + b"\n"
    assert raw[65535:65537] == b"\r\n"
    path = tmp_path / "data.jsonl"
    path.write_bytes(raw)
    public = api(tmp_path)
    first = public.validate(DatasetValidateRequest(PROJECT, str(path), limit=1))
    assert data._decode_cursor(first.next_cursor) == "1"
    second = public.validate(DatasetValidateRequest(PROJECT, str(path), first.next_cursor, 1))
    assert (second.rows_checked, second.rows_valid, second.truncated) == (1, 1, False)


def test_oversized_first_line_never_reaches_json_parser(tmp_path, monkeypatch):
    path = tmp_path / "data.jsonl"
    path.write_bytes(b"x" * 9)
    monkeypatch.setattr(data, "_MAX_VALIDATION_LINE_BYTES", 8)
    monkeypatch.setattr(data, "_validate_row", lambda _: pytest.fail("oversized row parsed"))
    with pytest.raises(DataOperationError) as error:
        api(tmp_path).validate(DatasetValidateRequest(PROJECT, str(path)))
    assert error.value.code is DataOperationCode.MANIFEST_INVALID
