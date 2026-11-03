"""Closed error vocabulary for the weight-patch capability."""

from __future__ import annotations

ERROR_CODES = frozenset(
    {
        "schema",  # malformed or unsupported spec/inspect document
        "path",  # missing/unsafe/escaping path, symlink, non-regular file
        "index",  # malformed safetensors header or checkpoint index
        "binding",  # declared source/revision/hash does not match local bytes
        "base_hash_mismatch",  # targeted base tensor bytes differ from the pin
        "payload",  # payload hash/shape/dtype/index problem
        "nonfinite",  # NaN/Inf in a payload, base, or result
        "budget",  # patch exceeds a declared finite budget
        "output_exists",  # export destination already exists
        "apply",  # in-memory application failed (rolled back)
    }
)


class WeightPatchError(Exception):
    """A refusal. ``code`` is one of ``ERROR_CODES``; nothing was applied."""

    def __init__(self, code: str, message: str):
        if code not in ERROR_CODES:  # pragma: no cover - programming error
            raise ValueError(f"unknown weight-patch error code {code!r}")
        super().__init__(message)
        self.code = code
        self.message = message

    def __str__(self) -> str:
        return f"[{self.code}] {self.message}"
