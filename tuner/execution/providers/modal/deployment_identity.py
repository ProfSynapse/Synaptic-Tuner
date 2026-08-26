"""Provider-visible identity for one immutable Modal deployment."""
from __future__ import annotations

import re

from ...contracts import safe_ref


DEPLOYMENT_REF_PATTERN = re.compile(r"^modal-deployment-[0-9a-f]{32}$")
FUNCTION_PREFIX = "run_sft_v1_"


def modal_function_name(deployment_ref: str) -> str:
    """Derive the exact provider entrypoint from a host-owned deployment ref."""
    deployment_ref = safe_ref(deployment_ref, "deployment_ref")
    if DEPLOYMENT_REF_PATTERN.fullmatch(deployment_ref) is None:
        raise ValueError("Modal deployment_ref is invalid")
    return FUNCTION_PREFIX + deployment_ref.removeprefix("modal-deployment-")


def validate_modal_function_identity(
    deployment_ref: str, function_name: str
) -> tuple[str, str]:
    deployment_ref = safe_ref(deployment_ref, "deployment_ref")
    function_name = safe_ref(function_name, "function_name")
    if function_name != modal_function_name(deployment_ref):
        raise ValueError("Modal function name does not bind deployment_ref")
    return deployment_ref, function_name


__all__ = [
    "DEPLOYMENT_REF_PATTERN", "FUNCTION_PREFIX", "modal_function_name",
    "validate_modal_function_identity",
]
