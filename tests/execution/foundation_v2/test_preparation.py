import json

import pytest

from tuner.execution.foundation_v2.canonical import canonical_bytes
from tuner.execution.foundation_v2.preparation import CanonicalPreparationV2
from tests.execution.foundation_v2.helpers import D, prep


def test_execution_binding_digest_is_mandatory_and_changes_preparation_identity():
    original = prep()
    changed = prep(execution_binding_digest=D[9])
    assert original.execution_binding_digest == D[8]
    assert changed.preparation_digest != original.preparation_digest


@pytest.mark.parametrize("mutation", ("missing", "extra", "malformed"))
def test_execution_binding_digest_schema_is_exact(mutation):
    document = prep().to_dict()
    if mutation == "missing":
        del document["execution_binding_digest"]
    elif mutation == "extra":
        document["unexpected"] = D[9]
    else:
        document["execution_binding_digest"] = "not-a-digest"
    with pytest.raises((TypeError, ValueError)):
        CanonicalPreparationV2.parse(canonical_bytes(document))


def test_noncanonical_preparation_bytes_are_rejected():
    raw = json.dumps(prep().to_dict(), indent=2).encode("utf-8")
    with pytest.raises(ValueError):
        CanonicalPreparationV2.parse(raw)
