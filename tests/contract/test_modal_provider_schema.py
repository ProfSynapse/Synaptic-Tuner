from __future__ import annotations

import json
from pathlib import Path

import yaml
from jsonschema.validators import validator_for


ROOT=Path(__file__).parents[2]


def test_modal_provider_example_is_config_first_and_contains_no_secret_values():
    schema=json.loads((ROOT/"schemas"/"synaptic-modal-provider-v1.schema.json").read_text(encoding="utf-8"))
    document=yaml.safe_load((ROOT/"examples"/"host-project"/"providers"/"modal-a10-v1.yaml").read_text(encoding="utf-8"))
    validator=validator_for(schema);validator.check_schema(schema);validator(schema).validate(document)
    rendered=json.dumps(document,sort_keys=True).lower()
    assert "token_value" not in rendered and "api_key" not in rendered and "password" not in rendered
    assert document["deployment"]["app_name"]=="synaptic-training-v1"
    assert document["deployment"]["function_name"]=="run_sft_v1"
