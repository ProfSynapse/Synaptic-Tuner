"""Checkout and packaged catalogs have identical bytes and fail closed."""
import json
from pathlib import Path

import pytest

from shared import tool_catalog_resources as resources
from shared.environments import tool_executor
from shared.validation import dataset_validator
from SynthChat.scripts import embed_tool_schemas

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def installed_resources(tmp_path, monkeypatch):
    package = tmp_path / 'package'
    for relative in ('cli-first-tool-schemas.json', 'Tools/tool_schemas.json'):
        target = package / 'runtime' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / relative).read_bytes())
    monkeypatch.setattr(resources, '_SOURCE_ROOT', tmp_path / 'absent-checkout')
    monkeypatch.setattr(resources, 'files', lambda name: package)
    clear = tool_executor.cli_command_catalog.cache_clear
    clear()
    yield package
    clear()


def test_installed_and_checkout_catalogs_are_identical(installed_resources):
    assert dataset_validator.load_tool_schemas() == json.loads((ROOT / 'Tools/tool_schemas.json').read_text())
    assert embed_tool_schemas.load_tool_schemas() == json.loads((ROOT / 'cli-first-tool-schemas.json').read_text())
    assert len(tool_executor.cli_command_catalog()) > 0
    for kind, relative in resources._CATALOGS.items():
        with resources.tool_catalog_path(kind) as selected:
            assert selected.read_bytes() == (ROOT / relative).read_bytes()


@pytest.mark.parametrize('fault', ['missing', 'invalid_json', 'empty'])
@pytest.mark.parametrize('kind', ['cli', 'tools'])
def test_selected_installed_default_errors(installed_resources, kind, fault):
    path = installed_resources / 'runtime' / resources._CATALOGS[kind]
    if fault == 'missing':
        path.unlink()
    else:
        path.write_text('not json' if fault == 'invalid_json' else '{}')
    loader = tool_executor.cli_command_catalog if kind == 'cli' else dataset_validator.load_tool_schemas
    with pytest.raises((FileNotFoundError, ValueError)):
        loader()


def test_explicit_path_is_authoritative_even_when_bundle_exists(installed_resources, tmp_path):
    path = tmp_path / 'caller.json'
    with pytest.raises(FileNotFoundError): dataset_validator.load_tool_schemas(path)
    with pytest.raises(FileNotFoundError): embed_tool_schemas.load_tool_schemas(path)
    path.write_text('bad')
    with pytest.raises(ValueError): dataset_validator.load_tool_schemas(path)
    path.write_text('{"custom": {"parameters": []}}')
    assert dataset_validator.load_tool_schemas(path) == {'custom': {'parameters': []}}
    path.write_text('{}')
    assert dataset_validator.load_tool_schemas(path) == {}  # Explicit no-tool configuration.


def test_missing_checkout_default_does_not_select_bundle(installed_resources, tmp_path, monkeypatch):
    checkout = tmp_path / 'checkout'
    checkout.mkdir()
    (checkout / 'pyproject.toml').write_text('')
    monkeypatch.setattr(resources, '_SOURCE_ROOT', checkout)
    with pytest.raises(FileNotFoundError): dataset_validator.load_tool_schemas()


def test_ordinary_nonwrapper_does_not_require_cli_catalog(installed_resources, monkeypatch):
    def forbidden(): raise AssertionError('ordinary tool selected CLI catalog')
    monkeypatch.setattr(tool_executor, 'cli_command_catalog', forbidden)
    assert tool_executor.expand_cli_wrapper_commands('unrelated_tool', {'unrelated': 'value'}) is None
