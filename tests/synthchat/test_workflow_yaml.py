"""Admission and consumption share exact bounded YAML meaning."""
from pathlib import Path

import pytest
import yaml

from SynthChat.utils.yaml_loader import parse_workflow_yaml


@pytest.mark.parametrize('raw', [
    b'a: 1\na: 2', b'1: value', b'a: .nan', b'a: .inf',
    b'a: 2026-10-13', b'a: &cycle [*cycle]', b'a: [', b'- item',
    b'a: !!python/object:builtins.object {}', b'', b'root: ' + b'[' * 33 + b'0' + b']' * 33,
    b'root: [' + b'0,' * 20001 + b'0]', b'root: ' + b'x' * (4 * 1024 * 1024),
])
def test_invalid_or_unbounded_documents_reject(raw):
    with pytest.raises(ValueError): parse_workflow_yaml(raw)


def test_fractional_unicode_and_fresh_copies():
    raw = 'values: [0.125, true, null, café]\n'.encode()
    first = parse_workflow_yaml(raw)
    assert first == yaml.safe_load(raw)
    first['values'].append('changed')
    assert parse_workflow_yaml(raw) == {'values': [.125, True, None, 'café']}


@pytest.mark.parametrize('name', [
    'validation', 'tool_call_formats', 'workspace_formats', 'label_mappings', 'privacy_profiles',
])
def test_checked_in_documents_match_existing_yaml_semantics(name):
    raw = (Path(__file__).resolve().parents[2] / 'SynthChat/config' / (name + '.yaml')).read_bytes()
    assert parse_workflow_yaml(raw) == yaml.safe_load(raw)


def test_nonconflicting_yaml_merge_matches_legacy_parser():
    raw = b'base: &base {first: 1}\nmerged: {<<: *base, second: 0.125}\n'
    assert parse_workflow_yaml(raw) == yaml.safe_load(raw)


def test_conflicting_merge_cannot_hide_duplicate_keys():
    raw = b'base: &base {first: 1}\nmerged: {<<: *base, first: 2}\n'
    with pytest.raises(ValueError, match='unique'):
        parse_workflow_yaml(raw)
