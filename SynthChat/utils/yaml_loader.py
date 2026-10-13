"""YAML configuration loader."""

import yaml
from pathlib import Path
from typing import Dict, Any


def load_yaml(file_path: str) -> Dict[str, Any]:
    """
    Load YAML configuration file.

    Args:
        file_path: Path to YAML file

    Returns:
        Dictionary with configuration data

    Raises:
        FileNotFoundError: If file doesn't exist
        yaml.YAMLError: If YAML is invalid
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"YAML file not found: {file_path}")

    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_config(config_name: str) -> Dict[str, Any]:
    """
    Load configuration from config directory.

    Args:
        config_name: Name of config file (without .yaml extension)

    Returns:
        Dictionary with configuration data
    """
    config_dir = Path(__file__).parent.parent / "config"
    config_path = config_dir / f"{config_name}.yaml"
    return load_yaml(str(config_path))


class _WorkflowLoader(yaml.SafeLoader):
    pass


def _workflow_mapping(loader, node, deep=False):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if type(key) is not str or key in result:
            raise ValueError("Workflow YAML requires unique string object keys")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_WorkflowLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _workflow_mapping)


def parse_workflow_yaml(raw: bytes) -> Dict[str, Any]:
    """Pure bounded unique YAML parser for admitted workflow document bytes.

    Path-mode load_yaml keeps its legacy semantics. Admission and consumption
    share this parser so aliases, duplicate keys and nonfinite values cannot
    change meaning between those boundaries.
    """
    import math

    if type(raw) is not bytes or not 0 < len(raw) <= 4 * 1024 * 1024:
        raise ValueError("Workflow YAML bytes exceed their bound")
    budget, active = [20000], set()

    def tree(value, depth=0):
        budget[0] -= 1
        if budget[0] < 0 or depth > 32:
            raise ValueError("Workflow YAML tree exceeds its bound")
        if type(value) in (dict, list):
            if id(value) in active:
                raise ValueError("Workflow YAML contains a cycle")
            active.add(id(value))
            try:
                for item in value.values() if type(value) is dict else value:
                    tree(item, depth + 1)
            finally:
                active.remove(id(value))
        elif value is not None and type(value) not in (str, int, float, bool):
            raise ValueError("Workflow YAML must contain JSON values")
        elif type(value) is float and not math.isfinite(value):
            raise ValueError("Workflow YAML numbers must be finite")

    try:
        depth = 0
        for count, event in enumerate(yaml.parse(raw, Loader=yaml.SafeLoader)):
            if count > 20000:
                raise ValueError("Workflow YAML events exceed their bound")
            if isinstance(event, (yaml.events.MappingStartEvent, yaml.events.SequenceStartEvent)):
                depth += 1
                if depth > 32:
                    raise ValueError("Workflow YAML nesting exceeds its bound")
            elif isinstance(event, (yaml.events.MappingEndEvent, yaml.events.SequenceEndEvent)):
                depth -= 1
        document = yaml.load(raw, Loader=_WorkflowLoader)
        tree(document)
        if type(document) is not dict:
            raise ValueError("Workflow YAML must contain an object")
        return document
    except (yaml.YAMLError, UnicodeError, RecursionError):
        raise ValueError("Invalid bounded workflow YAML document") from None
