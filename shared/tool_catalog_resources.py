"""Resolve canonical checkout catalogs or their exact installed resources."""
from contextlib import contextmanager
from importlib.resources import as_file, files
from pathlib import Path


_CATALOGS = {"cli": "cli-first-tool-schemas.json", "tools": "Tools/tool_schemas.json"}
_SOURCE_ROOT = Path(__file__).resolve().parents[1]


@contextmanager
def tool_catalog_path(kind: str, explicit_path: Path | None = None):
    """Explicit paths never fall back; missing selected defaults fail closed."""
    relative = _CATALOGS[kind]
    if explicit_path is not None:
        path = Path(explicit_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        yield path
    elif (_SOURCE_ROOT / "pyproject.toml").is_file():
        path = _SOURCE_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        yield path
    else:
        resource = files("synaptic_tuner").joinpath("runtime", *relative.split("/"))
        if not resource.is_file():
            raise FileNotFoundError(f"Missing installed tool catalog: {relative}")
        with as_file(resource) as path:
            yield path
