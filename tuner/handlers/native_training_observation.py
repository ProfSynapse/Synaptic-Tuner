"""Bounded, stdlib-only observations for the native training doctor.

Distribution metadata is an installation hint, not wheel provenance or a
training, Metal, model, or tokenizer qualification.
"""

from importlib import metadata
import platform
import re
import sys


_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+\-]{0,63}\Z")
_DEPENDENCIES = ("mlx", "mlx-lm", "numpy", "transformers", "torch")


def _observed(value: object) -> dict[str, str]:
    if isinstance(value, str) and _TOKEN.fullmatch(value):
        return {"status": "observed", "value": value}
    return {"status": "malformed"}


def _call(observation) -> dict[str, str]:
    try:
        return _observed(observation())
    except Exception:
        return {"status": "observation_failed"}


def _os_version() -> dict[str, str]:
    try:
        value = platform.version()
        if isinstance(value, str) and 1 <= len(value) <= 128 and all(32 <= ord(char) <= 126 for char in value):
            return {"status": "observed", "value": value}
        return {"status": "malformed"}
    except Exception:
        return {"status": "observation_failed"}


def collect_native_training_observation() -> dict[str, object]:
    """Read local facts without importing SDKs or initializing compute devices."""
    system = _call(platform.system)
    architecture = _call(platform.machine)
    release = _call(platform.release)
    os_version = _os_version()
    macos = _call(lambda: platform.mac_ver()[0]) if system.get("value") == "Darwin" else {"status": "not_applicable"}
    implementation = _call(lambda: sys.implementation.name)
    python_version = _call(lambda: ".".join(str(part) for part in sys.version_info[:3]))
    dependencies: dict[str, dict[str, str]] = {}
    for name in _DEPENDENCIES:
        try:
            dependencies[name] = _observed(metadata.version(name))
        except metadata.PackageNotFoundError:
            dependencies[name] = {"status": "absent"}
        except Exception:
            dependencies[name] = {"status": "observation_failed"}
    return {
        "schema_version": 1,
        "status": "DIAGNOSTIC_ONLY",
        "native_apple_candidate": system.get("value") == "Darwin" and architecture.get("value", "").lower() == "arm64",
        "platform": {"system": system, "architecture": architecture, "release": release, "version": os_version, "macos_version": macos},
        "python": {"implementation": implementation, "version": python_version},
        "dependencies": dependencies,
    }
