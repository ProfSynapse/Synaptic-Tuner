"""Fixed inventory metadata must never be normalized by a hash refresh."""
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/regenerate_hf_parent_closure.py"
spec = importlib.util.spec_from_file_location("hf_parent_maintenance", SCRIPT)
maintenance = importlib.util.module_from_spec(spec)
spec.loader.exec_module(maintenance)


@pytest.mark.parametrize("damage", ["count", "bool_count", "extra_member_field"])
def test_malformed_metadata_write_preserves_original(tmp_path, monkeypatch, damage):
    document = json.loads((maintenance.ROOT / maintenance.MANIFEST).read_bytes())
    if damage == "count":
        document["member_count"] -= 1
    elif damage == "bool_count":
        document["member_count"] = True
    else:
        document["members"][0]["extra"] = "must not be stripped"
    path = tmp_path / maintenance.MANIFEST
    path.parent.mkdir(parents=True)
    original = json.dumps(document).encode()
    path.write_bytes(original)
    monkeypatch.setattr(maintenance, "ROOT", tmp_path)
    monkeypatch.setattr("sys.argv", [str(SCRIPT), "--write"])
    with pytest.raises(SystemExit, match="inventory mismatch"):
        maintenance.main()
    assert path.read_bytes() == original


def test_valid_fixed_inventory_is_current(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", [str(SCRIPT)])
    maintenance.main()
    assert capsys.readouterr().out == "CURRENT\n"
