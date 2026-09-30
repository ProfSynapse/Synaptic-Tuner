from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import pytest

from tuner.runtime.packaged_sft_execution import _require_trainer_assets


@pytest.mark.parametrize("mutation", [None, "missing", "unowned", "hash", "wheel"])
def test_trainer_default_asset_requires_exact_wheel_membership(tmp_path, mutation):
    names = ("Trainers/sft/train_sft.py", "Trainers/sft/configs/config.yaml")
    contents = {names[0]: b"# installed trainer\n", names[1]: b"training: {}\n"}
    archive_bytes = BytesIO()
    with ZipFile(archive_bytes, "w") as archive:
        for name, content in contents.items():
            if mutation != "wheel" or name != names[1]:
                archive.writestr(name, content)
            path = tmp_path / name
            path.parent.mkdir(parents=True, exist_ok=True)
            if mutation != "missing" or name != names[1]:
                path.write_bytes(content if mutation != "hash" or name != names[1] else b"hostile: true\n")
    distribution = SimpleNamespace(files=names[:1] if mutation == "unowned" else names,
                                   locate_file=lambda name: tmp_path / name)
    with ZipFile(BytesIO(archive_bytes.getvalue())) as archive:
        if mutation is None:
            assert _require_trainer_assets(distribution, archive) == tmp_path / names[0]
        else:
            with pytest.raises((ValueError, OSError)):
                _require_trainer_assets(distribution, archive)
