from Trainers.sft.configs import config_loader


def test_yaml_suffix_uses_declarative_loader(tmp_path, monkeypatch):
    path = tmp_path / "job.yaml"
    path.write_text("model: {}\n", encoding="utf-8")
    sentinel = object()
    seen = []

    monkeypatch.setattr(config_loader, "load_config", lambda value: seen.append(value) or sentinel)

    assert config_loader.load_external_config(str(path)) is sentinel
    assert seen == [str(path)]


def test_legacy_python_config_keeps_config_constructor_contract(tmp_path):
    path = tmp_path / "legacy_config.py"
    path.write_text(
        "class Config:\n"
        "    def __init__(self):\n"
        "        self.marker = 'legacy'\n",
        encoding="utf-8",
    )

    loaded = config_loader.load_external_config(str(path))
    assert loaded.marker == "legacy"
