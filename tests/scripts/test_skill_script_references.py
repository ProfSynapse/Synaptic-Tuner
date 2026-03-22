from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_canonical_skill_docs_do_not_reference_old_script_locations() -> None:
    targets = [REPO_ROOT / ".skills", REPO_ROOT / "AGENTS.md", REPO_ROOT / "CLAUDE.md"]
    text_suffixes = {".md", ".py", ".sh", ".ps1", ".yaml", ".yml", ".txt"}
    banned_tokens = (
        "Trainers/scripts/",
        "tools/validate_syngen.py",
        "Tools/validate_syngen.py",
        "tools/split_for_gspo.py",
        "src/upload_to_hf.py",
    )

    for target in targets:
        paths = (
            [target]
            if target.is_file()
            else [
                path
                for path in target.rglob("*")
                if path.is_file() and path.suffix in text_suffixes
            ]
        )
        for path in paths:
            text = _read_text(path)
            for token in banned_tokens:
                assert token not in text, f"{path} still references deprecated skill script path: {token}"


def test_generalizable_skill_scripts_exist_in_canonical_tree() -> None:
    expected = [
        REPO_ROOT / ".skills" / "fine-tuning" / "scripts" / "battle_of_models.py",
        REPO_ROOT / ".skills" / "fine-tuning" / "scripts" / "hf_jobs_hardware.py",
        REPO_ROOT / ".skills" / "fine-tuning" / "scripts" / "prune_dataset_from_loss.py",
        REPO_ROOT / ".skills" / "fine-tuning" / "scripts" / "split_for_gspo.py",
        REPO_ROOT / ".skills" / "synethetic-data-generation" / "scripts" / "validate_syngen.py",
        REPO_ROOT / ".skills" / "upload-deployment" / "scripts" / "upload_model.py",
    ]

    for path in expected:
        assert path.exists(), f"Missing canonical skill script: {path}"
