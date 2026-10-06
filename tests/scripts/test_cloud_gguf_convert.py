import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from shared.upload.converters import gguf_reliable
from shared.upload.converters.gguf_reliable import GGUF_MANIFEST_NAME, ReliableGGUFConverter

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "cloud_gguf_convert.py"
RECIPE = REPO_ROOT / "Trainers" / "recipes" / "gguf_conversion.yaml"


def _load_script():
    spec = importlib.util.spec_from_file_location("cloud_gguf_convert_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def script(monkeypatch, tmp_path):
    module = _load_script()
    calls = {"download_model": [], "uploads": [], "ensure_build_tools": 0, "download_calibration": []}

    def download_model(repo_id, local_dir):
        calls["download_model"].append(repo_id)
        local_dir.mkdir(parents=True, exist_ok=True)
        (local_dir / "config.json").write_text("{}")
        return local_dir

    def clone(dest):
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "convert_hf_to_gguf.py").write_text("# fake\n")
        return dest

    def ensure_build_tools():
        calls["ensure_build_tools"] += 1

    monkeypatch.setattr(module, "download_model", download_model)
    monkeypatch.setattr(module, "clone_llama_cpp", clone)
    monkeypatch.setattr(module, "install_gguf_package", lambda: None)
    monkeypatch.setattr(module, "ensure_build_tools", ensure_build_tools)
    monkeypatch.setattr(module, "upload_gguf", lambda path, repo: calls["uploads"].append((path.name, repo)))
    module.calls = calls
    return module


class FakeRun:
    def __init__(self):
        self.calls = []

    def __call__(self, cmd, **kwargs):
        cmd = [str(c) for c in cmd]
        self.calls.append(cmd)
        if "--outfile" in cmd:
            Path(cmd[cmd.index("--outfile") + 1]).write_bytes(b"gguf")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")


def test_parse_quant_list_accepts_spaces_and_commas():
    module = _load_script()
    assert module.parse_quant_list(["q4_k_m,q8_0", "Q8_0", " q5_k_m "]) == ["Q4_K_M", "Q8_0", "Q5_K_M"]
    assert module.is_fast_path(["Q8_0", "BF16"])
    assert not module.is_fast_path(["Q8_0", "Q4_K_M"])


def test_default_quant_is_q8_0():
    module = _load_script()
    args = module.parse_args(["--model-repo", "org/m"])
    assert args.quant == ["q8_0"]


@pytest.mark.parametrize("argv", [
    ["--calibration-repo", "org/data"],
    ["--calibration-file", "train.jsonl"],
    ["--calibration-path", "x.jsonl", "--calibration-repo", "org/data", "--calibration-file", "t.jsonl"],
])
def test_calibration_argument_validation(argv):
    module = _load_script()
    with pytest.raises(SystemExit):
        module.parse_args(["--model-repo", "org/m", *argv])


def test_fast_path_skips_llama_cpp_build(script, tmp_path, monkeypatch):
    fake = FakeRun()
    monkeypatch.setattr(gguf_reliable.subprocess, "run", fake)
    monkeypatch.setattr(
        ReliableGGUFConverter, "convert_merged_model",
        lambda *a, **k: pytest.fail("fast path must not use the quantize path"),
    )

    script.main(["--model-repo", "org/tiny", "--quant", "q8_0,bf16", "--work-dir", str(tmp_path / "w")])

    converts = [c for c in fake.calls if "--outtype" in c]
    assert [c[c.index("--outtype") + 1] for c in converts] == ["q8_0", "bf16"]
    assert all("--use-temp-file" in c for c in converts)
    assert script.calls["ensure_build_tools"] == 0
    assert script.calls["uploads"] == [
        ("tiny-Q8_0.gguf", "org/tiny"),
        ("tiny-BF16.gguf", "org/tiny"),
        (GGUF_MANIFEST_NAME, "org/tiny"),
    ]
    manifest = json.loads((tmp_path / "w" / "gguf" / GGUF_MANIFEST_NAME).read_text())
    assert manifest["source_model"] == "org/tiny"
    assert manifest["base_dtype"] is None
    assert [f["quant_type"] for f in manifest["files"]] == ["Q8_0", "BF16"]


def test_kquant_uses_quantize_path_with_calibration_repo(script, tmp_path, monkeypatch):
    seen = {}
    cal_file = tmp_path / "train.jsonl"
    cal_file.write_text("{}\n")
    monkeypatch.setattr(
        script, "download_calibration",
        lambda repo, filename, local_dir: seen.setdefault("download", (repo, filename)) and cal_file,
    )

    def fake_convert_merged(self, merged_model_path, gguf_dir, quantizations, model_name, work_dir, **kwargs):
        seen.update(quants=quantizations, name=model_name, **kwargs)
        gguf_dir.mkdir(parents=True, exist_ok=True)
        created = []
        for q in quantizations:
            path = gguf_dir / f"{model_name}-{q}.gguf"
            path.write_bytes(b"q")
            created.append(path)
        self.manifest_path = gguf_dir / GGUF_MANIFEST_NAME
        self.manifest_path.write_text("{}")
        return created

    monkeypatch.setattr(ReliableGGUFConverter, "convert_merged_model", fake_convert_merged)

    script.main([
        "--model-repo", "org/tiny", "--quant", "q4_k_m", "q8_0",
        "--upload-to", "org/tiny-GGUF",
        "--calibration-repo", "org/data", "--calibration-file", "sft/train.jsonl",
        "--imatrix-chunks", "32",
        "--work-dir", str(tmp_path / "w"),
    ])

    assert seen["download"] == ("org/data", "sft/train.jsonl")
    assert seen["quants"] == ["Q4_K_M", "Q8_0"]
    assert seen["include_base"] is False
    assert seen["use_temp_file"] is True
    assert seen["dtype"] == "bf16"
    assert seen["source_model"] == "org/tiny"
    calibration = seen["calibration"]
    assert calibration.dataset_path == cal_file
    assert calibration.source == "hf://datasets/org/data/sft/train.jsonl"
    assert calibration.chunks == 32
    assert script.calls["ensure_build_tools"] == 1
    assert script.calls["uploads"] == [
        ("tiny-Q4_K_M.gguf", "org/tiny-GGUF"),
        ("tiny-Q8_0.gguf", "org/tiny-GGUF"),
        (GGUF_MANIFEST_NAME, "org/tiny-GGUF"),
    ]


def test_missing_quant_output_fails_before_upload(script, tmp_path, monkeypatch):
    def fake_convert_merged(self, merged_model_path, gguf_dir, quantizations, model_name, work_dir, **kwargs):
        gguf_dir.mkdir(parents=True, exist_ok=True)
        path = gguf_dir / f"{model_name}-Q4_K_M.gguf"
        path.write_bytes(b"q")
        return [path]

    monkeypatch.setattr(ReliableGGUFConverter, "convert_merged_model", fake_convert_merged)
    with pytest.raises(RuntimeError, match="tiny-Q5_K_M.gguf"):
        script.main(["--model-repo", "org/tiny", "--quant", "q4_k_m,q5_k_m", "--work-dir", str(tmp_path / "w")])
    assert script.calls["uploads"] == []


def test_iq_without_calibration_fails_before_download(script, tmp_path):
    with pytest.raises(ValueError, match="importance"):
        script.main(["--model-repo", "org/tiny", "--quant", "iq2_xxs", "--work-dir", str(tmp_path / "w")])
    assert script.calls["download_model"] == []


def _render_recipe_step(env_overrides):
    """Render the recipe like cloud-run does (str.format_map + export + bash)."""
    recipe = yaml.safe_load(RECIPE.read_text())
    run_cfg = recipe["run"]
    variables = {"repo_dir": str(REPO_ROOT)}
    env = {k: str(v).format_map(variables) for k, v in run_cfg["env"].items()}
    env.update(env_overrides)
    (step,) = run_cfg["steps"]
    rendered = step.format_map(variables)
    rendered = rendered.replace("python scripts/cloud_gguf_convert.py", "printf '%s\\n'")
    exports = " && ".join(f"export {k}={json.dumps(v)}" for k, v in env.items())
    out = subprocess.run(
        ["bash", "-c", f"{exports} && {rendered}"], capture_output=True, text=True, check=True
    )
    return out.stdout.splitlines()


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash required")
def test_recipe_omits_empty_calibration_flags():
    argv = _render_recipe_step({"GGUF_QUANT_TYPE": "q4_k_m q8_0"})
    assert argv == ["--model-repo", "professorsynapse/gemma-4-e4b-sft", "--quant", "q4_k_m", "q8_0"]


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash required")
def test_recipe_passes_calibration_flags_when_set():
    argv = _render_recipe_step({
        "GGUF_QUANT_TYPE": "q4_k_m",
        "GGUF_CALIBRATION_REPO": "org/data",
        "GGUF_CALIBRATION_FILE": "sft train.jsonl",
    })
    assert argv[-4:] == ["--calibration-repo", "org/data", "--calibration-file", "sft train.jsonl"]
