import json
import subprocess
import sys
from pathlib import Path

import pytest

from shared.upload.converters import ConverterRegistry
from shared.upload.converters import gguf_reliable
from shared.upload.converters.calibration import CalibrationSpec
from shared.upload.converters.gguf_reliable import (
    GGUF_MANIFEST_NAME,
    ReliableGGUFConverter,
    file_sha256,
    validate_quantizations,
)

FAKE_COMMIT = "0123456789abcdef0123456789abcdef01234567"


def _make_exe(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\n", encoding="utf-8")
    path.chmod(0o755)


def _llama_cpp(tmp_path: Path, *, imatrix: bool = True) -> Path:
    root = tmp_path / "llama.cpp"
    _make_exe(root / "build" / "bin" / "llama-quantize")
    if imatrix:
        _make_exe(root / "build" / "bin" / "llama-imatrix")
    (root / "convert_hf_to_gguf.py").write_text("# fake\n", encoding="utf-8")
    return root


def _merged_model(tmp_path: Path) -> Path:
    model = tmp_path / "merged"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"architectures": ["LlamaForCausalLM"]}))
    return model


class FakeRun:
    """Records llama.cpp commands and creates the files they would write."""

    def __init__(self):
        self.calls = []

    def __call__(self, cmd, **kwargs):
        cmd = [str(c) for c in cmd]
        self.calls.append(cmd)
        if cmd[0] == "git":
            return subprocess.CompletedProcess(cmd, 0, stdout=FAKE_COMMIT + "\n", stderr="")
        name = Path(cmd[0]).name
        if name.startswith("python") or cmd[0] == sys.executable:
            out = Path(cmd[cmd.index("--outfile") + 1])
            out.write_bytes(b"base-gguf")
        elif name == "llama-imatrix":
            Path(cmd[cmd.index("-o") + 1]).write_bytes(b"imatrix")
        elif name == "llama-quantize":
            positional = cmd[3:] if cmd[1] == "--imatrix" else cmd[1:]
            Path(positional[1]).write_bytes(f"quant-{positional[2]}".encode())
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    def by_tool(self, tool):
        return [c for c in self.calls if Path(c[0]).name == tool]


@pytest.fixture
def fake_run(monkeypatch):
    fake = FakeRun()
    monkeypatch.setattr(gguf_reliable.subprocess, "run", fake)
    return fake


def _calibration_file(tmp_path: Path) -> Path:
    path = tmp_path / "train.jsonl"
    rows = [
        {"messages": [{"role": "user", "content": f"q{i}"}, {"role": "assistant", "content": f"a{i}"}]}
        for i in range(5)
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


def test_registry_instantiates_reliable_converter():
    converter = ConverterRegistry.get("gguf", model_loader=object())
    assert isinstance(converter, ReliableGGUFConverter)


def test_validate_quantizations_normalizes_and_rejects_iq_without_calibration():
    assert validate_quantizations(["q4_k_m", " Q8_0 "], has_calibration=False) == ["Q4_K_M", "Q8_0"]
    for quant in ["IQ2_XXS", "iq3_xxs", "IQ1_S", "Q2_K_S", "IQ3_XS", "IQ2_M"]:
        with pytest.raises(ValueError, match="importance"):
            validate_quantizations([quant], has_calibration=False)
    assert validate_quantizations(["iq3_xxs"], has_calibration=True) == ["IQ3_XXS"]
    # IQ types that do not require an imatrix stay allowed.
    assert validate_quantizations(["IQ4_XS", "IQ3_S"], has_calibration=False) == ["IQ4_XS", "IQ3_S"]


def test_quantize_places_imatrix_flag_before_positional_args(tmp_path, fake_run):
    converter = ReliableGGUFConverter(llama_cpp_dir=_llama_cpp(tmp_path))
    imatrix = tmp_path / "m.imatrix.gguf"
    assert converter.quantize_gguf(tmp_path / "in.gguf", tmp_path / "out.gguf", "Q4_K_M", imatrix=imatrix)

    cmd = fake_run.by_tool("llama-quantize")[0]
    assert cmd[1:3] == ["--imatrix", str(imatrix)]
    assert cmd[3:6] == [str(tmp_path / "in.gguf"), str(tmp_path / "out.gguf"), "Q4_K_M"]


def test_quantize_without_imatrix_has_no_flag(tmp_path, fake_run):
    converter = ReliableGGUFConverter(llama_cpp_dir=_llama_cpp(tmp_path))
    assert converter.quantize_gguf(tmp_path / "in.gguf", tmp_path / "out.gguf", "Q8_0")
    cmd = fake_run.by_tool("llama-quantize")[0]
    assert "--imatrix" not in cmd
    assert cmd[1:4] == [str(tmp_path / "in.gguf"), str(tmp_path / "out.gguf"), "Q8_0"]


def test_quantize_iq_without_imatrix_fails_fast(tmp_path, fake_run):
    converter = ReliableGGUFConverter(llama_cpp_dir=_llama_cpp(tmp_path))
    with pytest.raises(ValueError, match="importance"):
        converter.quantize_gguf(tmp_path / "in.gguf", tmp_path / "out.gguf", "IQ2_XXS")
    assert fake_run.calls == []


def test_compute_imatrix_command(tmp_path, fake_run):
    converter = ReliableGGUFConverter(llama_cpp_dir=_llama_cpp(tmp_path))
    out = tmp_path / "m.imatrix.gguf"
    assert converter.compute_imatrix(
        tmp_path / "base.gguf", tmp_path / "cal.txt", out, chunks=64, ctx_size=256, parse_special=True
    )
    cmd = fake_run.by_tool("llama-imatrix")[0]
    assert cmd[1:7] == ["-m", str(tmp_path / "base.gguf"), "-f", str(tmp_path / "cal.txt"), "-o", str(out)]
    assert cmd[cmd.index("-c") + 1] == "256"
    assert cmd[cmd.index("--chunks") + 1] == "64"
    assert "--parse-special" in cmd


def test_compute_imatrix_omits_optional_flags(tmp_path, fake_run):
    converter = ReliableGGUFConverter(llama_cpp_dir=_llama_cpp(tmp_path))
    converter.compute_imatrix(tmp_path / "b.gguf", tmp_path / "c.txt", tmp_path / "o.gguf", chunks=None)
    cmd = fake_run.by_tool("llama-imatrix")[0]
    assert "--chunks" not in cmd
    assert "--parse-special" not in cmd


def test_base_conversion_uses_current_interpreter(tmp_path, fake_run):
    converter = ReliableGGUFConverter(llama_cpp_dir=_llama_cpp(tmp_path))
    out = tmp_path / "base.gguf"
    assert converter.convert_to_gguf_base(tmp_path / "merged", out, "bf16", use_temp_file=True)
    cmd = fake_run.calls[0]
    assert cmd[0] == sys.executable
    assert cmd[cmd.index("--outtype") + 1] == "bf16"
    assert "--use-temp-file" in cmd
    assert cmd[-1] == str(tmp_path / "merged")


def test_setup_does_not_rebuild_when_imatrix_not_needed(tmp_path, fake_run, monkeypatch):
    monkeypatch.setattr(gguf_reliable.shutil, "which", lambda name: "/usr/bin/" + name)
    converter = ReliableGGUFConverter(llama_cpp_dir=_llama_cpp(tmp_path, imatrix=False))
    assert converter.setup_llama_cpp(with_imatrix=False)
    assert fake_run.calls == []


def test_setup_builds_imatrix_when_missing_without_reconfiguring(tmp_path, monkeypatch):
    llama = _llama_cpp(tmp_path, imatrix=False)
    (llama / "build" / "CMakeCache.txt").write_text("GGML_CUDA:BOOL=ON\n")
    monkeypatch.setattr(gguf_reliable.shutil, "which", lambda name: "/usr/bin/" + name)
    calls = []

    def fake(cmd, **kwargs):
        calls.append([str(c) for c in cmd])
        if "--build" in cmd:
            _make_exe(llama / "build" / "bin" / "llama-imatrix")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(gguf_reliable.subprocess, "run", fake)
    converter = ReliableGGUFConverter(llama_cpp_dir=llama)
    assert converter.setup_llama_cpp(with_imatrix=True)

    assert len(calls) == 1, "existing CMake cache must not be reconfigured"
    build = calls[0]
    assert build[:3] == ["cmake", "--build", "."]
    targets = build[build.index("--target") + 1:]
    assert targets == ["llama-quantize", "llama-imatrix"]
    assert converter.imatrix_path == llama / "build" / "bin" / "llama-imatrix"


def test_setup_reports_missing_cmake(tmp_path, monkeypatch, capsys, fake_run):
    monkeypatch.setattr(gguf_reliable.shutil, "which", lambda name: None)
    converter = ReliableGGUFConverter(llama_cpp_dir=_llama_cpp(tmp_path, imatrix=False))
    assert not converter.setup_llama_cpp(with_imatrix=True)
    assert "cmake not found" in capsys.readouterr().out
    assert fake_run.calls == []


def test_convert_merged_model_with_calibration_writes_manifest(tmp_path, fake_run):
    llama = _llama_cpp(tmp_path)
    converter = ReliableGGUFConverter(llama_cpp_dir=llama)
    gguf_dir = tmp_path / "out" / "gguf"
    work = tmp_path / "work"
    calibration = CalibrationSpec(
        dataset_path=_calibration_file(tmp_path),
        source="hf://datasets/org/data/train.jsonl",
        chunks=16,
        ctx_size=128,
    )

    created = converter.convert_merged_model(
        _merged_model(tmp_path),
        gguf_dir,
        ["q4_k_m", "Q8_0"],
        "tiny",
        work,
        calibration=calibration,
        source_model="org/tiny",
    )

    assert [p.name for p in created] == ["tiny.gguf", "tiny-Q4_K_M.gguf", "tiny-Q8_0.gguf"]

    # imatrix computed once from the base GGUF; used only where it matters.
    imatrix_cmd = fake_run.by_tool("llama-imatrix")[0]
    assert imatrix_cmd[imatrix_cmd.index("-m") + 1] == str(work / "tiny.gguf")
    quant_cmds = {c[-2]: c for c in fake_run.by_tool("llama-quantize")}
    assert quant_cmds["Q4_K_M"][1:3] == ["--imatrix", str(work / "tiny.imatrix.gguf")]
    assert "--imatrix" not in quant_cmds["Q8_0"]

    manifest_path = gguf_dir / GGUF_MANIFEST_NAME
    assert converter.manifest_path == manifest_path
    manifest = json.loads(manifest_path.read_text())
    assert manifest["schema_version"] == 1
    assert manifest["model_name"] == "tiny"
    assert manifest["source_model"] == "org/tiny"
    assert manifest["base_dtype"] == "bf16"
    assert manifest["llama_cpp_commit"] == FAKE_COMMIT

    files = {f["filename"]: f for f in manifest["files"]}
    assert set(files) == {"tiny.gguf", "tiny-Q4_K_M.gguf", "tiny-Q8_0.gguf"}
    assert files["tiny.gguf"]["role"] == "base"
    assert files["tiny-Q4_K_M.gguf"]["imatrix_used"] is True
    assert files["tiny-Q8_0.gguf"]["imatrix_used"] is False
    for entry in manifest["files"]:
        path = gguf_dir / entry["filename"]
        assert entry["size_bytes"] == path.stat().st_size
        assert entry["sha256"] == file_sha256(path)
        assert set(entry) == {"filename", "role", "quant_type", "size_bytes", "sha256", "imatrix_used"}

    cal = manifest["calibration"]
    assert cal["source"] == "hf://datasets/org/data/train.jsonl"
    assert cal["rows_used"] == 5
    assert cal["chunks"] == 16
    assert cal["ctx_size"] == 128
    assert cal["imatrix_sha256"] == file_sha256(work / "tiny.imatrix.gguf")
    assert cal["text_sha256"] == file_sha256(work / "tiny.calibration.txt")
    # Calibration text stays in the work dir, never next to the uploadable GGUFs.
    assert not any("calibration" in p.name for p in gguf_dir.iterdir())


def test_convert_merged_model_without_calibration(tmp_path, fake_run):
    converter = ReliableGGUFConverter(llama_cpp_dir=_llama_cpp(tmp_path, imatrix=False))
    gguf_dir = tmp_path / "gguf"
    created = converter.convert_merged_model(
        _merged_model(tmp_path), gguf_dir, ["Q4_K_M"], "tiny", tmp_path / "work", include_base=False
    )
    assert [p.name for p in created] == ["tiny-Q4_K_M.gguf"]
    assert fake_run.by_tool("llama-imatrix") == []
    manifest = json.loads((gguf_dir / GGUF_MANIFEST_NAME).read_text())
    assert manifest["calibration"] is None
    assert [f["imatrix_used"] for f in manifest["files"]] == [False]


def test_calibration_skipped_when_no_quant_uses_it(tmp_path, fake_run):
    converter = ReliableGGUFConverter(llama_cpp_dir=_llama_cpp(tmp_path, imatrix=False))
    calibration = CalibrationSpec(dataset_path=_calibration_file(tmp_path))
    converter.convert_merged_model(
        _merged_model(tmp_path), tmp_path / "gguf", ["Q8_0"], "tiny", tmp_path / "work",
        calibration=calibration,
    )
    assert fake_run.by_tool("llama-imatrix") == []


def test_convert_merged_model_iq_without_calibration_fails_before_work(tmp_path, fake_run):
    converter = ReliableGGUFConverter(llama_cpp_dir=_llama_cpp(tmp_path))
    with pytest.raises(ValueError, match="importance"):
        converter.convert_merged_model(
            _merged_model(tmp_path), tmp_path / "gguf", ["IQ3_XXS"], "tiny", tmp_path / "work"
        )
    assert fake_run.calls == []


def test_convert_merges_then_delegates(tmp_path, fake_run, monkeypatch):
    converter = ReliableGGUFConverter(llama_cpp_dir=_llama_cpp(tmp_path))
    adapter = tmp_path / "final_model"
    adapter.mkdir()
    merged = _merged_model(tmp_path)
    monkeypatch.setattr(converter, "merge_lora_to_16bit", lambda model_path, output_dir: merged)
    seen = {}

    def fake_convert_merged(merged_model_path, gguf_dir, quantizations, model_name, work_dir, **kwargs):
        seen.update(
            merged=merged_model_path, gguf_dir=gguf_dir, quants=quantizations,
            name=model_name, **kwargs,
        )
        return []

    monkeypatch.setattr(converter, "convert_merged_model", fake_convert_merged)
    calibration = CalibrationSpec(dataset_path=_calibration_file(tmp_path))
    converter.convert(adapter, tmp_path / "out", quantizations=["q4_k_m"], calibration=calibration)

    assert seen["merged"] == merged
    assert seen["gguf_dir"] == tmp_path / "out" / "gguf"
    assert seen["quants"] == ["Q4_K_M"]
    assert seen["name"] == "final_model"
    assert seen["calibration"] is calibration
    assert seen["source_model"] == adapter.name
    assert seen["include_base"] is True


def test_orchestrator_passes_calibration_and_uploads_manifest(tmp_path, monkeypatch):
    orchestrator_module = pytest.importorskip("shared.upload.orchestrator")
    from shared.upload.core.config import ConversionConfig, SaveConfig, UploadConfig

    manifest = tmp_path / "gguf" / GGUF_MANIFEST_NAME
    quant = tmp_path / "gguf" / "m-Q4_K_M.gguf"
    seen = {}

    class FakeConverter:
        manifest_path = None

        def convert(self, model_path, output_dir, **kwargs):
            seen.update(kwargs)
            quant.parent.mkdir(parents=True, exist_ok=True)
            quant.write_bytes(b"q")
            manifest.write_text("{}")
            self.manifest_path = manifest
            return [quant]

    class FakeUploader:
        def upload_files(self, files, repo_id, credential):
            seen["uploaded"] = [Path(f).name for f in files]

    monkeypatch.setattr(orchestrator_module.ConverterRegistry, "get", lambda name, model_loader=None: FakeConverter())
    monkeypatch.setattr(orchestrator_module.UploaderRegistry, "get", lambda name: FakeUploader())

    calibration = CalibrationSpec(dataset_path=tmp_path / "train.jsonl")
    orch = object.__new__(orchestrator_module.UploadOrchestrator)
    orch.upload_config = UploadConfig(model_path=tmp_path, repo_id="user/m", credential="x")
    orch.save_config = SaveConfig()
    orch.conversion_config = ConversionConfig(quantizations=["IQ3_XXS"], calibration=calibration)
    orch.model_loader = None
    orch.output_dir = tmp_path
    orch.formats_created, orch.artifacts_created, orch.gguf_files = [], [], []
    orch.gguf_converter = None

    orch._convert_formats()
    orch._upload_converted_files()

    assert seen["calibration"] is calibration
    assert seen["quantizations"] == ["IQ3_XXS"]
    assert seen["uploaded"] == ["m-Q4_K_M.gguf", GGUF_MANIFEST_NAME]
