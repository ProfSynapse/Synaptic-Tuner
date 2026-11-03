"""CLI: ``python -m synaptic_tuner.experimental.weight_editor <command>``.

Exit codes: 0 ok/complete, 1 refused or invalid input, 3 incomplete and
reconciliation needed, 4 incomplete for another reason, 5 verification failed.
``--ports`` names an operator-trusted ``module:factory``; it is never read from
the config or from a model request.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import broker, replay
from .canon import create_only, strict_loads
from .config import load_config
from .errors import EditorError, PartialTail
from .fakes import write_fixture
from .ports import PortContext, load_ports


def _kv(items: list[str]) -> dict[str, str]:
    out = {}
    for item in items or []:
        if "=" not in item:
            raise EditorError(f"--ports-arg expects key=value, got {item!r}", code="usage")
        k, v = item.split("=", 1)
        out[k] = v
    return out


def _binding(path: str | None) -> dict:
    if not path:
        return {}
    try:
        data = strict_loads(Path(path).read_bytes())
    except (OSError, ValueError) as exc:
        raise EditorError(f"cannot read binding file: {exc}", code="usage") from exc
    if not isinstance(data, dict):
        raise EditorError("binding file must be a JSON object", code="usage")
    return data


def _ports(args, config):
    if not args.ports:
        raise EditorError("--ports module:factory is required", code="usage")
    return load_ports(args.ports, PortContext(_kv(args.ports_arg), config.model.requested()))


def _emit(obj: dict) -> None:
    print(json.dumps(obj, indent=2, sort_keys=True))


def _exit_for(summary: dict) -> int:
    if summary["status"] == "complete":
        return 0 if summary["capture_qualification"]["qualified"] else 6
    halt = summary.get("halt") or {}
    return 3 if halt.get("code") == "reconciliation_needed" else 4


def cmd_validate(args) -> int:
    config = load_config(Path(args.config))
    ports = _ports(args, config)
    report = broker.validate_ports(config, ports)
    ports.executor.close()
    _emit({
        "ok": True, "config_sha256": config.sha256, "capsule_manifest_sha256": config.capsule.manifest_sha256,
        "capsule_objects": len(config.capsule.objects), "operations": sorted(config.operations),
        "output_root": str(config.output_root), "limits": dict(config.limits) or "none configured (caller-owned)",
        "ports": report, "effects": "none: validate performs no executor or model call",
    })
    return 0


def cmd_run(args) -> int:
    config = load_config(Path(args.config))
    ports = _ports(args, config)
    run = broker.EditorRun.create(config, ports, args.run_id, _binding(args.binding_file))
    try:
        summary = run.run()
    finally:
        run.close()
    _emit(summary)
    return _exit_for(summary)


def cmd_resume(args) -> int:
    try:
        config = load_config(Path(args.config))
    except EditorError as exc:
        _audit_pre_open_refusal(args, exc)
        raise
    ports = _ports(args, config)
    run = broker.EditorRun.open(config, ports, args.run_id, _binding(args.binding_file),
                                recover_tail=args.recover_partial_tail)
    try:
        summary = run.run()
    finally:
        run.close()
    _emit(summary)
    return _exit_for(summary)


def _audit_pre_open_refusal(args, exc: EditorError) -> None:
    """Config/capsule drift is refused before any port exists; still leave audit evidence."""
    try:
        data = strict_loads(Path(args.config).read_bytes())
        out = (Path(args.config).resolve().parent / data["output_root"]).resolve()
    except Exception:
        return
    run_dir = out / args.run_id
    if (run_dir / "manifest.json").is_file():
        broker.record_refusal(run_dir, "config_or_capsule_invalid", {"code": exc.code, "message": str(exc)})


def cmd_verify(args) -> int:
    report = replay.verify_run(Path(args.run_dir), anchor_file=Path(args.anchor_file) if args.anchor_file else None)
    _emit(report)
    return (5 if not report["ok"] else 6 if report["status"] == "complete"
            and not report["capture_qualification"]["qualified"] else 0)


def cmd_status(args) -> int:
    return cmd_verify(args)


def cmd_replay(args) -> int:
    run_dir = Path(args.run_dir)
    manifest, journal, raw = replay.load_verified(run_dir)
    text = replay.render_markdown(run_dir, journal, raw)
    out = Path(args.output) if args.output else run_dir / "transcript.md"
    if out.exists() and out.read_bytes() != text.encode() and not args.force:
        raise EditorError(f"{out} exists with different content; pass --force to replace the derived view",
                          code="output_exists")
    if args.force and out.exists():
        out.write_text(text)
    elif not out.exists():
        create_only(out, text.encode())
    _emit({"ok": True, "transcript": str(out), "journal_records": len(journal), "raw_records": len(raw)})
    return 0


def cmd_advance(args) -> int:
    _emit(broker.advance_phase(Path(args.run_dir), args.to, args.reference,
                              anchor_file=Path(args.anchor_file) if args.anchor_file else None))
    return 0


def cmd_anchor(args) -> int:
    _emit(broker.write_anchor(Path(args.run_dir), Path(args.output)))
    return 0


def cmd_smoke(args) -> int:
    workdir = Path(args.workdir)
    if workdir.exists():
        raise EditorError(f"{workdir} exists; smoke never overwrites", code="run_exists")
    fx = write_fixture(workdir)
    config = load_config(fx["config"])
    ports = load_ports("synaptic_tuner.experimental.weight_editor.fakes:build_ports",
                       PortContext({"script": str(fx["script"]), "effects_dir": str(fx["effects"])},
                                   config.model.requested()))
    run = broker.EditorRun.create(config, ports, args.run_id, {"smoke": "synthetic"})
    try:
        summary = run.run()
    finally:
        run.close()
    report = replay.verify_run(Path(summary["run_dir"]))
    effects = [json.loads(x) for x in (fx["effects"] / "effects.log").read_text().splitlines()]
    ok = summary["status"] == "complete" and report["ok"] and summary["capture_qualification"]["qualified"]
    _emit({"ok": ok, "summary": summary, "verify": report,
           "effect_count": len(effects), "unique_dispatches": len({e["dispatch_id"] for e in effects}),
           "transcript": str(Path(summary["run_dir"]) / "transcript.md")})
    return 0 if ok else 6 if report["ok"] else 5


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="weight_editor", description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    def ports_args(sp):
        sp.add_argument("--config", required=True)
        sp.add_argument("--ports", help="operator-trusted module:factory returning PortSet")
        sp.add_argument("--ports-arg", action="append", default=[], metavar="KEY=VALUE")
        sp.add_argument("--binding-file", help="JSON object of source/runtime identifiers bound into the run")

    sp = sub.add_parser("validate", help="validate config, capsule and ports; no run, no effects")
    ports_args(sp)
    sp.set_defaults(fn=cmd_validate)
    sp = sub.add_parser("run", help="register a NEW run and drive it")
    ports_args(sp)
    sp.add_argument("--run-id", required=True)
    sp.set_defaults(fn=cmd_run)
    sp = sub.add_parser("resume", help="resume a run under identical bytes; reconciles in-flight dispatches")
    ports_args(sp)
    sp.add_argument("--run-id", required=True)
    sp.add_argument("--recover-partial-tail", action="store_true",
                    help="quarantine (never delete) an incomplete final log line and record an audit entry")
    sp.set_defaults(fn=cmd_resume)
    for name, fn in (("verify", cmd_verify), ("status", cmd_status)):
        sp = sub.add_parser(name, help="verify hash chains, raw references, receipts and artifacts")
        sp.add_argument("--run-dir", required=True)
        sp.add_argument("--anchor-file")
        sp.set_defaults(fn=fn)
    sp = sub.add_parser("replay", help="export a readable Markdown transcript from the retained run")
    sp.add_argument("--run-dir", required=True)
    sp.add_argument("--output")
    sp.add_argument("--force", action="store_true")
    sp.set_defaults(fn=cmd_replay)
    sp = sub.add_parser("advance", help="host action: final_patch_frozen -> independent_evaluation -> closed")
    sp.add_argument("--run-dir", required=True)
    sp.add_argument("--to", required=True, choices=["independent_evaluation", "closed"])
    sp.add_argument("--reference", required=True)
    sp.add_argument("--anchor-file", help="externally retained freeze anchor; required before evaluation")
    sp.set_defaults(fn=cmd_advance)
    sp = sub.add_parser("anchor", help="write hash-only freeze anchor outside the run output tree")
    sp.add_argument("--run-dir", required=True)
    sp.add_argument("--output", required=True)
    sp.set_defaults(fn=cmd_anchor)
    sp = sub.add_parser("smoke", help="deterministic synthetic smoke with fake ports (no model call)")
    sp.add_argument("--workdir", required=True)
    sp.add_argument("--run-id", default="smoke-1")
    sp.set_defaults(fn=cmd_smoke)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.fn(args)
    except EditorError as exc:
        _emit({"ok": False, "error": {"code": exc.code, "message": str(exc), **{k: v for k, v in exc.details.items()}}})
        return 1 if not isinstance(exc, PartialTail) and exc.code not in (
            "journal_corrupt", "hash_mismatch", "manifest_tampered", "raw_reference_broken") else 5


if __name__ == "__main__":
    sys.exit(main())
