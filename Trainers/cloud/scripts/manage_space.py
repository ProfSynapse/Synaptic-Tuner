#!/usr/bin/env python3
"""Render, deploy, and manage reusable Hugging Face Spaces."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict

from huggingface_hub import HfApi, SpaceHardware, SpaceStorage
from huggingface_hub.errors import HfHubHTTPError


REPO_ROOT = Path(__file__).resolve().parents[3]
TEMPLATES_ROOT = REPO_ROOT / "Trainers" / "cloud" / "spaces"
SUPPORTED_TEMPLATES = {"vllm_warm"}


def _parse_key_value(raw: str) -> tuple[str, str]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError(f"Expected KEY=VALUE, got: {raw}")
    key, value = raw.split("=", 1)
    key = key.strip()
    if not key:
        raise argparse.ArgumentTypeError(f"Expected KEY=VALUE, got: {raw}")
    return key, value


def _pairs_to_dict(pairs: list[tuple[str, str]] | None) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for key, value in pairs or []:
        result[key] = value
    return result


def _apply_space_variables(api: HfApi, args: argparse.Namespace) -> None:
    for key in getattr(args, "unset_var", []) or []:
        api.delete_space_variable(args.space_id, key, token=args.token)
    for key, value in _pairs_to_dict(getattr(args, "var", [])).items():
        api.add_space_variable(args.space_id, key, value, token=args.token)


def _render_template(contents: str, replacements: Dict[str, str]) -> str:
    rendered = contents
    for key, value in replacements.items():
        rendered = rendered.replace(f"{{{{{key}}}}}", value)
    return rendered


def render_space_template(
    *,
    template: str,
    output_dir: Path,
    base_image: str,
    app_port: int,
    space_title: str,
    space_emoji: str,
    color_from: str,
    color_to: str,
) -> None:
    if template not in SUPPORTED_TEMPLATES:
        raise ValueError(f"Unsupported template '{template}'. Supported: {sorted(SUPPORTED_TEMPLATES)}")
    template_dir = TEMPLATES_ROOT / template
    output_dir.mkdir(parents=True, exist_ok=True)
    replacements = {
        "BASE_IMAGE": base_image,
        "APP_PORT": str(app_port),
        "SPACE_TITLE": space_title,
        "SPACE_EMOJI": space_emoji,
        "COLOR_FROM": color_from,
        "COLOR_TO": color_to,
    }
    for name in ("README.md.tmpl", "Dockerfile.tmpl"):
        src = template_dir / name
        dest = output_dir / name.replace(".tmpl", "")
        dest.write_text(_render_template(src.read_text(encoding="utf-8"), replacements), encoding="utf-8")
    for name in ("entrypoint.sh", "sync_bucket_prefix.py"):
        shutil.copy2(template_dir / name, output_dir / name)


def _coerce_hardware(value: str | None) -> SpaceHardware | None:
    return SpaceHardware(value) if value else None


def _coerce_storage(value: str | None) -> SpaceStorage | None:
    return SpaceStorage(value) if value else None


def _warn_provisioning(action: str, exc: Exception) -> None:
    print(f"[manage_space] Warning: {action} failed: {exc}", file=sys.stderr)


def _retry_hub_action(action_name: str, fn, retries: int = 3, delay_seconds: float = 5.0):
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            return fn()
        except HfHubHTTPError as exc:
            last_exc = exc
            if attempt == retries:
                break
            print(
                f"[manage_space] {action_name} attempt {attempt}/{retries} failed; retrying in {delay_seconds:.1f}s: {exc}",
                file=sys.stderr,
            )
            time.sleep(delay_seconds)
    if last_exc is not None:
        raise last_exc
    return None


def deploy_space(args: argparse.Namespace) -> int:
    api = HfApi(token=args.token)
    with tempfile.TemporaryDirectory(prefix="hf-space-render-") as tmp:
        rendered_dir = Path(tmp)
        render_space_template(
            template=args.template,
            output_dir=rendered_dir,
            base_image=args.base_image,
            app_port=args.app_port,
            space_title=args.space_title,
            space_emoji=args.space_emoji,
            color_from=args.color_from,
            color_to=args.color_to,
        )
        api.create_repo(
            repo_id=args.space_id,
            repo_type="space",
            space_sdk="docker",
            private=args.private,
            exist_ok=True,
            token=args.token,
            space_hardware=_coerce_hardware(args.hardware),
            space_storage=_coerce_storage(args.storage),
            space_sleep_time=args.sleep_time,
        )
        api.upload_folder(
            repo_id=args.space_id,
            repo_type="space",
            folder_path=rendered_dir,
            commit_message=f"Deploy {args.template} Space scaffold",
            token=args.token,
        )
    _apply_space_variables(api, args)
    for key, value in _pairs_to_dict(args.secret).items():
        api.add_space_secret(args.space_id, key, value, token=args.token)
    _apply_runtime_settings(api, args)
    print(json.dumps({"space_id": args.space_id, "status": "deployed"}, indent=2))
    return 0


def _apply_runtime_settings(api: HfApi, args: argparse.Namespace) -> None:
    if getattr(args, "storage", None):
        try:
            _retry_hub_action(
                "request_space_storage",
                lambda: api.request_space_storage(args.space_id, _coerce_storage(args.storage), token=args.token),
            )
        except HfHubHTTPError as exc:
            _warn_provisioning("request_space_storage", exc)
    if getattr(args, "hardware", None):
        try:
            _retry_hub_action(
                "request_space_hardware",
                lambda: api.request_space_hardware(
                    args.space_id,
                    _coerce_hardware(args.hardware),
                    token=args.token,
                    sleep_time=args.sleep_time,
                ),
            )
        except HfHubHTTPError as exc:
            _warn_provisioning("request_space_hardware", exc)
    elif getattr(args, "sleep_time", None) is not None:
        try:
            _retry_hub_action(
                "set_space_sleep_time",
                lambda: api.set_space_sleep_time(args.space_id, args.sleep_time, token=args.token),
            )
        except HfHubHTTPError as exc:
            _warn_provisioning("set_space_sleep_time", exc)
    if getattr(args, "dev_mode", False):
        try:
            _retry_hub_action(
                "enable_space_dev_mode",
                lambda: api.enable_space_dev_mode(args.space_id, token=args.token),
            )
        except HfHubHTTPError as exc:
            _warn_provisioning("enable_space_dev_mode", exc)
    if getattr(args, "disable_dev_mode", False):
        try:
            _retry_hub_action(
                "disable_space_dev_mode",
                lambda: api.disable_space_dev_mode(args.space_id, token=args.token),
            )
        except HfHubHTTPError as exc:
            _warn_provisioning("disable_space_dev_mode", exc)


def render_space(args: argparse.Namespace) -> int:
    render_space_template(
        template=args.template,
        output_dir=Path(args.output_dir),
        base_image=args.base_image,
        app_port=args.app_port,
        space_title=args.space_title,
        space_emoji=args.space_emoji,
        color_from=args.color_from,
        color_to=args.color_to,
    )
    print(Path(args.output_dir))
    return 0


def pause_space(args: argparse.Namespace) -> int:
    runtime = HfApi(token=args.token).pause_space(args.space_id, token=args.token)
    print(json.dumps({"space_id": args.space_id, "stage": getattr(runtime, "stage", None), "action": "paused"}, indent=2))
    return 0


def restart_space(args: argparse.Namespace) -> int:
    api = HfApi(token=args.token)
    runtime = api.restart_space(args.space_id, token=args.token, factory_reboot=args.factory_reboot)
    if args.dev_mode:
        api.enable_space_dev_mode(args.space_id, token=args.token)
    print(json.dumps({"space_id": args.space_id, "stage": getattr(runtime, "stage", None), "action": "restarted"}, indent=2))
    return 0


def configure_space(args: argparse.Namespace) -> int:
    api = HfApi(token=args.token)
    _apply_space_variables(api, args)
    _apply_runtime_settings(api, args)
    runtime = api.get_space_runtime(args.space_id, token=args.token)
    payload = {
        "space_id": args.space_id,
        "stage": getattr(runtime, "stage", None),
        "hardware": getattr(runtime, "hardware", None),
        "requested_hardware": getattr(runtime, "requested_hardware", None),
        "sleep_time": getattr(runtime, "sleep_time", None),
        "storage": getattr(runtime, "storage", None),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def info_space(args: argparse.Namespace) -> int:
    api = HfApi(token=args.token)
    info = api.space_info(args.space_id, token=args.token)
    runtime = api.get_space_runtime(args.space_id, token=args.token)
    variables = api.get_space_variables(args.space_id, token=args.token)
    payload = {
        "space_id": args.space_id,
        "sha": getattr(info, "sha", None),
        "subdomain": getattr(info, "subdomain", None),
        "sdk": getattr(info, "sdk", None),
        "runtime": {
            "stage": getattr(runtime, "stage", None),
            "hardware": getattr(runtime, "hardware", None),
            "requested_hardware": getattr(runtime, "requested_hardware", None),
            "sleep_time": getattr(runtime, "sleep_time", None),
            "storage": getattr(runtime, "storage", None),
        },
        "variables": {key: getattr(value, "value", None) for key, value in variables.items()},
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--token", default=None, help="HF token override. Defaults to HF CLI auth/env.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    render = subparsers.add_parser("render", help="Render a local Space scaffold")
    render.add_argument("--template", default="vllm_warm", choices=sorted(SUPPORTED_TEMPLATES))
    render.add_argument("--output-dir", required=True)
    render.add_argument("--base-image", required=True)
    render.add_argument("--app-port", type=int, default=7860)
    render.add_argument("--space-title", default="Warm vLLM Eval")
    render.add_argument("--space-emoji", default="🔥")
    render.add_argument("--color-from", default="indigo")
    render.add_argument("--color-to", default="blue")
    render.set_defaults(func=render_space)

    deploy = subparsers.add_parser("deploy", help="Create/update a Space and upload a rendered scaffold")
    deploy.add_argument("--space-id", required=True)
    deploy.add_argument("--template", default="vllm_warm", choices=sorted(SUPPORTED_TEMPLATES))
    deploy.add_argument("--base-image", required=True)
    deploy.add_argument("--app-port", type=int, default=7860)
    deploy.add_argument("--space-title", default="Warm vLLM Eval")
    deploy.add_argument("--space-emoji", default="🔥")
    deploy.add_argument("--color-from", default="indigo")
    deploy.add_argument("--color-to", default="blue")
    deploy.add_argument("--hardware", choices=[item.value for item in SpaceHardware], default=None)
    deploy.add_argument("--storage", choices=[item.value for item in SpaceStorage], default=None)
    deploy.add_argument("--sleep-time", type=int, default=None, help="Sleep time in seconds.")
    deploy.add_argument("--private", action="store_true")
    deploy.add_argument("--dev-mode", action="store_true")
    deploy.add_argument("--disable-dev-mode", action="store_true")
    deploy.add_argument("--var", action="append", type=_parse_key_value, default=[], help="Repeatable KEY=VALUE variable.")
    deploy.add_argument("--unset-var", action="append", default=[], help="Repeatable variable name to delete.")
    deploy.add_argument("--secret", action="append", type=_parse_key_value, default=[], help="Repeatable KEY=VALUE secret.")
    deploy.set_defaults(func=deploy_space)

    pause = subparsers.add_parser("pause", help="Pause a Space")
    pause.add_argument("--space-id", required=True)
    pause.set_defaults(func=pause_space)

    restart = subparsers.add_parser("restart", help="Restart a Space")
    restart.add_argument("--space-id", required=True)
    restart.add_argument("--factory-reboot", action="store_true")
    restart.add_argument("--dev-mode", action="store_true")
    restart.add_argument("--disable-dev-mode", action="store_true")
    restart.set_defaults(func=restart_space)

    configure = subparsers.add_parser("configure", help="Apply hardware/storage/dev-mode settings to an existing Space")
    configure.add_argument("--space-id", required=True)
    configure.add_argument("--hardware", choices=[item.value for item in SpaceHardware], default=None)
    configure.add_argument("--storage", choices=[item.value for item in SpaceStorage], default=None)
    configure.add_argument("--sleep-time", type=int, default=None, help="Sleep time in seconds.")
    configure.add_argument("--dev-mode", action="store_true")
    configure.add_argument("--disable-dev-mode", action="store_true")
    configure.add_argument("--var", action="append", type=_parse_key_value, default=[], help="Repeatable KEY=VALUE variable.")
    configure.add_argument("--unset-var", action="append", default=[], help="Repeatable variable name to delete.")
    configure.set_defaults(func=configure_space)

    info = subparsers.add_parser("info", help="Show Space metadata and runtime state")
    info.add_argument("--space-id", required=True)
    info.set_defaults(func=info_space)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
