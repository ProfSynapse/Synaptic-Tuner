"""Chat directly with a selected model; no training API or run state required.

Run on the execution machine (local GPU or inside a bounded cloud Sandbox).
The consumer owns configuration and the private output directory. This script
does not provision a provider, train, publish, or recover another process.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import time

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Evaluator.chat_session import ChatSessionPolicy
from Evaluator.vllm_runtime import (
    ExplicitNetworkLoRA,
    ExplicitNetworkVLLMSource,
    VLLMStartupSpec,
)
from tuner.inference.model_chat import _source, open_model_chat


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        raise ValueError("chat_arguments_invalid")


def load_configuration(path: Path):
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as selected:
        info = os.fstat(selected.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > 65536:
            raise ValueError("chat_configuration_invalid")
        raw = selected.read(65537)
        after = os.fstat(selected.fileno())
        if (
            any(
                getattr(info, field) != getattr(after, field)
                for field in (
                    "st_dev",
                    "st_ino",
                    "st_size",
                    "st_mtime_ns",
                    "st_ctime_ns",
                )
            )
            or len(raw) != info.st_size
        ):
            raise ValueError("chat_configuration_changed")
    if len(raw) > 65536:
        raise ValueError("chat_configuration_invalid")

    def unique(pairs):
        result = {}
        for name, value in pairs:
            if name in result:
                raise ValueError("chat_configuration_invalid")
            result[name] = value
        return result

    def invalid_constant(value):
        raise ValueError("chat_configuration_invalid")

    document = json.loads(
        raw, object_pairs_hook=unique, parse_constant=invalid_constant
    )
    if type(document) is not dict or set(document) != {
        "model",
        "revision",
        "adapter_path",
        "prompt",
        "max_tokens",
        "lifetime_seconds",
    }:
        raise ValueError("chat_configuration_invalid")
    if (
        type(document["prompt"]) is not str
        or not 1 <= len(document["prompt"].encode("utf-8")) <= 4096
    ):
        raise ValueError("chat_configuration_invalid")
    if (
        type(document["max_tokens"]) is not int
        or not 1 <= document["max_tokens"] <= 1024
    ):
        raise ValueError("chat_configuration_invalid")
    if (
        type(document["lifetime_seconds"]) is not int
        or not 1 <= document["lifetime_seconds"] <= 900
    ):
        raise ValueError("chat_configuration_invalid")
    adapter = document["adapter_path"]
    if adapter is not None:
        if type(adapter) is not str:
            raise ValueError("chat_configuration_invalid")
        selected = Path(adapter)
        if (
            not selected.is_absolute()
            or not selected.is_dir()
            or selected.resolve(strict=True) != selected
        ):
            raise ValueError("chat_configuration_invalid")
        adapter = ExplicitNetworkLoRA("selected-model", selected)
    startup = VLLMStartupSpec(
        source=ExplicitNetworkVLLMSource(
            document["model"], document["revision"], lora=adapter
        ),
        served_model_name="selected-model",
        gpu_memory_utilization=0.5,
        startup_timeout_s=min(600, document["lifetime_seconds"]),
        python_executable=sys.executable,
    )
    _source(startup)
    return document, startup, hashlib.sha256(raw).hexdigest()


def main(argv=None):
    stream = sys.stdout
    directory = None
    try:
        parser = _Parser(description=__doc__)
        parser.add_argument("--configuration", type=Path, required=True)
        parser.add_argument("--output-directory", type=Path)
        parser.add_argument("--check", action="store_true")
        args = parser.parse_args(argv)
        document, startup, digest = load_configuration(args.configuration)
        if args.check:
            stream.write('{"status":"CHAT_INPUTS_CHECKED","training_required":false}\n')
            return 0
        output = args.output_directory
        if (
            output is None
            or not output.is_absolute()
            or output.resolve(strict=True) != output
            or not output.is_dir()
            or output.stat().st_uid != os.geteuid()
            or output.stat().st_mode & 0o077
        ):
            raise ValueError("chat_output_invalid")
        directory = os.open(output, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        retained = os.fstat(directory)
        if retained.st_uid != os.geteuid() or retained.st_mode & 0o077:
            raise ValueError("chat_output_invalid")
        # Exclusive creation relative to the retained directory is the one-shot claim.
        descriptor = os.open(
            "chat-result.jsonl",
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=directory,
        )
        with os.fdopen(descriptor, "w", encoding="utf-8") as saved:
            current = output.stat(follow_symlinks=False)
            if output.resolve(strict=True) != output or (
                current.st_dev,
                current.st_ino,
            ) != (retained.st_dev, retained.st_ino):
                raise ValueError("chat_output_changed")
            result = {
                "status": "CLAIMED",
                "configuration_sha256": digest,
                "training_required": False,
            }
            saved.write(json.dumps(result) + "\n")
            saved.flush()
            os.fsync(saved.fileno())
            deadline = time.monotonic() + document["lifetime_seconds"]
            policy = ChatSessionPolicy(60, 120, document["lifetime_seconds"], 1, 16384)
            environment = {
                "PATH": str(Path(sys.executable).parent) + ":/usr/bin:/bin",
                "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
                "HF_HUB_DISABLE_TELEMETRY": "1",
                "VLLM_NO_USAGE_STATS": "1",
            }
            with (
                open(os.devnull, "w") as sink,
                contextlib.redirect_stdout(sink),
                contextlib.redirect_stderr(sink),
            ):
                with open_model_chat(
                    startup,
                    policy,
                    cwd=Path.cwd(),
                    environment=environment,
                    max_tokens=document["max_tokens"],
                    max_request_bytes=8192,
                    max_response_bytes=16384,
                    deadline=deadline,
                ) as chat:
                    response = chat.chat(document["prompt"]).message
                    if (
                        type(response) is not str
                        or len(response.encode("utf-8")) > 16384
                    ):
                        raise ValueError("chat_response_invalid")
                    result = {
                        "status": "REPLY_SAVED",
                        "configuration_sha256": digest,
                        "model": document["model"],
                        "revision": document["revision"],
                        "response": response,
                        "training_required": False,
                    }
                    saved.write(json.dumps(result, ensure_ascii=False) + "\n")
                    saved.flush()
                    os.fsync(saved.fileno())
            saved.write(
                '{"status":"CHAT_CONTEXT_CLOSED","provider_shutdown_proof":false}\n'
            )
            saved.flush()
            os.fsync(saved.fileno())
        stream.write('{"status":"CHAT_SAVED_AND_CLOSED","training_required":false}\n')
        return 0
    except KeyboardInterrupt:
        stream.write('{"status":"INTERRUPTED","retry_authorized":false}\n')
        return 130
    except Exception:
        stream.write('{"status":"CHAT_FAILED","retry_authorized":false}\n')
        return 1
    finally:
        if directory is not None:
            os.close(directory)


if __name__ == "__main__":
    raise SystemExit(main())
