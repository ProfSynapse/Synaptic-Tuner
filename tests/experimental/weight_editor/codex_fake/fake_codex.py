"""Synthetic stand-in for the codex CLI. Never contacts a model or reads credentials.

Copied next to a ``scenario.json`` by the tests and launched through a generated
shebang. Each ``exec`` invocation consumes the next scenario turn and appends its
argv/stdin/env names to ``calls.jsonl`` for assertions.
"""

import base64
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def main() -> int:
    scen = json.load(open(os.path.join(HERE, "scenario.json")))
    argv = sys.argv[1:]
    if argv == ["--version"]:
        print(scen.get("version", "codex-cli 0.166.0"))
        return 0
    if argv[:2] == ["login", "status"]:
        print(scen.get("login", "Logged in using ChatGPT"))
        return 0
    if not argv or argv[0] != "exec":
        print("fake codex: unsupported invocation", file=sys.stderr)
        return 2
    stdin = sys.stdin.buffer.read()
    calls = os.path.join(HERE, "calls.jsonl")
    n = sum(1 for _ in open(calls)) if os.path.exists(calls) else 0
    with open(calls, "a") as fh:
        fh.write(json.dumps({"argv": argv, "stdin_b64": base64.b64encode(stdin).decode(), "cwd": os.getcwd(),
                             "env_names": sorted(os.environ)}) + "\n")
    turn = scen["turns"][n]
    if turn.get("spawn_child"):
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)"])
        with open(os.path.join(HERE, "pids.json"), "w") as fh:
            json.dump({"parent": os.getpid(), "child": child.pid}, fh)
    if "stderr_b64" in turn:
        sys.stderr.buffer.write(base64.b64decode(turn["stderr_b64"]))
        sys.stderr.buffer.flush()
    out = base64.b64decode(turn["stdout_b64"]) if "stdout_b64" in turn else b""
    for i in range(0, len(out), turn.get("chunk", 1 << 20)):  # arbitrary pipe chunking
        sys.stdout.buffer.write(out[i:i + turn.get("chunk", 1 << 20)])
        sys.stdout.buffer.flush()
    if turn.get("write_into_cwd"):
        open("native-write.txt", "w").write("x")
    if turn.get("sleep"):
        time.sleep(turn["sleep"])
    return int(turn.get("exit", 0))


if __name__ == "__main__":
    sys.exit(main())
