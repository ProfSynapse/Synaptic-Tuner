# Native owned-process CPU qualification

`native_cpu_process.py` is a private, provider-free one-shot probe of the same
`NativeOwnedProcess` and stdlib guardian used by native SFT. The fixed worker
uses only `os`, `stat`, and `sys`: it reads the checked-in exact request and
creates one fixed owner-only result. It loads no model, tokenizer, ML package,
dataset, credential, or GPU and makes no network request. This observes a
normal short worker exit; it does not grant training or public cancellation.

Copy [`native_cpu_process.example.json`](native_cpu_process.example.json) to a
fresh owner-only 0600 config outside the clean source checkout. Replace every
placeholder with reviewed real paths and hashes, and write canonical JSON with
no trailing newline. `engine_root` must be the CLI's clean checkout, and its
local `HEAD`, selected branch ref, and origin URL must match the config. The
reviewer separately verifies that exact commit is pushed; this CLI performs
no network lookup. `launcher` is the actual private venv `bin/python` path;
`executable` is its owner-owned regular target. They must resolve exactly and
the target bytes must match `executable_sha256`. The guardian, worker, and
request are selected by fixed repository paths and exact config hashes.

The config also binds an unused `state_root` under an existing owner-private
0700 parent, a unique attempt, the actual platform/Python version, and finite
limits. `timeout_seconds` is at most 30 and termination grace at most 5;
output is at most 4 KiB per worker stream and file size at most 1 MiB. The
reviewed first selection uses 20 seconds plus 5 seconds grace. The core
guardian transport adds its own bounded response/wait allowance, so the
caller must separately bound its SSH session and preserve uncertainty if that
outer boundary fires.

From the selected clean source, after root review of the exact config:

```sh
"$private_python" -I -B examples/training_qualification/native_cpu_process.py inspect "$private_config"
# Run once only after a separate root admission of this CPU-only effect:
"$private_python" -I -B examples/training_qualification/native_cpu_process.py run "$private_config"
```

`inspect` only checks local source and input bytes, then emits a closed
selection digest; it does not create state. `run` creates the state root,
`request.json`, `workflow.sqlite3`, a create-only `worker-result.json`, and on
normal verified exit a create-only `qualification-result.json`. It checks the
exact retained claim/spawn/outcome via the existing SQLite record port and
checks the complete result membership. Guardian failures retain only a closed
diagnostic record. A failure after root creation consumes that root; never
reset claims, adopt a partial root, or rerun it. CLI output has no exception,
path, argv, environment, or worker text. The selected config path tells the
operator where to inspect the private records and receipt.

The first Darwin CPU observation retained claim/spawn and a guardian exit in
the broad termination phase (code 126), but no process outcome. Its worker
result file is not a supervised-exit receipt, and that state stays consumed.
The guardian now distinguishes future termination failures with closed exit
codes 128 (SIGTERM), 129 (grace observation), 130 (SIGKILL), and 131 (wait).
The next CPU observation stopped at SIGTERM code 128, still without an errno.
On a fresh attempt only, codes 132-135 classify SIGTERM and 136-139 classify
SIGKILL as EPERM, other OS error, missing API, or non-OS failure respectively.
An EPERM result may reflect an empty group or a denied live member; it does not
authorize skipping cleanup or reporting a supervised outcome. These codes do
not reveal exception details or prove a historical cause. Any
fresh CPU attempt needs its own reviewed clean source, config, unused root and
one-use admission; the existing `inspect` and `run` actions remain the only
consumer commands.
