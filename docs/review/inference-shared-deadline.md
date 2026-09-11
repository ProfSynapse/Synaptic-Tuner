# Shared inference deadline

Status: locally qualified source `c2efd11d30716f7dde7f2a76e3b68f193f1444a4`,
2026-09-11. Independent source and exact-wheel audits passed.
Engine-only; no cloud/server/GPU execution or EHR changes.

## Boundary

The existing startup, verified-chat composition and session controller now
accept the same optional absolute monotonic `deadline`. This is process-local
timing data, not UTC evidence, runtime authentication or a new grant. Omission
retains separate startup/session behavior. An injected `ChatSession` clock must
use the same domain as its deadline; the verified composition uses the process
monotonic clock throughout.

Startup validates exact finite deadline numbers before filesystem
projection, checks expiry again after projection/port probing, and caps startup
and readiness polling at the earlier local timeout/deadline. Clock bools,
numeric subclasses, non-finite values and conversion overflow are rejected when
a deadline is supplied. Expiry after process acquisition closes only that owner;
unresolved cleanup retains its exact handle. Control-flow interruptions survive.

The verified composition checks the unchanged deadline after readiness and before
yield. Session absolute lifetime is the earlier configured lifetime/deadline;
idle and request limits still apply. Watchdog and request waits use remaining
time. Responses crossing expiry during transport or validation cannot append
history or renew activity. This bounds acceptance and cleanup initiation, not
arbitrary syscall duration, thread scheduling, cleanup completion or billing.

## Qualification

The final selected runtime, HTTP, chat and local materialization regression
passed 387 tests in 1.84 seconds
under CPython 3.12.9 / pytest 8.4.2 with plugins disabled. It includes full/LoRA
real-target/argv/client composition with fake OS/HTTP effects, both with and
without a deadline, across success, error, interruption and timeout. The earlier
370-case run and Sol's 27-case subset are not additive qualification totals.
The separate signed Modal worker full/LoRA and maximum-alias integration cases
passed 3 tests in 12.32 seconds with fake effects, not provider access. Independent
Sol review passed, with 148 focused tests in 1.24 seconds (a subset of the 387).

The existing evaluator/cloud caller lane passed 105 tests in 3.94 seconds on
CPython 3.12.9 / pytest 8.4.2. It overlaps the primary selection and uses existing
system-site evaluator dependencies; it is not fresh CPU CI qualification and its
count must not be added to the primary total.

An offline wheel built from the exact source commit's archive is 2,077,689 bytes,
SHA-256 `85d288ceb9eb2e93af2cd2362c8f69e761a0a8153bdabadf75f73c98190c0e53`.
Independent audit verified 725 unique ZIP entries, all 725 RECORD rows, and byte
parity for all 717 packaged Python files against that archive, with no packaged
tests. The runtime lock's 97 members and offline SFT closure's 66 members
(679,487 payload bytes) remain CURRENT; packaged resources match the archive.
These are existing training inventories, not inference runtime qualification.
Filename-only private-artifact checks passed; this is not a content/secret scan.

The wheel was installed offline without dependencies into the existing isolated
declared-dependencies-only wheel environment. This reused environment is not a
fresh dependency resolution. From a neutral directory with isolated Python,
the checked-in CI probe passed 44 engine-module imports and two lock-resource
checks. Three keyword-only deadline signatures bound exact sentinel arguments
without `**kwargs`; pure validators rejected malformed deadlines. These probes
do not invoke startup, a session or a provider. Modal, Torch, NumPy, pandas and
pytest were absent. Installed source hashes matched:

| Source | SHA-256 |
| --- | --- |
| `Evaluator/vllm_runtime.py` | `61d025aca38b9d9702a4ae167e249032fc0385689b52e1d013166f991cd70573` |
| `Evaluator/chat_session.py` | `a35506ca677d8e6b193f9fdbc24799165bdff89f3b374898a8b8a4d685efee9c` |
| `Evaluator/verified_vllm_chat.py` | `8d9e92153c76398485ca95435b1911fe70948b721607c32e6da0a86cf5598d9e` |

Canonical fine-tuning guidance and its two mirrors passed skill synchronization
checks. Black's single-worker Python 3.12 check passed for all six changed Python
files, and the source diff passed whitespace checks. No cloud object, credential,
model download, server, GPU, EHR checkout, push or merge was involved.

## Remaining work

The Modal bootstrap must freshly authenticate/rederive the original claim and
serving projections and conservatively convert its remaining lifetime into this
clock domain without renewal. Distinct inspected inference image, Python,
dependency and complete bootstrap/source pins are still missing; the current
training lock, SFT closure and source staging cannot be relabeled for inference.
Runtime verification must gate serving and SDK activation. Authenticated remote
access, exact Sandbox ownership/cleanup and live qualification remain unfinished.
