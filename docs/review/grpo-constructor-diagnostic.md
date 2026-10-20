# GRPO constructor diagnostic admission — 2026-10-20

## Boundary and current finding

Run `modal-f5d1b7f10ce5d4196ccd13f9`, call
`fc-01M5CTF2FJ3AP4MCQ58646HCYG`, is consumed. Its returned closed stage
`GRPO_TRAINER_CHILD_EXEC_TYPE_TRAINER_SETUP` and empty artifacts do not identify
the constructor cause. Keep its journal, read keys, Volumes, Secrets, image and
application intact. Do not replay or reinterpret it as a successful update.

The published `unsloth_zoo==2026.1.2` wheel (PyPI SHA256
`dc377c5a4adb5c9feffa21d9323b0087188953b49859885134892b594024593f`)
resolves `DEVICE_TYPE` at module import and raises `NotImplementedError` when
Torch reports no accelerator. The matching published `unsloth==2026.1.2`
wheel SHA256 is
`1e2cab339676c99c17b7948f05199d3602119e10b66ce5479d39a54357b26a18`.
These are source inspection facts, not installed-image attestation. An ordinary
CPU-only call cannot reach the Unsloth-patched `GRPOTrainer` constructor, so
there is no useful CPU provider probe for this question. No import-order change
or cause conclusion follows from this observation.

## Checked-in diagnostic

`scripts/qualify_derived_training_image.py release plan-grpo-constructor-input`
derives a four-row constructor input from the existing normal plan, recipe and
original JSONL. The GRPO compiler and snapshot admission check the same
training declaration and prepared identity. Against the retained 0d515 host
files, this provider-free command returned input SHA256
`3c5c179a0aca0e2ab9899c6a11cb6064761ec2df766e76c60899192093c89416`
and 1,873 bytes. It did not import Unsloth or touch a provider.

`release diagnose-grpo-constructor` belongs to the existing derived-image
qualification CLI. Its default is plan-only. Before execution it reopens the
profile, build receipt, candidate, image verification, final runtime capture
and runtime release; derives the same input; verifies the
reviewed admission's source commit and committed probe, helper and CLI bytes; hashes the exact
recipe, plan and dataset; and checks every regular member of an offline model
snapshot against a closed inventory, then recopies and rechecks it into a new
private read-only tree before mounting it. The admission binds the runtime-release
digest and derived image config digest. It refuses absent or changed inputs.

The effectful mode creates a unique consumed claim before `docker create`,
uses an exact image ID with `--pull=never`, a private read-only snapshot mount,
no network, one selected GPU, two CPUs, 8 GiB RAM, a 120-second wait, and an
8 KiB Docker log limit. The probe imports Unsloth before TRL through the actual
runtime dependency function; records the installed patched class/signature
digests; constructs the exact `GRPOConfig`, model/LoRA objects, four formatted
rows, reward wrapper and callback types; and stops after the constructor. It
never calls `.train()` or generation. It returns only fixed stage/category
labels and hashes. The owner checks the terminal container identity and exit,
then removes only its nonce-labelled container. A timeout or uncertain cleanup
leaves the claim consumed and reports no success. The normal training/worker
locks and model/profile pins are unchanged.

The CLI is an operator-side **local Docker GPU diagnostic**, not a new public
training API, a Modal deployment, or a qualification of model updates. The
machine running this checkout has no usable GPU and does not have the exact
image or retained model snapshot locally, so `--execute` has not run. A
container-only construction pass still leaves the original failure unexplained
if its model composition or provider environment differs.

## Smallest next GPU admission for root review

Use one fresh diagnostic claim and a new resource name; never use the old run's
nonce, journal, keys or resources. Bind the exact clean pushed diagnostic source
commit, probe source SHA256, input SHA256 above, recipe/plan/dataset hashes,
`smollm2-135m-grpo-candidate-v1` profile, exact derived image config digest,
runtime release, and a freshly verified inventory of the retained offline
`HuggingFaceTB/SmolLM2-135M-Instruct` revision
`a91318be21aeaf0879874faa161dcb40c68847e9`. Verify the source wheel and
current image from their independent records before executing source; do not
accept a manifest's own claimed hash as that check.

Expose **one L40S GPU**, two CPUs, 8 GiB RAM and a 120-second workload deadline;
disable workload network and credentials, prohibit download, training,
generation and optimizer updates, and retain at most one 8 KiB closed result.
Use the already verified offline snapshot on the execution machine after
source/model admission. Inspect the exact terminal process/container identity
and remove only newly owned resources; preserve an uncertain submission and
all old resources. A returned constructor category narrows a *new* diagnostic
boundary. It does not retroactively establish the cause of the consumed call.

The current CLI can execute this admission only where the verified image and
snapshot are accessible to Docker on the same GPU machine. The retained Modal
snapshot is provider-side. Presenting it to the checked-in Docker command, or
joining the same probe to the normal Modal qualification worker, needs a
separately reviewed source/model custody step. Do not submit a Modal call using
an improvised launcher or copy an unverified snapshot to satisfy this proposal.

## Verification and limits

Focused CPU tests cover the real GRPO compiler projection, four-row and source
mutations, closed constructor results, image/source/input/pin admission
mutations, an owned Docker lifecycle substitute, and actual hung/flooding
subprocess termination. These tests do not attest the installed Unsloth patch,
Docker daemon, mounted model or provider GPU. The pinned published wheel source
only explains why a CPU constructor attempt is uninformative.
