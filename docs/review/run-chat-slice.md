# Verified run chat consumer slice

This internal consumer composition joins the existing authenticated `RunsAPI`
artifact path to serving-target preparation and a selected chat runtime.  It
adds no run lookup, persistence, authentication authority, provider registry,
download cache, retry, fallback, or hidden inference prompt.

`open_run_chat` validates its inputs, delegates the exact run to the selected
runtime once, and checks a freshly reconstructed `PreparedRunChat` inside the
acquired context. Wrong result/session types, run substitution and acquisition-time
input mutation still receive the adapter's normal teardown. The selected trusted
adapter owns mandatory current reverification and artifact admission; generic
dispatch is not an authenticator.

`PreparedRunChat` exposes the session, run, canonical five-artifact inventory and
prepared model identity. Its optional `local_model` preserves locally saved-file
access but is absent for remote results. Runtime selection is not authentication:
the consuming host must supply an already-authenticated `RunsAPI`, and a future
provider runtime must honestly preserve the same target boundary.  This slice
does not turn the existing Modal training deployment into inference serving.

Runtime/session ownership ends with the context.  Successfully materialized
artifacts remain under the caller-owned private destination afterward.  The
composition performs no automatic deletion, retry, or local/provider fallback,
and preserves runtime failures and control exceptions without imposing a new
cleanup policy.

## Initial target-first qualification (2026-09-10, source 8032f5e)

`Evaluator.local_run_chat.LocalVLLMRunChatRuntime` implements that seam with
the existing `verified_vllm_chat` context. Its immutable configuration snapshots
reject source/name overrides, nonprimitive startup options and forbidden child
environment names. Range validation remains in the existing startup/generation
boundary before process creation, not before artifact retrieval. The adapter
does not access model paths or validate model files: preparation and immediate
pre-spawn runtime validation retain their separate checks.

The focused integrated selection passed **109 tests in 0.56 seconds** on clean
CPython 3.12.9 / pytest 8.4.2 without Modal or PyTorch. This includes 29 new tests:
eight generic orchestration, four minimal-host consumer acceptance, fourteen
local-adapter and three complete local-chain tests. The complete chain uses
real materialization/preparation, adapter, runtime projection, HTTP client and
chat controller with only process/readiness and HTTP effects faked. Full/LoRA
success and backend failure prove one spawn, no hidden inference request,
credential-free offline child configuration, owned cleanup and retained files.

Independent reviewers checked the orchestration, adapter and consumer acceptance
boundaries and executed their 26 focused tests. The embedding guide's constructor
fields bind against the actual code without opening a session. The 97-member
Modal lock and 66-member offline closure remain CURRENT without regeneration;
no locked production member changed in this slice. CI selects the new tests and
imports both new modules from the installed wheel, but CI has not been run.

The broader established provider-free selection plus these 29 additions passed
**2,288 tests in 265.15 seconds** in the same clean environment. After final
test-only formatting, the 109-test focused selection passed again in **0.55
seconds**. This is not the whole evaluator suite: the pre-existing exploratory
selection failures documented in `verified-vllm-chat-slice.md` remain outside
this selection. The standalone offline-closure generator was checked separately
because the local WSL/Windows worktree layout does not support the Git subprocess
in one of that generator's contract tests; CI retains that contract selection.

The workflow stays embedded in consumer composition; it does not invent a
standalone CLI authority loader. See `../architecture/verified-run-chat.md` for
usage and the exact ownership/lifetime limits. Real GPU loading, inference
quality and a Modal-hosted inference adapter remain unqualified. Local deadlines
are not provider-side cost controls. No provider/credential access, GPU run,
publication, push, merge or consuming-project edit is part of this checkpoint.

## Initial immutable-source package check

Source commit `8032f5eb0255165cbf267fb64fc9ceb742a488f2` was archived and built
offline, without dependency resolution, into `synaptic_tuner-1.1.0-py3-none-any.whl`.
Its size is **2,031,440 bytes** and SHA-256 is
`eb276815360aa725f3a07784560c5296c2d981ee9aad33dbe29e6b5bf4eff0f1`.
The wheel was installed without dependencies into the existing disposable,
non-system-site package-verification environment, replacing only its previous
engine wheel. The archived CI neutral-directory resource/import step passed:
**34 installed engine/evaluator module imports and both resources**, with Modal
and PyTorch absent. Both new module paths resolved from `site-packages`.

Independent read-only ZIP audit verified 715 unique members, a clean ZIP CRC
check, complete RECORD coverage with every declared hash/size correct, and all
707 packaged Python files byte-identical to the immutable archive. The new
run-chat modules and 97/66 source-inventory resources are present and exact;
the retired legacy modules remain absent. This is local package integrity and
import evidence, not a CI run, GPU/model-load test or live provider qualification.

## Runtime-first correction (2026-09-10)

The initial helper materialized local files before selecting a runtime, so it
could not support remote-first Modal preparation. The internal signature is now
`open_run_chat(runs, run, *, runtime)`, and `RunChatRuntime.open(runs, run)` yields
the complete result context. Destination/preparer moved to the local adapter
constructor; no old signature, provider-specific wrapper or compatibility alias
remains. Model/artifact metadata is reconstructed without filesystem I/O, while
the optional factory-issued local model is checked by direct field projection.
Local materialization, serving-target preparation and pre-spawn validation retain
their existing distinct checks. No new downloader or Foundation authority system
is needed.

The focused integrated selection passed **118 tests in 0.54 seconds**, and an
independent reviewer ran the same selection: **118 passed in 0.61 seconds**.
Real local adapter regressions prove failed reverification prevents preparation
and startup and leaves no admitted attempt, while a missing private destination
fails before run reads. A remote-style consumer fixture does current metadata
reverification/outcome reads but never streams artifact bodies or prepares local
weights. This proves the interface permits remote-first execution, not that a
Modal runtime has been implemented or deployed. The earlier wheel evidence above
belongs only to its named immutable source and predates this correction.

The established broader provider-free selection passed **2,297 tests in 271.74
seconds** after this correction (net nine more cases than the previous checkpoint).
The 97-member Modal and 66-member offline source-inventory checks remain CURRENT
without regeneration; canonical skill mirrors and six touched Python files'
format checks pass. The three constructors in the updated embedding example bind
against the real signatures without executing them. Existing exploratory evaluator
baseline exclusions remain unchanged; this is not a whole-repository test claim.
