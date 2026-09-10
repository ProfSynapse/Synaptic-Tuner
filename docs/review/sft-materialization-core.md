# Shared post-admission SFT materialization

Engineering slice, 2026-09-10: implementation, independent review, combined
provider-free qualification and immutable package qualification complete locally.
This is engine-only code reuse, not a new public retrieval API or Modal deployment.

The existing public `materialize_verified_sft_model` retains exact RunsAPI
verification, successful outcome and canonical five-artifact admission before
artifact reads or writes. A private core accepts an exact owned run/artifact
projection and an explicit byte reader only after its caller authenticates the
source. That reader is transport, not a grant or a claim that source paths are
safe. Stream metadata, exact hash/size, bounded raw archives, semantic validation,
exclusive writes, retained filesystem identities and proven-owned cleanup remain
part of materialization. No arbitrary-path receipt constructor is added.

A future Modal worker must authenticate its launch and native mounted inventory
before invoking this core; that adapter is not implemented by this slice. Its
resulting filesystem receipt and `ServingTarget` belong to the Modal machine,
where the existing pinned-base preparer and vLLM process controller can run
unchanged. Operator-local filesystem receipts are never remote proof.

Reader owns the single production-file refactor, conformance owns a new boundary
test module, effects independently reviews the implementation, and the lead
integrates and qualifies it. Existing hostile archive and cleanup tests remain
required. No provider calls, credentials, cloud objects, dependency upgrades,
pushes, merges or EHR changes are authorized by this implementation slice.

## Boundary review and tests

The private helper requires a borrowed destination-root descriptor in addition
to the canonical path. It checks their identities before writes and never
closes the caller's descriptor. The public wrapper retains its original type
signature and imports; no alternate signature, public receipt constructor,
parallel vLLM controller or compatibility layer was introduced.

The byte reader receives reconstructed request values. Correspondence is checked
before writing, during each yielded chunk and at EOF; original caller inputs
are also checked between artifact reads. Exact byte hashes and semantic archive
validation remain mandatory even after source admission. Existing raw archive,
descriptor-relative extraction and conservative cleanup tests are unchanged.

The lead passed 175 focused tests before the final boundary additions, then
179 after the first four additions. The final new module has 29 cases and passed
independently in 0.19 seconds and on the lead in 0.16 seconds. A measured
pre-correction run of the caller-run mutation regression failed because outcome
was called after reverification changed the caller's original run reference.
The corrected wrapper checks an independent snapshot immediately after each
admission call and around stream acquisition; the final module passes.
This is distinct from mutation of RunsAPI's presented operation argument, which
the existing facade already rejected. Concurrent hostile mutation after the
last callback can still cause a final denial with a private orphan; no new
unproven deletion or cleanup authority is added to address that case.

Both training source locks remain CURRENT (97 Modal pins; 66 offline trainer
members). They do not include this inference materializer and are not inference
runtime qualification. The corrected full selection passed **2,368 tests in
332.87 seconds**, on clean CPython 3.12.9 / pytest 8.4.2 without Modal or PyTorch.
This is the defined coordinator/Foundation/provider/training/runtime/inference
selection, not every repository test. The immutable installed package check is
recorded below.

Launch correction: the first combined attempt reported 2,367 passes and one
failure in 344.50 seconds. The lead had invoked embedded pytest through Python
stdin. The FIFO test's multiprocessing child tried to reopen a nonexistent
`<stdin>` path and failed before entering the runtime under test. Its exact
82-test module passed separately in 3.59 seconds using the established
`python -c` launch. The full frozen selection then passed with that same
launch form; this required no repository-code change and is not evidence of a
flaky runtime test.

## Immutable package qualification

Exact source commit `ad8eafd497365231347a0ff7aa19572192943ce7` was archived and
built offline without dependency downloads. The wheel is 2,042,313 bytes with
SHA-256
`f587c232c6d186ec40056024759ef13e841327a75557d565cf9785f923d312fa`.
It passed the checked-in neutral-directory CI snippet's 36 installed
engine/Evaluator imports and two resource checks, plus public/private signature
binding and public type-hint resolution. The private root descriptor remains
required; the public signature is unchanged. Modal and PyTorch were absent from
the disposable package-check environment; the trainer environment was untouched.

Independent audit passed all 717 unique ZIP/RECORD entries and verified all
709 Python members against the immutable source archive. The materializer's
SHA-256 is
`ba59341ef0708d3c40a2e7f0f5facd9317aa2a997ff768dd8aefcbb163883f22`.
The source commit contains exactly the nine intended paths. No unexpected
credential/private artifact filenames were found; this is not a content-level
secret scan. No CI dispatch, provider execution, model-load or live-chat
qualification is implied.
