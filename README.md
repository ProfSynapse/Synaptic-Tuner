# Modal SFT lifecycle preparation — 2026-10-20

The genuine engine gitlink selects `0d515ae1ae235d0858802228ce4af8dba8fccd03`
on `release/api-v1-completion`. This branch is a consuming host fixture;
it must not be merged into the engine's main branch.

The existing ingestion and preparation configurations retain all 16 authored
documents and produce eight messages/v2 rows, with six training rows and two
validation rows. The semantic dataset digest is
`58f367d391782375b2fb04c31c5b7c50fec804b35e4a7a54c668433d9018971d`.

The inherited `recipe-completion.yaml` is historical fixture configuration.
This preparation selects only `recipe-cancellation.yaml`; no completion,
provider CPU/GPU, cancellation or inference execution is authorized here.

`recipe-cancellation.yaml` is a separate configuration with 32 steps, initial
cancellation opt-in and a capture budget of 32 records. Use the checked-in caller
supervisor only under an exact reviewed admission. A CPU transport proof does
not establish training, pre-completion metric delivery, cancellation or complete
image qualification. This fixture does not promote a logs capability or claim
successful qualification.

Each run requires fresh resources and claims, original signing keys, and current
account, scope and source evidence. Previous attempts and receipts remain
historical; never replay them. Candidate source and local preparation remain
provisional until their independent tests, CI and root review pass.

This host is a provider-free preparation checkpoint. Active runtime fixes may
change the completion source; refresh its exact gitlink, installed wheel and
normal local preparation after final integration before reviewing a live
admission. Private launcher, planning cache, supervisor configuration and unused
execution roots stay in ignored `private/`, separate from all prior attempts.
