# Derived Training Image Qualification

Use this workflow when a training recipe needs a reviewed package overlay that
must not be re-resolved at launch time. It is config-first and model-agnostic.
It does not update the protected HF image lock or the Modal runtime lock.

The workflow has four explicit stages:

1. `plan` validates an exact digest-pinned official Unsloth base, exact index
   versions, and full 40-hex VCS commits. It renders the deterministic
   Dockerfile without calling Docker or the network.
2. `build --execute` uses local Docker Buildx with `--pull`, `--no-cache`,
   `--load`, and maximum BuildKit provenance output. Its review-only receipt
   retains bounded canonical BuildKit metadata and records the final
   OCI manifest digest, image-config digest, base digest, Dockerfile digest,
   and BuildKit metadata digest. Omitting `--execute` prints the build plan.
3. `capture --execute` rechecks the local image identity and labels, then runs
   a network-disabled metadata probe. The candidate records every configured
   installed version plus PEP 610 `direct_url.json` evidence for VCS packages.
   Omitting `--execute` prints the capture plan.
4. `verify` is offline. It rejects version, URL, commit, base, config, or final
   OCI identity mismatches and exclusively writes a separate diagnostic report
   with status `DIAGNOSTIC_PASS`. The report is review evidence, not launch
   authority, and verification never changes a provider runtime lock.

## Qwen 3.5 4B SFT: verified named runtime profile

The Qwen-specific derived-image overlay framing above is historical and
superseded for the supported 32K prompt/completion SFT path. Use
`Trainers/recipes/qwen35_4b_32k_prompt_completion.yaml`, which selects
`qwen35-sft-v1`. That profile admits only
`Qwen/Qwen3.5-4B` at revision
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` for SFT, and binds an immutable
image plus the complete captured installed-distribution inventory and runtime
facts. The inventory, rather than a short manually chosen package list, is the
runtime's transitive dependency lock.

Inspect the resolved profile and lineage inputs without Docker effects, then
run the normal dry smoke:

```bash
python tuner.py local-run \
  --job-config Trainers/recipes/qwen35_4b_32k_prompt_completion.yaml \
  --json

python tuner.py local-run \
  --job-config Trainers/recipes/qwen35_4b_32k_prompt_completion.yaml \
  --yes
```

Do not use `unsloth/unsloth:latest`, `setup.pip`, or manual dependency overlays
for this verified Qwen path. `qwen35-sft-v1` does not qualify Qwen 3.5 0.8B,
2B, 9B, another revision, or another method; each exact model/revision/method
requires its own successful smoke and explicitly admitting immutable profile.

## Recipe safety contract

A recipe can opt into effectful enforcement without changing other local-run
recipes:

```yaml
job:
  image: sha256:<real-local-image-config-digest>
  pull_policy: never
  image_qualification:
    required: true
    profile: Trainers/image_profiles/my-profile.yaml
    verification_report: private/image-qualifications/my-verification.json
```

Provider-free `local-run --json` compilation remains available if `job.image`
or the report is absent. Offline validation occurs before confirmation. After
confirmation, and before project/artifact/cache preparation or the normal pull
and run flow, the launcher uses a fresh empty Docker config and identity-stable
absolute Docker executable to inspect the immutable reference, probe the exact
returned image-config digest with networking disabled and Python `-I -S`, and
re-inspect that digest. Installed versions and PEP 610 VCS evidence must match
the profile and report. The launch reference must equal either the captured
final OCI reference or local image-config digest; tags are never accepted.
Qualified recipes must not declare `setup.pip`; launch-time package mutation
would invalidate the exact versions and VCS provenance that were just probed.
Do not put a fake digest or runnable-looking placeholder into a checked-in
template. Omit `job.image` until a real built and captured reference exists.

The diagnostic report is self-contained local review evidence. It is not
unforgeable, a signature, provider identity, GPU qualification, performance
result, training authorization, or publication receipt, and it cannot unlock
an effectful run without the fresh Docker checks above. A later Modal image
change still requires the independent Modal lock and qualification process.

## Trust boundary

This workflow is an honest-local-operator consistency check. It detects stale
tags, changed image identities, package drift, mismatched reports, and a Docker
client/config swap during the checked run. The candidate and diagnostic report
carry the build-receipt and Dockerfile digests plus bounded canonical BuildKit
metadata and its digest so a reviewer can trace the local evidence chain.

It does **not** cryptographically prove the official base, Dockerfile, BuildKit
execution, or installed-package metadata against a malicious image, compromised
Docker daemon, or hostile local administrator. Image labels and Python dist-info
are image-controlled. Release or Modal promotion therefore remains blocked
until a separate reviewer-controlled or CI provenance path binds protected build
evidence to the published registry digest. Do not invent local signatures or
treat `DIAGNOSTIC_PASS` or `LIVE_DOCKER_VERIFIED` as release authority.
