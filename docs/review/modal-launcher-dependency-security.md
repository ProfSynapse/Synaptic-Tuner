# Modal launcher dependency security scope

Review date: 2026-10-15. This is a scoped dependency correction, not a declaration
that every dependency, model or runtime image is free of vulnerabilities.

## Launcher correction

The [publisher advisory](https://github.com/aio-libs/multidict/security/advisories/GHSA-54p9-h82j-f925)
identifies a C-extension reference leak in reflected items-view union and
subtraction, affecting multidict 6.7.0 through 6.9.0 and fixed in 6.9.1.
The launcher changes only its 6.7.1 pin to the reviewed 6.9.1 CPython 3.11 Linux
x86_64 wheel, SHA-256
`e8f1e362c9352b50ed120f001046fdbb80810c9d56580f4c3fc13bbe30823387`.
The other 39 dependency lines, Modal 1.5.4, Python 3.11.14 and image/ML pins stay
unchanged. The wheel requires Python 3.10 or later; its conditional dependency
does not apply to Python 3.11. Existing aiohttp and yarl requirements admit it.

A copied Python 3.11.14 environment installs the exact 40 hashed wheels and
passes dependency checks, including the engine's unchanged base requirements.
A 32-item subprocess check uses the actual C extension: the original 6.7.1 leaks
32 references for each affected operator, while 6.9.1 reports zero; an
intersection control reports zero for both. The test runs in the existing
locked-controller CI step. This small regression does not run an exhaustion
attack or claim complete SDK behavior coverage.

## Immutable runtime images remain separate

The captured Qwen SFT inventory contains multidict 6.9.0 and the captured Smol
GRPO inventory contains 6.7.0. Both versions are within the advisory range.
The launcher correction does not rewrite those observations, change their
inventories or patch either image.

Review of the selected first-party trainer/source/transport code found no
multidict import or direct items-view reflected-union/subtraction call. The
exact installed Modal 1.5.4 client and its aiohttp/grpclib sources also showed no
direct affected operator callsite. This is limited static evidence: aliases,
native code and dynamic dependency paths prevent concluding that the operation
is unreachable throughout an image or SDK.

The packaged trainer consumes committed local dataset/model inputs and enforces
`HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`; reporting is disabled for the
selected recipes. Provider model preparation remains a separate networked parent
operation. These controls narrow the supported path but do not fix multidict or
prove a general absence of dependency risk. The image finding remains disclosed
for release review; serving arbitrary HTTP traffic, accepting arbitrary plugins
or claiming the full ML image advisory-free is outside this evidence.

## Test-only pytest finding

The test environment's pytest 8.4.2 is affected by
[CVE-2025-71176](https://github.com/advisories/GHSA-6w46-j5rx-g56g).
The [upstream fix](https://github.com/pytest-dev/pytest/commit/95d8423)
addresses symlink handling while examining and changing permissions on a
predictable temporary root; the advisory identifies pytest 9.0.3 as patched.
This is a real local temporary-directory risk, not a false positive or a
runtime-engine dependency. The base engine and operational launcher do not
require pytest; it is an optional test extra and CI tool.

Supported tests execute reviewed repository inputs in an isolated CI job or
caller-owned private scratch directory. A private, caller-selected `--basetemp`
avoids the shared default temporary-root path for that invocation. Trusted test
inputs alone do not eliminate a hostile local user's symlink race. Existing
tests are not a sandbox for untrusted code or evidence that every historical
pytest invocation used private scratch. No pytest upgrade or advisory suppression
is included in this launcher correction; its test-tool disposition remains
explicit in final release review.
