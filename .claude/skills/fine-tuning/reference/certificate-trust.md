# Certificate trust maintenance

Reviewed 2026-10-22. This is dependency preparation, not cloud qualification.

The eMudhra/emSign incident affects separate trust stores. Updating the standalone
`certifi` distribution does not update pip's vendored copy, uv's Rust bundle, an
OS CA store, Node's bundle, or an already-built cloud image.

- Require `certifi>=2026.10.19` for new engine and HF helper installations. The
  [maintainer advisory](https://github.com/certifi/python-certifi/security/advisories/GHSA-6q48-6c7w-9gw2)
  lists 2026.10.18 as the first fix; 2026.10.19 is the chosen maintenance target.
- Use pip 26.2.2 or newer when pip is installed. The offline Modal wheel builder
  pins 26.2.2 by wheel hash. An environment without pip does not need pip added
  solely to update certifi.
- Use uv 0.13.4 or newer, or explicitly select a verified patched system/custom
  CA store. See [uv's advisory](https://github.com/astral-sh/uv/security/advisories/GHSA-hq85-5cpj-mr62).
- Review system CA/NSS vendor revisions, not just interpreter versions.

Use the existing `scripts/regenerate_modal_runtime_lock.py` after intentionally
changing `requirements/modal-launcher-v1.lock`, then sync skill mirrors. The
result attests source-hash consistency only. It does not qualify a rebuilt image.

Do not edit measured distribution inventories or inference provenance to make
them appear patched. Frozen image digests, measured Python/distribution maps,
and accepted evidence remain historical facts until a fresh image is built and
captured using the existing runtime/profile workflow. Inspect certifi and pip's
vendored certifi in that capture. Keep trainer/Hub-client isolation intact.

The protected HF training-smoke launcher still binds Windows CPython 3.12.7
(`win_amd64`/`cp312`) and pip 24.2. Its canonical
`requirements-hf-training-smoke-installed.json` is the normative installed-package
allowlist consumed by setup/preflight, bound to `requirements-hf-training-smoke.lock`.
It is also an exact-target evidence claim: preserve both files as a pair until a
true capture is reviewed. They retain their historical certifi version; this path
is explicitly unpatched by this maintenance change. The newer engine certifi
floor is incompatible with that historical installed environment.

`scripts/capture_hf_training_smoke_launcher_lock.py` probes the exact Windows
Python/pip target, resolves and downloads hash-bound wheels, installs them offline
in a fresh verifier, and inspects the installed distributions. It emits only
external `.candidate` artifacts; it never promotes them. A manual version/hash
edit or a Linux install is not measured exact-target evidence. Review a real
candidate and its bound lock/installed evidence together before any promotion.
A separate reviewed launcher contract modernization and exact-target recapture
are required before using this path, including pip's vendored CA bundle and the
legacy interpreter. Do not run a paid smoke as evidence for this dependency change.

For an isolated validation environment, install the exact wheels with hashes,
run `pip check`, and inspect the actual roots loaded into an SSL context. Require
`emSign ECC Root CA - G3` to be absent while preserving unaffected emSign roots.
An unaffected root's presence is not evidence of a failed update. A version string
alone does not verify the trust store a client actually loads.

Sources: [PSF pip/virtual-environment guidance](https://blog.python.org/2026/10/python-3151-3149-31317/),
[Ubuntu CA fixes](https://ubuntu.com/security/notices/USN-8944-1),
[Ubuntu NSS fixes](https://ubuntu.com/security/notices/USN-8947-1).
