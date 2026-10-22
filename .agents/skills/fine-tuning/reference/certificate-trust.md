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

The protected HF training-smoke launcher still intentionally binds CPython 3.12.7
and pip 24.2. Its certifi lock/allowlist can advance together, but this does not fix
its vendored pip CA bundle or qualify the legacy interpreter. Do not run a paid
smoke as evidence for this dependency change. A separate reviewed launcher
contract update and exact-target recapture are required before using that path.

For an isolated validation environment, install the exact wheels with hashes,
run `pip check`, and inspect the actual roots loaded into an SSL context. Require
`emSign ECC Root CA - G3` to be absent while preserving unaffected emSign roots.
An unaffected root's presence is not evidence of a failed update. A version string
alone does not verify the trust store a client actually loads.

Sources: [PSF pip/virtual-environment guidance](https://blog.python.org/2026/10/python-3151-3149-31317/),
[Ubuntu CA fixes](https://ubuntu.com/security/notices/USN-8944-1),
[Ubuntu NSS fixes](https://ubuntu.com/security/notices/USN-8947-1).
