# Modal worker dispatch codec slice

Status: private provider-free codec implemented; not registered with a Modal
deployment and not a public API or provider transport.

`coordinator_dispatch.py` encodes one canonical
`synaptic.modal-worker-dispatch/v1` byte argument for a future provider SDK
spawn. It carries the complete immutable `ModalWorkerLaunchExpectation`, the
bounded canonical host launch claim, and its canonical Base64 tag. Exact fields,
types, bounds, canonical JSON/Base64, claim projections, and reconstruction are
checked on every parse. Unknown fields and explicit credential-value field names
are refused; key references and requirement digests remain allowed.

The builder first serializes the exact caller expectation to plain canonical
JSON fields and reconstructs a fresh exact expectation. It rejects subclasses,
malformed caller mutation, reconstruction differences, and bundle sizes above
the same policy bound used by worker parsing; only the rebuilt value is used for
claim comparison and persisted output.

The expectation contains exact submit/deployment bytes, scope, app/function and
executor targets, Volume identities/names, key reference, and stage claim and
bundle digest/size commitments. The builder confirms that the embedded launch
claim projects to those independently supplied host values. It does not verify
the launch MAC or rederive Foundation evidence: the root composition must call
host launch admission first, then construct this codec from that admitted launch
and its trusted dispatch expectation.

These bytes are trusted host dispatch only when delivered as the one argument
of an authenticated provider invocation prepared through that host path. The
parser alone establishes canonical structure, not authenticity, freshness,
provider execution, Foundation authority, or permission to train. No SDK,
filesystem, process, catalog, grant service, signing key, credential value,
legacy Modal lifecycle, or runtime operation is included here.
