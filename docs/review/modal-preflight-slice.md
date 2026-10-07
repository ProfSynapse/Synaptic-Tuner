# Modal operational preflight slice

This private candidate adapter combines the provider-free preparation adapter
with current authenticated operational facts. It does not change the public
Modal descriptor or register or qualify any advertised read/lifecycle surface.

The quote has a deliberately one-way identity. Its canonical body binds the
provider, profile, scope, resources, USD maximum cost, validity window, and
Host evidence identity. The quote digest is the domain hash of that body and is
required to equal the digest already retained by the preparation adapter. The
body therefore contains neither its own digest nor a plan fingerprint that
would introduce a cycle. An existing Host evidence authority authenticates it;
payload issuer fields never establish trust.
Quote evidence has a closed five-minute lifetime and maximum age with no future
skew. Composition must refresh it rather than stretching or silently reusing an
expired quote.

A ready result additionally requires fresh and authentic pushed-source and
deployment evidence, the packaged runtime lock, exact execution-source links,
the explicit Modal client and scope, provider-observed deployment equality,
hydrated exact volume identities, and successful hydration of every configured
named Secret with its required-key subset. Secret values are never requested
or returned. Modal's public lookup proves that the requested keys exist, not
that they are the Secret's complete key inventory.

Replay admission remains a composition/factory responsibility and is not
repeated by preflight. Production composition, public registration, and
qualification of the six advertised capability surfaces remain outside this
slice. Those flags do not authorize start; exact preflight and consumer effect
grants are separate requirements.
