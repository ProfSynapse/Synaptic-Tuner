# Modal coordinator retention slice

This disabled internal slice wraps, but does not replace, the existing
Foundation. It authenticates the exact execution grant against the complete
command before consulting a consumer store. It then reconstructs and
authenticates the command binding and publishes it with publish-if-absent and
exact readback semantics. Invalid grants and conflicting readbacks reach
neither Foundation execution nor a provider effect.

Stage execution first rebuilds the semantic eight-member bundle from the
consumer-retained preparation material, recipe registry, log/terminal policy,
and worker closure. The signed stage material is retained and read back before
the unchanged Foundation is called. Submit execution resolves that exact stage
material and the predecessor Foundation record. A missing launch is built from
a freshly authenticated record assessment and retained as one complete launch
envelope; a present launch is fully admitted and reused without reassessment or
resigning. Cancel retains only its complete command binding.

The catalogs and preparation source are consumer-injected durable ports. A
fresh wrapper instance has no local facts, signing keys, database, or recovery
authority. A concurrent publisher may win, but a different winner is rejected
before Foundation execution; a later retry can resolve and authenticate that
winner. A crash before publication may repeat host-only construction, while a
published stage or launch preserves its original tags and assessment across
restart. Foundation remains the sole owner of effect records and mutation
authority; this slice does not serialize records or convert legacy Modal
commands.

The tests are provider-free composition checks. They do not register this
wrapper, call Modal, establish a production database, or qualify live training,
artifact quality, credentials, cost controls, or restart durability of any
particular consumer store.
