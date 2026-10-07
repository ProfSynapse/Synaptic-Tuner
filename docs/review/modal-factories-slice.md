# Internal Modal coordinator factories

This slice packages already-constructed Foundation-native Modal components for
`LazyProviderRegistryV2`. Registration retains the preparation adapter's exact
provider descriptor, execution descriptor, profile digest, account, and
namespace. Executor and reconciliation objects must match every retained field.
The existing resolver classes perform the registry-native resolved-value
minting.

The reader factory accepts only an exact `ModalCoordinatorRunReader` and checks
the complete reader factory request before returning
`ResolvedProviderReaderV1`. Exact reader type is not evidence of provider
authentication: composition owns its catalog, Foundation authentication,
evidence authority, and transport, and the reader reauthenticates each request.

Construction, registration, listing, and factory inspection perform no SDK or
provider calls. The retained public descriptor continues to advertise its six
read, lifecycle, artifact-streaming, and cost-quote flags as false. Those flags
do not govern authenticated start, and factory construction grants no effect
authority. Public registration and qualification of those advertised surfaces
remain an atomic-cutover responsibility outside this internal slice.
