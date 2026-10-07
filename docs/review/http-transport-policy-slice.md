# HTTP transport policy slice

The existing backend client now accepts three keyword-only transport controls:
`trust_environment`, `allow_redirects`, and `max_response_bytes`. Defaults retain
the historical module-level requests behavior. Verified-local composition can
set `False`, `False`, and a finite bound while also using `api_key=None` and
`retries=0`.

Non-default policies use an owned short-lived `requests.Session`;
`trust_environment=False` disables ambient proxy/netrc trust. Chat, model listing, and health checks share the
same redirect policy. Bounded JSON is streamed through a fixed-size chunk loop
before decoding; responses and sessions close on success, rejection, transport
failure, and interruption. No response or environment value is logged.

The requests timeout applies to socket phases; it is not a hard wall-clock
deadline. Conversation lifetime and request ownership remain the responsibility
of `ChatSession`.

Cleanup attempts both response and session closure. An active exception wins
over teardown errors; absent an active exception, cleanup-origin interrupts
remain interrupts and ordinary cleanup failures use a closed error message.
Generic message mappings are copied to ordinary JSON dictionaries before HTTP
serialization, preserving the existing backend protocol without assuming a
dataset-specific field set.
