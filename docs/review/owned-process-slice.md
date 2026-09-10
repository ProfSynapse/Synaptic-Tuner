# Owned POSIX process-family slice

`Evaluator.owned_process.OwnedProcessLease` is an internal Linux lease for an exact
`subprocess.Popen` started with an explicit argument vector, working directory,
environment, `shell=False`, and a new POSIX session. Linux procfs identity and
the unreaped group leader anchor the numeric process group against reuse. Cleanup uses
bounded logical budgets and a bounded, streaming procfs census:
`SIGTERM` is followed by observation and then `SIGKILL` when necessary.

The lease never adopts a process or numeric process-group ID. If signaling is
denied, ownership remains visibly unresolved and later `close()` calls do not
retarget the numeric ID. If post-spawn procfs identity establishment fails, the
original exception carries a `cleanup_lease`; callers must retain and close it
after identity observation recovers. Unsupported platforms fail closed.

The containment boundary is the POSIX process group. A descendant that creates
a new session escapes it, and ownership does not survive an OS restart. This
primitive is not GPU cleanup or runtime qualification. The vLLM caller remains
unchanged pending a separate atomic migration.

The deadlines bound userspace iteration and waits, but cannot impose a hard
wall-clock bound on a kernel filesystem or signal syscall that itself stalls.
