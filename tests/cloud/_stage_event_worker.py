"""Spawned-process target for the cross-process stage-event append test.

It lives in its own module so each spawned child imports only this file and
``shared.cloud_stage_logging``, not the whole experiment-handler test module
(with torch and the tracking stack), which made child start-up slow enough on
a loaded machine for the parent to give up waiting.
"""

from __future__ import annotations

from pathlib import Path


def emit_stage_event_process(log_dir: str, index: int, start, results) -> None:
    try:
        from shared.cloud_stage_logging import CloudStageLogger

        if not start.wait(timeout=120):
            raise TimeoutError("start event was never set")
        CloudStageLogger(Path(log_dir), stage="evaluation").emit(f"process-{index}")
        results.put((0, index, ""))
    except BaseException as exc:
        results.put((1, index, repr(exc)))
