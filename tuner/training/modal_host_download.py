"""Save an authenticated public-API artifact into a private POSIX directory."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import secrets
import stat

from synaptic_tuner.api.v1.results import TrainingRunRef
from synaptic_tuner.api.v1.runs_facade import RunArtifactRequest


class ModalArtifactDownloadUnavailable(RuntimeError):
    """A fixed diagnostic; provider and filesystem exception text is suppressed."""


def download_verified_modal_artifact(
    runs: object, run: TrainingRunRef, *, role: str, output_root: Path,
    maximum_bytes: int = 256 * 1024 * 1024,
) -> Path:
    """Stream one verified role without following links or overwriting a file.

    The returned path is usable only after its length and digest match the
    authenticated artifact metadata.  Failure leaves a private partial file
    for inspection; it never publishes a misleading completed filename.
    """
    if os.name != "posix":
        raise ModalArtifactDownloadUnavailable("modal_artifact_download_posix_required")
    if (type(run) is not TrainingRunRef or role not in {
        "final_model", "tokenizer", "training_lineage", "training_metrics",
        "workload_record",
    } or type(output_root) is not type(Path()) or not output_root.is_absolute()
            or type(maximum_bytes) is not int or maximum_bytes < 1):
        raise ModalArtifactDownloadUnavailable("modal_artifact_download_invalid")
    directory = -1
    partial = None
    try:
        output_root.mkdir(mode=0o700, exist_ok=True)
        root_info = output_root.lstat()
        if (not stat.S_ISDIR(root_info.st_mode) or stat.S_ISLNK(root_info.st_mode)
                or root_info.st_uid != os.getuid()
                or stat.S_IMODE(root_info.st_mode) & 0o077):
            raise ValueError
        directory = os.open(output_root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        opened = os.fstat(directory)
        if (opened.st_dev, opened.st_ino) != (root_info.st_dev, root_info.st_ino):
            raise ValueError
        stream = runs.artifacts(RunArtifactRequest(run, role, maximum_bytes))
        artifact = stream.artifact
        if artifact.role != role or artifact.size_bytes > maximum_bytes:
            raise ValueError
        partial = f".{role}.{secrets.token_hex(16)}.partial"
        fd = os.open(partial, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                     0o600, dir_fd=directory)
        digest = hashlib.sha256()
        size = 0
        try:
            with os.fdopen(fd, "wb", closefd=True) as handle:
                for chunk in stream.iter_bytes():
                    if type(chunk) is not bytes or not chunk:
                        raise ValueError
                    size += len(chunk)
                    if size > artifact.size_bytes or size > maximum_bytes:
                        raise ValueError
                    digest.update(chunk)
                    handle.write(chunk)
                handle.flush()
                os.fsync(handle.fileno())
        except BaseException:
            raise
        if size != artifact.size_bytes or digest.hexdigest() != artifact.sha256:
            raise ValueError
        completed = f"{role}.artifact"
        os.link(partial, completed, src_dir_fd=directory, dst_dir_fd=directory,
                follow_symlinks=False)
        os.unlink(partial, dir_fd=directory)
        os.fsync(directory)
        return output_root / completed
    except Exception:
        raise ModalArtifactDownloadUnavailable("modal_artifact_download_failed") from None
    finally:
        if directory >= 0:
            os.close(directory)
