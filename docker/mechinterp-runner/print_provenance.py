"""Print one line of runtime provenance JSON to stdout.

Called by entrypoint.sh at container start. Kept as a standalone script
(rather than inline shell) so any downstream project can invoke it directly,
e.g. to append its output to a run log without re-deriving the same fields.

The image digest is a property of a pulled/tagged reference, not something
baked into the image filesystem, so it cannot be discovered by introspecting
the running container alone. Callers that know the digest (see README.md)
should pass it via IMAGE_DIGEST at `docker run` time. When absent, this
script falls back to the git revision the image was built from, which is
baked in at build time via a Dockerfile ARG/LABEL and is enough to locate the
exact Dockerfile that produced the image, even if the digest itself is
unrecorded.
"""

import json
import os
import sys


def _versions():
    out = {}
    try:
        import torch

        out["torch"] = torch.__version__
        out["cuda_available"] = bool(torch.cuda.is_available())
        out["cuda_version"] = torch.version.cuda
    except Exception as exc:  # pragma: no cover - diagnostic path
        out["torch_error"] = str(exc)
        out["cuda_available"] = False

    try:
        import transformers

        out["transformers"] = transformers.__version__
    except Exception as exc:  # pragma: no cover - diagnostic path
        out["transformers_error"] = str(exc)

    out["python"] = sys.version.split()[0]
    return out


def main():
    record = {
        "event": "mechinterp_runner_provenance",
        "image_digest": os.environ.get("IMAGE_DIGEST", "unknown (pass IMAGE_DIGEST at `docker run`)"),
        "image_git_revision": os.environ.get("MECHINTERP_RUNNER_GIT_REVISION", "unknown"),
    }
    record.update(_versions())
    print(json.dumps(record, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
