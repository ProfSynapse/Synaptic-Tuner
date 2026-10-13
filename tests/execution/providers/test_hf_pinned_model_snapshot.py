"""Actual pinned Hub model preparation with entirely controlled HTTP transport."""

import hashlib
import os

import pytest

sdk = pytest.importorskip("huggingface_hub")
if sdk.__version__ != "1.32.0":
    pytest.skip("dedicated base-image Hub1.32 compatibility lane", allow_module_level=True)
import httpx

from huggingface_hub import HfApi, __version__, snapshot_download
from huggingface_hub.utils import close_session, set_client_factory
from tuner.execution.providers.modal.model_snapshot import (
    _bind_hub_api,
    prepare_model_snapshot,
)


@pytest.mark.parametrize("umask", [0o022, 0o002])
def test_actual_hub_132_private_anonymous_snapshot(tmp_path, umask):
    assert __version__ == "1.32.0"
    _bind_hub_api(HfApi, snapshot_download)
    revision = "a" * 40
    files = {"config.json": b'{"model_type":"fixture"}',
             "nested/model.safetensors": b"synthetic weights"}
    git = {name: hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
           for name, raw in files.items()}
    siblings = [{"rfilename": name, "size": len(raw), "blobId": git[name],
                 **({"lfs": {"sha256": hashlib.sha256(raw).hexdigest(),
                             "size": len(raw), "pointerSize": 128}}
                    if name.endswith("safetensors") else {})}
                for name, raw in files.items()]
    requests = []

    def serve(request):
        requests.append(request)
        assert "authorization" not in request.headers
        assert request.url.host == "huggingface.co"
        path = request.url.path
        if path == f"/api/models/fixture/tiny/revision/{revision}":
            assert request.url.params["blobs"] == "true"
            return httpx.Response(200, json={"id": "fixture/tiny", "sha": revision,
                                            "siblings": siblings})
        if path == f"/api/models/fixture/tiny/tree/{revision}":
            return httpx.Response(200, json=[{"type": "file", "path": name,
                                             "size": len(raw), "oid": git[name]}
                                            for name, raw in files.items()])
        prefix = f"/fixture/tiny/resolve/{revision}/"
        assert path.startswith(prefix), path
        name = path[len(prefix):]
        raw = files[name]
        headers = {"X-Repo-Commit": revision, "ETag": git[name],
                   "Content-Length": str(len(raw))}
        return httpx.Response(200, headers=headers,
                              content=b"" if request.method == "HEAD" else raw)

    close_session()
    set_client_factory(lambda: httpx.Client(transport=httpx.MockTransport(serve)))
    roots = {name: tmp_path / name for name in
             ("persistent_root", "destination_root", "scratch_root")}
    for root in roots.values():
        root.mkdir(mode=0o700)
    previous = os.umask(umask)
    try:
        destination = prepare_model_snapshot(model_ref="fixture/tiny", revision=revision,
                                             token=False, **roots)
        assert {p.relative_to(destination).as_posix(): p.read_bytes()
                for p in destination.rglob("*") if p.is_file()} == files
        assert all(p.stat().st_mode & 0o077 == 0 for p in destination.rglob("*")
                   if p.is_dir())
        assert not (destination / ".cache").exists()
        assert {r.method for r in requests} == {"GET", "HEAD"}
    finally:
        os.umask(previous)
        close_session()
        set_client_factory(httpx.Client)
