"""Provider-free automatic preparation of exact packaged Modal build inputs."""
from __future__ import annotations

from contextlib import ExitStack
import hashlib
import json
import os
import platform
from importlib.metadata import version
from pathlib import Path
import secrets
import stat
import tempfile
import time
from urllib.parse import quote, urlsplit
from urllib.request import Request, build_opener, ProxyHandler

from tuner.cloud.derived_training_image import load_profile
from tuner.execution.providers.modal.modal_wheel_builder import _NoRedirect, _download_exact
from tuner.execution.providers.modal.runtime_build import (
    _bounded, plan_modal_build_material, prepare_current_source_wheel, prepare_capture_build_inputs,
    ModalBuildStageFailure,
)
from tuner.runtime.packaged_worker_closure import stable_read

_MAX_WHEEL = 256 * 1024 * 1024
_MAX_METADATA = 2 * 1024 * 1024
_ATTESTED = object()


class ModalLocalPreparationUnavailable(RuntimeError):
    """Closed, actionable local failure before paid/provider preparation."""
    CODES = frozenset({
        'STATE_ROOT_MODE', 'STATE_ROOT_INVALID', 'PROFILE_BINDING', 'BOOTSTRAP_CACHE_INVALID',
        'BOOTSTRAP_METADATA_INVALID', 'BOOTSTRAP_DOWNLOAD_UNAVAILABLE', 'BOOTSTRAP_HASH_INVALID',
        'SOURCE_WHEEL_UNAVAILABLE', 'BUILD_INPUTS_INVALID', 'LOCAL_INPUTS_CHANGED',
        'LAUNCHER_PYTHON', 'LAUNCHER_MODAL',
    })
    def __init__(self, code):
        if code not in self.CODES:
            raise ValueError('invalid_local_preparation_code')
        self.code = code
        super().__init__('modal_local_preparation_unavailable')


def validate_local_launcher():
    if os.name != 'posix' or platform.python_version() != '3.11.14':
        raise ModalLocalPreparationUnavailable('LAUNCHER_PYTHON')
    try:
        if version('modal') != '1.5.4':
            raise ValueError
    except Exception:
        raise ModalLocalPreparationUnavailable('LAUNCHER_MODAL') from None


def private_directory(path: Path, *, create=True) -> Path:
    """Explicit 0700 for every newly created component, independent of umask."""
    if os.name != 'posix' or not path.is_absolute():
        raise ModalLocalPreparationUnavailable('STATE_ROOT_INVALID')
    parent = os.open(path.anchor, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in path.parts[1:]:
            if part in ('.', '..'):
                raise ValueError
            if create:
                try:
                    os.mkdir(part, 0o700, dir_fd=parent)
                except FileExistsError:
                    pass
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
            os.close(parent)
            parent = child
        info = os.fstat(parent)
        if info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) != 0o700:
            raise ModalLocalPreparationUnavailable('STATE_ROOT_MODE')
        return path
    except ModalLocalPreparationUnavailable:
        raise
    except Exception:
        raise ModalLocalPreparationUnavailable('STATE_ROOT_INVALID') from None
    finally:
        os.close(parent)


def training_state_root(*, create=True):
    base = Path(os.environ.get('XDG_STATE_HOME', str(Path.home()/'.local'/'state')))
    return private_directory(base/'synaptic-training', create=create)


def _metadata_bytes(url):
    opener = build_opener(_NoRedirect, ProxyHandler({}))
    with opener.open(Request(url, headers={'User-Agent':'synaptic-modal-preparation/1'}), timeout=20) as response:
        if response.status != 200 or response.url != url:
            raise ValueError
        raw = response.read(_MAX_METADATA + 1)
    if not raw or len(raw) > _MAX_METADATA:
        raise ValueError
    return raw


def _public_artifact(item, *, deadline):
    endpoint = f"https://pypi.org/pypi/{quote(item['distribution'], safe='')}/{quote(item['version'], safe='')}/json"
    try:
        raw = _bounded(lambda:_metadata_bytes(endpoint), deadline=deadline,
                       code='modal_bootstrap_metadata_unavailable')
        document = json.loads(raw)
        info = document['info']
        normalize = lambda name: name.lower().replace('_','-').replace('.','-')
        if normalize(info['name']) != normalize(item['distribution']) or info['version'] != item['version']:
            raise ValueError
        rows = [row for row in document['urls'] if row['filename'] == item['filename']]
        if len(rows) != 1:
            raise ValueError
        row = rows[0]
        url, size = row['url'], row['size']
        parsed = urlsplit(url)
        if (row['packagetype'] != 'bdist_wheel' or row['digests']['sha256'] != item['sha256']
                or type(size) is not int or not 1 <= size <= _MAX_WHEEL
                or parsed.scheme != 'https' or parsed.netloc != 'files.pythonhosted.org'
                or parsed.query or parsed.fragment or Path(parsed.path).name != item['filename']):
            raise ValueError
    except Exception:
        raise ModalLocalPreparationUnavailable('BOOTSTRAP_METADATA_INVALID') from None
    try:
        raw = _bounded(lambda:_download_exact(url,size,opener=build_opener(_NoRedirect,ProxyHandler({}))), deadline=deadline,
                       code='modal_bootstrap_download_unavailable')
    except Exception:
        raise ModalLocalPreparationUnavailable('BOOTSTRAP_DOWNLOAD_UNAVAILABLE') from None
    if hashlib.sha256(raw).hexdigest() != item['sha256']:
        raise ModalLocalPreparationUnavailable('BOOTSTRAP_HASH_INVALID')
    return raw


def _verified(path, digest):
    try:
        raw = stable_read(path, _MAX_WHEEL)
        info = path.lstat()
        if (info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) & 0o077
                or hashlib.sha256(raw).hexdigest() != digest):
            raise ValueError
        return raw
    except Exception:
        raise ModalLocalPreparationUnavailable('BOOTSTRAP_CACHE_INVALID') from None


def bootstrap_paths(profile, root, *, prepare):
    paths = {}
    deadline = time.monotonic()+600
    for item in profile.packaged_runtime['bootstrap']:
        directory = root/'modal-bootstrap'/item['sha256']
        path = directory/item['filename']
        if path.exists() or path.is_symlink():
            private_directory(directory, create=False)
            _verified(path,item['sha256'])
        elif not prepare:
            paths[item['filename']] = None
            continue
        else:
            private_directory(directory)
            raw = _public_artifact(item,deadline=deadline)
            parent = os.open(directory,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
            temporary = '.prepare-'+secrets.token_hex(16)
            try:
                fd = os.open(temporary,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600,dir_fd=parent)
                with os.fdopen(fd,'wb') as stream:
                    stream.write(raw)
                    stream.flush()
                    os.fsync(stream.fileno())
                try:
                    os.link(temporary,path.name,src_dir_fd=parent,dst_dir_fd=parent,follow_symlinks=False)
                except FileExistsError:
                    pass  # Concurrent publication is accepted only after exact revalidation.
                os.unlink(temporary,dir_fd=parent)
                os.fsync(parent)
            finally:
                os.close(parent)
            _verified(path,item['sha256'])
        paths[item['filename']] = path
    return paths


class PreparedModalLocalInputs:
    """Process-local verified files with retained identities, never an execution grant."""
    def __init__(self, profile_path, intent, temporary, layout, *, token, source_root=None):
        if token is not _ATTESTED:
            raise ModalLocalPreparationUnavailable('LOCAL_INPUTS_CHANGED')
        self.profile_path, self.intent, self._temporary, self.layout = profile_path, intent, temporary, layout
        self.source_root = Path(__file__).resolve().parents[2] if source_root is None else Path(source_root).resolve(strict=True)
        self._explicit_source = source_root is not None
        self.staged = Path(temporary.name)
        self.wheel = next(self.staged.glob('synaptic_tuner-*-py3-none-any.whl'))
        self.wheel_sha256 = hashlib.sha256(stable_read(self.wheel,_MAX_WHEEL)).hexdigest()
        self._stack = ExitStack()
        self._files = []
        self._directories = []
        self._closed = False
        self._token = token
        self._snapshot = self._material_snapshot()
        try:
            self._retain_files()
            self._source_fd = os.open(self.source_root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            self._stack.callback(os.close, self._source_fd)
            held = os.fstat(self._source_fd)
            self._source_identity = (held.st_dev, held.st_ino)
        except Exception:
            self._stack.close()
            raise

    def _material_snapshot(self):
        return json.dumps({"intent":self.intent,"source_root":str(self.source_root),"staged":str(self.staged),"wheel":str(self.wheel),
            "wheel_sha256":self.wheel_sha256,"paths":[str(self.layout[1]),str(self.layout[2])],
            "packaged":self.layout[0],"command":self.layout[3],"recipe":self.layout[4],
            "material":self.layout[5]},sort_keys=True,separators=(",",":"),allow_nan=False).encode()

    def _retain_files(self):
        for path in (self.staged.parent, self.staged):
            private_directory(path, create=False)
            fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            self._stack.callback(os.close, fd)
            info = os.fstat(fd)
            self._directories.append((path, fd, info.st_dev, info.st_ino))
        for path in self.staged.iterdir():
            path.chmod(0o600)
            raw = stable_read(path,_MAX_WHEEL)
            fd = os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
            self._stack.callback(os.close,fd)
            info = os.fstat(fd)
            self._files.append((path,fd,info.st_dev,info.st_ino,hashlib.sha256(raw).hexdigest()))

    def revalidate(self, profile_path, expected_intent, *, source_root=None):
        try:
            if (self._closed or self._token is not _ATTESTED or profile_path != self.profile_path
                    or expected_intent != self.intent['intent_digest']
                    or (source_root is not None and Path(source_root).resolve(strict=True) != self.source_root)
                    or plan_modal_build_material(profile_path, **({"source_root": self.source_root} if self._explicit_source else {})) != self.intent
                    or self._material_snapshot() != self._snapshot):
                raise ValueError
            held, named = os.fstat(self._source_fd), self.source_root.lstat()
            if ((held.st_dev, held.st_ino) != self._source_identity
                    or (named.st_dev, named.st_ino) != self._source_identity):
                raise ValueError
            from tuner.training.modal_metrics_qualification import current_clean_commit
            if self._explicit_source:
                from tuner.execution.providers.modal.runtime_build import current_clean_source_commit
                current = current_clean_source_commit(self.source_root)
            else:
                current = current_clean_commit(self.source_root)
            if current != self.intent["engine_source_commit"]:
                raise ValueError
            for path,fd,device,inode in self._directories:
                held,named = os.fstat(fd),path.lstat()
                if ((held.st_dev,held.st_ino) != (device,inode)
                        or (named.st_dev,named.st_ino) != (device,inode)
                        or named.st_uid != os.geteuid() or stat.S_IMODE(named.st_mode) != 0o700):
                    raise ValueError
            for path,fd,device,inode,digest in self._files:
                held,named = os.fstat(fd),path.lstat()
                if ((held.st_dev,held.st_ino) != (device,inode)
                        or (named.st_dev,named.st_ino) != (device,inode)
                        or named.st_uid != os.geteuid() or stat.S_IMODE(named.st_mode) != 0o600
                        or hashlib.sha256(stable_read(path,_MAX_WHEEL)).hexdigest() != digest):
                    raise ValueError
            # No mutable dictionary can change provider material after preparation.
            if self.layout[4]['wheels'] != [{"filename":item['filename'],"sha256":item['sha256']}
                    for item in [self.layout[0]['wheel'],*self.layout[0]['bootstrap']]]:
                raise ValueError
        except Exception:
            raise ModalLocalPreparationUnavailable('LOCAL_INPUTS_CHANGED') from None

    def report(self):
        return {'schema_version':'synaptic-modal-local-preparation/v1','status':'local_inputs_ready',
                'source_commit':self.intent['engine_source_commit'],'intent_digest':self.intent['intent_digest'],
                'source_wheel_sha256':self.wheel_sha256,'bootstrap_wheels':len(self.layout[0]['bootstrap']),
                'provider_effects':False,'execution_authorized':False}

    def close(self):
        if not self._closed:
            self._closed = True
            self._stack.close()
            self._temporary.cleanup()


def prepare_local_inputs(profile_path: Path, *, root: Path, expected_intent: str, source_root: Path | None = None):
    temporary = None
    try:
        validate_local_launcher()
        private_directory(root)
        profile = load_profile(profile_path)
        intent = plan_modal_build_material(profile_path, **({"source_root": source_root} if source_root is not None else {}))
        if intent['intent_digest'] != expected_intent or profile.packaged_runtime is None:
            raise ModalLocalPreparationUnavailable('PROFILE_BINDING')
        paths = bootstrap_paths(profile,root,prepare=True)
        temporary = tempfile.TemporaryDirectory(prefix='modal-local-',dir=root)
        staged = Path(temporary.name)
        try:
            wheel,digest = prepare_current_source_wheel(Path(__file__).resolve().parents[2] if source_root is None else source_root,staged,
                expected_source_commit=intent['engine_source_commit'],builder_cache_root=root/'modal-wheel-builder')
        except Exception:
            raise ModalLocalPreparationUnavailable('SOURCE_WHEEL_UNAVAILABLE') from None
        try:
            layout = prepare_capture_build_inputs(profile,staged,wheel,digest,bootstrap_paths=paths)
        except ModalBuildStageFailure:
            raise ModalLocalPreparationUnavailable('BUILD_INPUTS_INVALID') from None
        prepared = PreparedModalLocalInputs(profile_path,intent,temporary,layout,token=_ATTESTED,source_root=source_root)
        prepared.revalidate(profile_path,expected_intent)
        return prepared
    except ModalLocalPreparationUnavailable:
        if temporary is not None:
            temporary.cleanup()
        raise
    except Exception:
        if temporary is not None:
            temporary.cleanup()
        raise ModalLocalPreparationUnavailable('BUILD_INPUTS_INVALID') from None


def local_readiness(profile_path: Path):
    """Read-only: missing state/cache is unprepared, never an implicit download."""
    base = Path(os.environ.get('XDG_STATE_HOME',str(Path.home()/'.local'/'state')))
    root = base/'synaptic-training'
    try:
        if not root.exists() and not root.is_symlink():
            return {'status':'unprepared','bootstrap_verified':0,'network_used':False}
        private_directory(root,create=False)
        profile = load_profile(profile_path)
        paths = bootstrap_paths(profile,root,prepare=False)
        verified = sum(path is not None for path in paths.values())
        return {'status':'bootstrap_cache_ready' if verified==len(paths) else 'unprepared',
                'bootstrap_verified':verified,'source_wheel_preflight':'execution_or_prepare_local',
                'network_used':False}
    except ModalLocalPreparationUnavailable as failure:
        return {'status':'invalid','diagnostic_code':failure.code,'network_used':False}
