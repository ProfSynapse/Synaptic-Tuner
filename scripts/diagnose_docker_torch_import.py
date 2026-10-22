"""Bounded, credential-free Torch import diagnosis for an exact packaged image.

This is a diagnostic, never training or qualification authority.  It retains
the new container and writes only closed results to a fresh private directory.
"""

from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import json
import os
import re
import secrets
import selectors
import shutil
import signal
import stat
import subprocess
import sys
import time
from pathlib import Path


SCHEMA = "synaptic-docker-torch-import-diagnostic/v1"
_SHA = re.compile(r"sha256:[0-9a-f]{64}\Z")
_CID = re.compile(r"[0-9a-f]{64}\Z")
_CATEGORIES = frozenset({"PASS", "MODULE_NOT_FOUND", "IMPORT", "OS", "RUNTIME", "OTHER"})
_REASONS = frozenset({"DAEMON_ID_INVALID", "INPUT_INVALID", "DOCKER_INVALID",
    "DOCKER_CHANGED", "DOCKER_COMMAND_REJECTED", "DIAGNOSTIC_OUTPUT_INVALID",
    "RELEASE_INVALID", "RELEASE_HASH_INVALID", "OUTPUT_INVALID", "SOURCE_INVALID", "DAEMON_INVALID", "IMAGE_INVALID",
    "CHILD_OUTPUT_INVALID", "CONTAINER_ID_INVALID", "CONTAINER_IMAGE_INVALID",
    "IMAGE_CHANGED", "SOURCE_IDENTITY_INVALID", "GPU_SCOPE_INVALID", "CREATE_UNCERTAIN", "START_UNCERTAIN",
    "DEADLINE_UNCERTAIN", "DEADLINE_STOPPED", "STOP_UNCONFIRMED", "CONTAINER_STATE_INVALID"})
_CLI_BUDGET = 8192
_RUN_SECONDS = 120
_HIDDEN_GPU_ENV = ",'CUDA_VISIBLE_DEVICES':'','NVIDIA_VISIBLE_DEVICES':'void'"
_R2_ROOTS = {"engine": "/source/engine", "project": "/source/project",
    "artifacts": "/artifacts/run/artifacts", "state": "/artifacts/run/state",
    "tracking": "/artifacts/run/tracking", "cache": "/artifacts/run/cache",
    "tmp": "/artifacts/run/tmp"}
_R2_ENV_KEYS = frozenset({"PATH", "PYTHONPATH", "PYTHONNOUSERSITE",
    "PYTHONSAFEPATH", "HF_HOME", "HF_HUB_OFFLINE", "TRANSFORMERS_CACHE",
    "TRANSFORMERS_OFFLINE", "WANDB_DISABLED", "SYNAPTIC_ENGINE_ROOT",
    "SYNAPTIC_PROJECT_ROOT", "SYNAPTIC_ARTIFACT_ROOT", "SYNAPTIC_STATE_ROOT",
    "SYNAPTIC_TRACKING_ROOT", "SYNAPTIC_CACHE_ROOT", "SYNAPTIC_TMP_ROOT"})
_ROOT_ENV = {name: "SYNAPTIC_" + name.upper() + "_ROOT" for name in _R2_ROOTS}
_ROOT_ENV["artifacts"] = "SYNAPTIC_ARTIFACT_ROOT"
_DIRECT_STOP = b"raise SystemExit(42)\n"
_DIRECT_TRAINER = "Trainers/sft/train_sft.py"
_INSPECT = "{{json .Id}}\n{{json .Name}}\n{{json .Image}}\n{{json .Config.Labels}}\n{{json .State.Status}}\n{{json .State.ExitCode}}"

# This fixed program never formats an exception, inspects its attributes, or
# imports an engine/trainer module.  The exact packaged interpreter imports
# Torch under the isolated, offline child environment.
_PROBE = """import json,os,sys
report_fd=os.dup(1)
null_fd=os.open('/dev/null',os.O_WRONLY)
os.dup2(null_fd,1);os.dup2(null_fd,2);os.close(null_fd)
os.environ.clear()
os.environ.update({'PATH':os.path.dirname(sys.executable)+':/usr/local/bin:/usr/bin:/bin','HOME':'/tmp','TMPDIR':'/tmp','LC_ALL':'C.UTF-8','PYTHONNOUSERSITE':'1','PYTHONSAFEPATH':'1','PYTHONDONTWRITEBYTECODE':'1','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','WANDB_DISABLED':'true','CUDA_VISIBLE_DEVICES':'','NVIDIA_VISIBLE_DEVICES':'void'})
try:
 import torch
 category='PASS'
except ModuleNotFoundError:
 category='MODULE_NOT_FOUND'
except ImportError:
 category='IMPORT'
except OSError:
 category='OS'
except RuntimeError:
 category='RUNTIME'
except BaseException:
 category='OTHER'
os.write(report_fd,(json.dumps({'schema_version':'synaptic-docker-torch-import-child/v1','category':category},sort_keys=True,separators=(',',':'))+'\\n').encode('ascii'))
os.close(report_fd)
"""

_GUARDED_PROBE = """import json,os,sys
report_fd=os.dup(1)
null_fd=os.open('/dev/null',os.O_WRONLY)
os.dup2(null_fd,1);os.dup2(null_fd,2);os.close(null_fd)
from pathlib import Path
os.environ.clear()
os.environ.update({'PATH':os.path.dirname(sys.executable)+':/usr/local/bin:/usr/bin:/bin','HOME':'/tmp','TMPDIR':'/tmp','LC_ALL':'C.UTF-8','PYTHONNOUSERSITE':'1','PYTHONSAFEPATH':'1','PYTHONDONTWRITEBYTECODE':'1','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','WANDB_DISABLED':'true','CUDA_VISIBLE_DEVICES':'','NVIDIA_VISIBLE_DEVICES':'void'})
root=Path('/source/engine')
sys.path.insert(0,str(root))
stage='IMPORT_WORKER'
try:
 from tuner.runtime.offline_sft_worker import load_offline_sft_worker_closure,install_owned_module_guard,verify_loaded_owned_module_origins
 stage='READ_MANIFEST'
 manifest=Path('/source/control/offline-sft-worker-v1.json')
 declared='EXPECTED_CLOSURE_DIGEST'
 stage='LOAD_CLOSURE'
 closure=load_offline_sft_worker_closure(manifest,expected_digest=declared,engine_root=root)
 stage='VERIFY_ORIGINS'
 verify_loaded_owned_module_origins(closure,engine_root=root)
 stage='INSTALL_GUARD'
 install_owned_module_guard(closure,engine_root=root)
 sys.path[:0]=[str(root),str(root/'Trainers/sft/src'),str(root/'Trainers/sft'),str(root)]
 stage='BOOTSTRAP_ENV'
 from shared.env_bootstrap import init_trainer_env
 init_trainer_env()
 stage='TORCH_IMPORT'
 import torch
 category='PASS'
except ModuleNotFoundError:
 category='MODULE_NOT_FOUND'
except ImportError:
 category='IMPORT'
except OSError:
 category='OS'
except RuntimeError:
 category='RUNTIME'
except BaseException:
 category='OTHER'
os.write(report_fd,(json.dumps({'schema_version':'synaptic-docker-torch-import-child/v1','category':category,'stage':stage},sort_keys=True,separators=(',',':'))+'\\n').encode('ascii'))
os.close(report_fd)
"""

# A second -I interpreter receives the reconstructed offline child environment.
# Only the authenticated trainer prefix is compiled.  Its final statement is
# import torch; the UNSLOTH_IMPORT marker and every later byte are excluded.
_WORKER_PREFIX_CHILD = """import hashlib,json,os,sys
report_fd=os.dup(1)
null_fd=os.open('/dev/null',os.O_WRONLY)
os.dup2(null_fd,1);os.dup2(null_fd,2);os.close(null_fd)
from pathlib import Path
def _environment_evidence(supplied,observed):
 if type(supplied) is not dict or type(observed) is not dict or 'LC_CTYPE' in supplied:
  raise ValueError()
 if any(type(key) is not str or type(value) is not str for key,value in supplied.items()):
  raise ValueError()
 if any(observed.get(key)!=value for key,value in supplied.items()):
  raise ValueError()
 additions={key:value for key,value in observed.items() if key not in supplied}
 if additions=={}:
  delta='NONE'
 elif additions=={'LC_CTYPE':'C.UTF-8'}:
  delta='LC_CTYPE_C_UTF8'
 else:
  raise ValueError()
 return {'environment_supplied_sha256':hashlib.sha256(json.dumps(supplied,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
         'environment_observed_sha256':hashlib.sha256(json.dumps(observed,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
         'startup_env_delta':delta}
stage='WORKER_IMPORT'
category='OTHER'
context={}
try:
 supplied_raw=sys.stdin.buffer.read(8193)
 if not supplied_raw or len(supplied_raw)>8192:
  raise ValueError()
 supplied_env=json.loads(supplied_raw)
 if _environment_evidence(supplied_env,dict(os.environ))['environment_supplied_sha256']!='EXPECTED_SUPPLIED_ENV_SHA256':
  raise ValueError()
 if sys.flags.isolated!=1 or sys.flags.no_user_site!=1 or not sys.flags.safe_path:
  raise ValueError()
 if os.environ.get('PYTHONNOUSERSITE')!='1' or os.environ.get('PYTHONSAFEPATH')!='1':
  raise ValueError()
 root=Path('/source/engine')
 import runpy
 worker=runpy.run_path(str(root/'tuner/runtime/offline_sft_worker.py'),run_name='__worker_diagnostic__')
 stage='WORKER_CLOSURE'
 closure=worker['load_offline_sft_worker_environment'](os.environ,engine_root=root)
 stage='WORKER_GUARD'
 worker['verify_loaded_owned_module_origins'](closure,engine_root=root)
 worker['install_owned_module_guard'](closure,engine_root=root)
 trainer=root/'Trainers/sft/train_sft.py'
 sys.path[:0]=[str(trainer.parent),str(root)]
 stage='TRAINER_PREFIX'
 with trainer.open('rb') as stream:
  source=stream.read(1024*1024+1)
 if len(source)>1024*1024:
  raise ValueError()
 if hashlib.sha256(source).hexdigest()!='EXPECTED_TRAINER_SHA256':
  raise ValueError()
 lines=source.decode('utf-8').splitlines(keepends=True)
 markers=('_mark_packaged_runtime_phase("BOOTSTRAP_ENV")','_mark_packaged_runtime_phase("TORCH_IMPORT")','_mark_packaged_runtime_phase("UNSLOTH_IMPORT")')
 positions=[next((i for i,line in enumerate(lines) if line.strip()==marker),-1) for marker in markers]
 if positions[0]<0 or not positions[0]<positions[1]<positions[2] or any(sum(line.strip()==marker for line in lines)!=1 for marker in markers):
  raise ValueError()
 if lines[positions[1]+1].strip()!='import torch  # noqa: E402' or lines[positions[2]+1].strip()!='from unsloth import is_bfloat16_supported  # noqa: E402':
  raise ValueError()
 prefix=''.join(lines[:positions[2]])
 import ast
 tree=ast.parse(prefix,filename=str(trainer))
 if not isinstance(tree.body[-1],ast.Import) or [alias.name for alias in tree.body[-1].names]!=['torch']:
  raise ValueError()
 if any(isinstance(node,ast.Import) and any(alias.name.split('.')[0] in ('unsloth','transformers','trl') for alias in node.names) for node in ast.walk(tree)) or any(isinstance(node,ast.ImportFrom) and node.module is not None and node.module.split('.')[0] in ('unsloth','transformers','trl') for node in ast.walk(tree)):
  raise ValueError()
 context.update(_environment_evidence(supplied_env,dict(os.environ)))
 context['python_isolated']=True
 context['cwd_sha256']=hashlib.sha256(os.getcwd().encode()).hexdigest()
 context['file_sha256']=hashlib.sha256(str(trainer).encode()).hexdigest()
 def mark(value):
  global stage
  if value not in ('BOOTSTRAP_ENV','TORCH_IMPORT'):
   raise ValueError()
  stage=value
  if value=='TORCH_IMPORT':
   context['sys_path_sha256']=hashlib.sha256(json.dumps(sys.path,separators=(',',':')).encode()).hexdigest()
 namespace={'__name__':'__main__','__file__':str(trainer),'__package__':'','__spec__':None,'__cached__':None,'__loader__':None,'__packaged_runtime_phase__':mark}
 sys.argv=[str(trainer)]
 exec(compile(tree,str(trainer),'exec'),namespace)
 if stage!='TORCH_IMPORT':
  raise ValueError()
 category='PASS'
except ModuleNotFoundError:
 category='MODULE_NOT_FOUND'
except ImportError:
 category='IMPORT'
except OSError:
 category='OS'
except RuntimeError:
 category='RUNTIME'
except BaseException:
 category='OTHER'
if category!='PASS':
 context={}
os.write(report_fd,(json.dumps({'schema_version':'synaptic-docker-torch-import-child/v1','category':category,'stage':stage,'context':context,'boundary':'BEFORE_UNSLOTH'},sort_keys=True,separators=(',',':'))+'\\n').encode('ascii'))
os.close(report_fd)
"""

# The optional entry probe enters the authenticated worker's real main and
# runpy trainer path. A line trace stops on the exact UNSLOTH_IMPORT marker;
# the independent import finder prevents any Unsloth module body from running
# if a future interpreter/library disables tracing. Neither hook replaces the
# worker's argument validation, closure guard or runpy.run_path call.
_WORKER_ENTRY_CHILD = """import builtins,hashlib,json,os,re,runpy,sys
from pathlib import Path
root=Path('DIAGNOSTIC_ENGINE_ROOT')
worker=root/'tuner/runtime/offline_sft_worker.py'
trainer=root/'Trainers/sft/train_sft.py'
stage='ENTRY_ADMISSION'
category='OTHER'
context={}
reached=False
blocked=False
class _StopAtMarker(BaseException):
 pass
class _BlockedImport(BaseException):
 pass
class _UnslothBarrier:
 def find_spec(self,fullname,path=None,target=None):
  global blocked
  if fullname=='unsloth' or fullname.startswith('unsloth.'):
   blocked=True
   raise _BlockedImport()
  return None
original_import=builtins.__import__
def guarded_import(name,globals=None,locals=None,fromlist=(),level=0):
 global blocked
 if name=='unsloth' or name.startswith('unsloth.'):
  blocked=True
  raise _BlockedImport()
 return original_import(name,globals,locals,fromlist,level)
def _digest(value):
 return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
try:
 observed=dict(os.environ)
 observed_sha=_digest(observed)
 if observed_sha=='EXPECTED_SUPPLIED_ENV_SHA256':
  delta='NONE'
 elif observed_sha=='EXPECTED_LC_ENV_SHA256' and observed.get('LC_CTYPE')=='C.UTF-8':
  delta='LC_CTYPE_C_UTF8'
 else:
  raise ValueError()
 if sys.flags.isolated!=1 or sys.flags.no_user_site!=1 or not sys.flags.safe_path or os.environ.get('PYTHONNOUSERSITE')!='1' or os.environ.get('PYTHONSAFEPATH')!='1':
  raise ValueError()
 if any(name=='unsloth' or name.startswith('unsloth.') for name in sys.modules):
  raise ValueError()
 with worker.open('rb') as stream:
  worker_raw=stream.read(1024*1024+1)
 if len(worker_raw)>1024*1024 or hashlib.sha256(worker_raw).hexdigest()!='EXPECTED_WORKER_SHA256':
  raise ValueError()
 with trainer.open('rb') as stream:
  source=stream.read(1024*1024+1)
 if len(source)>1024*1024 or hashlib.sha256(source).hexdigest()!='EXPECTED_TRAINER_SHA256':
  raise ValueError()
 lines=source.decode('utf-8').splitlines(keepends=True)
 markers=('_mark_packaged_runtime_phase("BOOTSTRAP_ENV")','_mark_packaged_runtime_phase("TORCH_IMPORT")','_mark_packaged_runtime_phase("UNSLOTH_IMPORT")')
 positions=[next((i for i,line in enumerate(lines) if line.strip()==marker),-1) for marker in markers]
 if positions[0]<0 or not positions[0]<positions[1]<positions[2] or any(sum(line.strip()==marker for line in lines)!=1 for marker in markers):
  raise ValueError()
 if lines[positions[1]+1].strip()!='import torch  # noqa: E402' or lines[positions[2]+1].strip()!='from unsloth import is_bfloat16_supported  # noqa: E402':
  raise ValueError()
 import ast
 prefix=ast.parse(''.join(lines[:positions[2]]),filename=str(trainer))
 if not isinstance(prefix.body[-1],ast.Import) or [alias.name for alias in prefix.body[-1].names]!=['torch']:
  raise ValueError()
 if any(isinstance(node,ast.ImportFrom) and node.module and node.module.split('.')[0] in ('unsloth','transformers','trl') for node in ast.walk(prefix)) or any(isinstance(node,ast.Import) and any(alias.name.split('.')[0] in ('unsloth','transformers','trl') for alias in node.names) for node in ast.walk(prefix)):
  raise ValueError()
 arguments=sys.argv[1:]
 if not arguments or arguments[0]!='--' or _digest([str(trainer),*arguments[1:]])!='EXPECTED_TRAINER_ARGV_SHA256':
  raise ValueError()
 context={'environment_supplied_sha256':'EXPECTED_SUPPLIED_ENV_SHA256','environment_observed_sha256':observed_sha,'startup_env_delta':delta,'python_isolated':True,'cwd_sha256':hashlib.sha256(os.getcwd().encode()).hexdigest(),'file_sha256':hashlib.sha256(str(trainer).encode()).hexdigest(),'argv_sha256':_digest([str(trainer),*arguments[1:]]),'runpy_main_bound':False}
 marker_line=positions[2]+1
 def trainer_lines(frame,event,arg):
  global stage,reached
  if event=='line' and frame.f_lineno==positions[1]+1:
   stage='TORCH_IMPORT'
   context['environment_before_torch_sha256']=_digest(dict(os.environ))
  if event=='line' and frame.f_lineno==marker_line:
   if frame.f_globals.get('__name__')!='__main__' or frame.f_globals.get('__file__')!=str(trainer) or sys.modules['__main__'].__dict__ is not frame.f_globals or sys.argv!=[str(trainer),*arguments[1:]]:
    raise ValueError()
   context['runpy_main_bound']=True
   context['environment_after_torch_sha256']=_digest(dict(os.environ))
   context['sys_path_sha256']=hashlib.sha256(json.dumps(sys.path,separators=(',',':')).encode()).hexdigest()
   reached=True
   stage='TORCH_IMPORT'
   raise _StopAtMarker()
  return trainer_lines
 def trace(frame,event,arg):
  if event=='call' and frame.f_code.co_filename==str(trainer):
   return trainer_lines
  return None
 sys.meta_path.insert(0,_UnslothBarrier())
 builtins.__import__=guarded_import
 sys.argv=[str(worker),*arguments]
 stage='WORKER_MAIN'
 sys.settrace(trace)
 try:
  runpy.run_path(str(worker),run_name='__main__')
 except SystemExit as error:
  if type(error.code) is not int or error.code!=2:
   raise ValueError()
 finally:
  sys.settrace(None)
 stderr_path=Path('DIAGNOSTIC_STDERR_PATH')
 with stderr_path.open('rb') as stream:
  stream.seek(0,2)
  stream.seek(max(0,stream.tell()-512))
  stderr_tail=stream.read(512)
 closed=stderr_tail.rsplit(b'\\n',2)
 label=closed[-2] if len(closed)>=2 and closed[-1]==b'' else b''
 match=re.fullmatch(rb'OFFLINE_SFT_WORKER_REJECTED phase=(BOOTSTRAP|CLOSURE|BOOTSTRAP_ORIGIN|ARGUMENTS|IMPORT_GUARD|TRAINER|POSTCHECK) category=(REJECTION|IMPORT|OS|VALUE|SYSTEM_EXIT|OTHER)(?: milestone=(BOOTSTRAP_ENV|TORCH_IMPORT|UNSLOTH_IMPORT|TRAINER_IMPORT|TRAIN_CALL))?',label)
 if blocked or match is None:
  raise ValueError()
 if reached and context['runpy_main_bound'] and match.groups()==(b'TRAINER',b'OTHER',b'TORCH_IMPORT'):
  category='PASS'
 else:
  context={}
  phase,kind,milestone=match.groups()
  stage=({'BOOTSTRAP':'WORKER_MAIN','CLOSURE':'WORKER_CLOSURE','BOOTSTRAP_ORIGIN':'WORKER_ORIGIN','ARGUMENTS':'WORKER_ARGUMENTS','IMPORT_GUARD':'WORKER_GUARD','TRAINER':'TORCH_IMPORT' if milestone==b'TORCH_IMPORT' else 'TRAINER_PREFIX','POSTCHECK':'WORKER_POSTCHECK'})[phase.decode('ascii')]
  category=kind.decode('ascii') if kind in (b'IMPORT',b'OS') else 'OTHER'
except ModuleNotFoundError:
 category='MODULE_NOT_FOUND'
except ImportError:
 category='IMPORT'
except OSError:
 category='OS'
except RuntimeError:
 category='RUNTIME'
except BaseException:
 category='OTHER'
if category!='PASS':
 context={}
os.write(1,(json.dumps({'schema_version':'synaptic-docker-torch-import-child/v1','category':category,'stage':stage,'context':context,'boundary':'BEFORE_UNSLOTH'},sort_keys=True,separators=(',',':'))+'\\n').encode('ascii'))
"""

_WORKER_CONTEXT_PROBE = """import base64,hashlib,json,os,sys,subprocess
report_fd=os.dup(1)
null_fd=os.open('/dev/null',os.O_WRONLY)
os.dup2(null_fd,1);os.dup2(null_fd,2);os.close(null_fd)
from pathlib import Path
stage='READ_WORKLOAD'
category='OTHER'
context={}
try:
 with Path('/source/control/workload.json').open('rb') as stream:
  raw=stream.read(1024*1024+1)
 if len(raw)>1024*1024:
  raise ValueError()
 if hashlib.sha256(raw).hexdigest()!='EXPECTED_WORKLOAD_SHA256':
  raise ValueError()
 workload=json.loads(raw)
 runtime=workload['execution_source']['runtime']
 roots=runtime['roots']
 if roots!=EXPECTED_ROOTS or runtime['environment']['clear_inherited'] is not True:
  raise ValueError()
 planned=runtime['environment']['variables']
 if set(planned)!=set(EXPECTED_ENV_KEYS) or any(type(v) is not str for v in planned.values()):
  raise ValueError()
 environment=dict(planned)
 environment.pop('PYTHONPATH')
 environment.update({'SYNAPTIC_WORKER_CLOSURE_MANIFEST':'/source/control/offline-sft-worker-v1.json','SYNAPTIC_WORKER_CLOSURE_DIGEST':'EXPECTED_CLOSURE_DIGEST'})
 environment.update({key:value for key,value in ROOT_BINDINGS.items()})
 model=workload['configuration']['document']['model']
 ref=model['ref']; revision=model['revision']
 if not isinstance(ref,str) or not isinstance(revision,str) or not __import__('re').fullmatch('[0-9a-f]{40}(?:[0-9a-f]{24})?',revision) or any(not __import__('re').fullmatch('[A-Za-z0-9][A-Za-z0-9._-]*',part) for part in ref.split('/')) or len(ref.split('/')) not in (1,2):
  raise ValueError()
 snapshot=roots['cache']+'/model/models--'+'--'.join(ref.split('/'))+'/snapshots/'+revision
 environment.update({'SYNAPTIC_WORKLOAD_FINGERPRINT':hashlib.sha256(b'synaptic-training-workload/v1\\0'+raw).hexdigest(),'PYTHONNOUSERSITE':'1','PYTHONSAFEPATH':'1','HF_HOME':roots['cache']+'/huggingface','TRANSFORMERS_CACHE':roots['cache']+'/transformers','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','SYNAPTIC_MODEL_SNAPSHOT':snapshot,'WANDB_DISABLED':'true'})
 os.umask(0o077)
 Path('/artifacts/run').mkdir(mode=0o700,exist_ok=DIRECT_MODE)
 for name in ('artifacts','state','tracking','cache','tmp'):
  Path(roots[name]).mkdir(mode=0o700,exist_ok=DIRECT_MODE)
 for name in ('huggingface','transformers'):
  (Path(roots['cache'])/name).mkdir(mode=0o700,exist_ok=True)
 stage='ISOLATED_CHILD'
 if ENTRY_MODE or DIRECT_MODE:
  import fcntl,runpy
  runtime_path=Path('/source/engine/Trainers/sft/runtime_v1.py')
  with runtime_path.open('rb') as stream:
   runtime_raw=stream.read(1024*1024+1)
  if len(runtime_raw)>1024*1024 or hashlib.sha256(runtime_raw).hexdigest()!='EXPECTED_RUNTIME_SHA256':
   raise ValueError()
  runtime_module=runpy.run_path(str(runtime_path),run_name='__diagnostic_runtime__')
  config=workload['configuration']['document'];dataset=config['dataset']
  if not dataset['ref'].startswith('prepared://sha256/'):
   raise ValueError()
  descriptor=os.memfd_create('synaptic-diagnostic-input',os.MFD_CLOEXEC|os.MFD_ALLOW_SEALING)
  try:
   if DIRECT_MODE:
    with (Path(roots['state'])/'prepared-inputs'/'EXPECTED_PREPARED_SHA256'/'private-dataset.jsonl').open('rb') as stream:
     content=stream.read(64*1024*1024+1)
    if len(content)!=EXPECTED_PREPARED_SIZE or hashlib.sha256(content).hexdigest()!='EXPECTED_PREPARED_SHA256':
     raise ValueError()
   else:
    content=b'0'
   view=memoryview(content)
   while view:
    written=os.write(descriptor,view)
    if written<=0: raise ValueError()
    view=view[written:]
   os.lseek(descriptor,0,os.SEEK_SET)
   seals=fcntl.F_SEAL_SEAL|fcntl.F_SEAL_SHRINK|fcntl.F_SEAL_GROW|fcntl.F_SEAL_WRITE
   fcntl.fcntl(descriptor,fcntl.F_ADD_SEALS,seals)
   arguments=['--','--model-name',ref,'--model-revision',revision,'--anonymous-model',
    '--model-cache-dir',roots['cache']+'/model','--model-snapshot',snapshot,
    '--local-file','/proc/self/fd/'+str(descriptor),
    '--output-root',roots['state']+'/runtime-v1-trainer/output',
    '--run-timestamp','runtime-v1','--no-dashboard','--quiet',
    '--runtime-v1-workload-fingerprint',environment['SYNAPTIC_WORKLOAD_FINGERPRINT'],
    '--runtime-v1-configuration-revision',workload['configuration']['revision'],
    '--runtime-v1-tokenizer-revision',model['tokenizer_revision'],
    '--runtime-v1-dataset-revision',dataset['revision'],
    '--runtime-v1-dataset-digest',dataset['content_digest']]
   runtime_module['_append_sft_arguments'](arguments,config['sft'],model,
    dataset_schema=dataset['format'] if dataset['format']=='synaptic-conversation-sft-row/v1' else None)
   trainer_argv=['/source/engine/Trainers/sft/train_sft.py',*arguments[1:]]
   argv_sha=hashlib.sha256(json.dumps(trainer_argv,sort_keys=True,separators=(',',':')).encode()).hexdigest()
   if DIRECT_MODE:
    worker_path=Path('/source/engine/tuner/runtime/offline_sft_worker.py')
    with worker_path.open('rb') as stream:
     worker_raw=stream.read(1024*1024+1)
    trainer_path=Path('/source/engine/Trainers/sft/train_sft.py')
    with trainer_path.open('rb') as stream:
     trainer_raw=stream.read(1024*1024+1)
    if (len(worker_raw)>1024*1024 or hashlib.sha256(worker_raw).hexdigest()!='EXPECTED_WORKER_SHA256'
        or len(trainer_raw)>1024*1024 or hashlib.sha256(trainer_raw).hexdigest()!='EXPECTED_DIRECT_TRAINER_SHA256'
        or not trainer_raw.endswith(b'raise SystemExit(42)\\n')):
     raise ValueError()
   else:
    child=base64.b64decode('CHILD_SOURCE_B64')
    token='EXPECTED'+'_'+'TRAINER_ARGV_SHA256'
    child_text=child.decode('utf-8').replace(token,argv_sha)
    if token in child_text:
     raise ValueError()
   stdout_path=Path(roots['tracking'])/'worker-entry.stdout'
   stderr_path=Path(roots['tracking'])/'worker-entry.stderr'
   if DIRECT_MODE:
    process=subprocess.Popen([sys.executable,'-I',str(worker_path),*arguments],
     cwd=roots['tmp'],env=environment,stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,
     stderr=subprocess.PIPE,pass_fds=(descriptor,))
    stderr_raw=process.stderr.read(257)
    if len(stderr_raw)>256: process.kill()
    returncode=process.wait()
    if returncode!=2 or len(stderr_raw)>256:
     raise ValueError()
    if stderr_raw==b'OFFLINE_SFT_WORKER_REJECTED phase=TRAINER category=SYSTEM_EXIT milestone=UNSLOTH_IMPORT\\n':
     result={'schema_version':'synaptic-docker-torch-import-child/v1','category':'PASS','stage':'UNSLOTH_IMPORT','context':{},'boundary':'BEFORE_UNSLOTH'}
    elif stderr_raw==b'OFFLINE_SFT_WORKER_REJECTED phase=TRAINER category=IMPORT milestone=TORCH_IMPORT\\n':
     result={'schema_version':'synaptic-docker-torch-import-child/v1','category':'IMPORT','stage':'TORCH_IMPORT','context':{},'boundary':'BEFORE_UNSLOTH'}
    else:
     result={'schema_version':'synaptic-docker-torch-import-child/v1','category':'OTHER','stage':'ISOLATED_CHILD','context':{},'boundary':'BEFORE_UNSLOTH'}
    payload=json.dumps(result,sort_keys=True,separators=(',',':')).encode()
   else:
    with stdout_path.open('xb') as stdout,stderr_path.open('xb') as stderr:
     process=subprocess.Popen([sys.executable,'-I','-c',child_text,*arguments],
      cwd=roots['tmp'],env=environment,stdin=subprocess.DEVNULL,stdout=stdout,
      stderr=stderr,pass_fds=(descriptor,))
     if process.wait()!=0: raise ValueError()
    with stdout_path.open('rb') as stream: payload=stream.read(4097)
    if len(payload)>4096: raise ValueError()
  finally:
   os.close(descriptor)
 else:
  supplied_raw=json.dumps(environment,sort_keys=True,separators=(',',':')).encode()
  if len(supplied_raw)>8192:
   raise ValueError()
  process=subprocess.Popen([sys.executable,'-I','-c',child.decode('utf-8')],cwd=roots['tmp'],env=environment,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL)
  process.stdin.write(supplied_raw)
  process.stdin.close()
  payload=process.stdout.read(4097)
  if len(payload)>4096:
   process.kill();process.wait()
   raise ValueError()
  if process.wait()!=0:
   raise ValueError()
 result=json.loads(payload)
 if set(result)!={'schema_version','category','stage','context','boundary'} or result['schema_version']!='synaptic-docker-torch-import-child/v1' or result['boundary']!='BEFORE_UNSLOTH' or result['category'] not in ('PASS','MODULE_NOT_FOUND','IMPORT','OS','RUNTIME','OTHER') or result['stage'] not in ('WORKER_IMPORT','WORKER_CLOSURE','WORKER_GUARD','TRAINER_PREFIX','BOOTSTRAP_ENV','TORCH_IMPORT','ENTRY_ADMISSION','WORKER_MAIN','WORKER_ORIGIN','WORKER_ARGUMENTS','WORKER_POSTCHECK','ISOLATED_CHILD','UNSLOTH_IMPORT') or not isinstance(result['context'],dict):
  raise ValueError()
 if DIRECT_MODE:
  if (result['context']!={} or
      (result['category']=='PASS' and result['stage']!='UNSLOTH_IMPORT') or
      (result['category']=='IMPORT' and result['stage']!='TORCH_IMPORT')): raise ValueError()
 elif result['category']=='PASS':
  expected_context={'environment_supplied_sha256','environment_observed_sha256','startup_env_delta','python_isolated','cwd_sha256','file_sha256','sys_path_sha256'}
  if ENTRY_MODE:
   expected_context.update(('argv_sha256','runpy_main_bound','environment_before_torch_sha256','environment_after_torch_sha256'))
  if result['stage']!='TORCH_IMPORT' or set(result['context'])!=expected_context or result['context']['python_isolated'] is not True or result['context']['startup_env_delta'] not in ('NONE','LC_CTYPE_C_UTF8') or (ENTRY_MODE and (result['context']['runpy_main_bound'] is not True or result['context']['argv_sha256']!=argv_sha)) or any(not isinstance(value,str) or not __import__('re').fullmatch('[0-9a-f]{64}',value) for key,value in result['context'].items() if key.endswith('_sha256')):
   raise ValueError()
 elif result['context']!={}:
  raise ValueError()
 category=result['category'];stage=result['stage'];context=result['context']
except BaseException:
 context={}
os.write(report_fd,(json.dumps({'schema_version':'synaptic-docker-torch-import-child/v1','category':category,'stage':stage,'context':context,'boundary':'BEFORE_UNSLOTH'},sort_keys=True,separators=(',',':'))+'\\n').encode('ascii'))
os.close(report_fd)
"""


def _read(path: Path, maximum: int) -> bytes:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > maximum:
        raise ValueError("INPUT_INVALID")
    with path.open("rb") as stream:
        raw = stream.read(maximum + 1)
    if not raw or len(raw) > maximum:
        raise ValueError("INPUT_INVALID")
    return raw


def _identity(path: Path):
    value = path.stat()
    return value.st_dev, value.st_ino, value.st_mode, value.st_size, value.st_mtime_ns


def _authority(docker: Path, docker_config: Path):
    if (not docker.is_absolute() or docker.is_symlink() or not docker.is_file()
            or not docker_config.is_absolute() or docker_config.is_symlink()
            or not docker_config.is_dir() or any(docker_config.iterdir())):
        raise ValueError("DOCKER_INVALID")
    return (docker, docker_config, _identity(docker), _identity(docker_config))


def _run(authority, args: tuple[str, ...], *, timeout: float = 15):
    """Bound both CLI lifetime and combined bytes before they enter memory."""
    docker, config, exe_stat, config_stat = authority
    if (_identity(docker) != exe_stat or _identity(config) != config_stat):
        raise ValueError("DOCKER_CHANGED")
    process = subprocess.Popen((str(docker), "--config", str(config),
        "--host", "unix:///var/run/docker.sock", *args),
        env={"PATH": os.defpath, "DOCKER_CONTENT_TRUST": "1"},
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        start_new_session=True)
    output = {process.stdout: bytearray(), process.stderr: bytearray()}
    deadline = time.monotonic() + timeout
    try:
        with selectors.DefaultSelector() as selector:
            for stream in output:
                os.set_blocking(stream.fileno(), False)
                selector.register(stream, selectors.EVENT_READ)
            while selector.get_map() or process.poll() is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise ValueError("DOCKER_COMMAND_REJECTED")
                for key, _ in selector.select(min(remaining, 0.1)):
                    chunk = os.read(key.fileobj.fileno(), _CLI_BUDGET + 1)
                    if not chunk:
                        selector.unregister(key.fileobj)
                    elif sum(map(len, output.values())) + len(chunk) > _CLI_BUDGET:
                        raise ValueError("DOCKER_COMMAND_REJECTED")
                    else:
                        output[key.fileobj].extend(chunk)
        if process.wait(timeout=max(0.01, deadline - time.monotonic())) != 0:
            raise ValueError("DOCKER_COMMAND_REJECTED")
        if (_identity(docker) != exe_stat or _identity(config) != config_stat):
            raise ValueError("DOCKER_CHANGED")
        return subprocess.CompletedProcess(args, 0, bytes(output[process.stdout]),
                                           bytes(output[process.stderr]))
    finally:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass
        for stream in output:
            stream.close()


def _one_line(raw: bytes) -> str:
    value = raw.decode("ascii")
    if value != value.strip() and value != value.strip() + "\n":
        raise ValueError("DIAGNOSTIC_OUTPUT_INVALID")
    if not value.strip() or "\n" in value.strip():
        raise ValueError("DIAGNOSTIC_OUTPUT_INVALID")
    return value.strip()


def _append(output_dir: Path, event: dict[str, object]) -> None:
    with (output_dir / "events.jsonl").open("ab") as stream:
        stream.write(json.dumps(event, sort_keys=True, separators=(",", ":")).encode("ascii") + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def _source_identity(source_root: Path, source_lock: Path, expected_lock: str,
                     expected_manifest: str, expected_closure: str) -> dict[str, str]:
    if (not source_root.is_absolute() or source_root.name != "engine"
            or source_root.is_symlink() or not source_root.is_dir()
            or not source_lock.is_absolute() or not re.fullmatch(r"[0-9a-f]{64}", expected_lock)
            or not re.fullmatch(r"[0-9a-f]{64}", expected_manifest)
            or not re.fullmatch(r"[0-9a-f]{64}", expected_closure)):
        raise ValueError("SOURCE_IDENTITY_INVALID")
    lock_raw = _read(source_lock, 128 * 1024)
    manifest = source_root.parent / "control/offline-sft-worker-v1.json"
    manifest_raw = _read(manifest, 128 * 1024)
    if (hashlib.sha256(lock_raw).hexdigest() != expected_lock
            or hashlib.sha256(manifest_raw).hexdigest() != expected_manifest):
        raise ValueError("SOURCE_IDENTITY_INVALID")
    lock = json.loads(lock_raw)
    if (lock.get("schema_version") != "synaptic-source-lock/v1"
            or lock.get("sources", {}).get("engine", {}).get("commit") !=
            "0d515ae1ae235d0858802228ce4af8dba8fccd03"):
        raise ValueError("SOURCE_IDENTITY_INVALID")
    # This trusted, checked-in verifier hashes every mounted worker member.
    # It is imported from this checkout, never from the staged source.
    checkout = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(checkout))
    from tuner.runtime import offline_sft_worker
    if Path(offline_sft_worker.__file__).resolve() != (
            checkout / "tuner/runtime/offline_sft_worker.py").resolve():
        raise ValueError("SOURCE_IDENTITY_INVALID")
    try:
        offline_sft_worker.load_offline_sft_worker_closure(
            manifest, expected_digest=expected_closure, engine_root=source_root)
    except Exception:
        raise ValueError("SOURCE_IDENTITY_INVALID") from None
    return {"source_lock_sha256": expected_lock, "manifest_sha256": expected_manifest,
            "closure_digest": expected_closure,
            "engine_commit": "0d515ae1ae235d0858802228ce4af8dba8fccd03"}


def _trainer_prefix_boundary(raw: bytes) -> int:
    """Validate the fixed marker sequence and return the first excluded line."""
    lines = raw.decode("utf-8").splitlines(keepends=True)
    markers = tuple(f'_mark_packaged_runtime_phase("{name}")' for name in
                    ("BOOTSTRAP_ENV", "TORCH_IMPORT", "UNSLOTH_IMPORT"))
    positions = []
    for marker in markers:
        matches = [i for i, line in enumerate(lines) if line.strip() == marker]
        if len(matches) != 1:
            raise ValueError("SOURCE_IDENTITY_INVALID")
        positions.append(matches[0])
    if (not positions[0] < positions[1] < positions[2]
            or lines[positions[1] + 1].strip() != "import torch  # noqa: E402"
            or lines[positions[2] + 1].strip() !=
            "from unsloth import is_bfloat16_supported  # noqa: E402"):
        raise ValueError("SOURCE_IDENTITY_INVALID")
    prefix = "".join(lines[:positions[2]])
    tree = ast.parse(prefix)
    if (not isinstance(tree.body[-1], ast.Import)
            or [alias.name for alias in tree.body[-1].names] != ["torch"]
            or any(isinstance(node, ast.ImportFrom) and node.module is not None and
                   node.module.split(".")[0] in {"unsloth", "transformers", "trl"}
                   for node in ast.walk(tree))
            or any(isinstance(node, ast.Import) and
                   any(alias.name.split(".")[0] in {"unsloth", "transformers", "trl"}
                       for alias in node.names) for node in ast.walk(tree))):
        raise ValueError("SOURCE_IDENTITY_INVALID")
    return positions[2]


def _worker_context_identity(source_root: Path, expected_workload: str,
                             expected_runtime: str, expected_trainer: str,
                             expected_closure: str, *, direct_stop: bool = False) -> dict[str, str]:
    if any(not re.fullmatch(r"[0-9a-f]{64}", value) for value in
           (expected_workload, expected_runtime, expected_trainer, expected_closure)):
        raise ValueError("SOURCE_IDENTITY_INVALID")
    workload_raw = _read(source_root.parent / "control/workload.json", 1024 * 1024)
    runtime_raw = _read(source_root / "Trainers/sft/runtime_v1.py", 1024 * 1024)
    trainer_raw = _read(source_root / "Trainers/sft/train_sft.py", 1024 * 1024)
    worker_raw = _read(source_root / "tuner/runtime/offline_sft_worker.py", 1024 * 1024)
    if (hashlib.sha256(workload_raw).hexdigest() != expected_workload
            or hashlib.sha256(runtime_raw).hexdigest() != expected_runtime
            or hashlib.sha256(trainer_raw).hexdigest() != expected_trainer):
        raise ValueError("SOURCE_IDENTITY_INVALID")
    workload = json.loads(workload_raw)
    runtime = workload["execution_source"]["runtime"]
    planned = runtime["environment"]["variables"]
    if (runtime["roots"] != _R2_ROOTS
            or runtime["environment"]["clear_inherited"] is not True
            or set(planned) != _R2_ENV_KEYS
            or any(type(value) is not str for value in planned.values())
            or any(planned[_ROOT_ENV[name]] != path for name, path in _R2_ROOTS.items())
            or planned["PYTHONPATH"] != _R2_ROOTS["engine"]
            or planned["HF_HUB_OFFLINE"] != "1"
            or planned["TRANSFORMERS_OFFLINE"] != "1"
            or runtime["interpreter"]["executable"] != "/opt/unsloth-venv/bin/python3"):
        raise ValueError("SOURCE_IDENTITY_INVALID")
    if direct_stop:
        if not trainer_raw.endswith(_DIRECT_STOP):
            raise ValueError("SOURCE_IDENTITY_INVALID")
    else:
        _trainer_prefix_boundary(trainer_raw)
    model = workload["configuration"]["document"]["model"]
    ref, revision = model["ref"], model["revision"]
    if (type(ref) is not str or type(revision) is not str
            or len(ref.split("/")) not in (1, 2)
            or any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", part)
                   for part in ref.split("/"))
            or not re.fullmatch(r"[0-9a-f]{40}(?:[0-9a-f]{24})?", revision)):
        raise ValueError("SOURCE_IDENTITY_INVALID")
    child_env = dict(planned)
    child_env.pop("PYTHONPATH")
    child_env.update({"SYNAPTIC_WORKER_CLOSURE_MANIFEST":
                      "/source/control/offline-sft-worker-v1.json"})
    child_env["SYNAPTIC_WORKER_CLOSURE_DIGEST"] = expected_closure
    child_env.update({_ROOT_ENV[name]: value for name, value in _R2_ROOTS.items()})
    child_env.update({"SYNAPTIC_WORKLOAD_FINGERPRINT": hashlib.sha256(
        b"synaptic-training-workload/v1\0" + workload_raw).hexdigest(),
        "PYTHONNOUSERSITE": "1", "PYTHONSAFEPATH": "1",
        "HF_HOME": _R2_ROOTS["cache"] + "/huggingface",
        "TRANSFORMERS_CACHE": _R2_ROOTS["cache"] + "/transformers",
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "SYNAPTIC_MODEL_SNAPSHOT": _R2_ROOTS["cache"] + "/model/models--" +
        "--".join(ref.split("/")) + "/snapshots/" + revision,
        "WANDB_DISABLED": "true"})
    child_env_lc_ctype = dict(child_env)
    child_env_lc_ctype["LC_CTYPE"] = "C.UTF-8"
    return {"workload_sha256": expected_workload,
            "runtime_sha256": expected_runtime,
            "trainer_sha256": expected_trainer,
            "worker_sha256": hashlib.sha256(worker_raw).hexdigest(),
            "child_env_sha256": hashlib.sha256(json.dumps(
                child_env, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
            "child_env_lc_ctype_sha256": hashlib.sha256(json.dumps(
                child_env_lc_ctype, sort_keys=True, separators=(",", ":")).encode()).hexdigest()}


def _direct_source_identity(source_root: Path, *, expected_trainer: str,
                            expected_manifest: str, expected_closure: str) -> dict[str, str]:
    """Authenticate the staged stop derivative with the trusted worker verifier."""
    manifest = source_root.parent / "control/offline-sft-worker-v1.json"
    raw = _read(manifest, 128 * 1024)
    trainer = _read(source_root / _DIRECT_TRAINER, 1024 * 1024)
    if (hashlib.sha256(raw).hexdigest() != expected_manifest
            or hashlib.sha256(trainer).hexdigest() != expected_trainer
            or not trainer.endswith(_DIRECT_STOP)):
        raise ValueError("SOURCE_IDENTITY_INVALID")
    from tuner.runtime.offline_sft_worker import load_offline_sft_worker_closure
    try:
        load_offline_sft_worker_closure(
            manifest, expected_digest=expected_closure, engine_root=source_root)
    except Exception:
        raise ValueError("SOURCE_IDENTITY_INVALID") from None
    return {"trainer_sha256": expected_trainer, "manifest_sha256": expected_manifest,
            "closure_digest": expected_closure}


def _stage_direct_source(original: Path, destination: Path, *, expected_trainer: str,
                         expected_manifest: str, expected_closure: str) -> dict[str, str]:
    """Copy only the authenticated closure and replace the post-Torch suffix."""
    original_manifest = json.loads(_read(
        original.parent / "control/offline-sft-worker-v1.json", 128 * 1024))
    trainer_raw = _read(original / _DIRECT_TRAINER, 1024 * 1024)
    lines = trainer_raw.splitlines(keepends=True)
    boundary = _trainer_prefix_boundary(trainer_raw)
    derived = b"".join(lines[:boundary + 1]) + _DIRECT_STOP
    if hashlib.sha256(derived).hexdigest() != expected_trainer:
        raise ValueError("SOURCE_IDENTITY_INVALID")
    document = json.loads(json.dumps(original_manifest))
    for member in document["members"]:
        if member["path"] == _DIRECT_TRAINER:
            member["sha256"] = expected_trainer
            member["size_bytes"] = len(derived)
            break
    else:
        raise ValueError("SOURCE_IDENTITY_INVALID")
    document["payload_bytes"] = sum(member["size_bytes"] for member in document["members"])
    from tuner.runtime.offline_sft_worker import closure_digest
    document["closure_digest"] = closure_digest(document)
    encoded = json.dumps(document, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"
    if (document["closure_digest"] != expected_closure
            or hashlib.sha256(encoded).hexdigest() != expected_manifest):
        raise ValueError("SOURCE_IDENTITY_INVALID")
    engine = destination / "engine"
    control = destination / "control"
    engine.mkdir(parents=True, mode=0o700)
    control.mkdir(mode=0o700)
    for member in document["members"]:
        target = engine / member["path"]
        target.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
        if member["path"] == _DIRECT_TRAINER:
            target.write_bytes(derived)
        else:
            shutil.copyfile(original / member["path"], target)
        target.chmod(int(member["git_mode"], 8) & 0o777)
    (control / "offline-sft-worker-v1.json").write_bytes(encoded)
    shutil.copyfile(original.parent / "control/workload.json", control / "workload.json")
    return _direct_source_identity(engine, expected_trainer=expected_trainer,
                                   expected_manifest=expected_manifest,
                                   expected_closure=expected_closure)


def _direct_retained_inputs(source_root: Path, artifact_root: Path, receipt_path: Path,
                            expected_receipt: str, expected_workload: str) -> dict[str, object]:
    """Verify the retained prepared input and snapshot without writing either."""
    if (not artifact_root.is_absolute() or artifact_root.is_symlink()
            or not artifact_root.is_dir() or not receipt_path.is_absolute()
            or not re.fullmatch(r"[0-9a-f]{64}", expected_receipt)):
        raise ValueError("SOURCE_IDENTITY_INVALID")
    receipt_raw = _read(receipt_path, 1024 * 1024)
    workload_raw = _read(source_root.parent / "control/workload.json", 1024 * 1024)
    if (hashlib.sha256(receipt_raw).hexdigest() != expected_receipt
            or hashlib.sha256(workload_raw).hexdigest() != expected_workload):
        raise ValueError("SOURCE_IDENTITY_INVALID")
    receipt = json.loads(receipt_raw)
    workload = json.loads(workload_raw)
    model = workload["configuration"]["document"]["model"]
    dataset = workload["configuration"]["document"]["dataset"]
    snapshot = artifact_root / "run/cache/model" / (
        "models--" + model["ref"].replace("/", "--")) / "snapshots" / model["revision"]
    prepared = artifact_root / "run/state/prepared-inputs" / dataset["content_digest"] / "private-dataset.jsonl"
    if (receipt.get("schema_version") != "synaptic-docker-model-preparation/v1"
            or receipt.get("artifact_host_root") != str(artifact_root)
            or receipt.get("snapshot") != str(snapshot)
            or receipt.get("model") != model
            or not re.fullmatch(r"[0-9a-f]{64}", dataset["content_digest"])
            or dataset.get("ref") != "prepared://sha256/" + dataset["revision"]
            or type(dataset.get("size_bytes")) is not int
            or not 0 < dataset["size_bytes"] <= 64 * 1024 * 1024):
        raise ValueError("SOURCE_IDENTITY_INVALID")
    try:
        if _readonly_snapshot_inventory(snapshot) != receipt["files"]:
            raise ValueError("SOURCE_IDENTITY_INVALID")
    except Exception:
        raise ValueError("SOURCE_IDENTITY_INVALID") from None
    content = _read(prepared, 64 * 1024 * 1024)
    if (len(content) != dataset["size_bytes"]
            or hashlib.sha256(content).hexdigest() != dataset["content_digest"]):
        raise ValueError("SOURCE_IDENTITY_INVALID")
    custody = tuple(_path_custody(path) for path in (artifact_root, snapshot, prepared, receipt_path))
    return {"snapshot": snapshot, "prepared": prepared,
            "prepared_sha256": dataset["content_digest"],
            "prepared_size_bytes": dataset["size_bytes"],
            "receipt_sha256": expected_receipt, "custody": custody}


def _readonly_snapshot_inventory(snapshot: Path) -> list[dict[str, object]]:
    """Hash bounded model bytes through held directory descriptors."""
    custody = _path_custody(snapshot)
    deadline = time.monotonic() + 300
    maximum_entries = 64 + 256
    maximum_name_bytes = maximum_entries * 255
    rows: list[dict[str, object]] = []
    total = 0
    directories = 0
    entries_seen = 0
    name_bytes_seen = 0
    root_fd = os.open(snapshot, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        def walk(directory_fd: int, prefix: str) -> None:
            nonlocal total, directories, entries_seen, name_bytes_seen
            if time.monotonic() >= deadline:
                raise ValueError("SOURCE_IDENTITY_INVALID")
            with os.scandir(directory_fd) as entries:
                names = []
                seen = set()
                for entry in entries:
                    if time.monotonic() >= deadline:
                        raise ValueError("SOURCE_IDENTITY_INVALID")
                    name = entry.name
                    if type(name) is not str or len(name) > 255:
                        raise ValueError("SOURCE_IDENTITY_INVALID")
                    encoded_name = os.fsencode(name)
                    entries_seen += 1
                    name_bytes_seen += len(encoded_name)
                    if (name in {"", ".", ".."} or "/" in name or "\0" in name or
                            name in seen or not 0 < len(encoded_name) <= 255 or
                            entries_seen > maximum_entries or
                            name_bytes_seen > maximum_name_bytes):
                        raise ValueError("SOURCE_IDENTITY_INVALID")
                    seen.add(name)
                    names.append(name)
                if time.monotonic() >= deadline:
                    raise ValueError("SOURCE_IDENTITY_INVALID")
                names.sort()
            for name in names:
                if time.monotonic() >= deadline:
                    raise ValueError("SOURCE_IDENTITY_INVALID")
                info = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                if stat.S_ISDIR(info.st_mode):
                    directories += 1
                    if directories > 256:
                        raise ValueError("SOURCE_IDENTITY_INVALID")
                    child_fd = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                                       dir_fd=directory_fd)
                    try:
                        walk(child_fd, prefix + name + "/")
                    finally:
                        os.close(child_fd)
                elif stat.S_ISREG(info.st_mode):
                    if len(rows) >= 64 or total + info.st_size > 16 * 1024 ** 3:
                        raise ValueError("SOURCE_IDENTITY_INVALID")
                    fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory_fd)
                    try:
                        before = os.fstat(fd)
                        if ((before.st_dev, before.st_ino, before.st_size,
                             before.st_mtime_ns) !=
                            (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)):
                            raise ValueError("SOURCE_IDENTITY_INVALID")
                        digest = hashlib.sha256()
                        size = 0
                        while True:
                            if time.monotonic() >= deadline:
                                raise ValueError("SOURCE_IDENTITY_INVALID")
                            chunk = os.read(fd, min(1024 * 1024, info.st_size - size + 1))
                            if not chunk:
                                break
                            size += len(chunk)
                            if size > info.st_size:
                                raise ValueError("SOURCE_IDENTITY_INVALID")
                            digest.update(chunk)
                        after = os.fstat(fd)
                        if (size != info.st_size or _identity_from_stat(after) !=
                                _identity_from_stat(before)):
                            raise ValueError("SOURCE_IDENTITY_INVALID")
                    finally:
                        os.close(fd)
                    rows.append({"path": prefix + name, "size_bytes": size,
                                 "sha256": digest.hexdigest()})
                    total += size
                else:
                    raise ValueError("SOURCE_IDENTITY_INVALID")
        walk(root_fd, "")
    finally:
        os.close(root_fd)
    if not rows or _path_custody(snapshot) != custody:
        raise ValueError("SOURCE_IDENTITY_INVALID")
    return sorted(rows, key=lambda row: row["path"])


def _identity_from_stat(info: os.stat_result) -> tuple[int, int, int, int]:
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns


def _worker_probe(identity: dict[str, str], closure: str, *, entry_mode: bool = False,
                  direct_mode: bool = False, prepared_sha256: str = "0" * 64,
                  prepared_size: int = 0) -> str:
    child = ("" if direct_mode else
             (_WORKER_ENTRY_CHILD if entry_mode else _WORKER_PREFIX_CHILD))
    child = child.replace("EXPECTED_TRAINER_SHA256", identity["trainer_sha256"])
    child = child.replace("EXPECTED_SUPPLIED_ENV_SHA256",
                          identity["child_env_sha256"])
    if entry_mode:
        child = child.replace("EXPECTED_LC_ENV_SHA256",
                              identity["child_env_lc_ctype_sha256"])
        child = child.replace("EXPECTED_WORKER_SHA256", identity["worker_sha256"])
        child = child.replace("DIAGNOSTIC_ENGINE_ROOT", "/source/engine")
        child = child.replace("DIAGNOSTIC_STDERR_PATH",
                              "/artifacts/run/tracking/worker-entry.stderr")
    probe = _WORKER_CONTEXT_PROBE
    replacements = {"EXPECTED_WORKLOAD_SHA256": identity["workload_sha256"],
                    "EXPECTED_CLOSURE_DIGEST": closure,
                    "EXPECTED_RUNTIME_SHA256": identity["runtime_sha256"],
                    "EXPECTED_ROOTS": repr(_R2_ROOTS),
                    "EXPECTED_ENV_KEYS": repr(sorted(_R2_ENV_KEYS)),
                    "ROOT_BINDINGS": repr({_ROOT_ENV[name]: value
                                          for name, value in _R2_ROOTS.items()}),
                    "CHILD_SOURCE_B64": base64.b64encode(child.encode()).decode(),
                    "EXPECTED_PREPARED_SHA256": prepared_sha256,
                    "EXPECTED_PREPARED_SIZE": str(prepared_size),
                    "EXPECTED_WORKER_SHA256": identity["worker_sha256"],
                    "EXPECTED_DIRECT_TRAINER_SHA256": identity["trainer_sha256"]}
    for key, value in replacements.items():
        if probe.count(key) != (2 if key == "EXPECTED_PREPARED_SHA256" else 1):
            raise ValueError("SOURCE_IDENTITY_INVALID")
        probe = probe.replace(key, value)
    probe = probe.replace("ENTRY_MODE", "True" if entry_mode else "False")
    probe = probe.replace("DIRECT_MODE", "True" if direct_mode else "False")
    return probe


def _path_custody(path: Path) -> tuple[tuple[int, int, int], ...]:
    if not path.is_absolute():
        raise ValueError("SOURCE_IDENTITY_INVALID")
    observed = []
    for member in (path, *path.parents):
        info = member.lstat()
        if stat.S_ISLNK(info.st_mode):
            raise ValueError("SOURCE_IDENTITY_INVALID")
        observed.append((info.st_dev, info.st_ino, info.st_mode))
    return tuple(observed)


def _require_daemon(authority, expected: str, *, timeout: float = 5) -> None:
    result = _run(authority, ("info", "--format", "{{.ID}}"), timeout=timeout)
    if result.stderr or _one_line(result.stdout) != expected:
        raise ValueError("DAEMON_INVALID")


def _inspect_owned(authority, identifier: str, *, name: str, label: str, image: str,
                   daemon_id: str | None = None, timeout: float = 15):
    if not _CID.fullmatch(identifier):
        raise ValueError("CONTAINER_ID_INVALID")
    deadline = time.monotonic() + timeout
    if daemon_id is not None:
        _require_daemon(authority, daemon_id, timeout=max(0.05, min(5, timeout / 2)))
    result = _run(authority, ("container", "inspect", "--format", _INSPECT, identifier),
                  timeout=max(0.05, deadline - time.monotonic()))
    if result.stderr:
        raise ValueError("CONTAINER_STATE_INVALID")
    rows = result.stdout.decode("ascii").splitlines()
    if len(rows) != 6:
        raise ValueError("CONTAINER_STATE_INVALID")
    cid, actual_name, actual_image, labels, status, code = map(json.loads, rows)
    if (cid != identifier or actual_name != "/" + name or actual_image != image
            or type(labels) is not dict or labels.get("ai.synapticlabs.torch-import.claim") != label
            or status not in {"created", "running", "exited", "dead"} or type(code) is not int):
        raise ValueError("CONTAINER_STATE_INVALID")
    return status, code


def _require_gpu_zero(authority, identifier: str) -> None:
    """Confirm the fixed device request before any GPU-exposed code starts."""
    result = _run(authority, ("container", "inspect", "--format",
                              "{{json .HostConfig.DeviceRequests}}", identifier))
    if result.stderr:
        raise ValueError("GPU_SCOPE_INVALID")
    try:
        requests = json.loads(_one_line(result.stdout))
    except Exception:
        raise ValueError("GPU_SCOPE_INVALID") from None
    if (type(requests) is not list or len(requests) != 1
            or type(requests[0]) is not dict
            or requests[0].get("DeviceIDs") != ["0"]
            or requests[0].get("Capabilities") != [["gpu"]]
            or requests[0].get("Count") != 0
            or requests[0].get("Driver") != ""
            or requests[0].get("Options") != {}):
        raise ValueError("GPU_SCOPE_INVALID")


def _stop_once(authority, identifier: str, *, name: str, label: str,
               image: str, daemon_id: str, output_dir: Path) -> bool:
    """One stop request; only a separate exact inspection can prove terminal state."""
    evidence_retained = True

    def record(phase: str) -> None:
        nonlocal evidence_retained
        try:
            _append(output_dir, {"phase": phase, "container_id": identifier})
        except Exception:
            # Storage failure cannot suppress cleanup of the already claimed
            # container. It does prevent claiming durable terminal evidence.
            evidence_retained = False

    try:
        status, _ = _inspect_owned(authority, identifier, name=name, label=label,
                                   image=image, daemon_id=daemon_id)
        if status in {"exited", "dead"}:
            record("TERMINAL_OBSERVED")
            return evidence_retained
        if status != "running":
            return False
        record("STOP_CLAIMED")
        try:
            _run(authority, ("container", "stop", "--timeout", "2", identifier), timeout=10)
        except Exception:
            # A lost CLI acknowledgement does not establish daemon state.
            # Inspect independently without issuing a second stop.
            pass
        status, _ = _inspect_owned(authority, identifier, name=name, label=label,
                                   image=image, daemon_id=daemon_id)
        if status in {"exited", "dead"}:
            record("TERMINAL_OBSERVED")
            return evidence_retained
    except Exception:
        pass
    record("STOP_UNCONFIRMED")
    return False


def diagnose(*, docker: Path, docker_config: Path, release_path: Path,
             daemon_id: str, output_dir: Path, expected_release_sha256: str,
             source_root: Path | None = None, source_lock: Path | None = None,
             expected_source_lock_sha256: str | None = None,
             expected_manifest_sha256: str | None = None,
             expected_closure_digest: str | None = None,
             gpu_device: str | None = None, worker_context: bool = False,
             worker_entry: bool = False, worker_direct: bool = False,
             expected_workload_sha256: str | None = None,
             expected_runtime_sha256: str | None = None,
             expected_trainer_sha256: str | None = None,
             expected_direct_trainer_sha256: str | None = None,
             expected_direct_manifest_sha256: str | None = None,
             expected_direct_closure_digest: str | None = None,
             artifact_root: Path | None = None, model_receipt: Path | None = None,
             expected_model_receipt_sha256: str | None = None) -> dict[str, object]:
    worker_values = (expected_workload_sha256, expected_runtime_sha256,
                     expected_trainer_sha256)
    direct_values = (expected_direct_trainer_sha256, expected_direct_manifest_sha256,
                     expected_direct_closure_digest, artifact_root, model_receipt,
                     expected_model_receipt_sha256)
    if (worker_entry and not worker_context) or (worker_direct and
            (worker_entry or not worker_context or gpu_device != "0" or
             any(value is None for value in direct_values))) or (
            not worker_direct and any(value is not None for value in direct_values)) or (
            (worker_context and (source_root is None or any(value is None for value in worker_values)))
            or (not worker_context and any(value is not None for value in worker_values))):
        raise ValueError("SOURCE_IDENTITY_INVALID")
    if worker_direct and any(not re.fullmatch(r"[0-9a-f]{64}", value) for value in
                             direct_values[:3]):
        raise ValueError("SOURCE_IDENTITY_INVALID")
    if gpu_device not in (None, "0") or (gpu_device is not None and source_root is None):
        raise ValueError("GPU_SCOPE_INVALID")
    if not re.fullmatch(r"[0-9a-f-]{36}", daemon_id):
        raise ValueError("DAEMON_ID_INVALID")
    release_raw = _read(release_path, 128 * 1024)
    if (not re.fullmatch(r"[0-9a-f]{64}", expected_release_sha256)
            or hashlib.sha256(release_raw).hexdigest() != expected_release_sha256):
        raise ValueError("RELEASE_HASH_INVALID")
    release = json.loads(release_raw)
    if (type(release) is not dict or
            json.dumps(release, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=True).encode("ascii") != release_raw):
        raise ValueError("RELEASE_INVALID")
    image = "sha256:" + release["image"]["digest"]
    python = release["python"]["executable"]
    digest = release["manifest_digest"]
    if (not _SHA.fullmatch(image) or type(python) is not str or not python.startswith("/")
            or type(digest) is not str or not re.fullmatch(r"[0-9a-f]{64}", digest)
            or release["image"]["reference"].split("@")[-1] != image):
        raise ValueError("RELEASE_INVALID")
    parent_info = output_dir.parent.stat() if output_dir.parent.is_dir() else None
    if (output_dir.exists() or output_dir.is_symlink() or parent_info is None
            or output_dir.parent.is_symlink() or parent_info.st_uid != os.getuid()
            or stat.S_IMODE(parent_info.st_mode) != 0o700):
        raise ValueError("OUTPUT_INVALID")
    try:
        _path_custody(output_dir.parent)
    except Exception:
        raise ValueError("OUTPUT_INVALID") from None
    source_values = (source_lock, expected_source_lock_sha256,
                     expected_manifest_sha256, expected_closure_digest)
    if (source_root is None and any(value is not None for value in source_values)) or (
            source_root is not None and any(value is None for value in source_values)):
        raise ValueError("SOURCE_INVALID")
    source_identity = ({} if source_root is None else _source_identity(
        source_root, source_lock, expected_source_lock_sha256,
        expected_manifest_sha256, expected_closure_digest))
    context_identity = ({} if not worker_context else _worker_context_identity(
        source_root, *worker_values, expected_closure_digest))
    if worker_entry or worker_direct:
        try:
            workload = json.loads(_read(source_root.parent / "control/workload.json", 1024 * 1024))
            dataset_ref = workload["configuration"]["document"]["dataset"]["ref"]
        except (KeyError, TypeError, ValueError):
            raise ValueError("SOURCE_IDENTITY_INVALID") from None
        if (type(dataset_ref) is not str or
                not re.fullmatch(r"prepared://sha256/[0-9a-f]{64}", dataset_ref)):
            raise ValueError("SOURCE_IDENTITY_INVALID")
    custody = (() if source_root is None else tuple(_path_custody(path) for path in
        (source_root, source_root.parent / "control/offline-sft-worker-v1.json", source_lock,
         *((source_root.parent / "control/workload.json",) if worker_context else ()))))
    direct_inputs = (None if not worker_direct else _direct_retained_inputs(
        source_root, artifact_root, model_receipt, expected_model_receipt_sha256,
        expected_workload_sha256))
    authority = _authority(docker, docker_config)
    info = _run(authority, ("info", "--format", "{{.ID}}"))
    if info.stderr or _one_line(info.stdout) != daemon_id:
        raise ValueError("DAEMON_INVALID")
    before = _run(authority, ("image", "inspect", "--format", "{{.Id}}", image))
    if before.stderr or _one_line(before.stdout) != image:
        raise ValueError("IMAGE_INVALID")
    output_dir.mkdir(mode=0o700)
    direct_identity = {}
    direct_custody = ()
    mounted_source = source_root
    if worker_direct:
        direct_identity = _stage_direct_source(source_root, output_dir / "derived-source",
            expected_trainer=expected_direct_trainer_sha256,
            expected_manifest=expected_direct_manifest_sha256,
            expected_closure=expected_direct_closure_digest)
        mounted_source = output_dir / "derived-source/engine"
        # A new writable root retains the r4 layout while the old model and
        # prepared input enter only through separate read-only nested mounts.
        scratch = output_dir / "artifacts"
        scratch.mkdir(mode=0o700)
        for leaf in ("artifacts", "state", "tracking", "cache", "tmp"):
            (scratch / "run" / leaf).mkdir(parents=True, mode=0o700)
        model_relative = direct_inputs["snapshot"].relative_to(artifact_root)
        (scratch / model_relative).mkdir(parents=True, mode=0o700)
        input_relative = direct_inputs["prepared"].relative_to(artifact_root)
        input_target = scratch / input_relative
        input_target.parent.mkdir(parents=True, mode=0o700)
        input_target.open("xb").close()
        input_target.chmod(0o600)
        direct_custody = tuple(_path_custody(path) for path in
            (mounted_source, mounted_source.parent / "control/offline-sft-worker-v1.json",
             scratch, direct_inputs["snapshot"], direct_inputs["prepared"]))
        context_identity = _worker_context_identity(
            mounted_source, expected_workload_sha256, expected_runtime_sha256,
            expected_direct_trainer_sha256, expected_direct_closure_digest,
            direct_stop=True)
    nonce = secrets.token_hex(16)
    name = "synaptic-torch-import-" + nonce
    probe = (_worker_probe(context_identity, (expected_direct_closure_digest if worker_direct else
             expected_closure_digest), entry_mode=worker_entry, direct_mode=worker_direct,
             prepared_sha256=(direct_inputs["prepared_sha256"] if worker_direct else "0" * 64),
             prepared_size=(direct_inputs["prepared_size_bytes"] if worker_direct else 0))
             if worker_context else (_PROBE if source_root is None else
             _GUARDED_PROBE.replace("EXPECTED_CLOSURE_DIGEST", expected_closure_digest)))
    if gpu_device is not None:
        if not worker_context and probe.count(_HIDDEN_GPU_ENV) != 1:
            raise ValueError("GPU_SCOPE_INVALID")
        # The r2 isolated child did not inherit either visibility variable.
        # The host device request is separately inspected before start.
        if not worker_context:
            probe = probe.replace(_HIDDEN_GPU_ENV, "")
    claim = {"schema_version": SCHEMA, "daemon_id": daemon_id, "image_id": image,
             "release_sha256": expected_release_sha256, "source_identity": source_identity,
             "context_identity": context_identity,
             "direct_identity": direct_identity,
             "retained_input_identity": ({"receipt_sha256": direct_inputs["receipt_sha256"],
                 "prepared_sha256": direct_inputs["prepared_sha256"],
                 "prepared_size_bytes": direct_inputs["prepared_size_bytes"]}
                 if worker_direct else {}),
             "name": name, "scope": ("worker_direct" if worker_direct else
             "worker_entry" if worker_entry else "worker_context") if worker_context else
             ("bare" if source_root is None else "guarded"),
             "helper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             "probe_sha256": hashlib.sha256(probe.encode("utf-8")).hexdigest(),
             "docker_file_identity": list(authority[2]),
             "docker_config_identity": list(authority[3])}
    if gpu_device is not None:
        claim["gpu_device"] = gpu_device
    claim_raw = json.dumps(claim, sort_keys=True, separators=(",", ":")).encode("ascii")
    label = hashlib.sha256(claim_raw).hexdigest()
    with (output_dir / "claim.json").open("xb") as stream:
        stream.write(claim_raw + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    _append(output_dir, {"phase": "CREATE_CLAIMED", "claim_sha256": label})
    mount = (() if source_root is None else
             ("--mount", f"type=bind,src={mounted_source.parent},dst=/source,readonly"))
    user = (() if source_root is None or worker_direct else
            ("--user", f"{source_root.stat().st_uid}:{source_root.stat().st_gid}"))
    visibility = (("--env=NVIDIA_VISIBLE_DEVICES=void", "--env=CUDA_VISIBLE_DEVICES=")
                  if gpu_device is None else ("--gpus", "device=0"))
    artifact_tmpfs = (("--tmpfs", "/artifacts:rw,nosuid,nodev,size=64m,mode=1777")
                      if worker_context and not worker_direct else ())
    direct_mounts = (() if not worker_direct else (
        "--mount", f"type=bind,src={scratch},dst=/artifacts",
        "--mount", f"type=bind,src={direct_inputs['snapshot']},dst=/artifacts/{model_relative},readonly",
        "--mount", f"type=bind,src={direct_inputs['prepared']},dst=/artifacts/{input_relative},readonly"))
    limits = (("--read-only", "--cap-drop=ALL", "--security-opt=no-new-privileges",
               "--pids-limit=64", "--memory=4g", "--tmpfs", "/tmp:rw,nosuid,nodev,size=64m,mode=1777")
              if not worker_direct else ("--memory=16g",))
    command = (
        "container", "create", "--name", name, "--label", "ai.synapticlabs.torch-import.claim=" + label,
        "--pull=never", "--network=none", "--log-driver", "json-file",
        "--log-opt", "max-size=1m", "--log-opt", "max-file=1",
        "--runtime=runc", *limits, "--cpus=2",
        *artifact_tmpfs,
        *mount, *direct_mounts, *user,
        *visibility,
        "--workdir=" + (_R2_ROOTS["tmp"] if worker_direct else "/tmp"), "--entrypoint", python,
        image, "-I", "-c", probe,
    )
    if source_root is not None and tuple(_path_custody(path) for path in
            (source_root, source_root.parent / "control/offline-sft-worker-v1.json",
             source_lock, *((source_root.parent / "control/workload.json",)
                           if worker_context else ()))) != custody:
        raise ValueError("SOURCE_IDENTITY_INVALID")
    _require_daemon(authority, daemon_id)
    try:
        created = _run(authority, command)
    except Exception:
        _append(output_dir, {"phase": "CREATE_UNCERTAIN"})
        raise ValueError("CREATE_UNCERTAIN") from None
    if created.stderr:
        _append(output_dir, {"phase": "CREATE_UNCERTAIN"})
        raise ValueError("CREATE_UNCERTAIN")
    cid = _one_line(created.stdout)
    if not _CID.fullmatch(cid):
        _append(output_dir, {"phase": "CREATE_UNCERTAIN"})
        raise ValueError("CREATE_UNCERTAIN")
    _append(output_dir, {"phase": "CREATED", "container_id": cid})
    if _inspect_owned(authority, cid, name=name, label=label, image=image,
                      daemon_id=daemon_id)[0] != "created":
        raise ValueError("CONTAINER_STATE_INVALID")
    if gpu_device is not None:
        _require_gpu_zero(authority, cid)
    # Recheck the exact source bytes immediately before starting mounted code.
    if source_root is not None and _source_identity(source_root, source_lock,
            expected_source_lock_sha256, expected_manifest_sha256,
            expected_closure_digest) != source_identity:
        raise ValueError("SOURCE_IDENTITY_INVALID")
    if worker_context and _worker_context_identity(
            (mounted_source if worker_direct else source_root),
            expected_workload_sha256, expected_runtime_sha256,
            (expected_direct_trainer_sha256 if worker_direct else expected_trainer_sha256),
            (expected_direct_closure_digest if worker_direct else expected_closure_digest),
            direct_stop=worker_direct) != context_identity:
        raise ValueError("SOURCE_IDENTITY_INVALID")
    if worker_direct:
        if (_direct_source_identity(mounted_source,
                expected_trainer=expected_direct_trainer_sha256,
                expected_manifest=expected_direct_manifest_sha256,
                expected_closure=expected_direct_closure_digest) != direct_identity
                or _direct_retained_inputs(source_root, artifact_root, model_receipt,
                    expected_model_receipt_sha256, expected_workload_sha256) != direct_inputs
                or tuple(_path_custody(path) for path in
                    (mounted_source, mounted_source.parent / "control/offline-sft-worker-v1.json",
                     scratch, direct_inputs["snapshot"], direct_inputs["prepared"])) != direct_custody):
            raise ValueError("SOURCE_IDENTITY_INVALID")
    if source_root is not None and tuple(_path_custody(path) for path in
            (source_root, source_root.parent / "control/offline-sft-worker-v1.json",
             source_lock, *((source_root.parent / "control/workload.json",)
                           if worker_context else ()))) != custody:
        raise ValueError("SOURCE_IDENTITY_INVALID")
    deadline = time.monotonic() + _RUN_SECONDS
    _append(output_dir, {"phase": "START_CLAIMED", "container_id": cid})
    cleanup_attempted = False
    start_acknowledged = False
    try:
        started = _run(authority, ("container", "start", cid), timeout=15)
        if started.stderr or _one_line(started.stdout) != cid:
            raise ValueError("START_UNCERTAIN")
        start_acknowledged = True
        _append(output_dir, {"phase": "STARTED", "container_id": cid})
        while True:
            if time.monotonic() >= deadline:
                cleanup_attempted = True
                confirmed = _stop_once(authority, cid, name=name, label=label,
                    image=image, daemon_id=daemon_id, output_dir=output_dir)
                raise ValueError("DEADLINE_STOPPED" if confirmed else "DEADLINE_UNCERTAIN")
            status, exit_code = _inspect_owned(authority, cid, name=name, label=label,
                image=image, daemon_id=daemon_id,
                timeout=max(0.1, min(5, deadline - time.monotonic())))
            if status in {"exited", "dead"}:
                _append(output_dir, {"phase": "TERMINAL_OBSERVED", "container_id": cid,
                                     "exit_code": exit_code})
                break
            if time.monotonic() >= deadline:
                cleanup_attempted = True
                confirmed = _stop_once(authority, cid, name=name, label=label,
                    image=image, daemon_id=daemon_id, output_dir=output_dir)
                raise ValueError("DEADLINE_STOPPED" if confirmed else "DEADLINE_UNCERTAIN")
            time.sleep(min(0.5, max(0, deadline - time.monotonic())))
    except BaseException as error:
        if not cleanup_attempted:
            cleanup_attempted = True
            confirmed = _stop_once(authority, cid, name=name, label=label,
                image=image, daemon_id=daemon_id, output_dir=output_dir)
            if not confirmed:
                raise ValueError("DEADLINE_UNCERTAIN") from None
        if isinstance(error, KeyboardInterrupt):
            raise
        if isinstance(error, ValueError) and error.args and error.args[0] in {"DEADLINE_UNCERTAIN", "DEADLINE_STOPPED"}:
            raise
        raise ValueError("DEADLINE_UNCERTAIN" if start_acknowledged else "START_UNCERTAIN") from None
    if exit_code != 0:
        raise ValueError("CHILD_OUTPUT_INVALID")
    result = _run(authority, ("container", "logs", cid))
    if result.stderr:
        raise ValueError("CHILD_OUTPUT_INVALID")
    child = json.loads(_one_line(result.stdout))
    fields = ({"schema_version", "category", "stage", "context", "boundary"}
              if worker_context else ({"schema_version", "category"} if source_root is None
              else {"schema_version", "category", "stage"}))
    if (type(child) is not dict or set(child) != fields
            or child["schema_version"] != "synaptic-docker-torch-import-child/v1"
            or child["category"] not in _CATEGORIES
            or (worker_context and (child["boundary"] != "BEFORE_UNSLOTH"
                or child["stage"] not in {"READ_WORKLOAD", "ISOLATED_CHILD",
                    "WORKER_IMPORT", "WORKER_CLOSURE", "WORKER_GUARD",
                    "TRAINER_PREFIX", "BOOTSTRAP_ENV", "TORCH_IMPORT",
                    "ENTRY_ADMISSION", "WORKER_MAIN", "WORKER_ORIGIN",
                    "WORKER_ARGUMENTS", "WORKER_POSTCHECK", "ISOLATED_CHILD",
                    "UNSLOTH_IMPORT"}
                or type(child["context"]) is not dict
                or (worker_direct and (child["context"] != {} or
                    (child["category"] == "PASS" and child["stage"] != "UNSLOTH_IMPORT") or
                    (child["category"] == "IMPORT" and child["stage"] != "TORCH_IMPORT") or
                    (child["category"] not in {"PASS", "IMPORT", "OTHER"})))
                or (child["category"] == "PASS" and not worker_direct and
                    (child["stage"] != "TORCH_IMPORT" or
                     set(child["context"]) != ({"environment_supplied_sha256",
                        "environment_observed_sha256", "startup_env_delta",
                        "python_isolated", "cwd_sha256", "file_sha256", "sys_path_sha256"}
                        | ({"argv_sha256", "runpy_main_bound",
                            "environment_before_torch_sha256",
                            "environment_after_torch_sha256"} if worker_entry else set()))
                     or child["context"]["python_isolated"] is not True
                     or (worker_entry and child["context"]["runpy_main_bound"] is not True)
                     or child["context"]["startup_env_delta"] not in
                        {"NONE", "LC_CTYPE_C_UTF8"}))
                or (child["category"] != "PASS" and child["context"] != {})
                or any(type(value) is not str or not re.fullmatch(r"[0-9a-f]{64}", value)
                       for key, value in child["context"].items() if key.endswith("_sha256"))))
            or (source_root is not None and not worker_context and child["stage"] not in {"IMPORT_WORKER", "READ_MANIFEST",
                "LOAD_CLOSURE", "VERIFY_ORIGINS", "INSTALL_GUARD", "BOOTSTRAP_ENV", "TORCH_IMPORT"})):
        raise ValueError("CHILD_OUTPUT_INVALID")
    if worker_context and child["category"] == "PASS" and not worker_direct:
        context = child["context"]
        expected_observed = (context_identity["child_env_sha256"]
            if context["startup_env_delta"] == "NONE"
            else context_identity["child_env_lc_ctype_sha256"])
        if (context["environment_supplied_sha256"] != context_identity["child_env_sha256"]
                or context["environment_observed_sha256"] != expected_observed):
            raise ValueError("CHILD_OUTPUT_INVALID")
    if _inspect_owned(authority, cid, name=name, label=label, image=image,
                      daemon_id=daemon_id)[0] not in {"exited", "dead"}:
        raise ValueError("CONTAINER_STATE_INVALID")
    after = _run(authority, ("image", "inspect", "--format", "{{.Id}}", image))
    if after.stderr or _one_line(after.stdout) != image:
        raise ValueError("IMAGE_CHANGED")
    report = {"schema_version": SCHEMA, "category": child["category"],
              "container_id": cid, "image_id": image, "daemon_id": daemon_id,
              "runtime_release_digest": digest,
              "training_executed": False, "gpu_used": None if gpu_device is not None else False,
              "scope": ("worker_direct" if worker_direct else
              "worker_entry" if worker_entry else "worker_context") if worker_context else
              ("bare" if source_root is None else "guarded"),
              "claim_sha256": label, "source_identity": source_identity,
              "context_identity": context_identity,
              "direct_identity": direct_identity,
              "retained_input_identity": claim["retained_input_identity"],
              "terminal_observed": True}
    if source_root is not None:
        report["stage"] = child["stage"]
    if worker_context:
        report["context"] = child["context"]
        report["boundary"] = child["boundary"]
    if gpu_device is not None:
        report["gpu_device"] = gpu_device
    with (output_dir / "result.json").open("x", encoding="ascii") as stream:
        json.dump(report, stream, sort_keys=True, separators=(",", ":"))
        stream.write("\n")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("docker", "docker-config", "release", "output-dir"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--daemon-id", required=True)
    parser.add_argument("--expected-release-sha256", required=True)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--source-lock", type=Path)
    parser.add_argument("--expected-source-lock-sha256")
    parser.add_argument("--expected-manifest-sha256")
    parser.add_argument("--expected-closure-digest")
    parser.add_argument("--gpu-device")
    parser.add_argument("--worker-context", action="store_true")
    parser.add_argument("--worker-entry", action="store_true")
    parser.add_argument("--worker-direct", action="store_true")
    parser.add_argument("--expected-workload-sha256")
    parser.add_argument("--expected-runtime-sha256")
    parser.add_argument("--expected-trainer-sha256")
    parser.add_argument("--expected-direct-trainer-sha256")
    parser.add_argument("--expected-direct-manifest-sha256")
    parser.add_argument("--expected-direct-closure-digest")
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--model-receipt", type=Path)
    parser.add_argument("--expected-model-receipt-sha256")
    args = parser.parse_args(argv)
    try:
        report = diagnose(docker=args.docker, docker_config=args.docker_config,
                          release_path=args.release, daemon_id=args.daemon_id,
                          output_dir=args.output_dir,
                          expected_release_sha256=args.expected_release_sha256,
                          source_root=args.source_root, source_lock=args.source_lock,
                          expected_source_lock_sha256=args.expected_source_lock_sha256,
                          expected_manifest_sha256=args.expected_manifest_sha256,
                          expected_closure_digest=args.expected_closure_digest,
                          gpu_device=args.gpu_device, worker_context=args.worker_context,
                          worker_entry=args.worker_entry, worker_direct=args.worker_direct,
                          expected_workload_sha256=args.expected_workload_sha256,
                          expected_runtime_sha256=args.expected_runtime_sha256,
                          expected_trainer_sha256=args.expected_trainer_sha256,
                          expected_direct_trainer_sha256=args.expected_direct_trainer_sha256,
                          expected_direct_manifest_sha256=args.expected_direct_manifest_sha256,
                          expected_direct_closure_digest=args.expected_direct_closure_digest,
                          artifact_root=args.artifact_root, model_receipt=args.model_receipt,
                          expected_model_receipt_sha256=args.expected_model_receipt_sha256)
    except BaseException as error:
        reason = error.args[0] if type(error) is ValueError and len(error.args) == 1 else None
        if type(reason) is not str or reason not in _REASONS:
            reason = "OTHER"
        print(json.dumps({"schema_version": SCHEMA, "status": "REJECTED",
                          "reason_code": reason}, sort_keys=True, separators=(",", ":")), file=sys.stderr)
        return 2
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
