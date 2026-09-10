from __future__ import annotations

import pytest

from tuner.execution.providers.modal.deployment_v1 import (
    APP_NAME, ARTIFACT_MOUNT, BOOTSTRAP_SOURCE_MODULES, CONTROL_MOUNT,
    ModalDeploymentSpecV1,
)
from tuner.execution.providers.modal.deployment_identity import modal_function_name


DEPLOYMENT_REF = "modal-deployment-" + "1" * 32
FUNCTION_NAME = modal_function_name(DEPLOYMENT_REF)


class Handle:
    def __init__(self,kind,value,kwargs):self.kind=kind;self.value=value;self.kwargs=kwargs;self.commits=0
    def commit(self):self.commits+=1


class Volume:
    calls=[]
    @classmethod
    def from_name(cls,value,**kwargs):h=Handle("volume",value,kwargs);cls.calls.append(h);return h


class Secret:
    calls=[]
    @classmethod
    def from_name(cls,value,**kwargs):h=Handle("secret",value,kwargs);cls.calls.append(h);return h


class ImageValue:
    def __init__(self,reference):self.reference=reference;self.entrypoint_value=None;self.environment=None;self.local_sources=None
    def entrypoint(self,value):self.entrypoint_value=value;return self
    def env(self,value):self.environment=value;return self
    def add_local_python_source(self,*modules,**kwargs):self.local_sources=(modules,kwargs);return self


class Image:
    calls=[]
    @classmethod
    def from_registry(cls,value):result=ImageValue(value);cls.calls.append(result);return result


class App:
    calls=[]
    def __init__(self,*args,**kwargs):self.args=args;self.kwargs=kwargs;self.function_kwargs=None;type(self).calls.append(self)
    def function(self,**kwargs):
        self.function_kwargs=kwargs
        def decorate(function):function.modal_kwargs=kwargs;return function
        return decorate


class SDK:
    __version__="1.5.4";Volume=Volume;Secret=Secret;Image=Image;App=App
    @staticmethod
    def current_function_call_id():return "fc-1"




@pytest.mark.parametrize(
    "environment",
    (
        {"HF_TOKEN": "literal-secret"},
        {"SAFE": "SYNAPTIC_EVIDENCE_MAC_KEY"},
        {"MODAL_TOKEN_SECRET": "literal-secret"},
        {"DATABASE_PASSWORD": "literal-secret"},
    ),
)
def test_deployment_rejects_raw_secret_environment(environment):
    with pytest.raises(ValueError, match="named Modal Secrets"):
        ModalDeploymentSpecV1(
            DEPLOYMENT_REF, FUNCTION_NAME,
            "registry.example/runtime@sha256:" + "a" * 64,
            "control-v1", "artifact-v1", "runtime-v1",
            ("HF_TOKEN", "SYNAPTIC_EVIDENCE_MAC_KEY"), environment,
        )


def test_deployment_environment_is_detached_and_immutable():
    environment = {"PYTHONNOUSERSITE": "1"}
    spec = ModalDeploymentSpecV1(
        DEPLOYMENT_REF, FUNCTION_NAME,
        "registry.example/runtime@sha256:" + "a" * 64,
        "control-v1", "artifact-v1", "runtime-v1",
        ("HF_TOKEN", "SYNAPTIC_EVIDENCE_MAC_KEY"), environment,
    )
    environment["PYTHONNOUSERSITE"] = "0"
    assert spec.environment["PYTHONNOUSERSITE"] == "1"
    with pytest.raises(TypeError):
        spec.environment["PYTHONNOUSERSITE"] = "0"
