"""Closed execution, stage, run, and cancellation references."""

from __future__ import annotations

from dataclasses import dataclass

from .canonical import safe_ref


@dataclass(frozen=True, slots=True)
class ExecutionScopeV1:
    account_ref: str
    namespace_ref: str

    def __post_init__(self) -> None:
        safe_ref(self.account_ref, "account_ref")
        safe_ref(self.namespace_ref, "namespace_ref")

    def to_dict(self) -> dict[str, object]:
        return {"account_ref": self.account_ref, "namespace_ref": self.namespace_ref}


@dataclass(frozen=True, slots=True)
class StageRefV1:
    stage_effect_id: str
    authenticated_receipt_digest: str

    def __post_init__(self) -> None:
        from .canonical import digest_text
        safe_ref(self.stage_effect_id, "stage_effect_id")
        digest_text(self.authenticated_receipt_digest, "authenticated_receipt_digest")

@dataclass(frozen=True, slots=True)
class StagePredecessorV2:
    provider_id:str;profile_ref:str;account_ref:str;namespace_ref:str;project_ref:str;run_id:str;plan_fingerprint:str;preparation_digest:str;workload_digest:str;stage_effect_id:str;authenticated_receipt_digest:str;record_digest:str
    def __post_init__(self):
        from .canonical import digest_text
        for name in ("provider_id","profile_ref","account_ref","namespace_ref","project_ref","run_id","stage_effect_id"):safe_ref(getattr(self,name),name)
        for name in ("plan_fingerprint","preparation_digest","workload_digest","authenticated_receipt_digest","record_digest"):digest_text(getattr(self,name),name)
    def to_dict(self):return {name:getattr(self,name) for name in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class ProviderRunRefV1:
    provider_job_ref: str

    def __post_init__(self) -> None:
        safe_ref(self.provider_job_ref, "provider_job_ref")

    def to_dict(self) -> dict[str, object]:
        return {"provider_job_ref": self.provider_job_ref}

@dataclass(frozen=True, slots=True)
class ProviderStageRefV1:
    provider_id:str;profile_ref:str;account_ref:str;namespace_ref:str;stage_ref:str
    def __post_init__(self):
        for name in self.__dataclass_fields__:safe_ref(getattr(self,name),name)
    def to_dict(self):return {name:getattr(self,name) for name in self.__dataclass_fields__}

@dataclass(frozen=True, slots=True)
class ScopedProviderRunRefV1:
    provider_id:str;profile_ref:str;account_ref:str;namespace_ref:str;provider_job_ref:str
    def __post_init__(self):
        for name in self.__dataclass_fields__:safe_ref(getattr(self,name),name)
    def to_dict(self):return {name:getattr(self,name) for name in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class CancellationRefV1:
    run: ProviderRunRefV1
    reason_digest: str

    def __post_init__(self) -> None:
        from .canonical import digest_text
        if not isinstance(self.run, ProviderRunRefV1):
            raise TypeError("run must be ProviderRunRefV1")
        digest_text(self.reason_digest, "reason_digest")

@dataclass(frozen=True, slots=True)
class WorkerRefV1:
    worker_ref: str
    def __post_init__(self) -> None: safe_ref(self.worker_ref,"worker_ref")

@dataclass(frozen=True, slots=True)
class OwnerRefV1:
    owner_ref: str
    def __post_init__(self) -> None: safe_ref(self.owner_ref,"owner_ref")
