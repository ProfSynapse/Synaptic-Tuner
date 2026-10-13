"""Retained-input bridge to the existing environment validator/session seams.

Preparing a validator is effect-free. The host supplies the runtime factory;
this module does not compose credentials, a worker, grants or public starts.
"""
from copy import deepcopy
import json
import time

from shared.environments.bounded_local_runtime import FilesystemLimits
from shared.environments.local_worker import WorkerLifecycleError
from shared.environments.tool_executor import SUPPORTED_ACTIONS
from shared.environments.validator import EnvironmentSession, EnvironmentValidator
from shared.validation.parsing.cli_commands import build_cli_command_catalog
from shared.validation.parsing.configured_formats import build_wrapper_specs
from SynthChat.config.format_resolver import parse_tool_call_formats
from SynthChat.utils.yaml_loader import parse_workflow_yaml

from synaptic_tuner.api.v1.data_facade import DataMode, DataOperationCode, DataOperationError
from synaptic_tuner.api.v1.environment_execution import EnvironmentMode
from .environment import _schema


def _invalid():
    return DataOperationError(DataOperationCode.ENVIRONMENT_INVALID)


def _document(inputs, role):
    dependency, = (item for item in inputs.binding.dependencies if item.role == role)
    return parse_workflow_yaml(inputs.document_bytes(role, dependency.ref))


def _project(value, path):
    for token in path:
        if type(token) is str and type(value) is dict:
            value = value[token]
        elif type(token) is int and type(value) is list:
            value = value[token]
        else:
            raise _invalid()
    return deepcopy(value)


class CapturedEnvironmentSessionFactory:
    """Host-configured factory; runtime construction happens only on session open."""

    def __init__(self, runtime_factory):
        if not callable(runtime_factory):
            raise TypeError("runtime_factory must be callable")
        self.runtime_factory = runtime_factory

    def prepare(self, plan, admitted_inputs, boundary):
        try:
            validator = CapturedEnvironmentValidator(admitted_inputs, self.runtime_factory, boundary)
            if plan.request.mode is DataMode.IMPROVE or plan.request.max_iterations > 1:
                validator.validate_projection()
            for target in plan.scenarios if plan.request.mode is DataMode.GENERATE else ():
                from SynthChat.input_snapshot import WorkflowInputSnapshot
                definitions = WorkflowInputSnapshot(admitted_inputs.workflow_documents()).validate_generation(
                    (target.scenario_ref,), max_iterations=plan.request.max_iterations)
                validator.validate_configuration(definitions[target.scenario_ref].get("environment", {}))
            return validator
        except DataOperationError:
            raise
        except Exception:
            raise _invalid() from None


class CapturedEnvironmentValidator(EnvironmentValidator):
    def __init__(self, inputs, runtime_factory, boundary):
        self.binding = inputs.binding
        self._pending_session = None
        self.boundary = boundary
        self._deadline = time.monotonic() + self.binding.bounds.max_seconds
        self.last_evaluation = None
        self.runtime_factory = runtime_factory
        self.tool_schema = _document(inputs, "tool_schema")
        policy = _document(inputs, "execution_policy")
        self.execution_config = deepcopy(policy.get("execution", {}))
        allowed_config = {"strict_schema", "default_action", "tool_action_hints", "key_hints",
                          "verb_rules", "read_output", "candidate_projection"}
        if set(self.execution_config) - allowed_config:
            raise _invalid()
        for key in ("tool_action_hints", "key_hints", "verb_rules", "read_output", "candidate_projection"):
            if key in self.execution_config and type(self.execution_config[key]) is not dict:
                raise _invalid()
        hints = self.execution_config.get("tool_action_hints", {})
        if any(type(action) is not str or action not in SUPPORTED_ACTIONS for action in hints.values()):
            raise _invalid()
        if set(self.execution_config.get("verb_rules", {})) - SUPPORTED_ACTIONS:
            raise _invalid()
        default = self.execution_config.get("default_action", "simulate")
        if default not in SUPPORTED_ACTIONS:
            raise _invalid()
        self.execution_config["default_action"] = default
        self.execution_config["strict_schema"] = True
        if type(self.tool_schema.get("tools")) is not dict:
            raise _invalid()
        catalog_document = policy.get("cli_catalog", {"tools": []})
        if set(catalog_document) != {"tools"} or type(catalog_document["tools"]) is not list:
            raise _invalid()
        self.catalog = build_cli_command_catalog(catalog_document)
        if len(self.catalog) != len(catalog_document["tools"]):
            raise _invalid()
        registry = parse_tool_call_formats(parse_workflow_yaml(
            inputs.document_bytes("registry", "registry:tool_call_formats")), require_complete=True)
        self.wrapper_specs = build_wrapper_specs(registry)
        self.allowed_actions = self.binding.allowed_actions
        self.simulate_only = self.binding.selection.mode is EnvironmentMode.SIMULATE
        self.safe_errors = self.bounded_execution = True
        self.fixture_validator = _schema(_document(inputs, "fixture_schema"))
        bounds = self.binding.bounds
        self.limits = FilesystemLimits(bounds.max_fixture_files, bounds.max_fixture_bytes,
            bounds.max_output_bytes, bounds.max_snapshot_bytes, 32, 1024, bounds.max_seconds)

    def validate_projection(self):
        projection = self.execution_config.get("candidate_projection")
        if type(projection) is not dict or set(projection) - {"response", "system", "environment"} or "response" not in projection:
            raise _invalid()
        for path in projection.values():
            if type(path) is not list or len(path) > 32 or any(type(token) not in (str, int) for token in path):
                raise _invalid()

    def validate_configuration(self, config):
        if type(config) is not dict or set(config) & {"execution", "action_hints", "backend", "backend_ref",
                "scope_ref", "bounds", "allowed_actions", "mode", "credentials", "network"}:
            raise _invalid()
        try:
            self.fixture_validator.validate(config)
        except Exception:
            raise _invalid() from None
        max_steps = config.get("max_steps", self.binding.bounds.max_tool_calls)
        if type(max_steps) is not int or not 1 <= max_steps <= self.binding.bounds.max_tool_calls:
            raise _invalid()
        return deepcopy(config)

    def _create_runtime(self):
        self.check_boundary()
        return _RuntimeChecks(self.runtime_factory(limits=self.limits, boundary=self.check_boundary), self.check_boundary)

    def check_boundary(self):
        self.boundary()
        if time.monotonic() >= self._deadline:
            raise TimeoutError("environment workflow deadline exceeded")

    def start_session(self, system_prompt, environment_config=None):
        if self._pending_session is not None and not self._pending_session.closed:
            raise DataOperationError(DataOperationCode.ENVIRONMENT_UNAVAILABLE)
        config = self.validate_configuration(environment_config if environment_config is not None else {})
        config["max_steps"] = config.get("max_steps", self.binding.bounds.max_tool_calls)
        self.check_boundary()
        session = CapturedEnvironmentSession.__new__(CapturedEnvironmentSession)
        self._pending_session = session
        session.__init__(validator=self, system_prompt=system_prompt, environment_config=config)
        return session

    def candidate_evaluator(self, environment_config=None):
        self.validate_projection()
        projection = self.execution_config["candidate_projection"]
        def evaluate(candidate, candidate_digest, base_context):
            system = _project(candidate, projection["system"]) if "system" in projection else ""
            config = environment_config if environment_config is not None else (
                _project(candidate, projection["environment"]) if "environment" in projection else {})
            response = _project(candidate, projection["response"])
            result = self.validate_response(system, response, config)
            evidence = result.to_dict()
            evidence["candidate_digest"] = candidate_digest
            evidence["mode"] = self.binding.selection.mode.value
            self.last_evaluation = deepcopy(evidence)
            return {**base_context, "environment_result": evidence, "environment_passed": result.passed,
                    "environment_issue_summary": "\n".join(issue.message for issue in result.issues),
                    "candidate_digest": candidate_digest}
        return evaluate


class CapturedEnvironmentSession(EnvironmentSession):
    def __post_init__(self):
        try:
            super().__post_init__()
        except BaseException:
            if hasattr(self, "runtime"):
                self.close()
            raise

    def execute_response(self, response):
        if getattr(self, "_closing", False):
            raise DataOperationError(DataOperationCode.ENVIRONMENT_UNAVAILABLE)
        self.validator.check_boundary()
        if len(self.steps) >= self.validator.binding.bounds.max_turns:
            raise _invalid()
        raw = json.dumps(response, ensure_ascii=False, allow_nan=False).encode()
        if len(raw) > self.validator.binding.bounds.max_output_bytes:
            raise _invalid()
        result = super().execute_response(response)
        self.runtime.check_failure()
        self.validator.check_boundary()
        return result

    def finalize(self, **kwargs):
        self.validator.check_boundary()
        result = super().finalize(**kwargs)
        self.runtime.check_failure()
        result.mode = self.validator.binding.selection.mode.value
        for issue in result.issues:
            if issue.code == "missing_path":
                issue.message = "A requested fixture path does not exist"
            elif issue.code == "action_denied":
                issue.message = "The requested action is not permitted"
            elif issue.code == "assertion_read_failed":
                issue.message = "Unable to read an asserted fixture path"
            else:
                issue.message = "Fixture validation failed" if issue.level.lower() == "error" else "Environment evidence unavailable"
        raw = json.dumps(result.to_dict(), ensure_ascii=False, allow_nan=False).encode()
        if len(raw) > self.validator.binding.bounds.max_output_bytes + self.validator.binding.bounds.max_snapshot_bytes:
            raise _invalid()
        return result

    def close(self):
        if not self.closed:
            self._closing = True
            try:
                self.runtime.teardown()
            except WorkerLifecycleError:
                raise
            except Exception:
                raise DataOperationError(DataOperationCode.ENVIRONMENT_UNAVAILABLE) from None
            self.closed = True
            if self.validator._pending_session is self:
                self.validator._pending_session = None


class _RuntimeChecks:
    """Keep swallowed legacy snapshot/assertion transport failures fatal locally."""
    def __init__(self, runtime, boundary):
        self.runtime, self.boundary, self.failure = runtime, boundary, None

    def check_failure(self):
        if self.failure is not None:
            raise self.failure

    def __getattr__(self, name):
        callback = getattr(self.runtime, name)
        if not callable(callback):
            return callback
        def invoke(*args, **kwargs):
            if name != "teardown":
                self.check_failure()
                self.boundary()
            try:
                return callback(*args, **kwargs)
            except Exception as error:
                if isinstance(error, TimeoutError) or not isinstance(error, (OSError, ValueError, TypeError)):
                    self.failure = error
                raise
        return invoke
