"""No caller treats a ``BaseLLMClient`` completion as a string or a dict (slice 11).

``BaseLLMClient.chat`` returns ``LLMCompletionV1`` and ``structured_output``
returns ``LLMStructuredV1``. This test pins the migration three ways:

1. every bundled adapter's ``chat`` / ``structured_output`` returns the exact
   wrapper type on every return path, and the base class annotates them so;
2. the census of production modules that call ``.chat(`` / ``.structured_output(``
   is exactly the classified list below (a new caller must be classified here);
3. in every ``BaseLLMClient`` consumer the completion is consumed only through
   ``.text`` / ``.value`` / ``.usage`` (or discarded, or re-typed and returned
   by a proxy), never compared, parsed or passed on as the provider string.

Modules under other protocols (``Evaluator`` ``BackendClient.chat`` returning
``BackendResponse``; the Modal chat session; Trainer inference helpers) are
listed so the census stays exact but are not subject to rule 3.
"""

from __future__ import annotations

import ast
import io
from pathlib import Path
import re
import tokenize

ROOT = Path(__file__).resolve().parents[2]
PRODUCTION_DIRS = ("shared", "SynthChat", "Evaluator", "tuner", "synaptic_tuner", "scripts", "examples", "Trainers")
SKIP_DIRS = frozenset({"__pycache__", "node_modules", "scratch", "_tmp", "_worktrees", "tests"})
CALL_RE = re.compile(r"\.(chat|structured_output)\(")
METHODS = frozenset({"chat", "structured_output"})
ALLOWED_ATTRIBUTES = frozenset({"text", "value", "usage"})
WRAPPER = {"chat": "LLMCompletionV1", "structured_output": "LLMStructuredV1"}

BASE_CLIENT_CONSUMERS = frozenset({
    "Evaluator/shared_llm_adapters.py",
    "SynthChat/llm/caller.py",
    "SynthChat/services/core/improvement_service.py",
    "SynthChat/services/core/judge_service.py",
    "SynthChat/services/privacy_preprocess.py",
    "shared/agentic_judge.py",
    "shared/flywheel/experiment_loop.py",
    "shared/judge/judge_service.py",
    "shared/llm/metering.py",
    "shared/llm/providers/unsloth.py",
    "shared/prompt_optimization/service.py",
    "shared/stage_judges.py",
    "tuner/handlers/generate_handler.py",
})
OTHER_PROTOCOL_CALLERS = frozenset({
    "Evaluator/llamacpp_client.py",
    "Evaluator/mlc_client.py",
    "Evaluator/runner.py",
    "Evaluator/vllm_client.py",
    "Trainers/kto/src/inference.py",
    "Trainers/sft/src/inference.py",
    "examples/modal_chat/consumer.py",
    "scripts/chat_model.py",
    "synaptic_tuner/api/v1/reference/evaluation.py",
    "synaptic_tuner/api/v1/reference/chat.py",
    "tuner/execution/providers/modal/inference_channel.py",
    "tuner/inference/chat_session.py",
})
DOCUMENTATION_ONLY = frozenset({
    "shared/judge/schema_builder.py",
    "shared/llm/__init__.py",
    "shared/llm/factory.py",
})


def _production_files():
    for directory in PRODUCTION_DIRS:
        for path in sorted((ROOT / directory).rglob("*.py")):
            if SKIP_DIRS.isdisjoint(path.relative_to(ROOT).parts):
                yield path


def _code_text(source: str) -> str:
    """The source with every comment and string literal blanked out."""
    kept = []
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type in (tokenize.COMMENT, tokenize.STRING):
            continue
        kept.append(token.string)
    return " ".join(kept)


def _method_calls(tree: ast.AST):
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in METHODS:
            yield node


def _parents(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    parents: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    return parents


def _enclosing_function(node: ast.AST, parents: dict[ast.AST, ast.AST]):
    while node in parents:
        node = parents[node]
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            return node
    return None


def _returns_of(function: ast.FunctionDef):
    """``Return`` nodes of ``function`` itself, not of nested functions."""
    stack = list(function.body)
    while stack:
        node = stack.pop()
        if isinstance(node, ast.Return):
            yield node
        elif not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
            stack.extend(ast.iter_child_nodes(node))


def test_census_of_completion_call_sites_is_exactly_the_classified_list() -> None:
    found = set()
    for path in _production_files():
        if CALL_RE.search(path.read_text(encoding="utf-8")):
            found.add(path.relative_to(ROOT).as_posix())
    expected = BASE_CLIENT_CONSUMERS | OTHER_PROTOCOL_CALLERS | DOCUMENTATION_ONLY
    assert found == expected, (sorted(found - expected), sorted(expected - found))
    assert not (BASE_CLIENT_CONSUMERS & OTHER_PROTOCOL_CALLERS)
    assert not ((BASE_CLIENT_CONSUMERS | OTHER_PROTOCOL_CALLERS) & DOCUMENTATION_ONLY)
    for relative in DOCUMENTATION_ONLY:
        source = (ROOT / relative).read_text(encoding="utf-8")
        assert CALL_RE.search(_code_text(source)) is None, relative
        assert not list(_method_calls(ast.parse(source))), relative


def test_every_bundled_adapter_returns_the_wrapper_types_on_every_path() -> None:
    adapters = sorted(path for path in (ROOT / "shared/llm/providers").glob("*.py") if path.name != "__init__.py")
    assert [path.name for path in adapters] == [
        "lmstudio.py", "ollama.py", "openai_responses.py", "openrouter.py", "unsloth.py",
    ]
    for path in adapters:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
        methods = {
            node.name: node for cls in classes for node in cls.body
            if isinstance(node, ast.FunctionDef) and node.name in METHODS
        }
        assert set(methods) == METHODS, path.name
        for name, function in methods.items():
            returns = list(_returns_of(function))
            assert returns, (path.name, name)
            for node in returns:
                value = node.value
                assert isinstance(value, ast.Call) and isinstance(value.func, ast.Name), (path.name, name, node.lineno)
                assert value.func.id == WRAPPER[name], (path.name, name, node.lineno)
            annotation = function.returns
            assert isinstance(annotation, ast.Name) and annotation.id == WRAPPER[name], (path.name, name)

    base = ast.parse((ROOT / "shared/llm/base.py").read_text(encoding="utf-8"))
    (client,) = [node for node in base.body if isinstance(node, ast.ClassDef) and node.name == "BaseLLMClient"]
    for node in client.body:
        if isinstance(node, ast.FunctionDef) and node.name in METHODS:
            assert isinstance(node.returns, ast.Name) and node.returns.id == WRAPPER[node.name], node.name


def _violations(path: Path) -> list[str]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    parents = _parents(tree)
    problems: list[str] = []
    label = path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else path.name

    def use_is_allowed(name_node: ast.Name, function) -> bool:
        parent = parents.get(name_node)
        if isinstance(parent, ast.Attribute) and parent.attr in ALLOWED_ATTRIBUTES:
            return True
        if isinstance(parent, ast.Call) and isinstance(parent.func, ast.Name) and parent.func.id == "type":
            return True  # a proxy re-typing the wrapper before returning it
        if isinstance(parent, ast.Return) and getattr(function, "name", None) in METHODS:
            return True  # a proxy returning the wrapper unchanged
        return False

    for call in _method_calls(tree):
        parent = parents[call]
        line = call.lineno
        if isinstance(parent, ast.Expr):
            continue  # result discarded
        if isinstance(parent, ast.Attribute) and parent.attr in ALLOWED_ATTRIBUTES:
            continue
        if isinstance(parent, ast.Assign) and len(parent.targets) == 1 and isinstance(parent.targets[0], ast.Name):
            function = _enclosing_function(parent, parents)
            bound = parent.targets[0].id
            scope = function if function is not None else tree
            for node in ast.walk(scope):
                if isinstance(node, ast.Name) and node.id == bound and isinstance(node.ctx, ast.Load):
                    if not use_is_allowed(node, function):
                        problems.append(f"{label}:{node.lineno} uses `{bound}` as the raw completion")
            continue
        problems.append(f"{label}:{line} consumes the completion without .text/.value/.usage")
    return problems


def test_base_client_consumers_read_the_completion_only_through_its_fields() -> None:
    problems: list[str] = []
    for relative in sorted(BASE_CLIENT_CONSUMERS):
        path = ROOT / relative
        assert list(_method_calls(ast.parse(path.read_text(encoding="utf-8")))), relative
        problems.extend(_violations(path))
    assert problems == []


def test_checker_rejects_a_caller_that_treats_the_completion_as_a_string(tmp_path) -> None:
    offending = tmp_path / "offender.py"
    offending.write_text(
        "def run(client):\n"
        "    text = client.chat([])\n"
        "    return text.strip()\n"
        "\n"
        "def other(client):\n"
        "    return json.loads(client.structured_output([], {}))\n"
        "\n"
        "def fine(client):\n"
        "    completion = client.chat([])\n"
        "    return completion.text\n",
        encoding="utf-8",
    )
    problems = _violations(offending)
    assert len(problems) == 2 and all("offender.py" in item for item in problems)
