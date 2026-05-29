from __future__ import annotations

from tuner.cli.parser import create_parser
from tuner.cli.router import route_command


def test_prompt_optimize_parser_and_router_dispatch(monkeypatch, tmp_path):
    config = tmp_path / "prompt-opt.yaml"
    output_dir = tmp_path / "out"
    parser = create_parser()

    args = parser.parse_args(
        [
            "prompt-optimize",
            "--prompt-opt-config",
            str(config),
            "--prompt-opt-output-dir",
            str(output_dir),
            "--json",
        ]
    )

    assert args.command == "prompt-optimize"
    assert args.prompt_opt_config == str(config)
    assert args.prompt_opt_output_dir == str(output_dir)
    assert args.json is True

    calls = {}

    class FakePromptOptimizeHandler:
        def __init__(self, args):
            calls["args"] = args

        def handle(self):
            calls["handled"] = True
            return 0

    monkeypatch.setattr(
        "tuner.handlers.prompt_optimize_handler.PromptOptimizeHandler",
        FakePromptOptimizeHandler,
    )

    assert route_command(args) == 0
    assert calls["args"] is args
    assert calls["handled"] is True
