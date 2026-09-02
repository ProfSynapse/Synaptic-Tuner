from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_train_sft_uses_runtime_compatible_trl_tokenizer_kwarg() -> None:
    source = (REPO_ROOT / "Trainers" / "sft" / "train_sft.py").read_text(encoding="utf-8")

    assert "load_and_prepare_tokenized_dataset" in source
    assert "collate_prepared_sft_batch" in source
    assert "Trainer(" in source
    assert '"dataset_representation": "tokenized"' in source
    assert "dataset_text_field" not in source


def test_train_sft_seed_override_is_wired_with_is_not_none() -> None:
    # train_sft imports unsloth at module load, so verify the --seed flag and its
    # is-not-None override (honoring seed=0) at the source level.
    source = (REPO_ROOT / "Trainers" / "sft" / "train_sft.py").read_text(encoding="utf-8")

    assert '"--seed"' in source
    assert "if args.seed is not None:" in source
    assert "config.seed = args.seed" in source


def test_train_sft_numeric_overrides_are_hardened_is_not_none() -> None:
    # The fifth silent-substitution instance (focus item 7), hardened in #41:
    # batch_size / num_epochs / max_seq_length / gradient_accumulation /
    # learning_rate override guards must use is not None so an explicit 0 forwards
    # to config rather than being dropped to the default.
    source = (REPO_ROOT / "Trainers" / "sft" / "train_sft.py").read_text(encoding="utf-8")

    for guard in (
        "if args.batch_size is not None:",
        "if args.num_epochs is not None:",
        "if args.max_seq_length is not None:",
        "if args.gradient_accumulation is not None:",
        "if args.learning_rate is not None:",
    ):
        assert guard in source, f"SFT trainer missing hardened guard: {guard}"
    # The pre-hardening truthy guards must be gone.
    for stale in (
        "if args.batch_size:",
        "if args.num_epochs:",
        "if args.max_seq_length:",
        "if args.gradient_accumulation:",
        "if args.learning_rate:",
    ):
        assert stale not in source, f"SFT trainer still has truthy guard: {stale}"


def test_train_sft_honors_config_level_max_steps() -> None:
    # Python/YAML configs can carry training.max_steps for smoke runs. The CLI
    # value still takes precedence, but an unset CLI flag must not silently turn
    # a smoke config into a full epoch.
    source = (REPO_ROOT / "Trainers" / "sft" / "train_sft.py").read_text(encoding="utf-8")
    config_source = (
        REPO_ROOT / "Trainers" / "sft" / "configs" / "config_loader.py"
    ).read_text(encoding="utf-8")

    assert "max_steps: Optional[int] = None" in config_source
    assert "if args.max_steps is not None:" in source
    assert "config.training.max_steps = args.max_steps" in source
    assert 'effective_max_steps = getattr(config.training, "max_steps", None) or -1' in source
    assert '"max_steps": effective_max_steps' in source


def test_train_sft_threads_protected_revision_and_evidence_without_ambient_token_fallback() -> None:
    source = (REPO_ROOT / "Trainers" / "sft" / "train_sft.py").read_text(encoding="utf-8")

    assert '"--model-revision"' in source
    assert '"--anonymous-model"' in source
    assert '"--protected-smoke-config"' in source
    assert '"--protected-smoke-evidence"' in source
    assert 'model_revision=getattr(config.model, "model_revision", None)' in source
    assert 'require_resolved_revision=bool(args.protected_smoke_evidence)' in source
    assert '"--model-snapshot"' in source
    assert "model_snapshot=args.model_snapshot" in source
    assert "require_local_snapshot=args.model_snapshot is not None" in source
    assert "Runtime v1 requires one environment-bound offline model snapshot" in source
    assert "Protected smoke rejects ambient Hugging Face credentials" in source
    assert "capture_trainable_snapshot(model)" in source
    assert "finalize_protected_evidence(" in source


def test_train_sft_exposes_aux_head_cli_flags() -> None:
    # train_sft imports unsloth at module load, so verify the local-run lane's
    # aux_head argparse surface at the source level. All 12 flags must exist so
    # the recipe handler can forward an aux_head block end-to-end.
    source = (REPO_ROOT / "Trainers" / "sft" / "train_sft.py").read_text(encoding="utf-8")

    for flag in (
        '"--aux-head-enabled"',
        '"--no-aux-head-enabled"',
        '"--aux-head-layer"',
        '"--aux-head-token-position"',
        '"--aux-head-target-field"',
        '"--aux-head-loss"',
        '"--aux-head-head-type"',
        '"--aux-head-out-activation"',
        '"--aux-head-input-norm"',
        '"--aux-head-freeze-base"',
        '"--no-aux-head-freeze-base"',
        '"--aux-head-lm-loss-weight"',
        '"--aux-head-head-lr"',
        '"--aux-head-prompt-render"',
    ):
        assert flag in source, f"SFT trainer missing aux_head flag: {flag}"
    # Tri-state booleans default to None so absence never collapses to False.
    assert "aux_head_enabled=None" in source
    assert "aux_head_freeze_base=None" in source


def test_train_sft_aux_head_overrides_use_is_not_none() -> None:
    # Each aux_head override must be is-not-None guarded so an unset flag preserves
    # the loaded config (and a falsy-but-set value like lm_loss_weight=0.0 still
    # forwards). prompt_render is grouped with the aux knobs but targets training.
    source = (REPO_ROOT / "Trainers" / "sft" / "train_sft.py").read_text(encoding="utf-8")

    for guard in (
        "if args.aux_head_enabled is not None:",
        "if args.aux_head_layer is not None:",
        "if args.aux_head_token_position is not None:",
        "if args.aux_head_target_field is not None:",
        "if args.aux_head_loss is not None:",
        "if args.aux_head_head_type is not None:",
        "if args.aux_head_out_activation is not None:",
        "if args.aux_head_input_norm is not None:",
        "if args.aux_head_freeze_base is not None:",
        "if args.aux_head_lm_loss_weight is not None:",
        "if args.aux_head_head_lr is not None:",
        "if args.aux_head_prompt_render is not None:",
    ):
        assert guard in source, f"SFT trainer missing aux_head override guard: {guard}"
    assert "config.aux_head.enabled = args.aux_head_enabled" in source
    assert "config.training.prompt_render = args.aux_head_prompt_render" in source


def test_train_sft_threads_prompt_render_and_warns_on_off_anchor_combo() -> None:
    # prompt_render must reach the dataset-prep call, and the off-anchor combo
    # (end_of_prompt + full_conversation) must WARN (not error) since it is
    # legitimate for single-turn / inference-shaped rows.
    source = (REPO_ROOT / "Trainers" / "sft" / "train_sft.py").read_text(encoding="utf-8")
    config_source = (
        REPO_ROOT / "Trainers" / "sft" / "configs" / "config_loader.py"
    ).read_text(encoding="utf-8")

    assert 'prompt_render: str = "full_conversation"' in config_source
    assert "prompt_render=config.training.prompt_render" in source
    assert 'aux_head_cfg.token_position == "end_of_prompt"' in source
    assert 'config.training.prompt_render == "full_conversation"' in source
    assert "WARNING: aux_head token_position='end_of_prompt'" in source


def test_train_sft_revalidates_aux_head_coherence_after_cli_overrides() -> None:
    # Finding A remediation: the --aux-head-* overrides mutate config.aux_head
    # AFTER load_aux_head_config has run, so the YAML-load coherence guards would
    # otherwise be bypassed by a pure-CLI config. train_sft must re-run the shared
    # validate_aux_head_coherence on the post-override config. Asserted at source
    # level because train_sft imports unsloth and cannot load in-process.
    source = (REPO_ROOT / "Trainers" / "sft" / "train_sft.py").read_text(encoding="utf-8")

    # Imported from the shared config module (single implementation, no dup guard).
    assert "validate_aux_head_coherence" in source
    assert "validate_aux_head_coherence(" in source
    # The call reads the post-override config (not argparse fields).
    assert "enabled=config.aux_head.enabled" in source
    assert "freeze_base=config.aux_head.freeze_base" in source
    assert "lm_loss_weight=config.aux_head.lm_loss_weight" in source
    assert "out_activation=config.aux_head.out_activation" in source
    assert "loss=config.aux_head.loss" in source
    # A-M1 lane parity: the shared validator now also enforces layer-presence, so
    # the CLI lane must thread config.aux_head.layer (set from --aux-head-layer)
    # into the same call — otherwise the flag-only runner lane skips the check.
    assert "layer=config.aux_head.layer" in source

    # Ordering: the guard must run AFTER the last aux_head CLI override so it sees
    # the final config. The prompt_render override is the last line of the block.
    last_override_idx = source.index(
        "config.training.prompt_render = args.aux_head_prompt_render"
    )
    guard_call_idx = source.index("validate_aux_head_coherence(\n        enabled=")
    assert guard_call_idx > last_override_idx, (
        "coherence guard must be called after the aux_head CLI-override block"
    )

    # The B-M1 ERROR guard is DISTINCT from the prompt_render WARN guard — both
    # must coexist; the remediation must not merge or remove the WARN.
    assert "WARNING: aux_head token_position='end_of_prompt'" in source


def test_runtime_v1_projection_is_opt_in_atomic_and_post_save() -> None:
    source = (REPO_ROOT / "Trainers" / "sft" / "train_sft.py").read_text(encoding="utf-8")
    for flag in (
        "--runtime-v1-workload-fingerprint",
        "--runtime-v1-configuration-revision",
        "--runtime-v1-tokenizer-revision",
        "--runtime-v1-dataset-revision",
        "--runtime-v1-dataset-digest",
    ):
        assert flag in source
    assert "if any(present) and not all(present):" in source
    assert "if args.protected_smoke_evidence or runtime_v1_requested:" in source
    assert "os.replace(temporary, destination)" in source
    assert source.index("trainer.save_model(") < source.index("write_runtime_v1_projection_atomic(", source.index("def run("))
    assert 'lineage["synaptic_runtime_projection"] = runtime_projection' in source


def _stamp_block(source: str) -> str:
    """Return the adapter_config stamp block, sliced back from its write.

    The assertions below must hold for the STAMP specifically, not anywhere in
    the 1500-line module, so every check is scoped to this slice.
    """
    marker = 'document["base_model_name_or_path"] = locked_ref'
    assert source.count(marker) == 1, (
        "expected exactly one adapter_config base_model_name_or_path assignment, "
        f"found {source.count(marker)}"
    )
    end = source.index(marker) + len(marker)
    start = source.rindex("if runtime_v1_requested:", 0, end)
    return source[start:end]


def test_train_sft_stamps_locked_model_ref_into_adapter_config() -> None:
    # B-2. peft writes base_model_name_or_path as the offline SNAPSHOT PATH
    # (src/model_loader.py asserts _name_or_path is the snapshot dir), but the
    # host verifies that field for equality against the locked repo id. The
    # trainer re-reads and rewrites the file after the final save.
    #
    # train_sft imports unsloth at module load, so this is pinned at the source
    # level. A bare grep for "base_model_name_or_path" would pass against a
    # version that stamps the snapshot path — which IS the bug — so all three
    # properties are asserted, scoped to the stamp block.
    source = (REPO_ROOT / "Trainers" / "sft" / "train_sft.py").read_text(encoding="utf-8")
    block = _stamp_block(source)

    # 1. The stamped value is the locked model ref, never the snapshot path.
    assert "locked_ref = config.model.model_name" in block
    assert "args.model_snapshot" not in block, (
        "the stamp must not read the snapshot path; that value is the defect"
    )

    # 2. The write is gated on runtime_v1_requested, so every other lane
    #    (notably local --compute-losses, which feeds this string back to
    #    transformers_loss_loader as a path to load from) stays byte-identical.
    assert block.startswith("if runtime_v1_requested:")

    # 3. A non-string or empty ref raises; there is no fallback branch.
    assert "if not isinstance(locked_ref, str) or not locked_ref:" in block
    assert "raise RuntimeError(" in block
    assert (
        "runtime-v1 run has no usable model ref to stamp into adapter_config.json"
        in block
    )
    assert "runtime-v1 run produced no adapter_config.json to stamp" in block
    assert "adapter_config.json is not a JSON object" in block
    for fallback in ("except", "try:", "pass"):
        assert fallback not in block, (
            f"the stamp must fail closed; found a {fallback!r} in the stamp block"
        )


def test_train_sft_guards_locked_model_ref_before_training_starts() -> None:
    # The same fail-closed guard is repeated immediately after the LoRA attach
    # so an unusable ref fails in seconds rather than after a full training run.
    # It is repeated rather than hoisted because the two sites are ~280 lines
    # apart and each must be independently fail-closed.
    source = (REPO_ROOT / "Trainers" / "sft" / "train_sft.py").read_text(encoding="utf-8")

    attach_end = source.index("init_lora_weights=config.lora.init_lora_weights,")
    guard_region = source[attach_end : source.index("protected_before = None")]

    assert "if runtime_v1_requested:" in guard_region
    assert "locked_ref = config.model.model_name" in guard_region
    assert "if not isinstance(locked_ref, str) or not locked_ref:" in guard_region
    assert "raise RuntimeError(" in guard_region

    # The early guard must precede training, and the stamp must follow the save.
    assert attach_end < source.index("trainer.save_model(")
    assert source.index("trainer.save_model(") < source.index(
        'document["base_model_name_or_path"] = locked_ref'
    )
