"""
shared/flywheel/experiment_loop.py

Autonomous experiment loop: runs N training experiments, evaluates each,
and converges on the best hyperparameter configuration.

Uses an LLM advisor for early experiments and a LightGBM surrogate model
once enough data accumulates.

Used by: tuner CLI ``experiment-loop`` command
"""
from __future__ import annotations

import json
import logging
import random
import re
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from shared.flywheel.experiment_config import ExperimentConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ExperimentResult:
    """Outcome of a single experiment run."""

    experiment_id: str
    config: Dict[str, Any]
    eval_score: float
    training_loss: float
    duration_seconds: float
    status: str  # "completed" | "failed"


# ---------------------------------------------------------------------------
# LLM Advisor
# ---------------------------------------------------------------------------

class LLMAdvisor:
    """Proposes hyperparameter configs using an LLM backend.

    The advisor receives the experiment history (as a markdown table) and
    the search space, then asks the LLM to suggest the next configuration
    as a YAML block.
    """

    def __init__(
        self,
        search_space: Dict[str, List[Any]],
        llm_backend: str = "openrouter",
        program_md: str = "",
    ) -> None:
        self.search_space = search_space
        self.llm_backend = llm_backend
        self.program_md = program_md
        self._client: Any = None

    # -- lazy init to avoid import errors when LLM deps are missing --
    def _get_client(self) -> Any:
        if self._client is None:
            from shared.llm import create_client

            self._client = create_client(provider=self.llm_backend)
        return self._client

    def _build_prompt(
        self, results_history: "pd.DataFrame",  # noqa: F821
    ) -> str:
        """Build the advisor prompt with history and search space."""
        lines = ["You are a hyperparameter tuning expert."]
        if self.program_md:
            lines.append(f"\n## Instructions\n{self.program_md}")

        lines.append("\n## Search Space")
        for param, values in self.search_space.items():
            lines.append(f"- {param}: {values}")

        lines.append("\n## Experiment History (best first)")
        if len(results_history) == 0:
            lines.append("No experiments run yet.")
        else:
            sorted_df = results_history.sort_values(
                "eval_score", ascending=False,
            )
            try:
                lines.append(sorted_df.to_markdown(index=False))
            except ImportError:
                # tabulate not installed; fall back to CSV representation
                lines.append(sorted_df.to_csv(index=False))

        lines.append(
            "\n## Task\n"
            "Propose the next hyperparameter configuration to try. "
            "Return ONLY a YAML block (```yaml ... ```) mapping parameter "
            "names to values from the search space above. "
            "Choose values that you expect to improve eval_score based on "
            "the experiment history. Do not include any explanation outside "
            "the YAML block."
        )
        return "\n".join(lines)

    def propose_config(
        self, results_history: "pd.DataFrame",  # noqa: F821
    ) -> Dict[str, Any]:
        """Ask the LLM to propose the next experiment config.

        Falls back to random sampling on failure.
        """
        try:
            prompt = self._build_prompt(results_history)
            client = self._get_client()
            response = client.chat(
                [{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=512,
            )
            return self._extract_yaml(response)
        except Exception:
            logger.warning(
                "LLM advisor failed; falling back to random sampling",
                exc_info=True,
            )
            return _random_sample(self.search_space)

    def propose_candidates(
        self,
        results_history: "pd.DataFrame",  # noqa: F821
        n: int = 5,
    ) -> List[Dict[str, Any]]:
        """Propose *n* candidate configs via repeated LLM calls.

        If the LLM returns duplicates or errors, pads with random samples.
        """
        seen: set[str] = set()
        candidates: List[Dict[str, Any]] = []
        for _ in range(n * 2):  # allow some retries
            cfg = self.propose_config(results_history)
            key = json.dumps(cfg, sort_keys=True)
            if key not in seen:
                seen.add(key)
                candidates.append(cfg)
            if len(candidates) >= n:
                break
        # pad with random if needed
        while len(candidates) < n:
            cfg = _random_sample(self.search_space)
            candidates.append(cfg)
        return candidates[:n]

    # -- YAML extraction ---------------------------------------------------

    @staticmethod
    def _extract_yaml(text: str) -> Dict[str, Any]:
        """Extract the first YAML code block from *text*.

        Raises ``ValueError`` if no valid YAML block is found.
        """
        pattern = r"```(?:yaml)?\s*\n(.*?)```"
        match = re.search(pattern, text, re.DOTALL)
        if not match:
            raise ValueError("No YAML block found in LLM response")
        raw = match.group(1).strip()
        parsed = yaml.safe_load(raw)
        if not isinstance(parsed, dict):
            raise ValueError("YAML block did not parse to a dict")
        return parsed


# ---------------------------------------------------------------------------
# Surrogate Model (LightGBM)
# ---------------------------------------------------------------------------

class SurrogateModel:
    """LightGBM-based surrogate that predicts eval_score from config params.

    Falls back gracefully when ``lightgbm`` is not installed.
    """

    def __init__(self) -> None:
        self._pipeline: Any = None
        self._feature_names: List[str] = []
        self._available: Optional[bool] = None

    @property
    def available(self) -> bool:
        """Whether lightgbm + sklearn are importable."""
        if self._available is None:
            try:
                import lightgbm  # noqa: F401
                from sklearn.pipeline import Pipeline  # noqa: F401

                self._available = True
            except ImportError:
                self._available = False
        return self._available

    def fit(self, results_df: "pd.DataFrame") -> None:  # noqa: F821
        """Train the surrogate on experiment history.

        Expects *results_df* to contain numeric feature columns
        (everything except ``eval_score``, ``experiment_id``, ``status``,
        ``training_loss``, ``duration_seconds``).
        """
        if not self.available:
            logger.warning("lightgbm not installed; surrogate disabled")
            return

        import lightgbm as lgb
        from sklearn.pipeline import Pipeline

        exclude = {
            "eval_score", "experiment_id", "status",
            "training_loss", "duration_seconds",
        }
        feature_cols = [c for c in results_df.columns if c not in exclude]
        self._feature_names = feature_cols

        X = results_df[feature_cols].values
        y = results_df["eval_score"].values

        self._pipeline = Pipeline([
            ("lgbm", lgb.LGBMRegressor(
                n_estimators=50,
                max_depth=3,
                learning_rate=0.1,
                verbose=-1,
            )),
        ])
        self._pipeline.fit(X, y)
        logger.info("Surrogate model trained on %d experiments", len(X))

    def predict_candidates(
        self, candidates: List[Dict[str, Any]],
    ) -> List[float]:
        """Predict eval_score for each candidate config.

        Returns a list of predicted scores, parallel to *candidates*.
        """
        if self._pipeline is None or not self.available:
            return [0.0] * len(candidates)

        import pandas as pd

        df = pd.DataFrame(candidates)
        # Ensure column order matches training
        for col in self._feature_names:
            if col not in df.columns:
                df[col] = 0.0
        df = df[self._feature_names]
        return list(self._pipeline.predict(df.values))

    def feature_importance(self) -> Dict[str, float]:
        """Return feature importances from the underlying LightGBM model."""
        if self._pipeline is None or not self.available:
            return {}
        lgbm_model = self._pipeline.named_steps["lgbm"]
        importances = lgbm_model.feature_importances_
        return dict(zip(self._feature_names, importances.tolist()))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_sample(search_space: Dict[str, List[Any]]) -> Dict[str, Any]:
    """Sample one random config from the search space."""
    return {k: random.choice(v) for k, v in search_space.items()}


def _flatten_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten a nested dict using dot notation for keys."""
    flat: Dict[str, Any] = {}
    for key, value in config.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                flat[f"{key}.{sub_key}"] = sub_value
        else:
            flat[key] = value
    return flat


def _merge_config_overrides(
    base_config: Dict[str, Any],
    overrides: Dict[str, Any],
) -> Dict[str, Any]:
    """Merge dot-notation overrides into a nested config dict.

    Example::

        base = {"training": {"lr": 1e-4}}
        overrides = {"training.lr": 2e-4, "r": 16}
        result = {"training": {"lr": 2e-4}, "r": 16}
    """
    merged = _deep_copy_dict(base_config)
    for key, value in overrides.items():
        parts = key.split(".")
        target = merged
        for part in parts[:-1]:
            if part not in target or not isinstance(target[part], dict):
                target[part] = {}
            target = target[part]
        target[parts[-1]] = value
    return merged


def _deep_copy_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    """Simple recursive deep copy for nested dicts."""
    result: Dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, dict):
            result[k] = _deep_copy_dict(v)
        elif isinstance(v, list):
            result[k] = list(v)
        else:
            result[k] = v
    return result


def _trainer_script(trainer_type: str) -> str:
    """Return the training script path for the given trainer type."""
    scripts = {
        "sft": "Trainers/sft/train_sft.py",
        "kto": "Trainers/kto/train_kto.py",
    }
    if trainer_type not in scripts:
        raise ValueError(
            f"Unknown trainer_type '{trainer_type}'; expected one of {list(scripts)}"
        )
    return scripts[trainer_type]


def _extract_training_loss(run_dir: Path) -> float:
    """Best-effort extraction of final training loss from a run directory.

    Scans ``logs/training_latest.jsonl`` for the last ``loss`` entry.
    Returns ``float('inf')`` if nothing is found.
    """
    log_file = run_dir / "logs" / "training_latest.jsonl"
    if not log_file.exists():
        # Also try finding any .jsonl in the run dir
        candidates = list(run_dir.glob("**/*.jsonl"))
        if not candidates:
            return float("inf")
        log_file = candidates[0]

    last_loss = float("inf")
    try:
        with open(log_file) as fh:
            for line in fh:
                try:
                    entry = json.loads(line)
                    if "loss" in entry:
                        last_loss = float(entry["loss"])
                except (json.JSONDecodeError, ValueError, TypeError):
                    continue
    except OSError:
        pass
    return last_loss


# ---------------------------------------------------------------------------
# Experiment Loop
# ---------------------------------------------------------------------------

class ExperimentLoop:
    """Autonomous experiment loop.

    Runs up to ``config.max_experiments`` training experiments, selects
    hyperparameters via random / LLM advisor / LLM + surrogate, evaluates
    each run, and records the best configuration.

    Usage::

        cfg = load_experiment_config("configs/flywheel/experiment_loop.yaml")
        loop = ExperimentLoop(cfg)
        results = loop.run()
    """

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.results: List[ExperimentResult] = []
        self.best_score: float = float("-inf")
        self.best_config: Dict[str, Any] = {}

        # Lazy-init components
        self._advisor: Optional[LLMAdvisor] = None
        self._surrogate: Optional[SurrogateModel] = None

    # -- public API --------------------------------------------------------

    def run(self) -> List[ExperimentResult]:
        """Execute the full experiment loop.

        Returns:
            List of ExperimentResult, one per experiment.
        """
        issues = self.config.validate()
        if issues:
            raise ValueError(
                "Invalid ExperimentConfig:\n  " + "\n  ".join(issues)
            )

        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Starting experiment loop: %d experiments, strategy=%s",
            self.config.max_experiments,
            self.config.search_strategy,
        )

        for i in range(self.config.max_experiments):
            experiment_id = f"exp_{i:04d}_{uuid.uuid4().hex[:8]}"
            logger.info(
                "--- Experiment %d/%d [%s] ---",
                i + 1, self.config.max_experiments, experiment_id,
            )

            # 1. Select next config
            overrides = self._select_next_config()

            # 2. Run experiment
            result = self._run_single_experiment(experiment_id, overrides)
            self.results.append(result)

            # 3. Track best
            if (
                result.status == "completed"
                and result.eval_score > self.best_score
            ):
                self.best_score = result.eval_score
                self.best_config = dict(result.config)
                logger.info(
                    "New best score: %.4f (experiment %s)",
                    self.best_score, experiment_id,
                )
                self._save_best_config(output_dir)

            # 4. Persist results after each experiment
            self._save_results(output_dir)

            # 5. Retrain surrogate periodically
            if (
                self._surrogate is not None
                and self._surrogate.available
                and (i + 1) % self.config.surrogate_retrain_every == 0
            ):
                self._retrain_surrogate()

        logger.info(
            "Experiment loop complete. Best score: %.4f", self.best_score,
        )
        return self.results

    # -- internal ----------------------------------------------------------

    def _select_next_config(self) -> Dict[str, Any]:
        """Choose the next hyperparameter configuration to try."""
        import pandas as pd

        strategy = self.config.search_strategy
        n_completed = len(self.results)

        # Always random on first experiment
        if n_completed == 0 or strategy == "random":
            return _random_sample(self.config.search_space)

        # Build history dataframe
        history_df = self._build_history_df()

        # Phase 1: LLM advisor only
        if n_completed < self.config.surrogate_phase_threshold:
            advisor = self._get_advisor()
            return advisor.propose_config(history_df)

        # Phase 2: LLM proposes candidates, surrogate ranks them
        advisor = self._get_advisor()
        surrogate = self._get_surrogate()

        # Make sure surrogate is fitted
        if surrogate._pipeline is None and surrogate.available:
            self._retrain_surrogate()

        candidates = advisor.propose_candidates(history_df, n=5)
        if surrogate.available and surrogate._pipeline is not None:
            scores = surrogate.predict_candidates(candidates)
            best_idx = max(range(len(scores)), key=lambda j: scores[j])
            logger.info(
                "Surrogate ranked %d candidates, best predicted score: %.4f",
                len(candidates), scores[best_idx],
            )
            return candidates[best_idx]

        # Surrogate unavailable, just use the first LLM proposal
        return candidates[0]

    def _get_advisor(self) -> LLMAdvisor:
        if self._advisor is None:
            self._advisor = LLMAdvisor(
                search_space=self.config.search_space,
                llm_backend=self.config.llm_backend,
                program_md=self.config.program_md,
            )
        return self._advisor

    def _get_surrogate(self) -> SurrogateModel:
        if self._surrogate is None:
            self._surrogate = SurrogateModel()
        return self._surrogate

    def _retrain_surrogate(self) -> None:
        """Retrain the surrogate model on current results."""
        surrogate = self._get_surrogate()
        if not surrogate.available:
            return
        completed = [r for r in self.results if r.status == "completed"]
        if len(completed) < 3:
            logger.info("Not enough completed experiments to train surrogate")
            return

        df = self._build_history_df()
        surrogate.fit(df)

    def _build_history_df(self) -> "pd.DataFrame":  # noqa: F821
        """Build a DataFrame of completed experiment results."""
        import pandas as pd

        rows = []
        for r in self.results:
            if r.status != "completed":
                continue
            flat = _flatten_config(r.config)
            flat["eval_score"] = r.eval_score
            flat["training_loss"] = r.training_loss
            flat["duration_seconds"] = r.duration_seconds
            flat["experiment_id"] = r.experiment_id
            flat["status"] = r.status
            rows.append(flat)
        return pd.DataFrame(rows)

    def _run_single_experiment(
        self,
        experiment_id: str,
        config_overrides: Dict[str, Any],
    ) -> ExperimentResult:
        """Run one training experiment with the given config overrides.

        1. Merge overrides into base config
        2. Write temporary config YAML
        3. Run training subprocess (time-boxed via max_steps)
        4. Extract training loss from logs
        5. Run evaluation (if available) to get eval_score
        """
        start_time = time.time()

        # Load base config if specified
        base_config: Dict[str, Any] = {}
        if self.config.base_config_path:
            base_path = Path(self.config.base_config_path)
            if base_path.exists():
                from shared.utilities import load_yaml
                base_config = load_yaml(base_path)

        # Merge overrides
        merged = _merge_config_overrides(base_config, config_overrides)
        merged["max_steps"] = self.config.max_steps_per_experiment

        # Write temp config
        output_dir = Path(self.config.output_dir) / experiment_id
        output_dir.mkdir(parents=True, exist_ok=True)
        config_file = output_dir / "config.yaml"
        with open(config_file, "w") as fh:
            yaml.dump(merged, fh, default_flow_style=False)

        # Record what we're trying
        logger.info(
            "Running experiment %s with overrides: %s",
            experiment_id, config_overrides,
        )

        # Run training subprocess
        script = _trainer_script(self.config.trainer_type)
        cmd = [
            sys.executable, script,
            "--config", str(config_file),
            "--max-steps", str(self.config.max_steps_per_experiment),
        ]

        # Optionally pass dataset
        if self.config.dataset_path:
            cmd.extend(["--dataset-file", self.config.dataset_path])

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config.training_timeout_seconds,
            )
            if proc.returncode != 0:
                logger.warning(
                    "Training failed (exit %d) for %s:\n%s",
                    proc.returncode, experiment_id,
                    proc.stderr[-500:] if proc.stderr else "(no stderr)",
                )
                duration = time.time() - start_time
                return ExperimentResult(
                    experiment_id=experiment_id,
                    config=config_overrides,
                    eval_score=0.0,
                    training_loss=float("inf"),
                    duration_seconds=duration,
                    status="failed",
                )
        except subprocess.TimeoutExpired:
            logger.warning("Training timed out for %s", experiment_id)
            duration = time.time() - start_time
            return ExperimentResult(
                experiment_id=experiment_id,
                config=config_overrides,
                eval_score=0.0,
                training_loss=float("inf"),
                duration_seconds=duration,
                status="failed",
            )
        except FileNotFoundError:
            logger.error("Training script not found: %s", script)
            duration = time.time() - start_time
            return ExperimentResult(
                experiment_id=experiment_id,
                config=config_overrides,
                eval_score=0.0,
                training_loss=float("inf"),
                duration_seconds=duration,
                status="failed",
            )

        duration = time.time() - start_time

        # Extract training loss
        training_loss = _extract_training_loss(output_dir)

        # Evaluate
        eval_score = self._evaluate_experiment(output_dir)

        return ExperimentResult(
            experiment_id=experiment_id,
            config=config_overrides,
            eval_score=eval_score,
            training_loss=training_loss,
            duration_seconds=duration,
            status="completed",
        )

    def _evaluate_experiment(self, run_dir: Path) -> float:
        """Evaluate the trained checkpoint.

        Tries CheckpointEvaluator if available (Wave 1C), otherwise falls
        back to using inverted training loss as a proxy score.
        """
        try:
            import asyncio

            from shared.checkpoint_eval import CheckpointEvaluator
            from shared.eval_backend import create_eval_backend

            backend = create_eval_backend(
                backend_type=self.config.eval_backend,
                min_vram_gb=self.config.local_min_vram_gb,
            )
            evaluator = CheckpointEvaluator(
                run_dir=str(run_dir),
                eval_backend=backend,
                eval_scenario=self.config.eval_scenario,
            )

            # evaluate_checkpoints is async; run it in the current event loop
            # or create one if none exists
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                # Already in async context — create a new event loop in a thread
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    report = pool.submit(
                        asyncio.run, evaluator.evaluate_checkpoints(top_n=5)
                    ).result()
            else:
                report = asyncio.run(evaluator.evaluate_checkpoints(top_n=5))

            return report.best.eval_score
        except ImportError:
            logger.debug(
                "CheckpointEvaluator not available; "
                "using training loss as eval proxy"
            )
        except Exception:
            logger.warning(
                "CheckpointEvaluator failed; using training loss proxy",
                exc_info=True,
            )

        # Fallback: invert training loss (lower loss = higher score)
        loss = _extract_training_loss(run_dir)
        if loss == float("inf"):
            return 0.0
        return 1.0 / (1.0 + loss)

    # -- persistence -------------------------------------------------------

    def _save_results(self, output_dir: Path) -> None:
        """Write results to TSV file."""
        if not self.results:
            return

        tsv_path = output_dir / "results.tsv"

        # Collect all config keys across all experiments
        all_keys: set[str] = set()
        for r in self.results:
            flat = _flatten_config(r.config)
            all_keys.update(flat.keys())
        config_keys = sorted(all_keys)

        header_cols = (
            ["experiment_id"]
            + config_keys
            + ["eval_score", "training_loss", "duration_seconds", "status"]
        )

        lines = ["\t".join(header_cols)]
        for r in self.results:
            flat = _flatten_config(r.config)
            row = [r.experiment_id]
            for k in config_keys:
                row.append(str(flat.get(k, "")))
            row.extend([
                f"{r.eval_score:.6f}",
                f"{r.training_loss:.6f}" if r.training_loss != float("inf") else "inf",
                f"{r.duration_seconds:.1f}",
                r.status,
            ])
            lines.append("\t".join(row))

        tsv_path.write_text("\n".join(lines) + "\n")
        logger.info("Results saved to %s", tsv_path)

    def _save_best_config(self, output_dir: Path) -> None:
        """Write current best config as YAML."""
        best_path = output_dir / "best_config.yaml"
        with open(best_path, "w") as fh:
            yaml.dump(
                {"best_score": self.best_score, "config": self.best_config},
                fh,
                default_flow_style=False,
            )
        logger.info("Best config saved to %s", best_path)
