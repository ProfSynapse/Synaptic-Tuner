"""check-contamination handler: n-gram containment between training data and eval prompts.

Location: tuner/handlers/contamination_handler.py
Purpose: Config-driven train/eval leakage check (see shared/contamination.py)
Used by: Router (tuner/cli/router.py) for ``python tuner.py check-contamination``

Exit codes: 0 = no eval item flagged, 2 = at least one eval item at/above the
threshold or exactly duplicated in training, 1 = error.
"""

from __future__ import annotations

import json
from argparse import Namespace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import yaml

from tuner.handlers.base import BaseHandler
from tuner.project import PathRef, ProjectContext

DEFAULT_CONFIG = Path("configs/contamination/default.yaml")
EXIT_FLAGGED = 2


class ContaminationHandler(BaseHandler):
    """Report n-gram containment of eval items inside training rows."""

    def __init__(self, args: Optional[Namespace] = None, context: Optional[ProjectContext] = None):
        super().__init__(args=args, context=context)

    @property
    def name(self) -> str:
        return "check-contamination"

    def can_handle_direct_mode(self) -> bool:
        return True

    # -- config ---------------------------------------------------------------

    def _arg(self, name: str) -> Any:
        return getattr(self.args, name, None) if self.args is not None else None

    def _cli_path(self, value: str, access: Literal["read", "write"] = "read") -> Path:
        """CLI paths resolve against the invocation directory (scheme refs allowed)."""
        return PathRef.parse(str(value)).resolve(self.context, from_cli=True, access=access)

    def _config_path(
        self, value: str, declaring_file: Path, access: Literal["read", "write"] = "read"
    ) -> Path:
        """Config paths: scheme refs (engine://, project://, artifact://, ...) resolve
        against their root; plain relative paths resolve against the engine root in
        standalone mode (as local-run recipes do) and the declaring file in host mode."""
        ref = PathRef.parse(str(value))
        if self.context.path_mode == "legacy" and ref.scheme is None and not Path(ref.value).is_absolute():
            return (self.engine_root / ref.value).resolve()
        return ref.resolve(self.context, declaring_file=declaring_file, access=access)

    def _load_config(self) -> Tuple[Path, Dict[str, Any]]:
        config_arg = self._arg("contamination_config")
        config_path = self._cli_path(config_arg) if config_arg else self.engine_root / DEFAULT_CONFIG
        if not config_path.exists():
            raise FileNotFoundError(f"Contamination config not found: {config_path}")
        data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            raise ValueError(f"Contamination config must be a YAML mapping: {config_path}")
        return config_path, data

    def _resolve_inputs(self, config_path: Path, data: Dict[str, Any]) -> Dict[str, Any]:
        from shared.contamination import ContaminationConfig

        def pick(cli_name: str, key: str, cast, default):
            cli_value = self._arg(cli_name)
            if cli_value is not None:
                return cast(cli_value)
            value = data.get(key)
            return cast(value) if value is not None else default

        defaults = ContaminationConfig()
        cfg = ContaminationConfig(
            ngram=pick("ngram", "ngram", int, defaults.ngram),
            threshold=pick("threshold", "threshold", float, defaults.threshold),
            min_item_tokens=pick("min_item_tokens", "min_item_tokens", int, defaults.min_item_tokens),
            top_k=pick("top_k", "top_k", int, defaults.top_k),
            train_roles=[str(r) for r in (data.get("train_roles") or defaults.train_roles)],
        )
        cfg.validate()

        cli_train = self._arg("train_data") or []
        if cli_train:
            train_paths = [self._cli_path(p) for p in cli_train]
        else:
            train_paths = [self._config_path(p, config_path) for p in (data.get("train_datasets") or [])]
        if not train_paths:
            raise ValueError("No training datasets: pass --train-data PATH (repeatable) or set train_datasets.")

        eval_cfg = data.get("eval") or {}
        if not isinstance(eval_cfg, dict):
            raise ValueError("Config key 'eval' must be a mapping.")
        cli_sources = self._arg("eval_source") or []
        cli_texts = self._arg("eval_text") or []
        if cli_sources or cli_texts:
            sources = [self._cli_path(p) for p in cli_sources]
            text_field = self._arg("eval_text_field") or "text"
            text_sources = [
                {"path": self._cli_path(p), "text_field": text_field, "id_field": "id"} for p in cli_texts
            ]
        else:
            sources = [self._config_path(p, config_path) for p in (eval_cfg.get("sources") or [])]
            text_sources = []
            for entry in eval_cfg.get("text_sources") or []:
                if isinstance(entry, str):
                    entry = {"path": entry}
                if not isinstance(entry, dict) or not entry.get("path"):
                    raise ValueError(f"eval.text_sources entries need a 'path': {entry!r}")
                text_sources.append(
                    {
                        "path": self._config_path(entry["path"], config_path),
                        "text_field": str(entry.get("text_field") or "text"),
                        "id_field": str(entry.get("id_field") or "id"),
                    }
                )
        if not sources and not text_sources:
            raise ValueError("No eval sources: pass --eval-source / --eval-text or set eval.sources.")

        config_dir = eval_cfg.get("config_dir")
        report_dir_arg = self._arg("report_dir")
        report_dir = (
            self._cli_path(report_dir_arg, access="write")
            if report_dir_arg
            else self._config_path(
                str(data.get("report_dir") or "artifact://scratch/contamination"), config_path, access="write"
            )
        )
        return {
            "cfg": cfg,
            "train_paths": train_paths,
            "sources": sources,
            "text_sources": text_sources,
            "config_dir": self._config_path(config_dir, config_path) if config_dir else None,
            "include_system": bool(eval_cfg.get("include_system", False)),
            "reference_fields": [str(f) for f in (eval_cfg.get("reference_fields") or [])],
            "report_dir": report_dir,
        }

    # -- run ------------------------------------------------------------------

    def handle(self) -> int:
        from shared import contamination as cont

        try:
            config_path, data = self._load_config()
            inputs = self._resolve_inputs(config_path, data)
            cfg = inputs["cfg"]

            labels = cont.dataset_labels(inputs["train_paths"])
            training_rows: List[cont.TrainingRow] = []
            for path, label in zip(inputs["train_paths"], labels):
                if not path.exists():
                    raise FileNotFoundError(f"Training dataset not found: {path}")
                training_rows.extend(cont.load_training_rows(path, label=label, train_roles=cfg.train_roles))

            eval_items: List[cont.EvalItem] = []
            source_counts: Dict[str, int] = {}
            for source in inputs["sources"]:
                config_dir = inputs["config_dir"]
                resolved = source.resolve()
                if resolved.parent.name == "scenarios":
                    config_dir = None  # use the scenario's own config dir
                items = cont.load_eval_source(
                    source,
                    config_dir=config_dir,
                    include_system=inputs["include_system"],
                    reference_fields=inputs["reference_fields"],
                )
                source_counts[str(source)] = len(items)
                eval_items.extend(items)
            for entry in inputs["text_sources"]:
                items = cont.load_text_source(
                    entry["path"], text_field=entry["text_field"], id_field=entry["id_field"]
                )
                source_counts[str(entry["path"])] = len(items)
                eval_items.extend(items)
            if not eval_items:
                raise ValueError("Eval sources produced no items.")

            report = cont.run_check(training_rows, eval_items, cfg)
            report["config_file"] = str(config_path)
            report["training_datasets"] = [
                {"path": str(p), "label": label, "rows": sum(1 for r in training_rows if r.dataset == label)}
                for p, label in zip(inputs["train_paths"], labels)
            ]
            report["eval_sources"] = source_counts

            decontam_arg = self._arg("write_decontaminated")
            if decontam_arg:
                report["decontaminated"] = self._write_decontaminated(
                    self._cli_path(decontam_arg, access="write"),
                    inputs["train_paths"],
                    labels,
                    report["flagged_training_rows"],
                )

            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = inputs["report_dir"] / stamp / "contamination_report.json"
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            report["report_path"] = str(report_path)
        except Exception as exc:  # noqa: BLE001 - surfaced to the user with a code
            self.output_error(f"Contamination check failed: {exc}", code="CONTAMINATION_CHECK_FAILED")
            return 1

        self.output(report, self._render(report))
        return 0 if report["passed"] else EXIT_FLAGGED

    def _write_decontaminated(
        self,
        target: Path,
        train_paths: List[Path],
        labels: List[str],
        flagged: Dict[str, List[Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        from shared.contamination import write_decontaminated

        if target.suffix == ".jsonl":
            if len(train_paths) != 1:
                raise ValueError(
                    "--write-decontaminated with a .jsonl file needs exactly one --train-data; "
                    "pass a directory for several datasets."
                )
            outputs = [target]
        else:
            outputs = [target / f"{Path(p).stem}.decontaminated.jsonl" for p in train_paths]
        for path, out in zip(train_paths, outputs):
            if out.resolve() == Path(path).resolve():
                raise ValueError(f"Refusing to overwrite the source dataset: {path}")
        return [
            write_decontaminated(path, label, flagged, out)
            for path, label, out in zip(train_paths, labels, outputs)
        ]

    # -- human output -----------------------------------------------------------

    @staticmethod
    def _render(report: Dict[str, Any]) -> str:
        s = report["summary"]
        cfg = report["config"]
        lines = [
            "",
            f"Contamination check (n={cfg['ngram']}, threshold={cfg['threshold']})",
            "-" * 72,
            f"Training rows:            {s['training_rows']}",
        ]
        for ds in report["training_datasets"]:
            formats = report["training_formats"].get(ds["label"], {})
            fmt = ", ".join(f"{k}={v}" for k, v in sorted(formats.items()))
            lines.append(f"  {ds['label']}: {ds['rows']} rows ({fmt})")
        lines += [
            f"Eval items:               {s['eval_items']} ({s['scored_items']} scored, {s['too_short_items']} too short)",
        ]
        for source, count in report["eval_sources"].items():
            lines.append(f"  {source}: {count}")
        lines += [
            f"Items with any overlap:   {s['items_with_any_overlap']}",
            f"Max / mean containment:   {s['max_containment']:.3f} / {s['mean_containment']:.3f}",
            f"Items >= threshold:       {s['items_over_threshold']}",
            f"Exact-duplicate prompts:  {s['exact_duplicate_prompts']}",
            f"Flagged training rows:    {s['flagged_training_rows']}",
            "Containment histogram:    "
            + "  ".join(f"{bucket}:{count}" for bucket, count in report["histogram"].items()),
        ]
        if report["top_pairs"]:
            lines += ["", "Top (eval item, training row) pairs:"]
            for pair in report["top_pairs"]:
                lines.append(
                    f"  {pair['containment']:.3f}  {pair['matched_ngrams']}/{pair['total_ngrams']}  "
                    f"{Path(pair['eval_source']).name}::{pair['eval_item_id']} [{pair['kind']}]  "
                    f"<- {pair['train_row_id']}"
                )
        if report["flagged_items"]:
            lines += ["", "Flagged eval items:"]
            for item in report["flagged_items"][:50]:
                dup = f" exact-dup x{len(item['exact_duplicate_row_ids'])}" if item["exact_duplicate_row_ids"] else ""
                lines.append(
                    f"  {item['max_containment']:.3f}{dup}  {Path(item['source']).name}::{item['item_id']} "
                    f"[{item['kind']}]  {item['excerpt']}"
                )
        for out in report.get("decontaminated", []):
            lines.append(
                f"Decontaminated: {out['output']} (kept {out['kept_rows']}, removed {out['removed_rows']}; "
                f"sidecar {out['sidecar']})"
            )
        lines += [
            "",
            f"Report: {report['report_path']}",
            "Result: " + ("PASS (no eval item flagged)" if report["passed"] else "FLAGGED (exit code 2)"),
        ]
        return "\n".join(lines)
