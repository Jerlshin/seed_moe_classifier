#!/usr/bin/env python
"""Run a stage-1 arm suite: train each arm, evaluate each arm, collect one table.

    python scripts/run_stage1_ablations.py --arms conf/stage1_arms/phase0.yaml
    python scripts/run_stage1_ablations.py --arms conf/stage1_arms/phase1.yaml
    python scripts/run_stage1_ablations.py --arms conf/stage1_arms/phase1.yaml --dry-run
    python scripts/run_stage1_ablations.py --arms conf/stage1_arms/phase1.yaml \
        --experiment pretrain_swinv2_tiny_dino --gpus 0,1
    python scripts/run_stage1_ablations.py --arms conf/stage1_arms/phase3.yaml --seeds 42 43 44
    python scripts/run_stage1_ablations.py --arms conf/stage1_arms/phase1.yaml -- \
        experiment.training.epochs=2 experiment.training.max_batches=4

Why this exists, and why ``scripts/run_ablations.py`` could not be reused.
``run_ablations.py`` runs stage-2 head variants: one trainer, one output tree,
one shared encoder that every variant reads. A stage-1 suite is the opposite
shape -- each arm *produces* an encoder, then needs a second process to evaluate
it, and both write to paths the other arms would otherwise overwrite. Without
per-arm ``experiment.training.save_path``, ``shared_backbone_path`` and
``experiment.evaluation.save_path``, four arms silently overwrite each other's
``outputs/eval_pretrain/`` and each other's
``outputs/checkpoints/dinov2_swinv2_pretrained.pth``, and the resulting table is
a comparison of one encoder against itself.

The arms are **data, not code**. A manifest under ``conf/stage1_arms/`` names a
base Hydra experiment, a list of overrides applied to every arm, and one entry
per arm; adding an arm is adding an entry. The manifests that ship correspond to
Phases 0-3 of ``STAGE1_CHANGES.md``, and each one carries the sequence's own
rules about what may and may not be combined.

Three arm shapes:

``train: true`` (the default)
    Run stage 1, then evaluate the encoder it published.
``train: false`` with ``evaluate_frozen: true``
    No training. Evaluate the configured trunk at its ImageNet initialisation --
    the P1-F reference that decides whether stage 1 is worth running at all.
``train: false`` with ``evaluate: true`` and an ``eval_experiment``
    A pure screening run: Phase 0's readout and backbone screens.

Every run is a subprocess, for the reasons ``src/trainers/runner.py`` documents:
Hydra initialises once per process, GPU memory is only truly released at process
exit, and one arm's crash must not take the suite with it.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.trainers.runner import output_root, parse_gpu_list  # noqa: E402

#: Directory every arm's artifacts land under, below the outputs root.
SUITE_DIRECTORY = "stage1_arms"


@dataclass
class ArmSpec:
    """One arm of a stage-1 suite."""

    name: str
    description: str = ""
    overrides: list[str] = field(default_factory=list)
    train: bool = True
    evaluate: bool = True
    evaluate_frozen: bool = False
    eval_experiment: str | None = None
    eval_overrides: list[str] = field(default_factory=list)
    seed: int | None = None

    @property
    def run_name(self) -> str:
        return self.name if self.seed is None else f"{self.name}/seed{self.seed}"

    def directory(self, root: Path) -> Path:
        base = root / SUITE_DIRECTORY / self.name
        return base if self.seed is None else base / f"seed{self.seed}"

    def with_seed(self, seed: int) -> "ArmSpec":
        # A frozen reference has nothing stochastic in it, so repeating it across
        # seeds would file the identical numbers under three names and inflate
        # the apparent evidence.
        if not self.train:
            return self
        return ArmSpec(
            name=self.name,
            description=self.description,
            overrides=list(self.overrides),
            train=self.train,
            evaluate=self.evaluate,
            evaluate_frozen=self.evaluate_frozen,
            eval_experiment=self.eval_experiment,
            eval_overrides=list(self.eval_overrides),
            seed=int(seed),
        )


@dataclass
class ArmSuite:
    """A parsed manifest: the base experiment, the shared overrides and the arms.

    ``evaluation`` and ``frozen_evaluation`` name the evaluation experiment each
    arm is scored with. They exist because the evaluation config carries the
    *trunk* and the *protocol*, so a suite built on a different trunk cannot
    reuse the default one: scoring a SwinV2-Tiny arm against an evaluation whose
    `imagenet_init` control is SwinV2-Small turns "what did self-distillation
    add" into an architecture delta. A per-arm ``eval_experiment`` still wins
    over both.
    """

    experiment: str
    common: list[str]
    arms: list[ArmSpec]
    path: Path
    evaluation: str = "eval_pretrain_representation"
    frozen_evaluation: str = "eval_frozen_reference"


def load_suite(path: str | Path, experiment: str | None = None) -> ArmSuite:
    """Parse an arm manifest. Unknown keys are an error, not a silent no-op."""
    manifest_path = Path(path)
    payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    known = {"experiment", "common", "arms", "evaluation", "frozen_evaluation"}
    unexpected = set(payload) - known
    if unexpected:
        raise ValueError(
            f"{manifest_path}: unknown top-level keys {sorted(unexpected)}; expected {sorted(known)}"
        )

    arms: list[ArmSpec] = []
    arm_keys = set(ArmSpec.__dataclass_fields__) - {"seed"}
    for entry in payload.get("arms") or []:
        extra = set(entry) - arm_keys
        if extra:
            raise ValueError(f"{manifest_path}: arm {entry.get('name')!r} has unknown keys {sorted(extra)}")
        arms.append(
            ArmSpec(
                name=str(entry["name"]),
                description=str(entry.get("description", "")).strip(),
                overrides=[str(item) for item in entry.get("overrides") or []],
                train=bool(entry.get("train", True)),
                evaluate=bool(entry.get("evaluate", True)),
                evaluate_frozen=bool(entry.get("evaluate_frozen", False)),
                eval_experiment=(
                    str(entry["eval_experiment"]) if entry.get("eval_experiment") else None
                ),
                eval_overrides=[str(item) for item in entry.get("eval_overrides") or []],
            )
        )
    if not arms:
        raise ValueError(f"{manifest_path}: no arms defined.")
    names = [arm.name for arm in arms]
    if len(set(names)) != len(names):
        raise ValueError(f"{manifest_path}: duplicate arm names {sorted(names)}")

    return ArmSuite(
        experiment=str(experiment or payload.get("experiment") or "pretrain_swinv2_dino"),
        common=[str(item) for item in payload.get("common") or []],
        arms=arms,
        path=manifest_path,
        evaluation=str(payload.get("evaluation") or "eval_pretrain_representation"),
        frozen_evaluation=str(payload.get("frozen_evaluation") or "eval_frozen_reference"),
    )


def train_command(arm: ArmSpec, suite: ArmSuite, directory: Path, extra: list[str]) -> list[str]:
    """The stage-1 command for one arm, with every path pinned per arm.

    ``shared_backbone_path`` is the one that matters most: left at its default,
    every arm would publish over ``outputs/checkpoints/dinov2_swinv2_pretrained.pth``
    and the last arm to finish would silently become the encoder every stage-2
    run reads.
    """
    overrides = [
        f"experiment={suite.experiment}",
        f"experiment.training.save_path={directory}",
        f"experiment.training.shared_backbone_path={directory / 'encoder.pth'}",
        f"hydra.run.dir={directory / 'hydra'}",
        *suite.common,
        *arm.overrides,
    ]
    if arm.seed is not None:
        overrides.append(f"seed={arm.seed}")
    overrides.extend(extra)
    return [sys.executable, "-m", "src.trainers.contrastive_pretrain", *overrides]


def eval_command(arm: ArmSpec, suite: ArmSuite, directory: Path, extra: list[str]) -> list[str]:
    """The stage-1 evaluation command for one arm.

    Three cases, and they differ in which encoder rows exist:

    * a screening arm names its own ``eval_experiment`` and is left alone;
    * a frozen reference evaluates the configured trunk at its ImageNet
      initialisation, with the stage-1 encoder rows removed -- there are none;
    * a trained arm points ``pretrain_run_dir`` at its own output directory, so
      the milestone rows resolve to that arm's checkpoints.
    """
    if arm.eval_experiment is not None:
        evaluation = arm.eval_experiment
    elif arm.evaluate_frozen:
        # A first-class experiment rather than a wall of CLI overrides: the
        # frozen reference has to remove every row that reads a stage-1 artifact,
        # and expressing that on the command line is exactly the sort of thing
        # that goes wrong silently.
        evaluation = suite.frozen_evaluation
    else:
        evaluation = suite.evaluation

    save_path = directory / "eval"
    overrides = [
        f"experiment={evaluation}",
        f"experiment.evaluation.save_path={save_path}",
        f"hydra.run.dir={save_path / 'hydra'}",
    ]
    if arm.eval_experiment is None:
        overrides.append(f"experiment.evaluation.pretrain_run_dir={directory}")
        # Nothing publishes to the shared handoff from a suite run, so the
        # comparison against it would always report a mismatch.
        overrides.append("experiment.evaluation.shared_backbone_path=null")

    # Only the `data.*` overrides carry over. They describe the CORPUS and the
    # augmentation, both of which the evaluation must reproduce -- `data.image_size`
    # and `data.augmentation.*` feed the alignment measurement's multi-crop
    # pipeline, and a mismatch there would compare each arm against a different
    # augmentation distribution. Training-side overrides (`experiment.training.*`)
    # have no meaning in an evaluation config and Hydra's struct mode rejects them.
    overrides.extend(item for item in data_overrides(suite.common) if arm.eval_experiment is None)
    overrides.extend(data_overrides(arm.overrides))
    overrides.extend(arm.eval_overrides)
    overrides.extend(extra)
    return [sys.executable, "-m", "src.trainers.pretrain_eval", *overrides]


def data_overrides(overrides: list[str]) -> list[str]:
    """The subset of ``overrides`` that describes the data rather than the training."""
    return [item for item in overrides if item.split("=", 1)[0].lstrip("+~").startswith("data.")]


def run(command: list[str], directory: Path, gpu: int | None, dry_run: bool) -> int:
    print(f"$ {' '.join(command)}", flush=True)
    if dry_run:
        return 0
    environment = dict(os.environ)
    if gpu is not None:
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    environment["SEED_RUN_ID"] = str(directory.name)
    directory.mkdir(parents=True, exist_ok=True)
    return subprocess.call(command, cwd=str(PROJECT_ROOT), env=environment)


def collect(root: Path, suite: ArmSuite, seeds: list[int]) -> list[dict[str, Any]]:
    """One row per arm, read from the artifacts each arm left behind.

    Reads only ``summary.json`` (stage 1) and ``tables/encoder_comparison.csv``
    (the evaluation), which is the same discipline ``scripts/generate_plots.py``
    follows for stage 2: the table and the run must be produced by the same code
    path, and an arm that failed must show up as a row with missing numbers
    rather than as an absent row nobody notices.
    """
    import csv

    rows: list[dict[str, Any]] = []
    for arm in suite.arms:
        for spec in (arm.with_seed(seed) for seed in seeds) if seeds else (arm,):
            directory = spec.directory(root)
            row: dict[str, Any] = {
                "arm": spec.run_name,
                "description": spec.description.replace("\n", " ")[:160],
                "trained": spec.train,
                "directory": str(directory),
            }
            summary_path = directory / "summary.json"
            if summary_path.exists():
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                metrics = summary.get("metrics") or {}
                flags = summary.get("loss_flags") or {}
                row.update(
                    {
                        # The learnable half of the objective. The raw loss is
                        # ~95 % target entropy and is not a learning curve.
                        "final_teacher_student_kl": metrics.get("final_teacher_student_kl"),
                        "min_teacher_student_kl": metrics.get("min_teacher_student_kl"),
                        "final_loss": metrics.get("final_loss"),
                        "epochs": metrics.get("epochs_completed"),
                        "hours": (metrics.get("wall_clock_seconds") or 0) / 3600.0,
                        "koleo_scope": flags.get("koleo_scope"),
                        "lambda_koleo": flags.get("lambda_koleo"),
                        "centering": flags.get("centering"),
                        "corpus_sha256": str(
                            ((summary.get("split") or {}).get("corpus") or {}).get("sha256", "")
                        )[:16],
                    }
                )

            table = directory / "eval" / "tables" / "encoder_comparison.csv"
            if table.exists():
                with table.open(newline="", encoding="utf-8") as handle:
                    entries = list(csv.DictReader(handle))
                primary = next(
                    (entry for entry in entries if entry.get("role") == "primary"), None
                )
                if primary is not None:
                    for column in (
                        "oof_probe_sub_accuracy",
                        "oof_probe_sub_accuracy_testable_classes",
                        "oof_probe_sub_f1_macro",
                        "oof_probe_sub_fold_std",
                        "oof_probe_sub_accuracy_at_stage3",
                        "alignment",
                        "same_image_minus_same_class",
                        "self_retrieval_top1",
                        "nuisance_photo_above_chance",
                    ):
                        value = primary.get(column)
                        row[column] = float(value) if value not in (None, "") else None
            rows.append(row)
    return rows


def write_table(rows: list[dict[str, Any]], path: Path) -> str:
    import csv

    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return str(path)


def print_table(rows: list[dict[str, Any]]) -> None:
    headline = "oof_probe_sub_accuracy_testable_classes"
    print("\n" + "=" * 108)
    print(
        f"{'arm':<22} {'trained':<8} {'KL final':>10} {headline[:22]:>22} "
        f"{'f1_macro':>9} {'fold SD':>8} {'nuisance':>9}"
    )
    print("-" * 108)

    def cell(value: Any, width: int, digits: int = 4) -> str:
        if value is None or value != value:
            return f"{'-':>{width}}"
        return f"{float(value):>{width}.{digits}f}"

    for row in rows:
        print(
            f"{str(row['arm']):<22} {str(row.get('trained')):<8} "
            f"{cell(row.get('final_teacher_student_kl'), 10)} "
            f"{cell(row.get(headline), 22)} "
            f"{cell(row.get('oof_probe_sub_f1_macro'), 9)} "
            f"{cell(row.get('oof_probe_sub_fold_std'), 8)} "
            f"{cell(row.get('nuisance_photo_above_chance'), 9)}"
        )
    print("=" * 108)
    print(
        "Headline is the photograph-disjoint out-of-fold probe restricted to classes with >= 2\n"
        "source photographs. 'KL final' is the LEARNABLE half of the objective -- the raw DINO\n"
        "loss is ~95 % target entropy and is not comparable across arms that move the centering.\n"
        "'nuisance' is within-class photograph decodability above chance: an arm that RAISES it\n"
        "may have won the probe by re-learning the confound the protocol punishes.\n"
        "A single arm cannot resolve a difference below ~2 pp; see Phase 3."
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a stage-1 arm suite and collect its results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--arms",
        default=str(PROJECT_ROOT / "conf" / "stage1_arms" / "phase1.yaml"),
        help="Arm manifest to run (default: conf/stage1_arms/phase1.yaml).",
    )
    parser.add_argument(
        "--experiment",
        default=None,
        help="Override the manifest's base Hydra experiment, e.g. pretrain_swinv2_tiny_dino.",
    )
    parser.add_argument("--only", nargs="*", default=None, help="Run only these arm names.")
    parser.add_argument(
        "--seeds",
        nargs="*",
        type=int,
        default=None,
        help="Repeat every trained arm at each seed (Phase 3). Frozen references are not repeated.",
    )
    parser.add_argument("--gpus", default=None, help="Comma-separated device ids, or 'auto'.")
    parser.add_argument("--dry-run", action="store_true", help="Print the commands and stop.")
    parser.add_argument(
        "--skip-eval", action="store_true", help="Train only; run the evaluations later."
    )
    parser.add_argument(
        "--collect-only",
        action="store_true",
        help="Skip every run and rebuild the table from the artifacts already on disk.",
    )
    parser.add_argument("--output", default=None, help="Where to write the results CSV.")
    arguments, extra = parser.parse_known_args()
    extra = [item for item in extra if item != "--"]

    suite = load_suite(arguments.arms, arguments.experiment)
    if arguments.only:
        wanted = set(arguments.only)
        suite.arms = [arm for arm in suite.arms if arm.name in wanted]
        if not suite.arms:
            parser.error(f"No arms named {sorted(wanted)} in {suite.path}.")

    root = output_root()
    seeds = list(arguments.seeds or [])
    gpus = parse_gpu_list(arguments.gpus)
    gpu = gpus[0] if gpus else None
    if len(gpus) > 1:
        # Deliberately not sharded. A stage-1 arm is a full self-distillation
        # run: it saturates one device on its own, so running two concurrently
        # halves each one's throughput rather than doubling the suite's. Use both
        # devices *within* an arm instead -- `python main.py pretrain --gpus 2`
        # with `effective_batch_size` pinned -- which is what G4 recommends.
        print(
            f"NOTE: {len(gpus)} devices given; arms run sequentially on device {gpus[0]}. "
            "A stage-1 arm saturates one GPU, so use both inside an arm (--gpus 2 on the "
            "trainer) rather than across arms.",
            flush=True,
        )

    results: list[tuple[str, int, float]] = []
    if not arguments.collect_only:
        specs = [
            spec
            for arm in suite.arms
            for spec in ((arm.with_seed(seed) for seed in seeds) if seeds else (arm,))
        ]
        # `with_seed` returns the same object for a frozen arm, so a multi-seed
        # suite would otherwise queue it once per seed.
        deduplicated: list[ArmSpec] = []
        for spec in specs:
            if not any(other.run_name == spec.run_name for other in deduplicated):
                deduplicated.append(spec)

        for spec in deduplicated:
            directory = spec.directory(root)
            print(f"\n=== {spec.run_name}: {spec.description or '(no description)'} ===", flush=True)
            started = time.perf_counter()
            code = 0
            if spec.train:
                code = run(train_command(spec, suite, directory, extra), directory, gpu, arguments.dry_run)
            if code == 0 and spec.evaluate and not arguments.skip_eval:
                code = run(eval_command(spec, suite, directory, extra), directory, gpu, arguments.dry_run)
            duration = time.perf_counter() - started
            status = "OK" if code == 0 else f"FAILED (exit {code})"
            print(f"--- {spec.run_name}: {status} in {duration:.1f}s ---", flush=True)
            results.append((spec.run_name, code, duration))

    if arguments.dry_run:
        return 0

    rows = collect(root, suite, seeds)
    output = Path(arguments.output or (root / SUITE_DIRECTORY / "stage1_arm_results.csv"))
    print(f"\nWrote {write_table(rows, output)}")
    print_table(rows)

    failures = [name for name, code, _ in results if code != 0]
    if failures:
        print(f"\n{len(failures)} of {len(results)} runs failed: {', '.join(failures)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
