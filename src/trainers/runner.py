"""Shared machinery for the ablation and baseline suites.

Both suites do the same thing: take a list of named variants, launch the *same*
finetune trainer once per variant with a different set of Hydra overrides, and
collect the results into one table. Keeping that logic here rather than
duplicating it in two scripts means the two suites cannot drift apart in how
they set output paths, reuse the pretrained checkpoint, or handle failures.

Each variant runs as a **subprocess** rather than an in-process function call.
Three reasons, in order of importance:

1. Hydra can only be initialised once per process. Running six variants in one
   process would need ``GlobalHydra.instance().clear()`` between them, which
   leaves stale config state behind and is precisely the sort of thing that
   makes an ablation table quietly wrong.
2. GPU memory is released completely when a process exits. Six models built in
   one process would accumulate.
3. A crash in one variant cannot take the suite down; the runner records the
   failure and continues.

Every variant reads the same pretrained encoder from
``experiment.pretrained_checkpoint``. :func:`ensure_pretrained_checkpoint`
refuses to start otherwise, because a suite that silently trained some variants
from random encoder weights would produce a comparison table that looks fine and
means nothing.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]

#: Where the pretrain stage publishes the encoder every downstream run reuses.
DEFAULT_PRETRAINED_CHECKPOINT = "checkpoints/dinov2_swinv2_pretrained.pth"


def output_root() -> Path:
    """Root of the outputs tree, honouring ``$SEED_OUTPUT_DIR``."""
    return Path(os.environ.get("SEED_OUTPUT_DIR", str(PROJECT_ROOT / "outputs")))


def default_checkpoint_path() -> Path:
    """Absolute path of the shared self-supervised SwinV2 checkpoint.

    The filename still says ``dinov2_`` for backward compatibility with existing
    checkpoints and ``$SEED_PRETRAIN_BACKBONE`` defaults. Stage 1 is DINO-style
    self-distillation, not DINOv2; see ``src/losses/dino.py``.
    """
    override = os.environ.get("SEED_PRETRAIN_BACKBONE")
    if override:
        return Path(override)
    return output_root() / DEFAULT_PRETRAINED_CHECKPOINT


def ensure_pretrained_checkpoint(path: str | Path, allow_missing: bool = False) -> Path | None:
    """Verify the shared encoder checkpoint exists, or explain how to produce it.

    Args:
        path: Checkpoint to check.
        allow_missing: Return ``None`` instead of raising. Only for smoke runs,
            where training from a random encoder is acceptable because nothing
            is being measured.
    """
    checkpoint = Path(path)
    if checkpoint.exists():
        return checkpoint
    if allow_missing:
        return None
    raise FileNotFoundError(
        f"Shared pretrained checkpoint not found: {checkpoint}\n\n"
        "Every variant in this suite must start from the same self-supervised SwinV2 encoder, "
        "otherwise the comparison measures self-supervised initialisation noise rather "
        "than the architectural change under test.\n\n"
        "Produce it once with:\n"
        "    python main.py pretrain\n\n"
        "or point --checkpoint / $SEED_PRETRAIN_BACKBONE at an existing file, or pass "
        "--allow-missing-checkpoint to train from a randomly initialised encoder "
        "(smoke runs only)."
    )


#: Training seeds every variant is repeated over.
#
# One run per variant cannot support the table it is being asked to support. On
# a 1,871-image test split at ~95 % accuracy the 95 % CI half-width on a single
# accuracy is +-0.99 pp and on a *difference* of two accuracies +-1.40 pp --
# before any training-seed variance (dropout, shuffling, router initialisation,
# and for a MoE specifically, which experts win the early race). Component
# contributions of 0.5-2 pp are the normal magnitude for a 27-class fine-grained
# task, so most of the gaps the ablation table needs to report sit inside the
# noise floor of a single split.
#
# Five seeds per variant with mean +- SD, plus McNemar's exact test on the shared
# test split, is what turns the table from anecdote into evidence. Stage 2 trains
# a head against a frozen encoder, so these runs are cheap relative to
# pretraining, and `run_suite`'s failure isolation already makes an unattended
# sweep safe.
DEFAULT_SEEDS = (42, 43, 44, 45, 46)


@dataclass
class VariantSpec:
    """One run in a suite: a name, a description, and its Hydra overrides."""

    name: str
    description: str
    overrides: list[str] = field(default_factory=list)
    group: str = "ablation"
    experiment: str = "finetune_hierarchical_moe"
    seed: int | None = None
    """Training seed. ``None`` uses the config's own value and writes to
    ``{group}/{variant}/``; an integer writes to ``{group}/{variant}/seed{n}/``,
    which is the extra directory level ``collect_run_summaries`` globs for."""

    def save_path(self, root: Path) -> Path:
        base = root / self.group_directory / self.name
        return base if self.seed is None else base / f"seed{self.seed}"

    @property
    def run_name(self) -> str:
        """Unique label for logs and the manifest."""
        return self.name if self.seed is None else f"{self.name}/seed{self.seed}"

    @property
    def group_directory(self) -> str:
        return "ablations" if self.group == "ablation" else "baselines"

    def with_seed(self, seed: int) -> "VariantSpec":
        """Copy of this spec pinned to ``seed``."""
        return VariantSpec(
            name=self.name,
            description=self.description,
            overrides=list(self.overrides),
            group=self.group,
            experiment=self.experiment,
            seed=int(seed),
        )


def expand_seeds(specs: Sequence[VariantSpec], seeds: Sequence[int]) -> list[VariantSpec]:
    """Cross every variant with every seed, preserving variant order.

    Variant-major rather than seed-major so a suite interrupted halfway has
    complete coverage of the variants it did reach, which is the more useful
    partial result.
    """
    if not seeds:
        return list(specs)
    return [spec.with_seed(seed) for spec in specs for seed in seeds]


@dataclass
class VariantResult:
    """Outcome of one variant run."""

    name: str
    returncode: int
    duration_seconds: float
    save_path: str
    command: list[str]

    @property
    def succeeded(self) -> bool:
        return self.returncode == 0


def build_command(
    spec: VariantSpec,
    save_path: Path,
    checkpoint: Path | None,
    extra_overrides: Sequence[str] = (),
) -> list[str]:
    """Assemble the full ``python -m src.trainers.moe_finetune ...`` command.

    ``extra_overrides`` is appended last so a caller's command-line override
    always wins over the suite's defaults -- Hydra takes the rightmost value.
    """
    overrides = [
        f"experiment={spec.experiment}",
        f"experiment.variant={spec.name}",
        f"experiment.group={spec.group}",
        f"experiment.training.save_path={save_path}",
        # Keep Hydra's own run directory (logs, config snapshot, TensorBoard
        # events, figures) next to the variant's checkpoints instead of in the
        # global timestamped tree, so one variant is one self-contained folder.
        f"hydra.run.dir={save_path / 'hydra'}",
        *spec.overrides,
    ]
    if spec.seed is not None:
        # Placed before spec.overrides' consumers but after the path settings so
        # a variant can still pin its own seed if it genuinely needs one.
        overrides.append(f"seed={spec.seed}")
    if checkpoint is not None:
        overrides.append(f"model.backbone.checkpoint_path={checkpoint}")
    overrides.extend(extra_overrides)
    return [sys.executable, "-m", "src.trainers.moe_finetune", *overrides]


def run_variant(
    spec: VariantSpec,
    root: Path,
    checkpoint: Path | None,
    extra_overrides: Sequence[str] = (),
    dry_run: bool = False,
) -> VariantResult:
    """Launch one variant and return its result. Never raises on a training failure."""
    save_path = spec.save_path(root)
    command = build_command(spec, save_path, checkpoint, extra_overrides)

    print(f"\n=== {spec.run_name}: {spec.description} ===", flush=True)
    print(f"$ {' '.join(command)}", flush=True)
    if dry_run:
        return VariantResult(spec.run_name, 0, 0.0, str(save_path), command)

    save_path.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    returncode = subprocess.call(command, cwd=str(PROJECT_ROOT))
    duration = time.perf_counter() - started

    status = "OK" if returncode == 0 else f"FAILED (exit {returncode})"
    print(f"--- {spec.run_name}: {status} in {duration:.1f}s ---", flush=True)
    return VariantResult(spec.run_name, returncode, duration, str(save_path), command)


def run_suite(
    specs: Sequence[VariantSpec],
    root: Path,
    checkpoint: Path | None,
    extra_overrides: Sequence[str] = (),
    dry_run: bool = False,
    stop_on_failure: bool = False,
) -> list[VariantResult]:
    """Run every variant in order and return their results.

    Continues past a failure by default: a long suite that aborts on variant two
    wastes the hours the remaining variants would have taken, and the failure is
    reported clearly either way.
    """
    results: list[VariantResult] = []
    for spec in specs:
        result = run_variant(spec, root, checkpoint, extra_overrides, dry_run)
        results.append(result)
        if stop_on_failure and not result.succeeded:
            print(f"Stopping: {spec.run_name} failed and --stop-on-failure is set.", flush=True)
            break
    return results


def write_suite_manifest(path: str | Path, specs: Sequence[VariantSpec], results: Sequence[VariantResult]) -> str:
    """Record what was run, with what overrides and what outcome."""
    by_name = {spec.run_name: spec for spec in specs}
    payload = {
        "variants": [
            {
                "name": result.name,
                "description": by_name[result.name].description if result.name in by_name else "",
                "overrides": by_name[result.name].overrides if result.name in by_name else [],
                "seed": by_name[result.name].seed if result.name in by_name else None,
                "returncode": result.returncode,
                "succeeded": result.succeeded,
                "duration_seconds": result.duration_seconds,
                "save_path": result.save_path,
                "command": result.command,
            }
            for result in results
        ]
    }
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return str(output)


def print_summary(results: Sequence[VariantResult]) -> int:
    """Print a per-variant status table. Returns a process exit code."""
    print("\n" + "=" * 72)
    print(f"{'variant':<32} {'status':<10} {'duration':>10}  output")
    print("-" * 72)
    for result in results:
        status = "ok" if result.succeeded else f"exit {result.returncode}"
        print(f"{result.name:<32} {status:<10} {result.duration_seconds:>9.1f}s  {result.save_path}")
    print("=" * 72)

    failures = [result for result in results if not result.succeeded]
    if failures:
        print(f"{len(failures)} of {len(results)} runs failed: {', '.join(f.name for f in failures)}")
        return 1
    print(f"All {len(results)} runs completed.")
    return 0
