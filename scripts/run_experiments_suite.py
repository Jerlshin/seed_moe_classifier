#!/usr/bin/env python
"""Run the whole stage-2 revision suite -- ablations and baselines -- resumably.

    python scripts/run_experiments_suite.py --all
    python scripts/run_experiments_suite.py --all --dry-run
    python scripts/run_experiments_suite.py --variants wo_moe wo_kl --seeds 42
    python scripts/run_experiments_suite.py --preset reviewer --kaggle
    python scripts/run_experiments_suite.py --status

``scripts/run_ablations.py`` and ``scripts/run_baselines.py`` still exist and
still work; they own the *definitions* of what an arm is, and this script
imports them rather than redeclaring anything. What it adds is the thing a
Kaggle or vast.ai session needs and a bare loop does not: a suite that can be
launched with the identical command line after every preemption and does the
right thing, which is to skip what finished, re-run what did not, and stop
cleanly before the session limit rather than being killed inside a run.

**No pretraining.** Every arm here consumes the already-published stage-1
encoder or builds its own ImageNet backbone. Nothing in this file can start
stage 1; ``--checkpoint`` points at the existing encoder and
``ensure_pretrained_checkpoint`` refuses to start if it is absent.

Resumption
----------

State lives in ``outputs/suite_status.json`` and is rewritten after every
transition, but it is a *cache*: completion is decided by opening the run's
``summary.json`` and ``test_predictions.npz`` and checking they are present,
parseable and mutually consistent. The consequences are the ones that matter on
a preemptible platform:

* a session killed between the two writes leaves a complete-looking
  ``summary.json`` with no predictions -- ``generate_plots.py`` re-scores from
  the predictions, so that run would silently drop out of the results table.
  It is re-run.
* a run finished by a bare ``python -m src.trainers.moe_finetune`` outside the
  suite is recognised and skipped.
* deleting ``suite_status.json`` loses the attempt counts and the timings, and
  nothing that decides what gets launched.

Within a run, ``experiment.training.resume=auto`` is passed by default, so a
variant killed at epoch 70 of 100 continues from its own epoch-boundary
checkpoint instead of restarting.

Session limits
--------------

``--max-session-minutes`` stops *between* runs once the remaining budget is
smaller than the mean run so far, rather than starting a variant that cannot
finish. It does not interrupt a running variant; ``--max-runtime-minutes`` is
the separate, per-run budget the trainer honours internally by checkpointing and
exiting cleanly. On Kaggle's 12-hour limit, ``--kaggle`` sets both.

Ordering
--------

Variant-major, seeds inner, and the ablation suite before the baselines. A suite
interrupted halfway then has *complete seed coverage* of the arms it reached,
which is the partial result worth having: five seeds of four variants supports a
mean +- SD and a McNemar test, where one seed of twenty variants supports
neither.
"""

from __future__ import annotations

import argparse
import itertools
import queue
import sys
import threading
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_ablations import ABLATION_VARIANTS, DEFAULT_VARIANTS as DEFAULT_ABLATIONS
from scripts.run_baselines import BASELINE_VARIANTS, END_TO_END_BASELINES
from src.trainers.runner import (
    DEFAULT_SEEDS,
    VariantSpec,
    build_command,
    default_checkpoint_path,
    ensure_pretrained_checkpoint,
    expand_seeds,
    output_root,
    parse_gpu_list,
    run_variant,
)
from src.trainers.suite import (
    COMPLETE,
    FAILED,
    PRIMARY_METRIC,
    SUITE_STATE_FILENAME,
    RunRecord,
    SuiteState,
    estimate_remaining_seconds,
    file_sha256,
    format_duration,
    format_progress,
)

# --------------------------------------------------------------- the matrix

#: Every arm this script knows about, ablations first.
#
# Both lists are imported, not restated. A variant added to `run_ablations.py`
# or a baseline added to `run_baselines.py` appears here automatically, which is
# what stops the two suites and the unified one from drifting apart in what they
# think the experiment is.
ALL_VARIANTS: list[VariantSpec] = [*ABLATION_VARIANTS, *BASELINE_VARIANTS]
VARIANTS_BY_NAME: dict[str, VariantSpec] = {spec.name: spec for spec in ALL_VARIANTS}

#: What `--all` runs: every crop-level ablation arm plus every baseline.
#
# `DEFAULT_ABLATIONS` has already dropped the arms that move
# `experiment.training.split_protocol` -- `leakage_grouped` is the only one, and
# it is registered but opt-in. That exclusion is the reason this study's table
# is internally comparable at all: every row here shares the crop-level
# stratified split byte for byte, which is what makes the McNemar column in
# `scripts/generate_plots.py` valid. A photograph-disjoint row does not share a
# test split with `full_model`, so it cannot be paired against it and must not
# sit in the same table. Naming it explicitly still runs it; `--all` never does.
DEFAULT_SUITE: list[VariantSpec] = [*DEFAULT_ABLATIONS, *BASELINE_VARIANTS]

#: Named subsets, for the two cases that are not "everything".
PRESETS: dict[str, tuple[str, list[str]]] = {
    "ablations": (
        "Component-wise ablations only (the crop-level arms).",
        [spec.name for spec in DEFAULT_ABLATIONS],
    ),
    "baselines": (
        "Baselines and controls only.",
        [spec.name for spec in BASELINE_VARIANTS],
    ),
    "reviewer": (
        "The minimum set that answers the review, if compute is short: the "
        "submitted six ablations, the backbone comparison both reviewers asked "
        "for, and the two floors.",
        [
            # Reviewer 1 & 2: "there is no ablation study". These are the six
            # the submitted manuscript named, with `wo_arcface` replaced by the
            # single-factor margin control it should always have been.
            "full_model",
            "wo_moe",
            "wo_margin_only",
            "wo_residual",
            "wo_kl",
            "wo_cross_attn",
            # Reviewer 2 major comment 1: ViT vs SwinV2, matched in every
            # respect but the trunk.
            "vit_small",
            "swinv2_tiny",
            # Reviewer 1 & 2: "zero comparisons with standard baselines".
            "resnet50",
            "linear_probe",
        ],
    ),
}


def resolve_selection(args: argparse.Namespace) -> list[VariantSpec]:
    """Which arms this invocation runs, from ``--all`` / ``--preset`` / ``--variants``."""
    if args.variants:
        unknown = [name for name in args.variants if name not in VARIANTS_BY_NAME]
        if unknown:
            raise SystemExit(
                f"Unknown variant(s): {', '.join(unknown)}\n"
                f"Known: {', '.join(sorted(VARIANTS_BY_NAME))}"
            )
        return [VARIANTS_BY_NAME[name] for name in args.variants]
    if args.preset:
        return [VARIANTS_BY_NAME[name] for name in PRESETS[args.preset][1]]
    return list(DEFAULT_SUITE)


def checkpoint_for(spec: VariantSpec, checkpoint: Path | None) -> Path | None:
    """The stage-1 encoder this arm should read, or ``None``.

    An end-to-end baseline owns its backbone; handing it the SwinV2 DINO
    checkpoint would be meaningless for a ResNet and a silent partial load for a
    shape-compatible trunk. ``swinv2_supervised`` is the shape-compatible case
    and is excluded on purpose -- its entire job is to *not* read stage 1.
    """
    if any(spec.name.startswith(name) for name in END_TO_END_BASELINES):
        return None
    return checkpoint


def suite_overrides(args: argparse.Namespace) -> list[str]:
    """Overrides this script adds to every arm, before the user's own.

    Kept minimal and all resumption- or precision-related. Anything that changes
    what is being measured belongs in the arm's own ``VariantSpec`` or
    experiment file, not here, where it would apply to every row at once and
    leave no trace in any single run's provenance.
    """
    overrides: list[str] = []
    if args.resume:
        # Continue an interrupted variant from its own epoch-boundary
        # checkpoint. `auto` starts fresh when there is none, which is why one
        # command line serves the first launch and every relaunch after it.
        overrides.append("experiment.training.resume=auto")
    if args.amp:
        overrides.append(f"experiment.training.amp={args.amp}")
    if args.max_runtime_minutes:
        overrides.append(
            f"experiment.training.max_runtime_minutes={args.max_runtime_minutes}"
        )
    if args.epochs is not None:
        overrides.append(f"experiment.training.epochs={args.epochs}")
    return overrides


# ------------------------------------------------------------------------ CLI


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--all",
        action="store_true",
        help="Every crop-level ablation arm plus every baseline (the default).",
    )
    selection.add_argument(
        "--preset",
        choices=sorted(PRESETS),
        help="; ".join(f"{name}: {description}" for name, (description, _) in PRESETS.items()),
    )
    selection.add_argument(
        "--variants",
        nargs="+",
        metavar="NAME",
        help="Explicit arms to run. Names come from run_ablations.py and "
        "run_baselines.py; --list prints them.",
    )

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_SEEDS),
        help="Seeds to repeat every arm over (default: 42-46). A single seed makes "
        "a smoke sweep and reports no dispersion, so the table cannot carry SDs.",
    )
    parser.add_argument("--checkpoint", default=None, help="Shared stage-1 encoder checkpoint.")
    parser.add_argument("--output-root", default=None, help="Root for outputs/ (default: $SEED_OUTPUT_DIR).")
    parser.add_argument(
        "--state-file",
        default=None,
        help=f"Suite state file (default: <output-root>/{SUITE_STATE_FILENAME}).",
    )
    parser.add_argument(
        "--allow-missing-checkpoint",
        action="store_true",
        help="Proceed without the stage-1 encoder. Smoke runs only: the arms that "
        "consume it would train from a random encoder and the table would be meaningless.",
    )

    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pass experiment.training.resume=auto to every arm (default: on).",
    )
    parser.add_argument(
        "--retry-failed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Re-run arms recorded as failed (default: on). A stage-2 failure here is "
        "usually a preemption or an OOM, not a deterministic bug.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run arms whose artifacts are already complete. Overwrites them.",
    )

    parser.add_argument(
        "--amp",
        choices=("auto", "bf16", "fp16", "off"),
        default=None,
        help="Autocast dtype for training. 'fp16' is the Tesla T4 setting (sm_75 has no "
        "hardware bf16); 'auto' resolves it from the card. Evaluation is always fp32.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override experiment.training.epochs for every arm.",
    )
    parser.add_argument(
        "--max-runtime-minutes",
        type=float,
        default=None,
        help="Per-run wall-clock budget. The trainer checkpoints and exits cleanly at it, "
        "so the next launch resumes instead of restarting.",
    )
    parser.add_argument(
        "--max-session-minutes",
        type=float,
        default=None,
        help="Suite-level budget. Stops BETWEEN runs once the remaining time is less than "
        "the mean run so far, rather than starting one that cannot finish.",
    )
    parser.add_argument(
        "--kaggle",
        action="store_true",
        help="Kaggle T4 x2 preset: --amp fp16, --max-session-minutes 690, "
        "--max-runtime-minutes 660, --gpus 0,1. Individually-passed flags win.",
    )
    parser.add_argument(
        "--gpus",
        default=None,
        help="Run arms concurrently, one per device: '0,1' or 'auto'. Preferred over DDP "
        "for stage 2 -- no gradient traffic, and each arm keeps single-GPU numerics.",
    )

    parser.add_argument("--dry-run", action="store_true", help="Print the exact commands and targets, run nothing.")
    parser.add_argument("--status", action="store_true", help="Print suite progress and exit.")
    parser.add_argument("--list", action="store_true", help="List every known arm and exit.")
    parser.add_argument(
        "--stop-on-failure",
        action="store_true",
        help="Abort at the first failing arm instead of continuing.",
    )
    parser.add_argument(
        "overrides",
        nargs=argparse.REMAINDER,
        help="Extra Hydra overrides applied to every arm, after a bare '--'.",
    )
    args = parser.parse_args(argv)

    if args.kaggle:
        # Kaggle's session limit is 12 h. 690 min leaves ~30 min of headroom for
        # the final run's evaluation, the efficiency sweep and the artifact
        # writes -- being killed during those is what produces the
        # summary-without-predictions state this suite has to detect and redo.
        if args.amp is None:
            args.amp = "fp16"
        if args.max_session_minutes is None:
            args.max_session_minutes = 690.0
        if args.max_runtime_minutes is None:
            args.max_runtime_minutes = 660.0
        if args.gpus is None:
            args.gpus = "0,1"
    return args


def print_variant_list() -> None:
    print(f"{'arm':<28} {'group':<10} experiment / description")
    print("-" * 100)
    for spec in ALL_VARIANTS:
        default = "" if spec in DEFAULT_SUITE else "  [opt-in]"
        print(f"{spec.name:<28} {spec.group:<10} {spec.experiment}{default}")
        print(f"{'':<28} {'':<10}   {spec.description}")
    print()
    for name, (description, members) in PRESETS.items():
        print(f"--preset {name}: {description}")
        print(f"    {' '.join(members)}")


def report_conflicts(state: SuiteState) -> None:
    """Warn loudly when finished runs disagree on the corpus or the split.

    Both are silent-by-default failures that invalidate the table rather than
    degrading it, so they are printed on every launch, not only when something
    goes wrong.
    """
    for label, conflicts, consequence in (
        (
            "CORPUS",
            state.corpus_conflicts(),
            "rows from two different corpora are not comparable -- see the "
            "Refined_Samples/Cropped_Samples re-baseline",
        ),
        (
            "TEST SPLIT",
            state.split_conflicts(),
            "McNemar's paired test is invalid across these rows",
        ),
    ):
        if not conflicts:
            continue
        print(f"\n!!! {label} MISMATCH across completed runs: {consequence}", flush=True)
        for digest, keys in conflicts.items():
            print(f"    {digest[:16]}...  {len(keys)} run(s): {', '.join(sorted(keys)[:6])}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.list:
        print_variant_list()
        return 0

    root = Path(args.output_root) if args.output_root else output_root()
    state_path = Path(args.state_file) if args.state_file else root / SUITE_STATE_FILENAME
    state = SuiteState.load(state_path)

    specs = expand_seeds(resolve_selection(args), args.seeds)
    records = state.sync(specs, root)

    if args.status:
        print(f"Suite state: {state_path}")
        print(format_progress(records))
        totals = state.totals()
        print(f"\n{totals.get(COMPLETE, 0)} complete, "
              f"{totals.get(FAILED, 0)} failed, "
              f"{len(records) - sum(1 for r in records if r.status == COMPLETE)} outstanding "
              f"of {len(records)} selected.")
        remaining = estimate_remaining_seconds(records)
        if remaining is not None:
            print(f"Estimated remaining: {format_duration(remaining)} (from this suite's own timings).")
        report_conflicts(state)
        return 0

    checkpoint = ensure_pretrained_checkpoint(
        args.checkpoint or default_checkpoint_path(),
        allow_missing=args.allow_missing_checkpoint,
    )
    extra = [item for item in args.overrides if item != "--"]
    extra = [*suite_overrides(args), *extra]
    if checkpoint is None:
        print("WARNING: no stage-1 encoder; arms that consume it train from a random one.", flush=True)
        extra = [*extra, "model.backbone.checkpoint_path=null"]

    # Recorded once per suite, so a state file carried across a session can be
    # compared against the encoder the new session resolved. Swapping the
    # encoder mid-suite is the stage-2 equivalent of swapping the corpus.
    encoder_sha = file_sha256(checkpoint) if checkpoint else None
    previous = state.reference.get("encoder_sha256")
    if previous and encoder_sha and previous != encoder_sha:
        print(
            f"\n!!! ENCODER CHANGED since this suite was started:\n"
            f"    was {previous}\n    now {encoder_sha}\n"
            "    Completed rows were trained on a different stage-1 encoder than the "
            "outstanding ones will be. Re-run the suite into a fresh --output-root, or "
            "restore the original checkpoint.",
            flush=True,
        )
    state.reference = {
        "encoder": str(checkpoint) if checkpoint else None,
        "encoder_sha256": encoder_sha,
        "seeds": list(args.seeds),
        "amp": args.amp,
        "suite_overrides": extra,
    }

    outstanding = records if args.force else state.pending(records, retry_failed=args.retry_failed)
    already = len(records) - len(outstanding)

    print(f"Suite: {len(records)} runs selected -> {root}")
    print(f"State file: {state_path}")
    print(f"Shared encoder: {checkpoint if checkpoint else '(none)'}")
    if extra:
        print(f"Applied to every arm: {' '.join(extra)}")
    print(f"{already} already complete, {len(outstanding)} to run.")
    report_conflicts(state)

    if not outstanding:
        print("\nNothing to do. Build the table with: python scripts/generate_plots.py")
        state.save()
        return 0

    gpus = parse_gpu_list(args.gpus)
    started = time.perf_counter()

    if args.dry_run:
        for index, record in enumerate(outstanding, start=1):
            spec = spec_for(specs, record)
            save_path = spec.save_path(root)
            command = build_command(spec, save_path, checkpoint_for(spec, checkpoint), extra)
            print(f"\n[{index}/{len(outstanding)}] {record.key}  ({record.status}: {record.reason})")
            print(f"    target: {save_path}")
            print(f"    $ {' '.join(command)}")
        print(f"\nDry run: {len(outstanding)} command(s) above, nothing executed.")
        return 0

    lock = threading.Lock()
    halt = threading.Event()
    failures: list[RunRecord] = []
    counter = itertools.count(1)

    def budget_exhausted() -> bool:
        """True when the remaining session budget cannot hold another mean run.

        Checked before *starting* a run, never during one. Stopping between runs
        leaves complete artifacts behind; being killed inside one leaves the
        summary-without-predictions state this suite then has to detect and
        redo, which costs the whole run rather than the tail of it.
        """
        if not args.max_session_minutes:
            return False
        remaining = args.max_session_minutes * 60.0 - (time.perf_counter() - started)
        mean = _mean_completed_duration(records)
        if mean is None:
            # Nothing has been timed yet, so there is no basis for "it will not
            # fit". Only a budget already spent stops the suite.
            return remaining <= 0
        return remaining < mean

    def execute(record: RunRecord, device: int | None) -> None:
        spec = spec_for(specs, record)
        encoder = checkpoint_for(spec, checkpoint)
        command = build_command(spec, spec.save_path(root), encoder, extra)
        index = next(counter)
        with lock:
            print(f"\n[{index}/{len(outstanding)}] {record.key}  (was {record.status}: {record.reason})")
            state.mark_running(record, command)

        result = run_variant(spec, root, encoder, extra, dry_run=False, gpu=device)

        with lock:
            state.mark_result(record, result)
            if record.status == COMPLETE:
                metric = record.primary_metric
                rendered = f"{metric:.4f}" if metric is not None else "?"
                print(
                    f"    OK  {record.key}  {PRIMARY_METRIC}={rendered}  "
                    f"({format_duration(record.duration_seconds)})"
                )
            else:
                failures.append(record)
                print(f"    FAILED  {record.key}  {record.reason}")
                if args.stop_on_failure:
                    print("Stopping: --stop-on-failure is set.", flush=True)
                    halt.set()
            remaining = estimate_remaining_seconds(records)
            if remaining is not None and remaining > 0:
                print(f"    estimated remaining: {format_duration(remaining)}")

    if len(gpus) > 1:
        # One arm per device. This is the right way to use a second GPU for
        # stage 2: the arms are already independent processes, so there is no
        # gradient traffic at all and each keeps the exact numerics of a
        # single-GPU run -- which is what a table of 0.5-2 pp gaps needs.
        # A shared work queue rather than a static split, so a slow arm does not
        # leave a device idle.
        print(f"\nSharding over GPUs {gpus} ({len(gpus)} concurrent, one arm per device).")
        pending_queue: queue.Queue[RunRecord] = queue.Queue()
        for record in outstanding:
            pending_queue.put(record)

        def worker(device: int) -> None:
            while not halt.is_set():
                if budget_exhausted():
                    with lock:
                        print(
                            f"\nGPU {device}: session budget spent; stopping between runs. "
                            "Relaunch with the same command line to continue.",
                            flush=True,
                        )
                    return
                try:
                    record = pending_queue.get_nowait()
                except queue.Empty:
                    return
                execute(record, device)

        threads = [threading.Thread(target=worker, args=(device,), daemon=True) for device in gpus]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    else:
        device = gpus[0] if gpus else None
        for record in outstanding:
            if halt.is_set():
                break
            if budget_exhausted():
                print(
                    f"\nStopping before {record.key}: session budget spent. "
                    "Relaunch with the same command line to continue.",
                    flush=True,
                )
                break
            execute(record, device)

    state.save()
    print("\n" + "=" * 78)
    print(format_progress(records))
    totals = state.totals()
    print("=" * 78)
    print(
        f"{totals.get(COMPLETE, 0)} complete, {totals.get(FAILED, 0)} failed "
        f"(state: {state_path})"
    )
    report_conflicts(state)
    if failures:
        print(f"\n{len(failures)} run(s) failed. Relaunch with the same command line to retry them.")
        return 1
    if all(record.status == COMPLETE for record in records):
        print("\nSuite complete. Build the table with: python scripts/generate_plots.py")
    else:
        print("\nSuite partially complete. Relaunch with the same command line to continue.")
    return 0


def spec_for(specs: list[VariantSpec], record: RunRecord) -> VariantSpec:
    """The spec whose output directory this record describes."""
    for spec in specs:
        if f"{spec.group_directory}/{spec.run_name}" == record.key:
            return spec
    raise KeyError(f"no spec for run {record.key!r}")


def _mean_completed_duration(records: list[RunRecord]) -> float | None:
    timed = [r.duration_seconds for r in records if r.status == COMPLETE and r.duration_seconds > 0]
    return sum(timed) / len(timed) if timed else None


if __name__ == "__main__":
    raise SystemExit(main())
