"""Persistent state for a resumable multi-variant, multi-seed experiment suite.

``src/trainers/runner.py`` knows how to launch one variant. This module knows
what a *suite* of them has and has not finished, across process restarts and
across preempted sessions, so ``scripts/run_experiments_suite.py`` can be
launched with the same command line every time and pick up where the last
session was killed.

The design rule is the one that makes a resumable suite trustworthy:

    **The artifacts on disk are the truth; the state file is a cache.**

:func:`inspect_run` never trusts a recorded status. It opens ``summary.json``
and ``test_predictions.npz`` and checks that they are present, parseable and
*mutually consistent* -- same number of predictions as the summary's test split,
a real value under the selection metric, the corpus digest the rest of the suite
was built on. A suite whose state file says "complete" for a directory somebody
wiped will re-run it; a run completed by a bare
``python -m src.trainers.moe_finetune`` outside the suite is recognised and
skipped. Deleting ``suite_status.json`` loses the timings and the attempt
counts, and nothing else.

Why validation has to go past "the file exists". Both artifacts are written at
the *end* of a run, but not atomically and not together: a session killed
between them leaves a ``summary.json`` with no predictions, and
``scripts/generate_plots.py`` re-scores every row from the raw predictions, so
that run would silently drop out of the table while the suite reported it
finished. A truncated ``.npz`` from a kill mid-write is the same failure with a
file that exists. Both are checked, and both are checked against the summary.

Three cross-checks beyond existence, each pinned to a failure this repository
has already had:

* **Corpus digest.** ``summary.json`` carries the SHA-256 of the corpus the run
  actually read. A suite spanning a dataset re-download, a remount, or a Kaggle
  input version bump would otherwise aggregate rows from two different corpora
  into one table. :meth:`SuiteState.corpus_conflicts` reports them.
* **Split identity.** Every McNemar p-value in the results table is valid only
  because the test split is byte-identical across variants. The summary records
  the split's protocol, seed and size; a run that disagrees with the suite's
  reference is flagged rather than aggregated.
* **Encoder identity.** Every stage-1-consuming variant must read the *same*
  published encoder. The suite records its SHA-256 once and compares.

Nothing here launches or inspects a training process.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from src.trainers.runner import VariantResult, VariantSpec

#: Artifacts a finished stage-2 run must leave behind. Both are part of the
#: "run artifacts are a contract" rule: `scripts/generate_plots.py` reads only
#: these two and re-scores the table from the raw predictions.
SUMMARY_FILENAME = "summary.json"
PREDICTIONS_FILENAME = "test_predictions.npz"

#: Default name of the suite's state file, written at the output root.
SUITE_STATE_FILENAME = "suite_status.json"

#: Bumped when the on-disk schema changes incompatibly. A state file from an
#: older version is discarded rather than half-read, because every field in it
#: is a cache of something recoverable from the run directories.
STATE_VERSION = 1

#: The metric the suite reports progress on. It is the study's selection metric
#: (`experiment.validation.monitor`) and the headline column of the results
#: table, so a run missing it has not produced the thing the suite exists for.
PRIMARY_METRIC = "sub_variety/f1_macro"

PENDING = "pending"
RUNNING = "running"
COMPLETE = "complete"
FAILED = "failed"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def file_sha256(path: str | Path, chunk_size: int = 1 << 20) -> str | None:
    """SHA-256 of a file, or ``None`` if it is absent or unreadable.

    Used on the shared encoder checkpoint. Reading it in chunks matters: the
    published encoder is ~110 MB and this runs once per suite launch, including
    on a Kaggle session whose whole memory budget is the thing being managed.
    """
    target = Path(path)
    if not target.is_file():
        return None
    digest = hashlib.sha256()
    try:
        with target.open("rb") as handle:
            for chunk in iter(lambda: handle.read(chunk_size), b""):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


# --------------------------------------------------------------- run inspection


@dataclass
class RunInspection:
    """What the artifacts in one run directory say about it.

    ``complete`` is the only field the scheduler reads. ``reason`` is what gets
    printed, and it names the *specific* artifact that is missing or
    inconsistent rather than saying "incomplete", because the two live failure
    modes -- a kill between the two writes, and a kill during the second write
    -- need different actions from whoever is watching the log.
    """

    complete: bool
    reason: str
    metrics: dict[str, float] = field(default_factory=dict)
    corpus_sha256: str | None = None
    split_signature: str | None = None
    num_test_predictions: int | None = None
    group: str | None = None
    variant: str | None = None


def _split_signature(summary: Mapping[str, Any]) -> str | None:
    """A compact identity for the test partition this run reported on.

    Protocol, seed, corpus size and test size decide the partition completely --
    the splitter is a pure function of ``cfg.seed`` over a fixed corpus. Two runs
    agreeing here are the precondition for the paired McNemar test the ablation
    table uses; two disagreeing must not share a table, whatever their metrics
    look like.
    """
    split = summary.get("split")
    if not isinstance(split, Mapping):
        return None
    parts = [
        str(split.get("protocol", "?")),
        str(split.get("seed", "?")),
        str(split.get("num_samples", "?")),
        str(split.get("num_classes", "?")),
        str(split.get("classes_present_in_test", "?")),
    ]
    return "|".join(parts)


def inspect_run(save_path: str | Path) -> RunInspection:
    """Decide whether the run under ``save_path`` finished, from its artifacts alone.

    Deliberately does not consult any recorded status: see the module docstring.
    """
    directory = Path(save_path)
    summary_path = directory / SUMMARY_FILENAME
    predictions_path = directory / PREDICTIONS_FILENAME

    if not summary_path.is_file():
        return RunInspection(False, f"no {SUMMARY_FILENAME}")
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return RunInspection(False, f"unreadable {SUMMARY_FILENAME}: {type(error).__name__}")
    if not isinstance(summary, Mapping):
        return RunInspection(False, f"{SUMMARY_FILENAME} is not an object")

    metrics_node = summary.get("metrics")
    metrics: dict[str, float] = {}
    if isinstance(metrics_node, Mapping):
        for key, value in metrics_node.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                metrics[str(key)] = float(value)

    corpus_sha = None
    split = summary.get("split")
    if isinstance(split, Mapping) and isinstance(split.get("corpus"), Mapping):
        corpus_sha = split["corpus"].get("sha256")
    signature = _split_signature(summary)
    group = summary.get("group") if isinstance(summary.get("group"), str) else None
    variant = summary.get("name") if isinstance(summary.get("name"), str) else None

    def partial(reason: str) -> RunInspection:
        return RunInspection(
            False, reason, metrics, corpus_sha, signature, None, group, variant
        )

    if PRIMARY_METRIC not in metrics:
        # A summary written by a crashed evaluation, or by a run whose head
        # emits a different metric set. Either way it cannot enter the table.
        return partial(f"{SUMMARY_FILENAME} has no {PRIMARY_METRIC!r}")

    if not predictions_path.is_file():
        # The live failure mode on a preemptible platform: killed between the
        # two writes. `generate_plots.py` re-scores from the predictions, so
        # this run would vanish from the table while looking finished here.
        return partial(f"no {PREDICTIONS_FILENAME} beside a complete {SUMMARY_FILENAME}")

    try:
        import numpy as np

        with np.load(predictions_path, allow_pickle=True) as archive:
            required = ("seed_true", "seed_pred", "sub_true", "sub_pred")
            missing = [key for key in required if key not in archive.files]
            if missing:
                return partial(f"{PREDICTIONS_FILENAME} missing {', '.join(missing)}")
            # Forces the arrays off disk: a file truncated by a kill mid-write
            # opens fine and raises here, which is exactly the case that must
            # not be reported as complete.
            lengths = {key: int(archive[key].shape[0]) for key in required}
    except Exception as error:  # noqa: BLE001 - any read failure means "re-run it"
        return partial(f"unreadable {PREDICTIONS_FILENAME}: {type(error).__name__}")

    if len(set(lengths.values())) != 1:
        return partial(f"{PREDICTIONS_FILENAME} arrays disagree in length: {lengths}")
    count = next(iter(lengths.values()))
    if count == 0:
        return partial(f"{PREDICTIONS_FILENAME} is empty")

    return RunInspection(
        True, "complete", metrics, corpus_sha, signature, count, group, variant
    )


# -------------------------------------------------------------------- the state


@dataclass
class RunRecord:
    """One (variant, seed) cell of the suite matrix, as persisted."""

    key: str
    group: str
    variant: str
    experiment: str
    seed: int | None
    overrides: list[str]
    save_path: str
    status: str = PENDING
    attempts: int = 0
    returncode: int | None = None
    duration_seconds: float = 0.0
    started_at: str | None = None
    finished_at: str | None = None
    reason: str = ""
    metrics: dict[str, float] = field(default_factory=dict)
    corpus_sha256: str | None = None
    split_signature: str | None = None
    num_test_predictions: int | None = None
    command: list[str] = field(default_factory=list)

    @classmethod
    def from_spec(cls, spec: VariantSpec, root: Path) -> "RunRecord":
        return cls(
            key=f"{spec.group_directory}/{spec.run_name}",
            group=spec.group,
            variant=spec.name,
            experiment=spec.experiment,
            seed=spec.seed,
            overrides=list(spec.overrides),
            save_path=str(spec.save_path(root)),
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RunRecord":
        known = {key: payload[key] for key in cls.__dataclass_fields__ if key in payload}
        known.setdefault("key", "")
        known.setdefault("group", "")
        known.setdefault("variant", "")
        known.setdefault("experiment", "")
        known.setdefault("seed", None)
        known.setdefault("overrides", [])
        known.setdefault("save_path", "")
        return cls(**known)  # type: ignore[arg-type]

    @property
    def primary_metric(self) -> float | None:
        value = self.metrics.get(PRIMARY_METRIC)
        return float(value) if isinstance(value, (int, float)) else None


class SuiteState:
    """The suite's ``suite_status.json``: load, reconcile with disk, persist.

    Instances are cheap and the file is small (one object per matrix cell), so
    it is rewritten after every state transition. That is deliberate: on a
    preemptible platform the process can disappear between any two lines, and a
    state file flushed only at the end would be empty exactly when it is needed.
    """

    def __init__(self, path: str | Path, records: dict[str, RunRecord] | None = None):
        self.path = Path(path)
        self.records: dict[str, RunRecord] = records or {}
        self.created: str = _now()
        self.reference: dict[str, Any] = {}

    # ------------------------------------------------------------- persistence

    @classmethod
    def load(cls, path: str | Path) -> "SuiteState":
        """Read the state file, or return an empty state.

        A missing, unreadable or older-schema file is not an error. Every field
        in it is a cache of something recoverable from the run directories, so
        starting empty costs the attempt counts and the recorded durations and
        nothing that affects which runs are launched.
        """
        state = cls(path)
        target = Path(path)
        if not target.is_file():
            return state
        try:
            payload = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return state
        if not isinstance(payload, Mapping) or int(payload.get("version", 0)) != STATE_VERSION:
            return state
        state.created = str(payload.get("created") or state.created)
        reference = payload.get("reference")
        state.reference = dict(reference) if isinstance(reference, Mapping) else {}
        runs = payload.get("runs")
        if isinstance(runs, Mapping):
            for key, entry in runs.items():
                if isinstance(entry, Mapping):
                    record = RunRecord.from_dict(entry)
                    record.key = str(key)
                    state.records[record.key] = record
        return state

    def save(self) -> Path:
        """Write the state file atomically.

        ``Path.write_text`` truncates its destination first, so a kill during
        the write would leave a zero-length ``suite_status.json`` *where the
        good one used to be* -- the same failure ``atomic_save`` exists to
        prevent for checkpoints. A sibling temp file plus ``os.replace`` makes
        the update atomic on every platform this runs on.
        """
        payload = {
            "version": STATE_VERSION,
            "created": self.created,
            "updated": _now(),
            "reference": self.reference,
            "totals": self.totals(),
            "runs": {key: asdict(record) for key, record in self.records.items()},
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temporary, self.path)
        return self.path

    # --------------------------------------------------------- reconciliation

    def sync(self, specs: Sequence[VariantSpec], root: Path) -> list[RunRecord]:
        """Register every spec and refresh each cell's status from its artifacts.

        Returns the records for ``specs``, in the order given. Records for cells
        not in ``specs`` are kept untouched, so running a subset with
        ``--variants`` never discards the rest of the matrix's history.
        """
        ordered: list[RunRecord] = []
        for spec in specs:
            fresh = RunRecord.from_spec(spec, root)
            record = self.records.get(fresh.key)
            if record is None:
                record = fresh
                self.records[record.key] = record
            else:
                # The spec is the authority on everything that describes the
                # run; only the history is carried forward. This is what makes a
                # renamed override or a moved output root take effect on resume
                # rather than being pinned by a stale file.
                record.group = fresh.group
                record.variant = fresh.variant
                record.experiment = fresh.experiment
                record.seed = fresh.seed
                record.overrides = fresh.overrides
                record.save_path = fresh.save_path
            self.refresh(record)
            ordered.append(record)
        return ordered

    def refresh(self, record: RunRecord) -> RunRecord:
        """Recompute one record's status from the artifacts on disk."""
        inspection = inspect_run(record.save_path)
        record.reason = inspection.reason
        if inspection.complete:
            record.status = COMPLETE
            record.metrics = inspection.metrics
            record.corpus_sha256 = inspection.corpus_sha256
            record.split_signature = inspection.split_signature
            record.num_test_predictions = inspection.num_test_predictions
            return record
        # Not complete. A cell previously marked complete whose directory was
        # wiped, or whose predictions were truncated, goes back to the queue.
        # FAILED is the one status preserved across a refresh, because
        # `--no-retry-failed` needs to be able to tell "never started" from
        # "started and did not finish".
        if record.status != FAILED:
            record.status = PENDING
        return record

    # ------------------------------------------------------------ transitions

    def mark_running(self, record: RunRecord, command: Sequence[str]) -> None:
        record.status = RUNNING
        record.attempts += 1
        record.started_at = _now()
        record.finished_at = None
        record.command = list(command)
        self.save()

    def mark_result(self, record: RunRecord, result: VariantResult) -> None:
        """Record a finished subprocess, then re-verify from the artifacts.

        Exit code 0 is necessary but not sufficient. The trainer can exit 0
        having written a summary and no predictions (a rank-0 evaluation that
        raised inside a ``try``, a disk that filled between the two writes), and
        that run must not be reported as complete. :meth:`refresh` has the last
        word.
        """
        record.returncode = result.returncode
        record.duration_seconds = result.duration_seconds
        record.finished_at = _now()
        self.refresh(record)
        if record.status != COMPLETE:
            record.status = FAILED
            if result.returncode == 0:
                record.reason = f"exit 0 but {record.reason}"
        self.save()

    # ----------------------------------------------------------------- queries

    def totals(self) -> dict[str, int]:
        counts = {PENDING: 0, RUNNING: 0, COMPLETE: 0, FAILED: 0}
        for record in self.records.values():
            counts[record.status] = counts.get(record.status, 0) + 1
        return counts

    def pending(self, records: Iterable[RunRecord], retry_failed: bool = True) -> list[RunRecord]:
        """The cells still to run, in the order given.

        Failed cells are retried by default. A stage-2 failure on this pipeline
        is overwhelmingly a preemption or an OOM rather than a deterministic
        bug, and the alternative -- a suite that quietly never revisits a cell
        the first Kaggle session happened to be killed in -- is worse than one
        wasted retry. ``--no-retry-failed`` is for the other case.
        """
        wanted = {PENDING, RUNNING} | ({FAILED} if retry_failed else set())
        return [record for record in records if record.status in wanted]

    def corpus_conflicts(self) -> dict[str, list[str]]:
        """Completed runs grouped by corpus digest, when more than one exists.

        An empty dict means every finished run read the same corpus. Anything
        else means the table under construction spans a re-baseline, and the
        rows are not comparable -- the exact condition ``corpus_fingerprint()``
        was added to stage 1 to make detectable.
        """
        by_digest: dict[str, list[str]] = {}
        for record in self.records.values():
            if record.status == COMPLETE and record.corpus_sha256:
                by_digest.setdefault(record.corpus_sha256, []).append(record.key)
        return by_digest if len(by_digest) > 1 else {}

    def split_conflicts(self) -> dict[str, list[str]]:
        """Completed runs grouped by test-split identity, when more than one exists.

        Every McNemar p-value in the results table assumes one shared split. Two
        signatures here means at least one row cannot be paired against
        ``full_model``, and reporting it in the same table would be invalid
        rather than merely noisy.
        """
        by_signature: dict[str, list[str]] = {}
        for record in self.records.values():
            if record.status == COMPLETE and record.split_signature:
                by_signature.setdefault(record.split_signature, []).append(record.key)
        return by_signature if len(by_signature) > 1 else {}


# ------------------------------------------------------------------- reporting


def format_progress(records: Sequence[RunRecord]) -> str:
    """A compact per-cell status table for the suite log."""
    if not records:
        return "(no runs)"
    width = max(len(record.key) for record in records)
    lines = [f"{'run':<{width}}  {'status':<9} {'metric':>8}  detail", "-" * (width + 30)]
    for record in records:
        metric = record.primary_metric
        rendered = f"{metric:.4f}" if metric is not None else "--"
        detail = "" if record.status == COMPLETE else record.reason
        lines.append(f"{record.key:<{width}}  {record.status:<9} {rendered:>8}  {detail}")
    return "\n".join(lines)


def estimate_remaining_seconds(records: Sequence[RunRecord]) -> float | None:
    """Seconds of work left, from this suite's own measured durations.

    ``None`` until at least one cell in this suite has finished and been timed.
    Estimating from anything else -- a hardcoded per-run figure, a different
    machine's history -- would produce a number whose only use is to be wrong on
    a platform whose session limit is the thing it is being used to plan around.
    """
    timed = [
        record.duration_seconds
        for record in records
        if record.status == COMPLETE and record.duration_seconds > 0
    ]
    if not timed:
        return None
    mean = sum(timed) / len(timed)
    outstanding = sum(1 for record in records if record.status != COMPLETE)
    return mean * outstanding


def format_duration(seconds: float) -> str:
    """``4512.0`` -> ``"1h 15m"``; the unit a session limit is expressed in."""
    seconds = max(0.0, float(seconds))
    hours, remainder = divmod(int(seconds), 3600)
    minutes = remainder // 60
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {int(seconds) % 60:02d}s"
    return f"{seconds:.0f}s"


def wall_clock_budget(started: float, limit_minutes: float | None) -> float | None:
    """Seconds left of a wall-clock budget, or ``None`` when there is none."""
    if not limit_minutes:
        return None
    return limit_minutes * 60.0 - (time.perf_counter() - started)
