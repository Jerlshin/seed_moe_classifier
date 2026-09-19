"""The resumable experiment-suite runner.

Nothing here launches training. What is pinned is the part that decides *what
gets launched*, because every failure mode of a resumable suite is silent:

* a run reported complete that is not leaves a hole in the results table, and
  ``scripts/generate_plots.py`` re-scores from ``test_predictions.npz``, so the
  row simply disappears rather than erroring;
* a run re-launched that was already complete overwrites a finished result and
  costs a whole Kaggle session;
* rows aggregated across two corpora or two test splits produce a table that is
  wrong rather than noisy -- McNemar's paired test is not defined across
  different splits at all.

The fixtures write the artifact shapes a killed session actually leaves behind,
not synthetic ones.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.trainers.runner import VariantResult, VariantSpec
from src.trainers.suite import (
    COMPLETE,
    FAILED,
    PENDING,
    PRIMARY_METRIC,
    STATE_VERSION,
    RunRecord,
    SuiteState,
    estimate_remaining_seconds,
    file_sha256,
    format_duration,
    inspect_run,
)

CORPUS_SHA = "013c04c5858e9815ac0bb71b9f5fde4c8be9176c514d28198e39727344b6f3d3"


def write_summary(directory, *, metric=0.7943, corpus=CORPUS_SHA, seed=42, name="full_model"):
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": name,
        "group": "ablation",
        "metrics": {PRIMARY_METRIC: metric, "sub_variety/accuracy": 0.8173},
        "split": {
            "protocol": "stratified",
            "seed": seed,
            "num_samples": 13492,
            "num_classes": 27,
            "classes_present_in_test": 27,
            "corpus": {"sha256": corpus},
        },
    }
    (directory / "summary.json").write_text(json.dumps(payload), encoding="utf-8")
    return directory


def write_predictions(directory, count=2699, truncate=False):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "test_predictions.npz"
    zeros = np.zeros(count, dtype=np.int64)
    np.savez(path, seed_true=zeros, seed_pred=zeros, sub_true=zeros, sub_pred=zeros)
    if truncate:
        raw = path.read_bytes()
        path.write_bytes(raw[: len(raw) // 2])
    return path


def complete_run(directory, **kwargs):
    write_summary(directory, **kwargs)
    write_predictions(directory)
    return directory


# ------------------------------------------------------------- run inspection


def test_a_run_with_both_artifacts_is_complete(tmp_path):
    inspection = inspect_run(complete_run(tmp_path / "run"))
    assert inspection.complete
    assert inspection.metrics[PRIMARY_METRIC] == pytest.approx(0.7943)
    assert inspection.corpus_sha256 == CORPUS_SHA
    assert inspection.num_test_predictions == 2699


def test_missing_directory_is_not_complete(tmp_path):
    inspection = inspect_run(tmp_path / "never-ran")
    assert not inspection.complete
    assert "summary.json" in inspection.reason


def test_summary_without_predictions_is_not_complete(tmp_path):
    """The live failure on a preemptible platform: killed between the two writes.

    ``summary.json`` alone looks finished, but ``scripts/generate_plots.py``
    re-scores every row from the raw predictions -- so a suite that accepted
    this would report the run done and then omit it from the table with no
    error anywhere.
    """
    directory = write_summary(tmp_path / "run")
    inspection = inspect_run(directory)
    assert not inspection.complete
    assert "test_predictions.npz" in inspection.reason
    # The metrics are still recovered, so the reason can be specific.
    assert inspection.metrics[PRIMARY_METRIC] == pytest.approx(0.7943)


def test_truncated_predictions_are_not_complete(tmp_path):
    """A kill *during* the second write leaves a file that exists and cannot be read."""
    directory = write_summary(tmp_path / "run")
    write_predictions(directory, truncate=True)
    inspection = inspect_run(directory)
    assert not inspection.complete
    assert "test_predictions.npz" in inspection.reason


def test_predictions_with_mismatched_lengths_are_not_complete(tmp_path):
    directory = write_summary(tmp_path / "run")
    np.savez(
        directory / "test_predictions.npz",
        seed_true=np.zeros(10, dtype=np.int64),
        seed_pred=np.zeros(10, dtype=np.int64),
        sub_true=np.zeros(9, dtype=np.int64),
        sub_pred=np.zeros(10, dtype=np.int64),
    )
    assert not inspect_run(directory).complete


def test_summary_without_the_selection_metric_is_not_complete(tmp_path):
    """A summary from a crashed evaluation carries no ``sub_variety/f1_macro``."""
    directory = tmp_path / "run"
    directory.mkdir()
    (directory / "summary.json").write_text(json.dumps({"metrics": {"loss": 1.2}}))
    write_predictions(directory)
    inspection = inspect_run(directory)
    assert not inspection.complete
    assert PRIMARY_METRIC in inspection.reason


def test_unparseable_summary_is_not_complete(tmp_path):
    directory = tmp_path / "run"
    directory.mkdir()
    (directory / "summary.json").write_text("{not json")
    write_predictions(directory)
    assert not inspect_run(directory).complete


# ------------------------------------------------------------- state machinery


def specs():
    return [
        VariantSpec(name="full_model", description="full", seed=42),
        VariantSpec(name="wo_moe", description="no moe", overrides=["model.head.use_moe=false"], seed=42),
    ]


def test_sync_registers_every_cell_and_reads_status_from_disk(tmp_path):
    root = tmp_path / "outputs"
    complete_run(root / "ablations" / "full_model" / "seed42")

    state = SuiteState(tmp_path / "suite_status.json")
    records = state.sync(specs(), root)

    assert [record.key for record in records] == [
        "ablations/full_model/seed42",
        "ablations/wo_moe/seed42",
    ]
    assert records[0].status == COMPLETE
    assert records[0].primary_metric == pytest.approx(0.7943)
    assert records[1].status == PENDING


def test_state_round_trips_through_the_file(tmp_path):
    root = tmp_path / "outputs"
    complete_run(root / "ablations" / "full_model" / "seed42")
    path = tmp_path / "suite_status.json"

    first = SuiteState(path)
    records = first.sync(specs(), root)
    first.mark_result(records[1], VariantResult("wo_moe/seed42", 1, 12.5, "x", []))
    first.save()

    second = SuiteState.load(path)
    assert json.loads(path.read_text())["version"] == STATE_VERSION
    assert second.records["ablations/wo_moe/seed42"].status == FAILED
    assert second.records["ablations/wo_moe/seed42"].duration_seconds == pytest.approx(12.5)


def test_a_state_file_from_a_future_schema_is_discarded_not_half_read(tmp_path):
    path = tmp_path / "suite_status.json"
    path.write_text(json.dumps({"version": STATE_VERSION + 99, "runs": {"x": {"key": "x"}}}))
    assert SuiteState.load(path).records == {}


def test_save_is_atomic_and_leaves_no_temp_file(tmp_path):
    """``write_text`` truncates first, so a kill mid-write would destroy the state."""
    path = tmp_path / "suite_status.json"
    state = SuiteState(path)
    state.sync(specs(), tmp_path / "outputs")
    state.save()
    assert path.is_file()
    assert not list(tmp_path.glob("*.tmp"))


def test_a_completed_cell_whose_directory_was_wiped_returns_to_pending(tmp_path):
    """The state file is a cache; the artifacts are the truth."""
    root = tmp_path / "outputs"
    directory = complete_run(root / "ablations" / "full_model" / "seed42")
    path = tmp_path / "suite_status.json"

    state = SuiteState(path)
    assert state.sync(specs(), root)[0].status == COMPLETE
    state.save()

    for artifact in directory.iterdir():
        artifact.unlink()

    assert SuiteState.load(path).sync(specs(), root)[0].status == PENDING


def test_a_run_completed_outside_the_suite_is_recognised(tmp_path):
    """`python -m src.trainers.moe_finetune ...` by hand still counts."""
    root = tmp_path / "outputs"
    state = SuiteState(tmp_path / "suite_status.json")
    assert state.sync(specs(), root)[0].status == PENDING

    complete_run(root / "ablations" / "full_model" / "seed42")
    assert state.sync(specs(), root)[0].status == COMPLETE


def test_exit_zero_without_predictions_is_recorded_as_failed(tmp_path):
    """A clean exit code is necessary but not sufficient.

    The trainer can exit 0 having written a summary and no predictions -- a
    rank-0 evaluation that raised inside a ``try``, or a disk that filled
    between the two writes. Trusting the return code would mark that complete.
    """
    root = tmp_path / "outputs"
    write_summary(root / "ablations" / "full_model" / "seed42")
    state = SuiteState(tmp_path / "suite_status.json")
    record = state.sync(specs(), root)[0]

    state.mark_result(record, VariantResult("full_model/seed42", 0, 10.0, "x", []))

    assert record.status == FAILED
    assert record.reason.startswith("exit 0 but")


def test_pending_retries_failed_cells_by_default(tmp_path):
    state = SuiteState(tmp_path / "suite_status.json")
    records = state.sync(specs(), tmp_path / "outputs")
    records[0].status = FAILED

    assert records[0] in state.pending(records)
    assert records[0] not in state.pending(records, retry_failed=False)


def test_running_a_subset_does_not_discard_the_rest_of_the_matrix(tmp_path):
    root = tmp_path / "outputs"
    state = SuiteState(tmp_path / "suite_status.json")
    state.sync(specs(), root)

    state.sync([specs()[0]], root)

    assert "ablations/wo_moe/seed42" in state.records


# ----------------------------------------------------------------- provenance


def test_two_corpora_among_completed_runs_are_reported(tmp_path):
    """The Refined_Samples / Cropped_Samples re-baseline is not comparable."""
    root = tmp_path / "outputs"
    complete_run(root / "ablations" / "full_model" / "seed42")
    complete_run(root / "ablations" / "wo_moe" / "seed42", corpus="0" * 64)

    state = SuiteState(tmp_path / "suite_status.json")
    state.sync(specs(), root)

    conflicts = state.corpus_conflicts()
    assert set(conflicts) == {CORPUS_SHA, "0" * 64}


def test_one_corpus_reports_no_conflict(tmp_path):
    root = tmp_path / "outputs"
    complete_run(root / "ablations" / "full_model" / "seed42")
    complete_run(root / "ablations" / "wo_moe" / "seed42")

    state = SuiteState(tmp_path / "suite_status.json")
    state.sync(specs(), root)
    assert state.corpus_conflicts() == {}
    assert state.split_conflicts() == {}


def test_two_test_splits_among_completed_runs_are_reported(tmp_path):
    """McNemar's paired test is undefined across different splits, not merely weaker."""
    root = tmp_path / "outputs"
    complete_run(root / "ablations" / "full_model" / "seed42")
    complete_run(root / "ablations" / "wo_moe" / "seed42", seed=7)

    state = SuiteState(tmp_path / "suite_status.json")
    state.sync(specs(), root)
    assert len(state.split_conflicts()) == 2


def test_file_sha256_of_a_missing_file_is_none(tmp_path):
    assert file_sha256(tmp_path / "absent.pth") is None
    target = tmp_path / "present.pth"
    target.write_bytes(b"weights")
    assert file_sha256(target) == file_sha256(target)
    assert len(file_sha256(target)) == 64


# -------------------------------------------------------------------- reporting


def test_remaining_estimate_uses_only_this_suite_s_own_timings(tmp_path):
    records = [
        RunRecord(key="a", group="ablation", variant="a", experiment="e", seed=42,
                  overrides=[], save_path="a", status=COMPLETE, duration_seconds=100.0),
        RunRecord(key="b", group="ablation", variant="b", experiment="e", seed=42,
                  overrides=[], save_path="b", status=COMPLETE, duration_seconds=200.0),
        RunRecord(key="c", group="ablation", variant="c", experiment="e", seed=42,
                  overrides=[], save_path="c", status=PENDING),
        RunRecord(key="d", group="ablation", variant="d", experiment="e", seed=42,
                  overrides=[], save_path="d", status=PENDING),
    ]
    assert estimate_remaining_seconds(records) == pytest.approx(300.0)


def test_remaining_estimate_is_none_before_anything_has_been_timed():
    records = [
        RunRecord(key="a", group="ablation", variant="a", experiment="e", seed=42,
                  overrides=[], save_path="a", status=PENDING)
    ]
    assert estimate_remaining_seconds(records) is None


@pytest.mark.parametrize(
    "seconds,expected",
    [(0, "0s"), (45, "45s"), (90, "1m 30s"), (4512, "1h 15m"), (43200, "12h 00m")],
)
def test_format_duration(seconds, expected):
    assert format_duration(seconds) == expected


# ----------------------------------------------------------------- the CLI wiring


def test_the_default_suite_excludes_every_split_protocol_arm():
    """Only the crop-level protocol is reported, and only one split makes the table valid."""
    from scripts.run_experiments_suite import DEFAULT_SUITE

    for spec in DEFAULT_SUITE:
        assert not any(
            override.startswith("experiment.training.split_protocol")
            for override in spec.overrides
        ), f"{spec.name} moves the split protocol and must not be in the default suite"
    assert "leakage_grouped" not in {spec.name for spec in DEFAULT_SUITE}


def test_the_default_suite_is_the_union_of_both_existing_suites():
    """The unified runner must not become a third, drifting definition of the experiment."""
    from scripts.run_ablations import DEFAULT_VARIANTS
    from scripts.run_baselines import BASELINE_VARIANTS
    from scripts.run_experiments_suite import DEFAULT_SUITE

    names = {spec.name for spec in DEFAULT_SUITE}
    assert names == {spec.name for spec in DEFAULT_VARIANTS} | {
        spec.name for spec in BASELINE_VARIANTS
    }


def test_end_to_end_baselines_are_never_handed_the_stage_one_encoder(tmp_path):
    """A ResNet would ignore a SwinV2 state dict; a SwinV2 would partially load it."""
    from scripts.run_baselines import END_TO_END_BASELINES
    from scripts.run_experiments_suite import VARIANTS_BY_NAME, checkpoint_for

    encoder = tmp_path / "dino_pretrained_encoder.pth"
    encoder.write_bytes(b"weights")

    for name in END_TO_END_BASELINES:
        assert checkpoint_for(VARIANTS_BY_NAME[name], encoder) is None, name
    # ...and the arms that must share it, do.
    for name in ("full_model", "wo_moe", "linear_probe", "hierarchical_cce"):
        assert checkpoint_for(VARIANTS_BY_NAME[name], encoder) == encoder, name


def test_the_reviewer_preset_covers_both_reviewers_named_requests():
    """If compute is short, this is the set that still answers the review."""
    from scripts.run_experiments_suite import PRESETS, VARIANTS_BY_NAME

    _, members = PRESETS["reviewer"]
    assert set(members) <= set(VARIANTS_BY_NAME)
    # Reviewer 1 and 2 both: "there is no ablation study".
    assert {"full_model", "wo_moe", "wo_kl", "wo_cross_attn", "wo_residual"} <= set(members)
    # Reviewer 2, major comment 1: ViT vs SwinV2, matched but for the trunk.
    assert {"vit_small", "swinv2_tiny"} <= set(members)
    # Reviewer 2: "zero comparisons with standard baselines".
    assert {"resnet50", "linear_probe"} <= set(members)


def test_kaggle_preset_sets_fp16_and_leaves_room_for_the_final_writes():
    """A T4 has no hardware bf16, and being killed during the artifact writes is the
    failure the whole resumable design exists to avoid."""
    from scripts.run_experiments_suite import parse_args

    args = parse_args(["--all", "--kaggle"])
    assert args.amp == "fp16"
    assert args.gpus == "0,1"
    assert args.max_session_minutes < 12 * 60
    assert args.max_runtime_minutes <= args.max_session_minutes


def test_explicit_flags_win_over_the_kaggle_preset():
    from scripts.run_experiments_suite import parse_args

    args = parse_args(["--all", "--kaggle", "--amp", "bf16", "--gpus", "0"])
    assert args.amp == "bf16"
    assert args.gpus == "0"


def test_suite_overrides_carry_resumption_and_precision_only():
    """Anything that changes what is *measured* belongs in the arm, not here."""
    from scripts.run_experiments_suite import parse_args, suite_overrides

    overrides = suite_overrides(parse_args(["--all", "--kaggle"]))
    assert "experiment.training.resume=auto" in overrides
    assert "experiment.training.amp=fp16" in overrides
    assert not any(override.startswith("model.head") for override in overrides)
    assert not any(override.startswith("experiment.training.split_protocol") for override in overrides)


def test_unknown_variant_names_are_refused_before_any_gpu_time():
    from scripts.run_experiments_suite import parse_args, resolve_selection

    with pytest.raises(SystemExit):
        resolve_selection(parse_args(["--variants", "wo_moe", "not_a_variant"]))
