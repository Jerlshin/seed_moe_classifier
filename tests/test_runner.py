"""The ablation and baseline suite runners.

Nothing here launches training. What is tested is command construction and
checkpoint handling, because those are the parts that can quietly invalidate a
whole suite: a variant pointed at the wrong output directory overwrites another,
and a variant that silently trains from a random encoder produces a comparison
table that looks fine and means nothing.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from src.trainers.runner import (
    VariantSpec,
    VariantResult,
    build_command,
    ensure_pretrained_checkpoint,
    print_summary,
    write_suite_manifest,
)


@pytest.fixture
def checkpoint(tmp_path) -> Path:
    path = tmp_path / "checkpoints" / "dino_pretrained_encoder.pth"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"not a real checkpoint")
    return path


# ----------------------------------------------------------- suite definitions


def test_ablation_suite_keeps_every_originally_requested_variant():
    """The submitted six must all survive, so the revision stays comparable."""
    from scripts.run_ablations import VARIANTS_BY_NAME

    for name in ("full_model", "wo_moe", "wo_residual", "wo_kl", "wo_cross_attn"):
        assert name in VARIANTS_BY_NAME
    # `wo_arcface` was renamed, because it never measured what its name claimed:
    # swapping ArcFace for a plain Linear removes the margin AND the embedding
    # normalisation AND the centre normalisation AND the logit scale.
    assert "wo_arcface" not in VARIANTS_BY_NAME
    assert "wo_angular_head" in VARIANTS_BY_NAME


def test_confounded_ablations_have_single_factor_counterparts():
    """Only ``wo_kl`` and ``wo_cross_attn`` were ever one-factor changes.

    Each confounded toggle now has a control that isolates the factor its name
    claims, so the table can attribute rather than assert.
    """
    from scripts.run_ablations import VARIANTS_BY_NAME

    controls = {
        # routing, holding active capacity fixed
        "wo_moe_capacity_matched": "model.head.dense_capacity_multiplier=2",
        # learned routing, holding sparse capacity fixed
        "moe_fixed_router": "model.head.router_mode=hash",
        # the angular margin alone, keeping the hypersphere geometry
        "wo_margin_only": "model.head.sub_head_variant=normface",
    }
    for name, expected in controls.items():
        assert name in VARIANTS_BY_NAME, f"missing single-factor control {name}"
        assert expected in VARIANTS_BY_NAME[name].overrides


def test_leakage_ablation_is_present_and_changes_only_the_split():
    """Quantifying the crop-level leak is a result, not a cleanup step.

    The primary protocol is crop-level, so the leakage row must be the
    photograph-disjoint one -- an arm that repeated the primary protocol would be
    a byte-identical copy of `full_model` wearing a different name.
    """
    from scripts.run_ablations import VARIANTS_BY_NAME

    spec = VARIANTS_BY_NAME["leakage_grouped"]
    assert spec.overrides == [
        "experiment.training.split_protocol=grouped_cv",
        "experiment.training.num_folds=5",
    ]
    assert "leakage_ungrouped" not in VARIANTS_BY_NAME
    assert all(
        not override.startswith("experiment.training.split_protocol")
        for name, variant in VARIANTS_BY_NAME.items()
        if name != "leakage_grouped"
        for override in variant.overrides
    ), "only the leakage row may move the split protocol"


def test_seed_expansion_gives_every_variant_its_own_directory():
    """Repeated seeds must not overwrite each other's summary.json."""
    from src.trainers.runner import DEFAULT_SEEDS, VariantSpec, expand_seeds

    specs = expand_seeds([VariantSpec("full_model", "reference")], DEFAULT_SEEDS)
    assert len(specs) == len(DEFAULT_SEEDS) == 5

    root = Path("/tmp/outputs")
    paths = {spec.save_path(root) for spec in specs}
    assert len(paths) == len(specs)
    assert all(path.name.startswith("seed") for path in paths)
    assert all(f"seed={spec.seed}" in build_command(spec, spec.save_path(root), None) for spec in specs)


def test_full_model_variant_overrides_nothing():
    """The reference row must be the unmodified default configuration."""
    from scripts.run_ablations import VARIANTS_BY_NAME

    assert VARIANTS_BY_NAME["full_model"].overrides == []


@pytest.mark.parametrize(
    ("variant", "flag"),
    [
        ("wo_moe", "use_moe"),
        ("wo_residual", "use_residual"),
        ("wo_kl", "use_kl_loss"),
        ("wo_cross_attn", "use_cross_attention"),
    ],
)
def test_each_ablation_disables_exactly_one_component(variant, flag):
    from scripts.run_ablations import VARIANTS_BY_NAME

    overrides = VARIANTS_BY_NAME[variant].overrides
    assert overrides == [f"model.head.{flag}=false"], (
        f"{variant} must change one thing only, got {overrides}"
    )


def test_baseline_suite_covers_the_requested_models_plus_the_missing_controls():
    """The three submitted baselines, plus the three the suite could not do without.

    ``linear_probe`` answers the question a reviewer asks before any other --
    does the head machinery beat a linear layer on the same frozen features? --
    and ``hierarchical_cce`` is not that control, because it keeps the residual
    and the SubVarietyEmbedding MLP.

    ``imagenet_frozen`` and ``swinv2_supervised`` are the pair that separates
    stage 1 from the architecture: same ImageNet initialisation, trunk frozen in
    one and unfrozen in the other, neither reading a stage-1 checkpoint. Stage 1
    is the most expensive thing in the pipeline, and without the frozen arm there
    is no measurement of what it bought.
    """
    from scripts.run_baselines import BASELINE_VARIANTS

    assert sorted(spec.name for spec in BASELINE_VARIANTS) == [
        "hierarchical_cce",
        "imagenet_frozen",
        "linear_probe",
        "resnet50",
        "swin_tiny",
        "swinv2_supervised",
    ]
    for spec in BASELINE_VARIANTS:
        # The control is a control, not a baseline, and is named and grouped as
        # one -- `generate_plots.py` groups the comparison table on `group`.
        if spec.name == "imagenet_frozen":
            assert spec.experiment == "control_imagenet_frozen"
            assert spec.group == "control"
            continue
        assert spec.experiment.startswith("baseline_")
        assert spec.group == "baseline"


def test_the_imagenet_controls_never_receive_the_stage_one_checkpoint():
    """Both ImageNet arms must ignore the shared encoder, or they stop being controls."""
    from scripts.run_baselines import END_TO_END_BASELINES

    assert {"imagenet_frozen", "swinv2_supervised"} <= END_TO_END_BASELINES


# ------------------------------------------------------- command construction


def test_command_targets_the_variant_directory(tmp_path, checkpoint):
    spec = VariantSpec(name="wo_moe", description="", overrides=["model.head.use_moe=false"])
    save_path = spec.save_path(tmp_path)
    command = build_command(spec, save_path, checkpoint)

    assert command[:3] == [sys.executable, "-m", "src.trainers.moe_finetune"]
    assert f"experiment.training.save_path={save_path}" in command
    assert f"hydra.run.dir={save_path / 'hydra'}" in command
    assert "experiment.variant=wo_moe" in command
    assert "model.head.use_moe=false" in command


def test_ablation_and_baseline_variants_land_in_separate_trees(tmp_path):
    ablation = VariantSpec(name="wo_kl", description="", group="ablation")
    baseline = VariantSpec(name="resnet50", description="", group="baseline")

    assert ablation.save_path(tmp_path) == tmp_path / "ablations" / "wo_kl"
    assert baseline.save_path(tmp_path) == tmp_path / "baselines" / "resnet50"


def test_every_variant_gets_its_own_directory(tmp_path):
    from scripts.run_ablations import ABLATION_VARIANTS

    paths = {spec.save_path(tmp_path) for spec in ABLATION_VARIANTS}
    assert len(paths) == len(ABLATION_VARIANTS), "two variants would overwrite each other"


def test_all_variants_are_pointed_at_the_same_checkpoint(tmp_path, checkpoint):
    """The premise of the whole comparison: one encoder, six architectures."""
    from scripts.run_ablations import ABLATION_VARIANTS

    referenced = {
        argument
        for spec in ABLATION_VARIANTS
        for argument in build_command(spec, spec.save_path(tmp_path), checkpoint)
        if argument.startswith("model.backbone.checkpoint_path=")
    }
    assert referenced == {f"model.backbone.checkpoint_path={checkpoint}"}


def test_checkpoint_override_is_omitted_when_absent(tmp_path):
    """End-to-end baselines own their backbone and must not be handed a SwinV2 file."""
    spec = VariantSpec(name="resnet50", description="", group="baseline")
    command = build_command(spec, spec.save_path(tmp_path), None)
    assert not any(argument.startswith("model.backbone.checkpoint_path") for argument in command)


def test_extra_overrides_come_last_so_they_win(tmp_path, checkpoint):
    """Hydra takes the rightmost value, so a caller's override must be appended."""
    spec = VariantSpec(name="wo_moe", description="", overrides=["model.head.use_moe=false"])
    command = build_command(
        spec, spec.save_path(tmp_path), checkpoint, extra_overrides=["data.batch_size=4"]
    )
    assert command[-1] == "data.batch_size=4"
    assert command.index("data.batch_size=4") > command.index("model.head.use_moe=false")


# -------------------------------------------------------- checkpoint handling


def test_existing_checkpoint_is_returned(checkpoint):
    assert ensure_pretrained_checkpoint(checkpoint) == checkpoint


def test_missing_checkpoint_explains_how_to_produce_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="python main.py pretrain"):
        ensure_pretrained_checkpoint(tmp_path / "absent.pth")


def test_missing_checkpoint_can_be_waived_for_smoke_runs(tmp_path):
    assert ensure_pretrained_checkpoint(tmp_path / "absent.pth", allow_missing=True) is None


# ------------------------------------------------------------------ reporting


def test_manifest_records_overrides_and_outcomes(tmp_path):
    specs = [VariantSpec(name="wo_kl", description="no KL", overrides=["model.head.use_kl_loss=false"])]
    results = [VariantResult("wo_kl", 0, 12.5, str(tmp_path / "wo_kl"), ["python"])]

    payload = json.loads(Path(write_suite_manifest(tmp_path / "m.json", specs, results)).read_text())
    entry = payload["variants"][0]
    assert entry["name"] == "wo_kl"
    assert entry["succeeded"] is True
    assert entry["overrides"] == ["model.head.use_kl_loss=false"]


def test_summary_exit_code_reflects_failures(capsys):
    ok = [VariantResult("a", 0, 1.0, "/tmp/a", [])]
    mixed = [*ok, VariantResult("b", 1, 1.0, "/tmp/b", [])]

    assert print_summary(ok) == 0
    assert print_summary(mixed) == 1
    assert "1 of 2 runs failed" in capsys.readouterr().out
