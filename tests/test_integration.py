"""End-to-end integration: data -> backbone -> head -> loss -> optimizer -> metrics.

These exercise the wiring the unit tests deliberately isolate: the trainer's
epoch loop, checkpoint round-tripping, and the figure/metric artifacts the paper
reports. The backbone is stubbed (see ``DummyFeatureExtractor``) so nothing here
touches the network.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset

from src.datasets.dataset import HierarchicalSeedDataset
from src.datasets.transforms import get_supervised_transforms
from src.losses.hierarchical import build_combined_loss
from src.models.builder import HierarchicalSeedClassifier, build_hierarchical_moe
from src.trainers.moe_finetune import (
    build_optimizer,
    clone_cpu_state_dict,
    resolve_monitor,
    run_epoch,
    save_split_manifest,
    split_dataset,
    stratification_labels,
)
from src.utils.training import CheckpointManager, ExperimentTracker
from tests.conftest import (
    PAPER_EMBED_DIM,
    PAPER_NUM_EXPERTS,
    PAPER_NUM_SEED_TYPES,
    PAPER_NUM_SUB_VARIETIES,
    REVISED_TOP_K,
)


@pytest.fixture
def dataset(synthetic_dataset_root) -> HierarchicalSeedDataset:
    return HierarchicalSeedDataset(
        root_dir=synthetic_dataset_root,
        transform=get_supervised_transforms(image_size=16, train=False),
    )


@pytest.fixture
def tracker(tmp_path) -> ExperimentTracker:
    cfg = OmegaConf.create(
        {
            "tracking": {
                "output_dir": str(tmp_path / "run"),
                "tensorboard": {"enabled": False},
                "wandb": {"enabled": False},
                "intervals": {"log_every_steps": 1},
                "artifacts": {
                    "log_gradient_norms": False,
                    "log_confusion_matrices": True,
                    "log_expert_utilization": True,
                    "log_per_class_tables": True,
                    "log_tsne": False,
                    "log_embeddings": False,
                    "max_tsne_samples": 100,
                },
            }
        }
    )
    tracker = ExperimentTracker(cfg, logging.getLogger("test"))
    yield tracker
    tracker.close()


@pytest.fixture
def trainer_pieces(dataset):
    torch.manual_seed(0)
    model = HierarchicalSeedClassifier(
        feature_dim=PAPER_EMBED_DIM,
        embed_dim=PAPER_EMBED_DIM,
        num_seed_types=PAPER_NUM_SEED_TYPES,
        num_sub_varieties=PAPER_NUM_SUB_VARIETIES,
        num_experts=PAPER_NUM_EXPERTS,
        top_k=REVISED_TOP_K,
        moe_hidden_dim=32,
        num_heads=4,
        dropout_rate=0.0,
    )
    criterion = build_combined_loss(
        OmegaConf.create({}),
        num_seed_types=PAPER_NUM_SEED_TYPES,
        num_sub_varieties=PAPER_NUM_SUB_VARIETIES,
        subvariety_to_seed_type=dataset.get_subvariety_to_seed_type(),
    )
    return model, criterion


# ------------------------------------------------------------------- splits


def test_splits_are_disjoint_and_cover_the_dataset(dataset):
    splits, test_indices, _ = split_dataset(
        dataset, test_size=0.2, num_folds=1, seed=42, protocol="stratified"
    )
    (train_indices, val_indices) = splits[0]

    assert not set(train_indices) & set(val_indices)
    assert not set(train_indices) & set(test_indices)
    assert not set(val_indices) & set(test_indices)
    assert len(train_indices) + len(val_indices) + len(test_indices) == len(dataset)


def test_kfold_produces_the_requested_number_of_folds(dataset):
    splits, _, _ = split_dataset(
        dataset, test_size=0.2, num_folds=3, seed=42, protocol="stratified"
    )
    assert len(splits) == 3
    for train_indices, val_indices in splits:
        assert not set(train_indices) & set(val_indices)


def test_stratification_key_is_the_sub_variety_label(dataset):
    """The composite ``seed * 1000 + sub`` key induced exactly these strata.

    The repository used to justify the composite key by claiming that
    "stratifying on sub-variety alone would not guarantee seed-type balance".
    That is false: sub-variety labels are global and each has exactly one parent,
    so ``seed = parent(sub)`` is a deterministic function of ``sub`` and the map
    ``sub -> seed*1000 + sub`` is a bijection. Same partition, simpler key, and
    the stated reason no longer says something untrue.
    """
    labels = stratification_labels(dataset)
    assert len(labels) == len(dataset)
    for key, (_, _, sub_label) in zip(labels, dataset.samples):
        assert key == sub_label

    composite = np.array([s * 1000 + b for _, s, b in dataset.samples])
    # A bijection induces identical strata: equal keys iff equal composite keys.
    assert len(set(zip(labels.tolist(), composite.tolist()))) == len(set(labels.tolist()))


def test_grouped_splitting_keeps_source_photographs_on_one_side(dataset):
    """The protocol the real dataset needs: 9,357 crops from 81 photographs.

    Under crop-level splitting, near-duplicate views of the same physical seeds
    -- same lighting, same background, overlapping bounding boxes -- land on both
    sides of the boundary, and the reported accuracy is substantially a
    memorisation score.
    """
    splits, test_indices, report = split_dataset(
        dataset, test_size=0.3, num_folds=1, seed=42, protocol="grouped"
    )
    groups = dataset.source_groups()
    train_groups = set(groups[splits[0][0]].tolist())
    test_groups = set(groups[test_indices].tolist())

    assert not (train_groups & test_groups)
    assert report["shared_source_groups"] == 0
    assert report["protocol"] == "grouped"


def test_split_report_quantifies_leakage_under_the_ungrouped_protocol(dataset):
    """The crop-level protocol is the primary one, and it carries its own indictment.

    Reporting the leak is what turns it from a hidden flaw into a measured
    result: ``leakage_grouped`` in the ablation suite runs the photograph-disjoint
    counterpart and the delta against ``full_model`` is the number the paper
    should quote.
    """
    _, _, report = split_dataset(
        dataset, test_size=0.3, num_folds=1, seed=42, protocol="stratified"
    )
    assert report["protocol"] == "stratified"
    assert report["shared_source_groups"] >= 0
    assert 0.0 <= report["leaked_test_fraction"] <= 1.0


def test_split_manifest_round_trips(dataset, tmp_path):
    splits, test_indices, _ = split_dataset(
        dataset, test_size=0.2, num_folds=1, seed=42, protocol="stratified"
    )
    path = save_split_manifest(tmp_path, splits, test_indices, dataset, protocol="stratified")

    payload = np.load(path, allow_pickle=True)
    assert "fold_1_train_indices" in payload
    assert np.array_equal(payload["test_indices"], test_indices)
    assert list(payload["subvariety_to_seed_type"]) == dataset.get_subvariety_to_seed_type()
    # Persisted so a reviewer can verify the grouping rather than trusting the name.
    assert np.array_equal(payload["source_groups"], dataset.source_groups())
    assert str(payload["split_protocol"]) == "stratified"


# --------------------------------------------------------------- epoch loop


def test_training_epoch_updates_parameters_and_reports_metrics(
    dataset, trainer_pieces, dummy_encoder, tracker
):
    model, criterion = trainer_pieces
    cfg = OmegaConf.create(
        {"experiment": {"training": {
            "optimizer": {"name": "AdamW"}, "learning_rate": 1e-3, "weight_decay": 0.0,
        }}}
    )
    optimizer = build_optimizer([model], cfg)
    before = [p.detach().clone() for p in model.parameters()]

    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    metrics, evaluation, accumulator, global_step = run_epoch(
        encoder=dummy_encoder,
        model=model, criterion=criterion, loader=loader,
        device=torch.device("cpu"), tracker=tracker, logger=logging.getLogger("test"),
        epoch=1, global_step=0, phase="train", dataset=dataset,
        optimizer=optimizer, max_batches=3, clip_grad=3.0,
        num_experts=PAPER_NUM_EXPERTS, max_tsne_samples=100,
    )

    assert global_step == 3
    assert accumulator.batches == 3
    assert np.isfinite(metrics["loss"])
    assert any(not torch.equal(a, b) for a, b in zip(before, model.parameters()))

    # Every paper metric family must be present.
    assert "seed_type/accuracy" in metrics
    assert "sub_variety/accuracy" in metrics
    assert "kl_alignment/overall" in metrics
    assert "moe/expert_0_utilization" in metrics
    assert 0.0 <= evaluation.alignment.overall <= 1.0


def test_evaluation_epoch_leaves_parameters_untouched(
    dataset, trainer_pieces, dummy_encoder, tracker
):
    model, criterion = trainer_pieces
    before = [p.detach().clone() for p in model.parameters()]

    loader = DataLoader(dataset, batch_size=8, shuffle=False)
    _, _, _, global_step = run_epoch(
        encoder=dummy_encoder,
        model=model, criterion=criterion, loader=loader,
        device=torch.device("cpu"), tracker=tracker, logger=logging.getLogger("test"),
        epoch=1, global_step=0, phase="validation", dataset=dataset,
        optimizer=None, max_batches=3,
        num_experts=PAPER_NUM_EXPERTS, max_tsne_samples=100,
    )

    assert global_step == 0  # evaluation must not advance the step counter
    for original, current in zip(before, model.parameters()):
        assert torch.equal(original, current)


def test_evaluation_applies_no_arcface_margin(dataset, trainer_pieces, dummy_encoder):
    """At evaluation the reported logits must be margin-free, or metrics are biased."""
    model, criterion = trainer_pieces
    model.eval()
    images, _, sub_labels = next(iter(DataLoader(dataset, batch_size=8)))
    features = dummy_encoder(images)

    with torch.no_grad():
        evaluated = model(features, sub_variety_labels=None)
        trained = model(features, sub_variety_labels=sub_labels)

    assert torch.equal(evaluated.sub_logits, evaluated.sub_margin_logits)
    assert not torch.equal(trained.sub_logits, trained.sub_margin_logits)
    # The un-margined logits are identical either way.
    assert torch.allclose(evaluated.sub_logits, trained.sub_logits, atol=1e-6)


def test_empty_loader_raises_a_clear_error(
    dataset, trainer_pieces, dummy_encoder, tracker
):
    model, criterion = trainer_pieces
    loader = DataLoader(Subset(dataset, []), batch_size=4)
    with pytest.raises(RuntimeError, match="No batches processed"):
        run_epoch(
            encoder=dummy_encoder,
            model=model, criterion=criterion, loader=loader,
            device=torch.device("cpu"), tracker=tracker, logger=logging.getLogger("test"),
            epoch=1, global_step=0, phase="train", dataset=dataset,
        )


def test_loss_decreases_over_repeated_steps_on_a_fixed_batch(
    dataset, trainer_pieces, dummy_encoder
):
    """A model that cannot overfit one batch has a broken gradient path somewhere."""
    model, criterion = trainer_pieces
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    images, seed_labels, sub_labels = next(iter(DataLoader(dataset, batch_size=16, shuffle=True)))
    features = dummy_encoder(images)

    model.train()
    losses = []
    for _ in range(30):
        optimizer.zero_grad(set_to_none=True)
        breakdown = criterion(model(features, sub_labels), seed_labels, sub_labels)
        breakdown.total.backward()
        optimizer.step()
        losses.append(float(breakdown.total.detach()))

    assert losses[-1] < losses[0], f"loss did not decrease: {losses[0]:.3f} -> {losses[-1]:.3f}"


# -------------------------------------------------------------- checkpoints


def test_checkpoint_round_trip_reproduces_predictions(dataset, trainer_pieces, tmp_path):
    """A saved model_state_dict alone must be enough to reproduce inference."""
    model, _ = trainer_pieces
    model.eval()
    features = torch.randn(6, PAPER_EMBED_DIM)
    with torch.no_grad():
        expected = model(features).sub_logits

    manager = CheckpointManager(tmp_path, keep_last_n=1)
    path = manager.save("model.pth", {"model_state_dict": model.state_dict()})

    restored = build_hierarchical_moe(
        OmegaConf.create(
            {
                "feature_dim": PAPER_EMBED_DIM, "embed_dim": PAPER_EMBED_DIM,
                "num_seed_types": PAPER_NUM_SEED_TYPES,
                "num_sub_varieties": PAPER_NUM_SUB_VARIETIES,
                "num_experts": PAPER_NUM_EXPERTS, "top_k": REVISED_TOP_K,
                "moe_hidden_dim": 32, "num_heads": 4, "dropout_rate": 0.0,
            }
        )
    )
    restored.load_state_dict(torch.load(path, weights_only=False)["model_state_dict"])
    restored.eval()
    with torch.no_grad():
        assert torch.allclose(restored(features).sub_logits, expected, atol=1e-6)


def test_checkpoint_manager_prunes_rolling_files(tmp_path):
    manager = CheckpointManager(tmp_path, keep_last_n=1)
    for epoch in range(3):
        manager.save(f"ckpt_epoch_{epoch:04d}.pth", {"epoch": epoch}, rolling_prefix="ckpt_epoch_")
    assert len(list(tmp_path.glob("ckpt_epoch_*.pth"))) == 1


# ----------------------------------------------------------------- tracking


def test_tracker_writes_events_and_figures(tmp_path, tracker):
    from src.utils.visualization import plot_expert_utilization

    tracker.log_metrics({"loss": 0.5, "ignored": "not a scalar"}, step=1, prefix="train")
    tracker.log_table("per_class", ["class", "f1"], [["Rice", 0.99]], step=1)
    saved = tracker.log_figure(
        "expert_utilization", plot_expert_utilization([0.2, 0.2, 0.2, 0.2, 0.1, 0.1]), step=1
    )

    assert saved is not None and saved.endswith(".png")
    lines = tracker.events_path.read_text(encoding="utf-8").strip().splitlines()
    assert any('"metrics"' in line and "train/loss" in line for line in lines)
    assert any('"table"' in line for line in lines)


def test_tracker_survives_missing_optional_backends(tmp_path):
    """Both integrations disabled must still produce a working jsonl sink."""
    cfg = OmegaConf.create(
        {"tracking": {
            "output_dir": str(tmp_path / "run2"),
            "tensorboard": {"enabled": False},
            "wandb": {"enabled": False},
        }}
    )
    with ExperimentTracker(cfg, logging.getLogger("test")) as tracker:
        assert tracker.writer is None
        assert tracker.wandb_run is None
        tracker.log_metrics({"loss": 1.0}, step=0)
    assert (tmp_path / "run2" / "events.jsonl").exists()


# ------------------------------------------- best-checkpoint state retention
#
# The defect these pin: `best_state` used to hold LIVE module references
# (`{"encoder": encoder, "model": model, ...}`). Those modules kept training for
# the remaining epochs, so the held-out evaluation, the efficiency profile and
# `hierarchical_moe_final.pth` all described the FINAL epoch while reporting the
# selected epoch's number -- and the checkpoint file carried the wrong `epoch`
# field. Measured on `outputs/finetune_hierarchical_moe`, a run selected at
# epoch 6: `hierarchical_moe_final.pth` was labelled epoch 6 and was
# tensor-identical to `model_fold1_epoch0100.pth`.


def _params(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def _assert_bit_identical(left, right, label: str) -> None:
    assert set(left) == set(right), label
    for key in left:
        assert torch.equal(left[key], right[key]), f"{label}: tensor {key} differs"


def test_clone_cpu_state_dict_does_not_alias_a_cpu_module():
    """`Tensor.cpu()` returns *self* for a CPU tensor, so the clone is load-bearing.

    Without it the "snapshot" aliases the live parameter on any CPU run and the
    bug survives the fix on exactly the configuration the tests exercise.
    """
    model = torch.nn.Linear(4, 3)
    snapshot = clone_cpu_state_dict(model)
    with torch.no_grad():
        model.weight.add_(1.0)
    assert not torch.equal(snapshot["weight"], model.weight.detach().cpu())


@pytest.mark.parametrize("num_folds", [1, 3])
def test_selected_checkpoint_is_restored_bit_identically(dataset, num_folds):
    """The restored weights are the selected epoch's, not the last epoch's.

    Mirrors the trainer's own loop: modules are rebuilt per fold, a snapshot is
    taken whenever the monitored metric improves, training continues, and the
    snapshot is restored into the live modules before evaluation. The monitor
    series is scripted so the winner is never the final epoch of the final fold
    -- which is precisely the case the old reference-holding code got wrong.
    """
    epochs = 4
    # Scripted validation macro-F1. The global maximum sits mid-run in every
    # configuration, so a restore that silently yielded the last epoch fails.
    scores = {
        (1, 1): 0.10, (1, 2): 0.55, (1, 3): 0.30, (1, 4): 0.20,
        (2, 1): 0.15, (2, 2): 0.40, (2, 3): 0.90, (2, 4): 0.25,  # global best
        (3, 1): 0.05, (3, 2): 0.35, (3, 3): 0.45, (3, 4): 0.50,
    }

    best_monitored = float("-inf")
    best_state = None
    reference = None  # independent copy of the weights at the winning epoch
    model = None

    for fold in range(1, num_folds + 1):
        torch.manual_seed(100 + fold)
        # Rebuilt per fold, exactly as `moe_finetune` does.
        model = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.Linear(8, 4))
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        for epoch in range(1, epochs + 1):
            # A real parameter update, so every epoch's weights are distinct.
            optimizer.zero_grad()
            model(torch.randn(4, 8)).sum().backward()
            optimizer.step()

            monitored = scores[(fold, epoch)]
            if monitored > best_monitored:
                best_monitored = monitored
                best_state = {
                    "model_state": clone_cpu_state_dict(model),
                    "epoch": epoch,
                    "fold": fold,
                }
                reference = _params(model)

    assert best_state is not None
    expected_fold = 2 if num_folds >= 2 else 1
    expected_epoch = 3 if num_folds >= 2 else 2
    assert (best_state["fold"], best_state["epoch"]) == (expected_fold, expected_epoch)

    # The winner is not the final epoch of the final fold, so a restore that
    # returned the live weights would differ.
    live = _params(model)
    assert any(not torch.equal(live[key], reference[key]) for key in reference)

    model.load_state_dict(best_state["model_state"], strict=True)
    _assert_bit_identical(_params(model), reference, "restored selected checkpoint")


def test_snapshot_is_unaffected_by_later_epochs(dataset):
    """A snapshot taken at the best epoch must not track subsequent training."""
    torch.manual_seed(7)
    model = torch.nn.Linear(6, 6)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

    snapshot = clone_cpu_state_dict(model)
    frozen = {key: value.clone() for key, value in snapshot.items()}

    for _ in range(5):
        optimizer.zero_grad()
        model(torch.randn(3, 6)).sum().backward()
        optimizer.step()

    _assert_bit_identical(snapshot, frozen, "snapshot after further training")
    assert not torch.equal(snapshot["weight"], model.weight.detach().cpu())


# ----------------------------------------------------- monitor configuration


@pytest.mark.parametrize(
    ("monitor", "higher_is_better"),
    [
        ("sub_variety/f1_macro", True),
        ("sub_variety/accuracy", True),
        ("seed_type/f1_macro", True),
        ("loss", False),
        ("arcface_loss", False),
    ],
)
def test_resolve_monitor_direction(monitor, higher_is_better):
    """A key containing "loss" minimises; every other metric maximises.

    `monitor` was documented in the config and read by nothing -- selection was
    hard-wired to the validation loss. This pins the direction rule so a metric
    cannot be selected in the wrong direction, which fails silently: the run
    still produces a checkpoint, just consistently the worst one seen.
    """
    cfg = OmegaConf.create({"experiment": {"validation": {"monitor": monitor}}})
    key, resolved = resolve_monitor(cfg)
    assert key == monitor
    assert resolved is higher_is_better
