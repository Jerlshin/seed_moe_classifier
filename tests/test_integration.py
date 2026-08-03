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
    """The stratified protocol is retained, but it must carry its own indictment.

    Reporting the leak is what turns it from a hidden flaw into a measured
    result: ``leakage_ungrouped`` in the ablation suite runs exactly this and the
    delta against ``full_model`` is the number the paper should quote.
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
