#!/usr/bin/env python
"""End-to-end pipeline smoke test on synthetic tensors.

    python scripts/dry_run.py
    python scripts/dry_run.py --backbone swinv2_base_window16_256 --epochs 3
    python scripts/dry_run.py --device cpu --output-dir /tmp/dry_run

Exercises every stage on random data, with no dataset, no checkpoint and no
network access::

    synthetic images
      -> self-supervised SwinV2 encoder       z in R^384        (Eq. 4)
      -> seed-type classifier                 s, p_s            (Eqs. 5-6)
      -> Top-2 MoE over 6 experts             h                 (Eq. 8)
      -> residual seed-type fusion            h' = h + P(p_s)   (Eq. 9)
      -> cross-attention refinement           h''               (Eqs. 11-12)
      -> ArcFace sub-variety classifier                         (Eq. 13)
      -> combined objective + backward pass
      -> hierarchical metrics
      -> efficiency profiling
      -> summary.json + summary_metrics.csv

The point is to fail fast. A shape mismatch, a broken toggle, a missing tensor
in ``HierarchicalOutput`` or a metric that cannot consume the model's output all
surface here in seconds, rather than after a data-loading stage and an epoch of
real training. Run it before launching anything long.

The backbone is built with ``pretrained=False``, so nothing is downloaded and
the weights are random -- the losses are meaningless by design. What is being
verified is that the pipeline *runs*, not that it learns.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.losses.hierarchical import CombinedHierarchicalLoss
from src.models.builder import (
    PAPER_EMBED_DIM,
    DinoV2SwinV2Encoder,
    HierarchicalSeedClassifier,
)
from src.utils.efficiency import profile_model
from src.utils.evaluation import RunSummary, write_summary_csv
from src.utils.metrics import evaluate_hierarchical
from src.utils.training import select_device

NUM_SEED_TYPES = 4
NUM_SUB_VARIETIES = 27
# 13 rice + 8 millet + 3 + 3 = 27, matching the paper's Section 3 hierarchy.
SUBVARIETY_COUNTS = (13, 8, 3, 3)


def build_hierarchy() -> list[int]:
    """Parent seed type of each sub-variety index."""
    mapping: list[int] = []
    for seed_index, count in enumerate(SUBVARIETY_COUNTS):
        mapping.extend([seed_index] * count)
    assert len(mapping) == NUM_SUB_VARIETIES
    return mapping


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--backbone",
        default="swinv2_tiny_window16_256",
        help="SwinV2 variant. Tiny by default so the dry run stays fast; the wiring "
        "it verifies is identical for Base.",
    )
    parser.add_argument("--image-size", type=int, default=256, help="Must match the backbone's window resolution.")
    parser.add_argument("--samples", type=int, default=10, help="Synthetic images to generate.")
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=2, help="Experts routed per token.")
    parser.add_argument("--num-experts", type=int, default=6)
    parser.add_argument(
        "--token-mode",
        choices=("grid", "pooled"),
        default="grid",
        help="grid keeps SwinV2's token grid through the head (attention is real); "
        "pooled reproduces the submitted architecture (attention is affine).",
    )
    parser.add_argument(
        "--weighting-mode",
        choices=("fixed", "uncertainty"),
        default="fixed",
        help="Task-term weighting. 'uncertainty' exercises the criterion's learnable scalars.",
    )
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Where to write summary.json and summary_metrics.csv "
        "(default: outputs/dry_run).",
    )
    parser.add_argument("--no-efficiency", action="store_true", help="Skip the efficiency profiler.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = select_device(args.device)
    output_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "outputs" / "dry_run"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(f"Dry run | device={device} backbone={args.backbone} top_k={args.top_k}")
    print("=" * 72)

    # ------------------------------------------------------- synthetic data
    images = torch.randn(args.samples, 3, args.image_size, args.image_size)
    seed_labels = torch.randint(0, NUM_SEED_TYPES, (args.samples,))
    hierarchy = build_hierarchy()
    # Sub-variety labels are drawn from their seed type's own children. Sampling
    # them independently would create label pairs no real sample can have, and
    # the KL alignment term -- whose whole job is to agree with the hierarchy --
    # would then be uninterpretable even as a smoke signal.
    children = {
        seed: [index for index, parent in enumerate(hierarchy) if parent == seed]
        for seed in range(NUM_SEED_TYPES)
    }
    sub_labels = torch.tensor(
        [
            children[int(seed)][int(torch.randint(0, len(children[int(seed)]), (1,)))]
            for seed in seed_labels
        ]
    )
    print(f"[1/7] Synthetic batch: {tuple(images.shape)}, {NUM_SEED_TYPES} seed types, "
          f"{NUM_SUB_VARIETIES} sub-varieties")

    # ------------------------------------------------------------- encoder
    started = time.perf_counter()
    encoder = DinoV2SwinV2Encoder(
        model_name=args.backbone,
        embed_dim=PAPER_EMBED_DIM,
        checkpoint_path=None,   # random weights: this verifies wiring, not accuracy
        pretrained=False,
        freeze_backbone=True,
        token_mode=args.token_mode,
    ).to(device)
    print(
        f"[2/7] Encoder built in {time.perf_counter() - started:.1f}s: "
        f"{encoder.backbone_dim} -> {encoder.embed_dim}"
    )

    with torch.no_grad():
        probe = encoder(images[:2].to(device))
    # The invariant is on the *last* axis: grid mode adds a token axis in front
    # of it, pooled mode does not, and both must land on the paper's 384.
    expected_ndim = 3 if args.token_mode == "grid" else 2
    if probe.ndim != expected_ndim or probe.shape[0] != 2 or probe.shape[-1] != PAPER_EMBED_DIM:
        raise AssertionError(
            f"Encoder in token_mode={args.token_mode!r} must emit "
            f"{'[B, tokens, ' if expected_ndim == 3 else '[B, '}{PAPER_EMBED_DIM}], "
            f"got {tuple(probe.shape)}"
        )
    print(
        f"      z shape verified: {tuple(probe.shape)}"
        + (f" ({probe.shape[1]} routing tokens per image)" if expected_ndim == 3 else "")
    )

    # ---------------------------------------------------------------- head
    model = HierarchicalSeedClassifier(
        feature_dim=PAPER_EMBED_DIM,
        embed_dim=PAPER_EMBED_DIM,
        num_seed_types=NUM_SEED_TYPES,
        num_sub_varieties=NUM_SUB_VARIETIES,
        num_experts=args.num_experts,
        top_k=args.top_k,
        token_mode=args.token_mode,
        router_noise_std=0.3,
    ).to(device)
    criterion = CombinedHierarchicalLoss(
        num_seed_types=NUM_SEED_TYPES,
        num_sub_varieties=NUM_SUB_VARIETIES,
        subvariety_to_seed_type=hierarchy,
        embed_dim=PAPER_EMBED_DIM,
        num_experts=args.num_experts,
        weighting_mode=args.weighting_mode,
    ).to(device)
    # The criterion joins the optimizer because uncertainty weighting owns three
    # learnable scalars; under fixed weighting it contributes nothing.
    optimizer = torch.optim.AdamW(
        [
            p
            for module in (encoder, model, criterion)
            for p in module.parameters()
            if p.requires_grad
        ],
        lr=1e-4,
    )
    print(f"[3/7] Head built: {args.num_experts} experts, top-{args.top_k}, "
          f"flags={model.component_flags()}")

    # ------------------------------------------------------------ training
    seed_pred: list[int] = []
    sub_pred: list[int] = []
    seed_true: list[int] = []
    sub_true: list[int] = []
    expert_indices: list[torch.Tensor] = []
    history = {"train_loss": [], "validation_loss": []}

    for epoch in range(1, args.epochs + 1):
        model.train()
        encoder.train()
        epoch_loss = 0.0
        batches = 0
        for collection in (seed_pred, sub_pred, seed_true, sub_true, expert_indices):
            collection.clear()

        for start in range(0, args.samples, args.batch_size):
            stop = min(start + args.batch_size, args.samples)
            batch_images = images[start:stop].to(device)
            batch_seed = seed_labels[start:stop].to(device)
            batch_sub = sub_labels[start:stop].to(device)

            optimizer.zero_grad(set_to_none=True)
            features = encoder(batch_images)
            output = model(features, sub_variety_labels=batch_sub)
            breakdown = criterion(output, batch_seed, batch_sub)
            breakdown.total.backward()
            # Sparse dispatch leaves unrouted experts with `grad is None`, which
            # AdamW skips entirely -- including weight decay. Materialising zeros
            # is what keeps sparse and dense dispatch equivalent under the
            # optimizer, not just in the forward pass.
            model.materialize_expert_grads()
            optimizer.step()

            epoch_loss += float(breakdown.total.detach())
            batches += 1
            seed_pred.extend(output.seed_type_logits.argmax(dim=1).detach().cpu().tolist())
            sub_pred.extend(output.sub_logits.argmax(dim=1).detach().cpu().tolist())
            seed_true.extend(batch_seed.cpu().tolist())
            sub_true.extend(batch_sub.cpu().tolist())
            expert_indices.append(output.top_k_indices.detach().cpu())

            if output.top_k_indices.shape[1] != args.top_k:
                raise AssertionError(
                    f"Router selected {output.top_k_indices.shape[1]} experts, expected {args.top_k}"
                )

        mean_loss = epoch_loss / max(batches, 1)
        history["train_loss"].append(mean_loss)
        history["validation_loss"].append(mean_loss)
        parts = breakdown.as_dict()
        print(
            f"[4/7] Epoch {epoch}/{args.epochs} | total={mean_loss:.4f} "
            f"seed={parts['seed_type_loss']:.4f} arcface={parts['arcface_loss']:.4f} "
            f"kl={parts['kl_loss']:.4f} load={parts['moe_load_balancing_loss']:.4f} "
            f"z={parts['moe_router_z_loss']:.4f} cosine={parts['cosine_loss']:.4f} "
            f"residual={parts['residual_magnitude_loss']:.4f} "
            f"dead_experts={int(parts['moe_dead_experts'])}"
        )

    # ------------------------------------------------------------- metrics
    evaluation = evaluate_hierarchical(
        seed_true=seed_true,
        seed_pred=seed_pred,
        sub_true=sub_true,
        sub_pred=sub_pred,
        subvariety_to_seed_type=hierarchy,
        num_seed_types=NUM_SEED_TYPES,
        num_sub_varieties=NUM_SUB_VARIETIES,
        seed_type_names=[f"seed_{i}" for i in range(NUM_SEED_TYPES)],
        sub_variety_names=[f"sub_{i:02d}" for i in range(NUM_SUB_VARIETIES)],
        top_k_indices=torch.cat(expert_indices).numpy(),
        num_experts=args.num_experts,
        tokens_per_sample=int(probe.shape[1]) if probe.ndim == 3 else 1,
    )
    print(
        f"[5/7] Metrics | seed_acc={evaluation.seed_type['accuracy']:.4f} "
        f"sub_acc={evaluation.sub_variety['accuracy']:.4f} "
        f"kl_alignment={evaluation.alignment.overall:.4f} "
        f"dead_experts={evaluation.dead_experts} "
        f"nmi_sub={evaluation.expert_nmi.get('sub_variety', float('nan')):.3f}\n"
        f"              expert_utilization="
        f"{[round(float(v), 3) for v in evaluation.expert_utilization]}"
    )

    # ---------------------------------------------------------- efficiency
    efficiency = None
    if not args.no_efficiency:
        efficiency = profile_model(
            model,
            images[: args.batch_size].to(device),
            device,
            name="dry_run",
            extra_modules=[encoder],
            batch_sizes=(1, args.batch_size),
            warmup=1,
            iterations=3,
            forward_fn=lambda batch: model(encoder(batch)),
        )
        print(f"[6/7] Efficiency | {efficiency.summary_line()}")
    else:
        print("[6/7] Efficiency | skipped")

    # ------------------------------------------------------------- artifacts
    summary = RunSummary(
        name="dry_run",
        group="dry_run",
        run_dir=str(output_dir),
        metrics=dict(evaluation.scalar_metrics()),
        efficiency=efficiency.as_dict() if efficiency is not None else {},
        history=history,
        component_flags=model.component_flags(),
        loss_flags=criterion.loss_flags(),
        config={
            "backbone": args.backbone,
            "head": "hierarchical_moe",
            "embed_dim": PAPER_EMBED_DIM,
            "num_experts": args.num_experts,
            "top_k": args.top_k,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "seed": args.seed,
        },
    )
    summary_path = summary.save(output_dir)
    csv_path = write_summary_csv(output_dir / "summary_metrics.csv", [summary])
    print(f"[7/7] Artifacts | {summary_path}\n                 {csv_path}")

    print("\n" + "=" * 72)
    print("Dry run completed with no runtime errors.")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
