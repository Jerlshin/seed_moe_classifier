#!/usr/bin/env python
"""Price the stage-2 suite on THIS machine, by measuring a step rather than guessing.

    python scripts/estimate_suite_cost.py                      # default suite, this device
    python scripts/estimate_suite_cost.py --preset reviewer --seeds 42
    python scripts/estimate_suite_cost.py --arms full_model vit_small --iterations 20
    python scripts/estimate_suite_cost.py --device cpu

Measured here: the per-step wall clock of a real forward + backward + optimiser
step, and a real evaluation forward, for each distinct *model shape* in the
selected suite, on the device this process resolved. Derived from those: the
per-run and whole-suite wall clock at the configured epochs and split sizes.

The distinction is the one ``src/utils/training/budget.py`` already draws for
stage 1, and for the same reason: a chart axis must not present a derived number
as a measured one. Every derived figure here is labelled ``[EST]``.

What this deliberately does *not* do
------------------------------------

**It does not model MLX or CoreML, because neither can run this suite.**

* **MLX** is a separate array framework with its own module system and its own
  optimisers. This repository is PyTorch end to end -- ``timm`` backbones,
  ``torch.compile``, DDP, ``GradScaler``, ``SDPA`` -- and none of that has an MLX
  equivalent it could be switched to with a flag. Porting stage 2 to MLX means
  reimplementing the encoder, the head and the loss, at which point the numbers
  would no longer be numbers about this codebase. The Apple-silicon path that
  *does* work is PyTorch's **MPS** backend, which ``select_device`` already
  resolves, and that is what this script measures when it runs on a Mac.
* **CoreML does not train.** It is an inference runtime: a converted model has no
  backward pass and no optimiser. It could serve the *latency* row of the
  efficiency table for a finished checkpoint, which is a different and much
  smaller job than running the suite. It cannot produce a single ablation row.

**Mixed precision is CUDA-only in this pipeline**, and that is not an oversight.
``amp`` is gated on ``device.type == "cuda"`` throughout, so on MPS and CPU the
suite runs in fp32 and ``--amp fp16`` is inert. On a T4 (``sm_75``, no hardware
bf16) fp16 with a ``GradScaler`` is the right setting and is what ``--kaggle``
passes; on Ampere and later ``auto`` resolves to bf16. Evaluation is fp32 on
every device regardless (``AMP_DISABLED``), because a reported metric that moved
with the autocast dtype would make the gaps between variants partly an artefact
of the card they ran on.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from hydra import compose, initialize_config_dir
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

from scripts.run_experiments_suite import DEFAULT_SUITE, PRESETS, VARIANTS_BY_NAME
from src.trainers.runner import DEFAULT_SEEDS, VariantSpec

#: Test fraction of the corpus, matching `experiment.training.test_size`. The
#: remainder splits train/validation, so a training epoch sees 0.8 * 0.8 and a
#: validation pass 0.8 * 0.2 -- which is what the step counts below assume.
TRAIN_FRACTION = 0.64
VALIDATION_FRACTION = 0.16
TEST_FRACTION = 0.20


@dataclass
class StepMeasurement:
    """Measured per-step cost of one model shape on one device."""

    label: str
    batch_size: int
    image_size: int
    train_ms: float
    train_ms_sd: float
    eval_ms: float
    parameters_millions: float

    def epoch_seconds(self, num_samples: int) -> float:
        """One training epoch plus one validation pass, in seconds. [EST]"""
        train_batches = max(1, round(num_samples * TRAIN_FRACTION / self.batch_size))
        eval_batches = max(1, round(num_samples * VALIDATION_FRACTION / self.batch_size))
        return (train_batches * self.train_ms + eval_batches * self.eval_ms) / 1000.0

    def run_seconds(self, num_samples: int, epochs: int) -> float:
        """One whole variant: every epoch, plus the final held-out pass. [EST]"""
        test_batches = max(1, round(num_samples * TEST_FRACTION / self.batch_size))
        return epochs * self.epoch_seconds(num_samples) + test_batches * self.eval_ms / 1000.0


def build_config(conf_dir: str, overrides: list[str]):
    """Compose the config the way a real run does, outside ``@hydra.main``.

    ``tracking/default.yaml`` interpolates ``${hydra:runtime.output_dir}``, which
    only resolves when Hydra's own config has been set. Composing with
    ``return_hydra_config=True`` and then dropping the ``hydra`` node -- Hydra
    marks it read-only, so resolution would fail on Hydra's internals rather
    than on anything this project owns -- is the same thing ``tests/`` does.
    """
    with initialize_config_dir(config_dir=conf_dir, version_base=None):
        composed = compose(config_name="config", overrides=overrides, return_hydra_config=True)
        HydraConfig.instance().set_config(composed)
    cfg = OmegaConf.create(OmegaConf.to_container(composed, resolve=False))
    cfg.pop("hydra", None)
    OmegaConf.resolve(cfg)
    return cfg


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def describe_device(device: torch.device) -> str:
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        return (
            f"{properties.name} (sm_{properties.major}{properties.minor}, "
            f"{properties.total_memory / 1e9:.1f} GB)"
        )
    if device.type == "mps":
        import platform

        return f"Apple silicon, PyTorch MPS ({platform.machine()})"
    return "CPU"


def amp_note(device: torch.device, requested: str | None) -> str:
    """What autocast will actually do here, which is not always what was asked."""
    if device.type != "cuda":
        return (
            f"fp32 (amp is CUDA-gated in this pipeline; "
            f"{'--amp ' + requested + ' would be inert here' if requested else 'nothing to downgrade'})"
        )
    major, _ = torch.cuda.get_device_capability(device)
    if requested in (None, "auto"):
        return "bf16" if major >= 8 else "fp16 + GradScaler (no hardware bf16 below sm_80)"
    if requested == "bf16" and major < 8:
        return "fp16 + GradScaler (bf16 requested, downgraded: sm_75 has no hardware bf16)"
    return str(requested)


def synchronize(device: torch.device) -> None:
    """Drain the queued work before stopping the clock.

    Without this the loop measures CPU enqueue speed rather than execution
    speed -- off by roughly an order of magnitude, which is the same mistake
    that produced the paper's wrong GPU-busy figure from
    ``loop_blocked_fraction``.
    """
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def shape_key(spec: VariantSpec) -> tuple[str, ...]:
    """Variants that build the identical model, so one measurement serves them all.

    Two arms differing only in a *loss* term (``wo_kl``, ``kl_jsd``,
    ``load_entropy``) or in a data augmentation run the same modules at the same
    shapes, so timing each one separately would spend minutes to re-measure a
    number that cannot differ. Anything touching the head's structure or the
    backbone gets its own measurement.
    """
    structural = tuple(
        sorted(
            override
            for override in spec.overrides
            if override.startswith(("model.head", "model.backbone", "data."))
        )
    )
    return (spec.experiment, *structural)


def measure(
    spec: VariantSpec,
    device: torch.device,
    iterations: int,
    warmup: int,
    conf_dir: str,
    extra: list[str],
) -> StepMeasurement:
    """Time one real training step and one real evaluation step for this arm."""
    from src.models.builder import HierarchicalOutput  # noqa: F401 - import cost is real
    from src.trainers.moe_finetune import build_model_and_encoder

    cfg = build_config(conf_dir, [f"experiment={spec.experiment}", *spec.overrides, *extra])

    batch_size = int(cfg.data.batch_size)
    image_size = int(cfg.data.image_size)
    encoder, model = build_model_and_encoder(cfg, device)

    parameters = [
        parameter
        for module in (encoder, model)
        for parameter in module.parameters()
        if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(parameters, lr=1e-4)
    total_millions = sum(
        parameter.numel() for module in (encoder, model) for parameter in module.parameters()
    ) / 1e6

    images = torch.randn(batch_size, 3, image_size, image_size, device=device)
    sub_labels = torch.randint(0, int(cfg.data.num_sub_varieties), (batch_size,), device=device)
    seed_labels = torch.randint(0, int(cfg.data.num_seed_types), (batch_size,), device=device)

    def train_step() -> None:
        optimizer.zero_grad(set_to_none=True)
        output = model(encoder(images), sub_variety_labels=sub_labels)
        loss = torch.nn.functional.cross_entropy(
            output.sub_margin_logits, sub_labels
        ) + torch.nn.functional.cross_entropy(output.seed_type_logits, seed_labels)
        loss.backward()
        optimizer.step()

    @torch.no_grad()
    def eval_step() -> None:
        model.eval()
        encoder.eval()
        model(encoder(images))
        model.train()
        encoder.train()

    encoder.train()
    model.train()
    for _ in range(warmup):
        train_step()
        eval_step()
    synchronize(device)

    timings: list[float] = []
    for _ in range(iterations):
        started = time.perf_counter()
        train_step()
        synchronize(device)
        timings.append((time.perf_counter() - started) * 1000.0)

    eval_timings: list[float] = []
    for _ in range(max(3, iterations // 2)):
        started = time.perf_counter()
        eval_step()
        synchronize(device)
        eval_timings.append((time.perf_counter() - started) * 1000.0)

    # The modules go out of scope with this frame; only the allocator needs a
    # nudge, or the next shape's build meets a cache still holding this one.
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()

    return StepMeasurement(
        label=spec.name,
        batch_size=batch_size,
        image_size=image_size,
        # Median, not mean: one outlier from a background process or an
        # allocator growth event should not set the figure the whole suite
        # estimate is built on.
        train_ms=statistics.median(timings),
        train_ms_sd=statistics.pstdev(timings) if len(timings) > 1 else 0.0,
        eval_ms=statistics.median(eval_timings),
        parameters_millions=total_millions,
    )


def format_duration(seconds: float) -> str:
    hours, remainder = divmod(int(max(0.0, seconds)), 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--preset", choices=sorted(PRESETS), help="Price a named preset.")
    selection.add_argument("--arms", nargs="+", metavar="NAME", help="Price specific arms.")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--device", default="auto", help="'auto' | 'cuda' | 'mps' | 'cpu'.")
    parser.add_argument("--iterations", type=int, default=12, help="Timed training steps per shape.")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument(
        "--num-samples",
        type=int,
        default=13492,
        help="Corpus size (default: Refined_Samples' 13,492 crops).",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override the configured epochs.")
    parser.add_argument("--gpus", type=int, default=1, help="Devices the suite will be sharded over.")
    parser.add_argument(
        "overrides", nargs=argparse.REMAINDER, help="Extra Hydra overrides, after a bare '--'."
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = resolve_device(args.device)
    extra = [item for item in args.overrides if item != "--"]
    conf_dir = str((PROJECT_ROOT / "conf").resolve())

    if args.arms:
        specs = [VARIANTS_BY_NAME[name] for name in args.arms]
    elif args.preset:
        specs = [VARIANTS_BY_NAME[name] for name in PRESETS[args.preset][1]]
    else:
        specs = list(DEFAULT_SUITE)

    # The measurement is per model *shape*, not per arm: two arms that differ
    # only in a loss term run identical modules at identical shapes.
    by_shape: dict[tuple[str, ...], list[VariantSpec]] = {}
    for spec in specs:
        by_shape.setdefault(shape_key(spec), []).append(spec)

    print(f"Device:     {describe_device(device)}")
    print(f"Precision:  {amp_note(device, None)}")
    print(f"Corpus:     {args.num_samples:,} crops "
          f"({TRAIN_FRACTION:.0%} train / {VALIDATION_FRACTION:.0%} val / {TEST_FRACTION:.0%} test)")
    print(f"Arms:       {len(specs)} x {len(args.seeds)} seed(s) = {len(specs) * len(args.seeds)} runs")
    print(f"Shapes:     {len(by_shape)} distinct model shape(s) to measure\n")

    measurements: dict[tuple[str, ...], StepMeasurement] = {}
    for key, group in by_shape.items():
        representative = group[0]
        print(f"  measuring {representative.name} ...", end="", flush=True)
        try:
            measurements[key] = measure(
                representative, device, args.iterations, args.warmup, conf_dir, extra
            )
            measurement = measurements[key]
            print(
                f" {measurement.train_ms:.1f} ms/step (+-{measurement.train_ms_sd:.1f}), "
                f"{measurement.eval_ms:.1f} ms/eval-batch, {measurement.parameters_millions:.1f} M params"
            )
        except Exception as error:  # noqa: BLE001 - one arm failing must not lose the rest
            print(f" FAILED: {type(error).__name__}: {error}")

    print("\n" + "=" * 92)
    print(f"{'arm':<26} {'params':>8} {'ms/step':>9} {'per run [EST]':>15} {'x seeds [EST]':>15}")
    print("-" * 92)

    total = 0.0
    for spec in specs:
        measurement = measurements.get(shape_key(spec))
        if measurement is None:
            print(f"{spec.name:<26} {'--':>8} {'--':>9} {'not measured':>15}")
            continue
        cfg = build_config(conf_dir, [f"experiment={spec.experiment}", *spec.overrides, *extra])
        epochs = args.epochs if args.epochs is not None else int(cfg.experiment.training.epochs)
        per_run = measurement.run_seconds(args.num_samples, epochs)
        per_arm = per_run * len(args.seeds)
        total += per_arm
        print(
            f"{spec.name:<26} {measurement.parameters_millions:>7.1f}M "
            f"{measurement.train_ms:>8.1f} {format_duration(per_run):>15} {format_duration(per_arm):>15}"
        )

    print("=" * 92)
    wall = total / max(1, args.gpus)
    print(f"{'TOTAL [EST]':<26} {'':>8} {'':>9} {'':>15} {format_duration(total):>15}")
    if args.gpus > 1:
        print(f"{'  sharded over ' + str(args.gpus) + ' GPUs':<26} {'':>8} {'':>9} {'':>15} {format_duration(wall):>15}")
    print()
    print("[EST] = derived from the measured per-step cost above, assuming every epoch")
    print("        costs the same and the dataloader never starves the device. On a")
    print("        cold page cache or a network-mounted corpus the first epoch is")
    print("        slower; run `--iterations 30` for a tighter per-step figure.")

    sessions = wall / (11.5 * 3600)
    if sessions > 1:
        print(f"\nKaggle: ~{sessions:.1f} sessions at the 12 h limit. Launch with")
        print("        python scripts/run_experiments_suite.py --all --kaggle")
        print("        and relaunch the identical command line after each one.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
