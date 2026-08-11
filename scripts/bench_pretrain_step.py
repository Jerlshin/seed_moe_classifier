#!/usr/bin/env python
"""A/B micro-benchmark of the exact stage-1 training micro-step, no dataset.

    python scripts/bench_pretrain_step.py
    python scripts/bench_pretrain_step.py --batch-size 64 --accum 1
    python scripts/bench_pretrain_step.py --no-sdpa --no-compile     # baseline
    python scripts/bench_pretrain_step.py --batch-size 64 --accum 1 --grad-checkpointing

    # single-GPU vs DDP, one command, printed as a scaling table
    python scripts/bench_pretrain_step.py --scaling 1,2 --batch-size 16 --accum 2

    # one rank of a DDP benchmark, launched by hand
    torchrun --standalone --nproc_per_node=2 scripts/bench_pretrain_step.py

Reproduces one micro-batch of ``src/trainers/contrastive_pretrain.py`` on
synthetic data, byte-for-byte in structure: pinned ``uint8`` views in the
collated ``[V, B, C, H, W]`` layout, the ``ViewBatcher`` H2D + normalise +
deferred local upsample, one fused teacher forward over ``2B`` globals, one
fused student forward over ``(2 + L)B`` views, the DINO loss with Sinkhorn
centering and KoLeo, backward, and — on accumulation boundaries — clip, fused
AdamW and the foreach EMA. Under ``torchrun`` it also reproduces the DDP wrapper
and the ``no_sync`` accumulation pattern, so the measured gradient traffic is the
traffic a real run pays.

What it does *not* reproduce is the CPU augmentation pipeline, which is the
point: this isolates the GPU-resident step so execution options can be compared
without dataloader noise. ``data_wait_fraction`` in a real run's step log tells
you which of the two you are limited by; this tells you how fast the other one
can go.

Timing runs after a warmup that absorbs ``torch.compile`` graph capture and
cuDNN autotuning, between explicit ``synchronize()`` calls — removing those
would measure enqueue speed rather than execution speed, off by roughly an
order of magnitude.

Reported per configuration: milliseconds per micro-batch, images/s and views/s
(**job-wide**, so the DDP number is directly comparable to the single-GPU one),
peak VRAM per rank, and mean/peak SM utilisation where NVML is available. Use it
to answer, on the actual server GPU, in minutes:

* what ``sdpa_attention`` buys (``--no-sdpa`` vs default);
* **the largest physical batch the card can hold** (``--find-batch-size``),
  which is the measurement the training configuration is waiting on: Sinkhorn
  and KoLeo are evaluated per micro-batch, so the physical batch is what their
  estimates are made from and accumulation cannot substitute for it;
* whether ``--grad-checkpointing`` is needed to fit, and what it costs;
* what a compile mode is worth (``--compile-mode max-autotune-no-cudagraphs``);
* what a second GPU actually buys (``--scaling 1,2``), which on a T4 pair
  without NVLink is a real question rather than a rhetorical one.

Peak memory is reported per configuration, so the largest batch that fits can
be found without sacrificing a real run to an OOM.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.dataset import MultiCropBatch  # noqa: E402
from src.losses.dino import CustomDINOLoss  # noqa: E402
from src.models.backbones.swinv2_dino import DINO  # noqa: E402
from src.trainers.contrastive_pretrain import ViewBatcher  # noqa: E402
from src.utils.training import (  # noqa: E402
    GpuUtilizationSampler,
    TeacherEmaUpdater,
    all_reduce_mean,
    autocast_context,
    build_grad_scaler,
    configure_backend,
    describe_accelerator,
    resolve_amp,
    setup_distributed,
    shutdown_distributed,
)

NORMALIZE_MEAN = (0.485, 0.456, 0.406)
NORMALIZE_STD = (0.229, 0.224, 0.225)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backbone", default="swinv2_tiny_window16_256")
    parser.add_argument("--feature-dim", type=int, default=768)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--local-crop-size", type=int, default=101)
    parser.add_argument("--local-crops", type=int, default=4)
    parser.add_argument("--out-dim", type=int, default=2048)
    parser.add_argument("--hidden-dim", type=int, default=1024, help="DINO head hidden width")
    parser.add_argument("--bottleneck-dim", type=int, default=256)
    parser.add_argument("--drop-path", type=float, default=0.1, help="student stochastic depth")
    parser.add_argument("--batch-size", type=int, default=32, help="physical images per micro-batch per rank")
    parser.add_argument("--accum", type=int, default=1, help="micro-batches per optimizer step")
    parser.add_argument(
        "--find-batch-size",
        default=None,
        help=(
            "Comma-separated physical batch sizes to try, e.g. '16,24,32,48,64'. Runs the real "
            "micro-step at each in a FRESH SUBPROCESS, reports peak VRAM and img/s per "
            "candidate, and names the largest that fits. This is the measurement to take "
            "before committing to a batch: an OOM here costs seconds, the same OOM twenty "
            "minutes into epoch 1 costs the run."
        ),
    )
    parser.add_argument("--steps", type=int, default=24, help="timed micro-batches")
    parser.add_argument("--warmup", type=int, default=8, help="untimed micro-batches (compile capture)")
    parser.add_argument("--amp", default="auto", choices=["auto", "bf16", "fp16", "off"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--sdpa", dest="sdpa", action="store_true", default=True)
    parser.add_argument("--no-sdpa", dest="sdpa", action="store_false")
    parser.add_argument("--compile", dest="compile", action="store_true", default=True)
    parser.add_argument("--no-compile", dest="compile", action="store_false")
    parser.add_argument("--compile-mode", default="default")
    parser.add_argument("--grad-checkpointing", action="store_true")
    parser.add_argument("--channels-last", action="store_true")
    parser.add_argument("--sinkhorn-iterations", type=int, default=3)
    parser.add_argument("--lambda-koleo", type=float, default=0.1)
    parser.add_argument(
        "--scaling",
        default=None,
        help=(
            "Comma-separated rank counts to compare, e.g. '1,2'. Re-invokes this script "
            "under torch.distributed.run for each and prints a scaling table. Every other "
            "flag is forwarded, so the per-rank micro-batch stays fixed and the job-wide "
            "effective batch scales with the rank count -- which is what a scaling number "
            "means."
        ),
    )
    parser.add_argument("--json", default=None, help="Write this run's measurements to a JSON file.")
    return parser.parse_args()


def synthetic_batch(args: argparse.Namespace, pin: bool) -> MultiCropBatch:
    """One collated batch exactly as the dataloader would emit it.

    ``uint8`` views in the view-major ``[V, B, C, H, W]`` layout, locals at
    their native ``local_crop_size`` (the upsample belongs to the GPU), pinned
    when the copy will be a real H2D transfer. One fixed batch is reused for
    every step: the GPU-side work is shape-dependent, not value-dependent.
    """
    generator = torch.Generator().manual_seed(0)

    def views(count: int, size: int) -> torch.Tensor:
        tensor = torch.randint(
            0, 256, (count, args.batch_size, 3, size, size), dtype=torch.uint8, generator=generator
        )
        return tensor.pin_memory() if pin else tensor

    return MultiCropBatch(
        global_views=views(2, args.image_size),
        local_views=views(args.local_crops, args.local_crop_size) if args.local_crops else None,
        targets=torch.zeros(args.batch_size, dtype=torch.long),
        paths=tuple(f"synthetic_{index}" for index in range(args.batch_size)),
        originals=None,
    )


# ------------------------------------------------------------------- scaling


def run_scaling(args: argparse.Namespace) -> int:
    """Re-invoke this script at each rank count and print the comparison.

    Each configuration runs in a **fresh process group**, which is the only way
    to measure them independently: a cached allocator, a warmed autotuner and an
    already-negotiated NCCL communicator would all leak from one measurement into
    the next.
    """
    counts = [int(item) for item in str(args.scaling).replace(" ", "").split(",") if item]
    forwarded = [
        argument
        for argument in sys.argv[1:]
        if not argument.startswith("--scaling") and argument not in {str(args.scaling)}
    ]
    # `--json` is repurposed per configuration; drop any caller-supplied one.
    forwarded = [
        argument for index, argument in enumerate(forwarded)
        if not argument.startswith("--json") and (index == 0 or not forwarded[index - 1].startswith("--json"))
    ]

    rows: list[dict] = []
    for count in counts:
        output = PROJECT_ROOT / f".bench_scaling_{count}.json"
        if count <= 1:
            command = [sys.executable, str(Path(__file__).resolve()), *forwarded, "--json", str(output)]
        else:
            command = [
                sys.executable, "-m", "torch.distributed.run",
                "--standalone", f"--nproc_per_node={count}",
                str(Path(__file__).resolve()), *forwarded, "--json", str(output),
            ]
        print(f"\n$ {' '.join(command)}", flush=True)
        environment = dict(os.environ)
        environment.setdefault("OMP_NUM_THREADS", str(max((os.cpu_count() or 2) // count, 1)))
        code = subprocess.call(command, cwd=str(PROJECT_ROOT), env=environment)
        if code != 0:
            print(f"Configuration with {count} rank(s) failed (exit {code}).")
            continue
        rows.append(json.loads(output.read_text()))
        output.unlink(missing_ok=True)

    if not rows:
        return 1

    print("\n" + "=" * 88)
    print(
        f"{'ranks':>5}  {'ms/micro-batch':>15}  {'img/s (job)':>12}  {'views/s (job)':>14}  "
        f"{'peak GiB/rank':>13}  {'GPU %':>6}  {'speedup':>8}"
    )
    print("-" * 88)
    baseline = rows[0]["images_per_second"]
    for row in rows:
        utilisation = row.get("gpu_utilization_mean")
        print(
            f"{row['world_size']:>5}  {row['ms_per_micro_batch']:>15.1f}  "
            f"{row['images_per_second']:>12.1f}  {row['views_per_second']:>14.1f}  "
            f"{row.get('peak_memory_gib', float('nan')):>13.2f}  "
            f"{(f'{utilisation:.0f}' if utilisation is not None else 'n/a'):>6}  "
            f"{row['images_per_second'] / baseline:>7.2f}x"
        )
    print("=" * 88)
    print(
        "Effective batch scales with the rank count here, because the per-rank micro-batch "
        "is held fixed -- that is what a scaling measurement means. For a REAL run, keep the "
        "effective batch at 64 by lowering experiment.training.gradient_accumulation_steps, "
        "or just set experiment.training.effective_batch_size and let it derive."
    )
    return 0


# ----------------------------------------------------------- batch-size search


def run_batch_size_search(args: argparse.Namespace) -> int:
    """Try each candidate physical batch in its own process; report what fits.

    A **fresh subprocess per candidate** is not fastidiousness. CUDA's caching
    allocator does not return freed blocks to the driver, so a batch of 64 that
    OOMs leaves the process holding a fragmented reservation, and every smaller
    candidate measured afterwards in the same process reports a peak that
    reflects the failed attempt rather than itself. An OOM is also not reliably
    recoverable in-process once autograd has partially unwound.

    Candidates run largest-last so the table reads in the order a reader
    chooses from, and a failure at one size does not stop the rest -- 48 can
    fail while 64 succeeds under a different allocator split, and seeing that is
    the point.
    """
    candidates = sorted(
        {int(item) for item in str(args.find_batch_size).replace(" ", "").split(",") if item}
    )
    skip_prefixes = ("--find-batch-size", "--batch-size", "--json", "--accum")
    forwarded: list[str] = []
    skip_next = False
    for argument in sys.argv[1:]:
        if skip_next:
            skip_next = False
            continue
        if any(argument.startswith(prefix) for prefix in skip_prefixes):
            skip_next = "=" not in argument
            continue
        forwarded.append(argument)

    rows: list[dict] = []
    for batch_size in candidates:
        output = PROJECT_ROOT / f".bench_batch_{batch_size}.json"
        command = [
            sys.executable, str(Path(__file__).resolve()), *forwarded,
            "--batch-size", str(batch_size), "--accum", "1", "--json", str(output),
        ]
        print(f"\n$ {' '.join(command)}", flush=True)
        code = subprocess.call(command, cwd=str(PROJECT_ROOT))
        if code != 0 or not output.exists():
            print(f"batch {batch_size}: FAILED (exit {code}) -- treat as does not fit.")
            rows.append({"batch_size": batch_size, "fits": False})
            output.unlink(missing_ok=True)
            continue
        row = json.loads(output.read_text())
        row["fits"] = True
        row["batch_size"] = batch_size
        rows.append(row)
        output.unlink(missing_ok=True)

    print("\n" + "=" * 78)
    print(f"{'batch':>6}  {'fits':>5}  {'peak GiB':>9}  {'ms/step':>9}  {'img/s':>9}  {'views/s':>9}")
    print("-" * 78)
    for row in rows:
        if not row["fits"]:
            print(f"{row['batch_size']:>6}  {'no':>5}  {'-':>9}  {'-':>9}  {'-':>9}  {'-':>9}")
            continue
        print(
            f"{row['batch_size']:>6}  {'yes':>5}  {row.get('peak_memory_gib', 0.0):>9.2f}  "
            f"{row['ms_per_micro_batch']:>9.1f}  {row['images_per_second']:>9.1f}  "
            f"{row['views_per_second']:>9.1f}"
        )
    print("=" * 78)

    fitting = [row["batch_size"] for row in rows if row["fits"]]
    if not fitting:
        print("Nothing fits. Lower the candidates, or enable grad_checkpointing.")
        return 1
    largest = max(fitting)
    print(
        f"Largest physical batch that fits: {largest}.\n"
        f"Use it with accumulation 1 and let the LR follow:\n"
        f"    python main.py pretrain data.batch_size={largest} "
        f"experiment.training.effective_batch_size={largest}\n"
        "Leave headroom: this measures one micro-step on an otherwise idle card, and a real "
        "run also holds the dataloader's pinned buffers and any allocator fragmentation that "
        "accumulates over an epoch. If the largest candidate fits with under ~10 % of the "
        "card free, take the next one down."
    )
    return 0


# ---------------------------------------------------------------------- main


def main() -> int:
    args = parse_args()
    if args.find_batch_size:
        return run_batch_size_search(args)
    if args.scaling:
        return run_scaling(args)

    context = setup_distributed(args.device)
    device = context.device
    configure_backend(device, allow_tf32=True, cudnn_benchmark=True, matmul_precision="high")
    amp = resolve_amp(device, args.amp)
    scaler = build_grad_scaler(amp)

    num_crops = 2 + args.local_crops
    if context.is_main:
        report = describe_accelerator(device)
        print(report.summary_line())
        print(
            f"device={device} ranks={context.world_size} amp={amp.label} backbone={args.backbone} "
            f"batch={args.batch_size}/rank x accum={args.accum} x {context.world_size} ranks "
            f"(effective {args.batch_size * args.accum * context.world_size}) "
            f"views/micro-batch/rank={args.batch_size * num_crops} "
            f"sdpa={args.sdpa} compile={args.compile}({args.compile_mode}) "
            f"grad_ckpt={args.grad_checkpointing}"
        )

    model = DINO(
        backbone_name=args.backbone,
        input_dim=args.feature_dim,
        hidden_dim=args.hidden_dim,
        bottleneck_dim=args.bottleneck_dim,
        out_dim=args.out_dim,
        # Weights are irrelevant to a shape-and-schedule benchmark, and skipping
        # the download keeps this runnable offline. Drop path IS relevant: it
        # changes what backward has to store.
        pretrained=False,
        drop_path_rate=args.drop_path,
    ).to(device)
    runtime = model.configure_runtime(
        compile_enabled=args.compile,
        compile_mode=args.compile_mode,
        grad_checkpointing=args.grad_checkpointing,
        channels_last=args.channels_last,
        sdpa_attention=args.sdpa,
    )
    model.configure_distributed(context)
    if context.is_main:
        print(f"runtime={runtime}")
        if args.sdpa and not runtime.get("sdpa_attention_modules"):
            print("WARNING: sdpa_attention converted 0 modules; benchmarking the eager path.")

    criterion = CustomDINOLoss(
        out_dim=args.out_dim,
        num_crops=num_crops,
        warmup_teacher_temp=0.04,
        teacher_temp=0.07,
        warmup_teacher_temp_epochs=30,
        num_epochs=300,
        centering="sinkhorn",
        sinkhorn_iterations=args.sinkhorn_iterations,
        lambda_koleo=args.lambda_koleo,
        context=context,
    ).to(device)
    criterion.metrics_enabled = False

    student_parameters = model.student_parameters()
    try:
        optimizer = torch.optim.AdamW(
            student_parameters, lr=5e-4, weight_decay=0.04, fused=device.type == "cuda"
        )
    except (RuntimeError, TypeError, ValueError):
        optimizer = torch.optim.AdamW(student_parameters, lr=5e-4, weight_decay=0.04)
    ema = TeacherEmaUpdater(model.ema_pairs())

    batcher = ViewBatcher(
        image_size=args.image_size, mean=NORMALIZE_MEAN, std=NORMALIZE_STD, device=device
    )
    batch = synthetic_batch(args, pin=device.type == "cuda")
    student_ids = list(range(num_crops))
    teacher_ids = [0, 1]

    def micro_step(index: int) -> None:
        """One micro-batch, ordered exactly as the trainer orders it."""
        is_step = (index + 1) % args.accum == 0
        student_views, teacher_views, batch_size = batcher(batch)
        # The same `no_sync` pattern the trainer uses: gradient traffic once per
        # optimizer step, not once per micro-batch. Benchmarking without it would
        # report `accum` times the real communication cost.
        with nullcontext() if is_step else model.no_sync():
            with autocast_context(amp):
                teacher_out = model.forward_teacher_views(teacher_views)
                student_out, bottleneck = model.forward_student_views(
                    student_views, return_bottleneck=True
                )
                embeddings = bottleneck[: 2 * batch_size] if criterion.lambda_koleo > 0 else None
                loss = criterion(
                    student_out,
                    teacher_out,
                    epoch=0,
                    student_view_ids=student_ids,
                    teacher_view_ids=teacher_ids,
                    student_embeddings=embeddings,
                )
            scaled = loss / args.accum
            if scaler is not None:
                scaler.scale(scaled).backward()
            else:
                scaled.backward()
        if is_step:
            if scaler is not None:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(student_parameters, max_norm=3.0)
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            ema.update(0.996)

    model.train()
    for index in range(args.warmup):
        micro_step(index)

    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    sampler = GpuUtilizationSampler(context.local_rank if device.type == "cuda" else 0)
    with sampler:
        started = time.perf_counter()
        for index in range(args.steps):
            micro_step(index)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started

    # Averaged across ranks: a straggler makes the whole job slower, so the mean
    # step time is the honest per-step cost and rank 0's own is not.
    elapsed_tensor = torch.tensor([elapsed], dtype=torch.float64, device=device)
    all_reduce_mean(elapsed_tensor, context)
    elapsed = float(elapsed_tensor.item())

    images = args.steps * args.batch_size * context.world_size
    utilisation = sampler.summary()
    measurement = {
        "world_size": context.world_size,
        "backend": context.backend,
        "batch_size_per_rank": args.batch_size,
        "accum": args.accum,
        "effective_batch": args.batch_size * args.accum * context.world_size,
        "amp": amp.label,
        "sdpa": bool(args.sdpa),
        "compile": bool(args.compile),
        "grad_checkpointing": bool(args.grad_checkpointing),
        "ms_per_micro_batch": elapsed / args.steps * 1000.0,
        "images_per_second": images / elapsed,
        "views_per_second": images * num_crops / elapsed,
        "peak_memory_gib": (
            torch.cuda.max_memory_allocated() / 1024**3 if device.type == "cuda" else 0.0
        ),
        "gpu_utilization_mean": utilisation["gpu_utilization_mean"],
        "gpu_utilization_peak": utilisation["gpu_utilization_peak"],
        "runtime": {key: str(value) for key, value in runtime.items()},
    }

    if context.is_main:
        print(
            f"{measurement['ms_per_micro_batch']:.1f} ms/micro-batch | "
            f"{measurement['images_per_second']:.1f} img/s | "
            f"{measurement['views_per_second']:.1f} views/s"
            + (
                f" | peak_mem={measurement['peak_memory_gib']:.2f} GiB/rank"
                if device.type == "cuda"
                else ""
            )
            + (
                f" | gpu={measurement['gpu_utilization_mean']:.0f}% mean, "
                f"{measurement['gpu_utilization_peak']:.0f}% peak"
                if measurement["gpu_utilization_mean"] is not None
                else " | gpu utilisation: NVML unavailable"
            )
        )
        if args.json:
            Path(args.json).write_text(json.dumps(measurement, indent=2), encoding="utf-8")

    shutdown_distributed(context)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
