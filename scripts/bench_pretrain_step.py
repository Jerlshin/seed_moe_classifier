#!/usr/bin/env python
"""A/B micro-benchmark of the exact stage-1 training micro-step, no dataset.

    python scripts/bench_pretrain_step.py
    python scripts/bench_pretrain_step.py --batch-size 64 --accum 1
    python scripts/bench_pretrain_step.py --no-sdpa --no-compile     # baseline
    python scripts/bench_pretrain_step.py --batch-size 64 --accum 1 --grad-checkpointing

Reproduces one micro-batch of ``src/trainers/contrastive_pretrain.py`` on
synthetic data, byte-for-byte in structure: pinned ``uint8`` views in the
collated ``[V, B, C, H, W]`` layout, the ``ViewBatcher`` H2D + normalise +
deferred local upsample, one fused teacher forward over ``2B`` globals, one
fused student forward over ``(2 + L)B`` views, the DINO loss with Sinkhorn
centering and KoLeo, backward, and — on accumulation boundaries — clip, fused
AdamW and the foreach EMA. What it does *not* reproduce is the CPU augmentation
pipeline, which is the point: this isolates the GPU-resident step so execution
options can be compared without dataloader noise. ``data_wait_fraction`` in a
real run tells you which of the two you are limited by.

Timing runs after a warmup that absorbs ``torch.compile`` graph capture and
cuDNN autotuning, between explicit ``synchronize()`` calls — removing those
would measure enqueue speed rather than execution speed, off by roughly an
order of magnitude.

Use it to answer, on the actual server GPU, in minutes:

* what ``sdpa_attention`` buys (``--no-sdpa`` vs default);
* whether a physical batch of 64 at ``--accum 1`` now fits, and its img/s
  (the effective batch — physical x accumulation — must stay 64: the collapse
  guards are batch statistics and the LR/momentum regime is tuned to it);
* whether ``--grad-checkpointing`` is needed to fit, and what it costs;
* what a compile mode is worth (``--compile-mode max-autotune-no-cudagraphs``).

Peak memory is reported per configuration, so the largest batch that fits can
be found without sacrificing a real run to an OOM.
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

from src.datasets.dataset import MultiCropBatch  # noqa: E402
from src.losses.dino import CustomDINOLoss  # noqa: E402
from src.models.backbones.swinv2_dino import DINO  # noqa: E402
from src.trainers.contrastive_pretrain import ViewBatcher  # noqa: E402
from src.utils.training import (  # noqa: E402
    TeacherEmaUpdater,
    autocast_context,
    build_grad_scaler,
    configure_backend,
    resolve_amp,
    select_device,
)

NORMALIZE_MEAN = (0.485, 0.456, 0.406)
NORMALIZE_STD = (0.229, 0.224, 0.225)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backbone", default="swinv2_base_window16_256")
    parser.add_argument("--feature-dim", type=int, default=1024)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--local-crop-size", type=int, default=101)
    parser.add_argument("--local-crops", type=int, default=4)
    parser.add_argument("--out-dim", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=16, help="physical images per micro-batch")
    parser.add_argument("--accum", type=int, default=4, help="micro-batches per optimizer step")
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


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    configure_backend(device, allow_tf32=True, cudnn_benchmark=True, matmul_precision="high")
    amp = resolve_amp(device, args.amp)
    scaler = build_grad_scaler(amp)

    num_crops = 2 + args.local_crops
    print(
        f"device={device} amp={amp.label} backbone={args.backbone} "
        f"batch={args.batch_size} x accum={args.accum} (effective {args.batch_size * args.accum}) "
        f"views/micro-batch={args.batch_size * num_crops} "
        f"sdpa={args.sdpa} compile={args.compile}({args.compile_mode}) "
        f"grad_ckpt={args.grad_checkpointing}"
    )

    model = DINO(
        backbone_name=args.backbone,
        input_dim=args.feature_dim,
        hidden_dim=2048,
        bottleneck_dim=256,
        out_dim=args.out_dim,
        pretrained=False,
    ).to(device)
    runtime = model.configure_runtime(
        compile_enabled=args.compile,
        compile_mode=args.compile_mode,
        grad_checkpointing=args.grad_checkpointing,
        channels_last=args.channels_last,
        sdpa_attention=args.sdpa,
    )
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
        student_views, teacher_views, batch_size = batcher(batch)
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
        if (index + 1) % args.accum == 0:
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
    started = time.perf_counter()
    for index in range(args.steps):
        micro_step(index)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    images = args.steps * args.batch_size
    print(
        f"{elapsed / args.steps * 1000:.1f} ms/micro-batch | "
        f"{images / elapsed:.1f} img/s | {images * num_crops / elapsed:.1f} views/s"
        + (
            f" | peak_mem={torch.cuda.max_memory_allocated() / 1024**3:.2f} GiB"
            if device.type == "cuda"
            else ""
        )
    )


if __name__ == "__main__":
    main()
