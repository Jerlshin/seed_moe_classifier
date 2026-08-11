#!/usr/bin/env python
"""Verify, on *this* machine, that the fast paths compute the intended function.

    python scripts/verify_runtime.py                 # everything available here
    python scripts/verify_runtime.py --skip-tests    # just the GPU-only checks
    python scripts/verify_runtime.py --gpus 2        # include the NCCL check

Run it once after provisioning a machine, and again after any change to
``sdpa_attention.py``, ``distributed.py`` or the precision plumbing. It answers
the question the test suite cannot answer from a developer laptop: *does this
hardware, this driver and this torch build still make the optimisations exact?*

Four checks, in increasing order of what they need:

1. **Capabilities.** Compute capability, bf16, TF32, which SDPA backends exist,
   whether Triton is importable. Everything downstream follows from these, and
   they are also what a later reader of ``summary.json`` needs in order to
   compare two runs.

2. **The portable contracts**, delegated to pytest: SDPA parity in fp64/bf16/fp16,
   AMP dtype selection and the fp32 pinning of the terms that need it, DDP
   gradient equality over Gloo, and exact resume. These run on CPU and are the
   same checks CI runs.

3. **SDPA parity on the real trunk**, per module, in fp64 on a copy -- the same
   probe the trainer runs at conversion time, printed rather than silently
   acted on. This is where a drifted timm shows up.

4. **DDP gradient equality over the real backend** (NCCL on a CUDA box), which
   the Gloo tests cannot cover. Needs ``--gpus N`` and relaunches itself under
   ``torch.distributed.run``.

A failure in 3 or 4 does not mean the run will be wrong -- both paths refuse or
fall back rather than proceeding -- but it does mean the run will be *slow*, and
knowing that before committing 300 epochs is the point.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.training import describe_accelerator, select_device  # noqa: E402

#: The portable contracts. Named individually so a failure names the claim.
TEST_TARGETS = (
    "tests/test_precision.py",
    "tests/test_distributed.py",
    "tests/test_checkpointing.py",
    "tests/test_throughput.py",
)

SEPARATOR = "=" * 78


def heading(text: str) -> None:
    print(f"\n{SEPARATOR}\n{text}\n{SEPARATOR}", flush=True)


# ------------------------------------------------------------ 1. capabilities


def report_capabilities(device: torch.device) -> int:
    heading("1. Hardware and build")
    report = describe_accelerator(device)
    print(report.summary_line())
    for key, value in report.as_dict().items():
        print(f"  {key:<32} {value}")

    if device.type != "cuda":
        print("\nNo CUDA device: checks 3 and 4 will be skipped.")
        return 0

    if not report.supports_bf16:
        print(
            "\nNOTE: no hardware bfloat16 (this is a Turing-class card, e.g. a T4).\n"
            "  amp: auto will select fp16 with a GradScaler. That is the intended\n"
            "  path, not a degradation: the Sinkhorn normaliser, the prototype\n"
            "  log-softmax and the KoLeo distances are pinned to fp32 inside the\n"
            "  autocast region, and tests/test_precision.py checks that they are."
        )
    if not report.supports_flash_sdpa:
        print(
            "\nNOTE: no FlashAttention SDPA backend (needs sm_80+). The converted\n"
            "  window attention will use the memory-efficient kernel instead, which\n"
            "  still avoids materialising the [B*nW, heads, N, N] matrices -- the\n"
            "  saving that makes the conversion worth doing is intact."
        )
    if not report.compile_available:
        print(f"\nNOTE: torch.compile unavailable ({report.compile_reason}).")
        print("  compile.enabled: auto will stay eager. Everything else is unaffected.")
    return 0


# ------------------------------------------------------- 2. portable contracts


def run_tests(extra_args: list[str]) -> int:
    heading("2. Portable numerical contracts (pytest)")
    command = [sys.executable, "-m", "pytest", *TEST_TARGETS, "-q", *extra_args]
    print(f"$ {' '.join(command)}", flush=True)
    return subprocess.call(command, cwd=str(PROJECT_ROOT))


# ----------------------------------------------------- 3. SDPA on this trunk


def run_sdpa_report(backbone: str, device: torch.device) -> int:
    heading("3. SDPA window-attention parity, per module, on this GPU")
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "diagnose_sdpa_parity.py"),
        "--backbone", backbone,
        "--device", str(device),
    ]
    print(f"$ {' '.join(command)}", flush=True)
    return subprocess.call(command, cwd=str(PROJECT_ROOT))


# --------------------------------------------------- 4. DDP on the real backend


def ddp_gradient_check() -> int:
    """One rank of the real-backend gradient-equality check.

    Same claim as ``tests/test_distributed.py`` proves over Gloo, re-checked over
    whatever backend this machine actually uses -- because NCCL is the one that
    will carry a real run's gradients, and a bad interconnect or a mismatched
    NCCL build shows up as a *wrong reduction* rather than as an error.
    """
    import torch.distributed as dist
    import torch.nn as nn

    from src.utils.training import setup_distributed, shutdown_distributed
    from torch.nn.parallel import DistributedDataParallel

    context = setup_distributed("auto")
    if not context.enabled:
        print("Not launched under torchrun; nothing to check.")
        return 0

    device = context.device
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 4)).to(device)
    ddp = DistributedDataParallel(
        model, device_ids=[context.local_rank] if device.type == "cuda" else None
    )

    generator = torch.Generator().manual_seed(4321)
    per_rank = 8
    inputs = torch.randn(context.world_size * per_rank, 16, generator=generator)
    targets = torch.randint(0, 4, (context.world_size * per_rank,), generator=generator)

    shard = slice(context.rank * per_rank, (context.rank + 1) * per_rank)
    ddp.zero_grad(set_to_none=True)
    loss = nn.functional.cross_entropy(
        ddp(inputs[shard].to(device)), targets[shard].to(device)
    )
    loss.backward()

    reference = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 4)).to(device)
    reference.load_state_dict({key: value.clone() for key, value in model.state_dict().items()})
    # The reference must be the *initial* weights; DDP has not stepped, so the
    # current ones are still the initial ones.
    reference.zero_grad(set_to_none=True)
    nn.functional.cross_entropy(reference(inputs.to(device)), targets.to(device)).backward()

    worst = 0.0
    for (name, parameter), (_, expected) in zip(
        model.named_parameters(), reference.named_parameters()
    ):
        difference = (parameter.grad - expected.grad).abs().max().item()
        worst = max(worst, difference)

    verdict = torch.tensor([worst], device=device)
    dist.all_reduce(verdict, op=dist.ReduceOp.MAX)
    worst = float(verdict.item())

    if context.is_main:
        tolerance = 1e-4 if device.type == "cuda" else 1e-6
        status = "PASS" if worst <= tolerance else "FAIL"
        print(
            f"{status}: DDP's averaged gradient over {context.world_size} ranks ({context.backend}) "
            f"differs from the single-process gradient on the concatenated batch by at most "
            f"{worst:.3e} (tolerance {tolerance:.0e})."
        )
        shutdown_distributed(context)
        return 0 if worst <= tolerance else 1

    shutdown_distributed(context)
    return 0


def run_ddp_check(processes: int) -> int:
    heading(f"4. DDP gradient equality over the real backend ({processes} ranks)")
    command = [
        sys.executable, "-m", "torch.distributed.run",
        "--standalone", f"--nproc_per_node={processes}",
        str(Path(__file__).resolve()), "--ddp-worker",
    ]
    print(f"$ {' '.join(command)}", flush=True)
    environment = dict(os.environ)
    environment.setdefault("OMP_NUM_THREADS", "1")
    return subprocess.call(command, cwd=str(PROJECT_ROOT), env=environment)


# ---------------------------------------------------------------------- main


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backbone", default="swinv2_tiny_window16_256")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--gpus",
        type=int,
        default=0,
        help="Ranks for the real-backend DDP check. 0 skips it; 2 is the useful value.",
    )
    parser.add_argument("--skip-tests", action="store_true", help="Skip the pytest section.")
    parser.add_argument("--skip-sdpa", action="store_true", help="Skip the per-module SDPA report.")
    parser.add_argument(
        "--ddp-worker", action="store_true", help=argparse.SUPPRESS
    )
    args, extra = parser.parse_known_args()

    if args.ddp_worker:
        return ddp_gradient_check()

    device = select_device(args.device)
    failures: list[str] = []

    if report_capabilities(device) != 0:
        failures.append("capabilities")
    if not args.skip_tests and run_tests(extra) != 0:
        failures.append("portable contracts")
    if not args.skip_sdpa and device.type == "cuda" and run_sdpa_report(args.backbone, device) != 0:
        failures.append("SDPA parity report")
    if args.gpus > 1 and run_ddp_check(args.gpus) != 0:
        failures.append("DDP gradient equality")

    heading("Verdict")
    if failures:
        print("FAILED: " + ", ".join(failures))
        return 1
    print("All requested checks passed on this machine.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
