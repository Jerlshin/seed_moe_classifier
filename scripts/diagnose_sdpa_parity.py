#!/usr/bin/env python
"""Per-module parity report for the SwinV2 -> SDPA attention conversion.

    python scripts/diagnose_sdpa_parity.py
    python scripts/diagnose_sdpa_parity.py --backbone swinv2_base_window16_256
    python scripts/diagnose_sdpa_parity.py --device cpu

For every ``WindowAttention`` module in the backbone this prints, on the
machine it runs on:

* whether the module sits in a shifted or non-shifted block, and its
  dim / heads / window / N geometry;
* ``logit_scale`` raw and after ``clamp(max=log 100).exp()``;
* the relative-position-bias shape, the probe mask shape, the SDPA
  ``attn_mask`` shapes, and the q/k shapes and strides the rewrite produces;
* the maximum absolute and relative forward error between the module's own
  stock forward and the SDPA rewrite, measured three ways:

  - **fp32 (flags as-is)** — what the pre-fix guard measured. On a CUDA card
    with TF32 enabled this includes cuBLAS/SDPA kernel selection noise
    (~5e-4), which is what refused the ``heads >= 16`` stages on a 12 GB card;
  - **fp32, TF32 disabled** — same probe with tensor-core rounding removed;
  - **fp64** — pure algebra, what the guard measures now (gate 1e-10);

* fp64 gradient error for the input and for every parameter (worst named);
* the current guard's verdict for the module.

The environment header records torch / timm versions, device, TF32 flags and
matmul precision so a report can be read without access to the box.
"""

from __future__ import annotations

import argparse
import copy
import sys
import types
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.backbones.sdpa_attention import (  # noqa: E402
    PARITY_ATOL,
    PARITY_RTOL,
    _parity_check,
    sdpa_window_attention_forward,
)
from src.utils.training import select_device  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backbone", default="swinv2_tiny_window16_256")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def environment_header(device: torch.device) -> str:
    import timm

    lines = [
        f"torch={torch.__version__} timm={timm.__version__} device={device}",
        f"float32_matmul_precision={torch.get_float32_matmul_precision()}",
    ]
    if device.type == "cuda":
        lines.append(
            f"gpu={torch.cuda.get_device_name(device)} "
            f"capability={'.'.join(map(str, torch.cuda.get_device_capability(device)))} "
            f"matmul.allow_tf32={torch.backends.cuda.matmul.allow_tf32} "
            f"cudnn.allow_tf32={torch.backends.cudnn.allow_tf32}"
        )
    return "\n".join(lines)


def forward_errors(module, dtype: torch.dtype, seed: int) -> dict[str, float] | str:
    """Max abs/rel error between stock forward and rewrite at ``dtype``.

    Probes a deep copy in eval mode, both mask cases, mirroring the guard's
    input construction. Returns an error dict, or the exception string when
    the probe cannot run (fp64 on MPS, for example).
    """
    try:
        probe = copy.deepcopy(module).to(dtype).eval()
        reference = next(probe.parameters())
        generator = torch.Generator(device=reference.device).manual_seed(seed)
        n = int(probe.window_size[0]) * int(probe.window_size[1])
        dim = int(probe.dim)

        def _randn(*shape: int) -> torch.Tensor:
            return torch.randn(
                *shape, generator=generator, device=reference.device, dtype=reference.dtype
            )

        results: dict[str, float] = {}
        with torch.no_grad():
            for label, batch, mask in (
                ("unmasked", 2, None),
                ("masked", 4, _randn(2, n, n)),
            ):
                x = _randn(batch, n, dim)
                stock = probe.forward(x, mask=mask)
                rewrite = sdpa_window_attention_forward(probe, x, mask=mask)
                delta = (stock - rewrite).abs()
                results[f"{label}_abs"] = float(delta.max())
                results[f"{label}_rel"] = float((delta / stock.abs().clamp_min(1e-12)).max())
        return results
    except Exception as exc:
        return repr(exc)


def gradient_errors(module, seed: int) -> dict[str, object] | str:
    """fp64 gradient parity: input gradient plus the worst parameter gradient."""
    try:
        stock = copy.deepcopy(module).double().eval()
        rewritten = copy.deepcopy(stock)
        rewritten.forward = types.MethodType(sdpa_window_attention_forward, rewritten)

        device = next(stock.parameters()).device
        generator = torch.Generator(device=device).manual_seed(seed)
        n = int(stock.window_size[0]) * int(stock.window_size[1])
        x = torch.randn(4, n, int(stock.dim), generator=generator, device=device, dtype=torch.float64)
        mask = torch.randn(2, n, n, generator=generator, device=device, dtype=torch.float64)

        stock_x = x.clone().requires_grad_(True)
        rewritten_x = x.clone().requires_grad_(True)
        stock(stock_x, mask=mask).square().sum().backward()
        rewritten(rewritten_x, mask=mask).square().sum().backward()

        worst_name, worst_err = "-", 0.0
        for (name, stock_param), (_, rewritten_param) in zip(
            stock.named_parameters(), rewritten.named_parameters()
        ):
            err = float((stock_param.grad - rewritten_param.grad).abs().max())
            if err > worst_err:
                worst_name, worst_err = name, err
        return {
            "input_grad_abs": float((stock_x.grad - rewritten_x.grad).abs().max()),
            "worst_param": worst_name,
            "worst_param_abs": worst_err,
        }
    except Exception as exc:
        return repr(exc)


def with_tf32_disabled(fn, *args, **kwargs):
    """Run ``fn`` with TF32 and reduced-precision fp32 matmul turned off."""
    saved = (
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
        torch.get_float32_matmul_precision(),
    )
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        return fn(*args, **kwargs)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = saved[0]
        torch.backends.cudnn.allow_tf32 = saved[1]
        torch.set_float32_matmul_precision(saved[2])


def format_errors(result: dict[str, float] | str) -> str:
    if isinstance(result, str):
        return f"EXCEPTION {result}"
    return (
        f"abs={result['unmasked_abs']:.3e}/{result['masked_abs']:.3e} "
        f"rel={result['unmasked_rel']:.3e}/{result['masked_rel']:.3e}"
    )


def main() -> None:
    import math

    import timm
    from timm.models.swin_transformer_v2 import SwinTransformerV2Block, WindowAttention

    args = parse_args()
    device = select_device(args.device)
    print(environment_header(device))
    print(f"guard gate: fp64 atol={PARITY_ATOL:g} rtol={PARITY_RTOL:g}\n")

    model = timm.create_model(args.backbone, pretrained=False, num_classes=0).to(device)

    shift_by_attention_id = {}
    for block in model.modules():
        if isinstance(block, SwinTransformerV2Block):
            shift = tuple(int(s) for s in getattr(block, "shift_size", (0, 0)))
            shift_by_attention_id[id(block.attn)] = any(s > 0 for s in shift)

    refused = 0
    for name, module in model.named_modules():
        if not isinstance(module, WindowAttention):
            continue
        n = int(module.window_size[0]) * int(module.window_size[1])
        heads = int(module.num_heads)
        shifted = shift_by_attention_id.get(id(module), False)
        raw_scale = module.logit_scale.detach()
        clamped = raw_scale.clamp(max=math.log(1.0 / 0.01)).exp()

        verdict_ok, verdict_detail = _parity_check(module)
        refused += 0 if verdict_ok else 1

        print(f"{name}")
        print(
            f"  block={'shifted' if shifted else 'non-shifted'} dim={int(module.dim)} "
            f"heads={heads} window={tuple(module.window_size)} N={n}"
        )
        print(
            f"  logit_scale raw=[{float(raw_scale.min()):.4f}, {float(raw_scale.max()):.4f}] "
            f"clamped.exp=[{float(clamped.min()):.4f}, {float(clamped.max()):.4f}]"
        )
        print(
            f"  rel_pos_bias=[{heads}, {n}, {n}] probe_mask=[2, {n}, {n}] "
            f"sdpa attn_mask=[1, {heads}, {n}, {n}] unmasked / [1, {2 * heads}, {n}, {n}] masked"
        )
        print(
            f"  q/k per path: stock [B_, {heads}, {n}, {int(module.dim) // heads}] "
            f"(strides from qkv permute); rewrite same, masked case merged to "
            f"[B, {2 * heads}, {n}, {int(module.dim) // heads}] contiguous"
        )
        print(f"  fp32 (flags as-is):  {format_errors(forward_errors(module, torch.float32, args.seed))}")
        print(
            "  fp32 (TF32 off):     "
            f"{format_errors(with_tf32_disabled(forward_errors, module, torch.float32, args.seed))}"
        )
        print(f"  fp64:                {format_errors(forward_errors(module, torch.float64, args.seed))}")
        gradients = gradient_errors(module, args.seed)
        if isinstance(gradients, str):
            print(f"  fp64 gradients:      EXCEPTION {gradients}")
        else:
            print(
                f"  fp64 gradients:      input={gradients['input_grad_abs']:.3e} "
                f"worst_param={gradients['worst_param']}={gradients['worst_param_abs']:.3e}"
            )
        print(f"  guard verdict:       {'CONVERT' if verdict_ok else 'REFUSE'} ({verdict_detail})")
        print()

    total = len(shift_by_attention_id)
    print(f"summary: {total - refused} of {total} attention modules pass the fp64 guard.")


if __name__ == "__main__":
    main()
