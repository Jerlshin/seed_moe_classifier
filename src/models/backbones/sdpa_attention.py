"""SDPA rewrite of timm's SwinV2 window attention. Same function, fused kernels.

Why this exists
---------------

``enable_fused_attention`` in ``src/utils/training/device.py`` reports **zero**
switchable modules on a stock SwinV2 trunk, because timm offers no fused path
for it: SwinV2's window attention is *cosine* attention — it L2-normalises ``q``
and ``k``, multiplies by a clamped, learned, **per-head** logit scale, and adds a
continuous relative-position bias produced by a small MLP. None of that matches
``F.scaled_dot_product_attention``'s ``softmax(q kᵀ / sqrt(d)) v`` signature, so
timm runs the whole thing eagerly.

Eager is expensive in a way FLOP counts hide. Per block, autograd records and
**saves for backward** two full ``[B·nW, heads, N, N]`` matrices:

* the pre-scale ``q̂ k̂ᵀ`` product — kept because the per-head ``logit_scale``
  multiply needs it for its own gradient;
* the softmax output — kept for both the softmax backward and the ``attn @ v``
  backward.

For ``swinv2_base_window16_256`` at the stage-1 student batch of 96 views
(``16 images x 6 crops``), summing ``B·nW·heads·N²`` over the 24 blocks gives
**~12 GB of bf16 activations per micro-batch** held from forward until
backward — the single largest consumer of both memory capacity and memory
bandwidth in the step, and the reason the physical batch cannot be raised.
``torch.compile`` does not remove it: inductor's SDPA pattern-matcher does not
recognise the cosine-attention chain (measured — zero ``fuse_attention``
rewrites on this module), and its pointwise fusion cannot eliminate tensors
that autograd has to keep.

The rewrite
-----------

The function *is* expressible with SDPA once two identities are applied:

1. The per-head scale commutes into ``q``:
   ``(q̂ k̂ᵀ) * s = (q̂ * s) k̂ᵀ`` for a scale broadcast as ``[heads, 1, 1]``.
2. The relative-position bias (and, in shifted blocks, the window mask) is an
   additive pre-softmax term, which is exactly SDPA's ``attn_mask``.

so

    softmax((q̂ s) k̂ᵀ + bias) v  ==  SDPA(q̂ s, k̂, v, attn_mask=bias, scale=1.0)

``scale=1.0`` is load-bearing: SDPA's default silently divides by
``sqrt(head_dim)``, which would apply the temperature twice. The shifted-window
mask is folded into the bias per *window*, and the window axis is merged into
the head axis (``[B·nW, H, N, d] -> [B, nW·H, N, d]``) so the mask stays
``[1, nW·H, N, N]`` and broadcasts over the batch instead of being tiled into a
``[B·nW, H, N, N]`` tensor.

This is the same arithmetic with a different reduction order — identical in
exact arithmetic, and ``tests/test_throughput.py`` pins it in fp64 at 1e-12
relative, values *and* gradients. The fused kernel keeps a per-row logsumexp
instead of the ``N²`` matrices, so the ~12 GB above simply stops existing, and
the softmax is computed in fp32 internally even under bf16 autocast.

Why every module is parity-checked at conversion time
-----------------------------------------------------

The rewrite reimplements the forward from the module's *attributes*, so it is
correct for the timm this repository pins and silently wrong for a future timm
that changes the semantics (a different bias transform, a moved clamp).
``convert_swinv2_attention_to_sdpa`` therefore runs each candidate module's own
stock forward against the rewrite on random inputs — with and without a shift
mask — and **refuses to convert a module that does not agree**, leaving it on
the eager path with a warning. A converted trunk is one whose every attention
module has just proven, on this machine and this timm, that the two paths
agree; a drifted timm degrades to current performance, not to a wrong model.

Why the probe runs in fp64, on a copy
-------------------------------------

The probe is a check of **algebra**, so it must not be contaminated by kernel
precision selection — and in the module's own dtype on a CUDA device, it is.
The trainer enables TF32 before the model is built, and cuBLAS then chooses
TF32 tensor-core kernels for the *stock* path's eager matmuls shape-by-shape,
while the SDPA path dispatches to a different backend entirely. Measured across
the four SwinV2-Base stage shapes: the two paths agree to ~4e-7 when both run
exact fp32, but a TF32-contaminated stock path lands ~5e-4 from the rewrite —
past any gate tight enough to catch real drift. Which stages trip it depends on
cuBLAS heuristics; on a 12 GB Ada card that was every ``heads >= 16`` module
(``layers.2`` and ``layers.3``, 40 of 48), refused for kernel rounding that the
run's own bf16 autocast dwarfs anyway.

The probe therefore runs on an **fp64 deep copy** of the module (device-local;
CPU when the accelerator cannot do fp64, e.g. MPS). fp64 has no TF32 path and
no fused SDPA backend — the math fallback is the same composition in fp64 — so
both sides are pure algebra: measured agreement ~1e-15, semantic drift O(1),
and the gate sits at 1e-10, six orders of magnitude *stricter* than the old
fp32 gate. The live training module is never cast, never re-moded and never
mutated; on refusal the warning now carries the measured error or the
exception, so a refusal is diagnosable from the log alone
(``scripts/diagnose_sdpa_parity.py`` produces the full per-module report).

Modules with ``attn_drop.p > 0`` are also refused: SDPA's internal dropout
matches the stock path in distribution but not in RNG draws, and this
repository does not use attention dropout, so refusing keeps the conversion
claim exact rather than exact-in-expectation.

State dict, EMA and ``torch.compile`` are all unaffected: conversion binds a
new ``forward`` onto each *instance* and touches no parameters, no buffers and
no module tree — the same reasoning that keeps compiled callables off the
module tree in :class:`~src.models.backbones.swinv2_dino.DINO`. Conversion must
run **before** ``torch.compile`` so the compiled graph traces the SDPA path.
"""

from __future__ import annotations

import copy
import logging
import math
import types
from typing import Optional

import torch
import torch.nn.functional as F

LOGGER = logging.getLogger(__name__)

#: Gate for the conversion-time parity probe, which runs in **fp64 on a copy**
#: of the module (see the module docstring). In fp64 both paths are the same
#: algebra to ~1e-15 measured, and semantic drift shows up at O(1) (a missing
#: 16x bias gain, a doubled temperature, ...), so 1e-10 separates the two by
#: five orders of magnitude on either side. The probe must NOT run in the
#: module's own dtype: an fp32 probe on a TF32-capable card measures cuBLAS
#: kernel selection (~5e-4 cross-path), not algebra, and refuses correct
#: modules stage-by-stage.
PARITY_ATOL = 1e-10
PARITY_RTOL = 1e-10


def sdpa_window_attention_forward(
    self,
    x: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Drop-in replacement for timm SwinV2 ``WindowAttention.forward``.

    Args:
        x: ``[num_windows * B, N, C]`` window tokens, exactly as the stock
            forward receives them.
        mask: ``[num_windows, N, N]`` additive shift mask for shifted blocks,
            or ``None``.
    """
    B_, N, C = x.shape

    # qkv projection — verbatim from the stock forward (k has a fixed zero
    # bias; only q and v carry learned biases).
    if self.q_bias is None:
        qkv = self.qkv(x)
    else:
        qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias))
        if self.qkv_bias_separate:
            qkv = self.qkv(x)
            qkv += qkv_bias
        else:
            qkv = F.linear(x, weight=self.qkv.weight, bias=qkv_bias)
    qkv = qkv.reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)

    # Cosine attention, refactored: (q̂ k̂ᵀ) * s == (q̂ s) k̂ᵀ, so the clamped
    # per-head scale rides into q and SDPA's own scaling is disabled below.
    logit_scale = torch.clamp(self.logit_scale, max=math.log(1.0 / 0.01)).exp()
    q = F.normalize(q, dim=-1) * logit_scale
    k = F.normalize(k, dim=-1)

    # Continuous relative-position bias — verbatim from the stock forward.
    relative_position_bias_table = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads)
    relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(
        self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1
    )
    relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # [H, N, N]
    relative_position_bias = 16 * torch.sigmoid(relative_position_bias)

    if mask is None:
        # [1, H, N, N], broadcast over the batch by SDPA.
        attn_mask = relative_position_bias.unsqueeze(0).to(q.dtype)
    else:
        # Shifted block. The stock forward views the batch as [B, nW, ...] with
        # the window index minor, so row i of B_ carries window i % nW. Folding
        # the window axis into the head axis preserves exactly that pairing and
        # keeps the mask at [1, nW·H, N, N] — broadcast over B — instead of a
        # [B_, H, N, N] tile.
        num_win = mask.shape[0]
        batch = B_ // num_win
        attn_mask = (relative_position_bias.unsqueeze(0) + mask.unsqueeze(1)).to(q.dtype)
        attn_mask = attn_mask.reshape(1, num_win * self.num_heads, N, N)
        q = q.reshape(batch, num_win * self.num_heads, N, -1)
        k = k.reshape(batch, num_win * self.num_heads, N, -1)
        v = v.reshape(batch, num_win * self.num_heads, N, -1)

    # scale=1.0: the temperature is already folded into q. SDPA's default of
    # 1/sqrt(head_dim) would apply a second, wrong scaling.
    x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, scale=1.0)

    if mask is not None:
        x = x.reshape(B_, self.num_heads, N, -1)
    x = x.transpose(1, 2).reshape(B_, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x


def _probe_parity(probe) -> tuple[bool, str]:
    """Stock forward vs. SDPA rewrite on an fp64 probe copy, both mask cases.

    Returns ``(ok, detail)`` where ``detail`` carries the measured maximum
    absolute errors, so a refusal in a training log states *how far apart* the
    paths were rather than only that they disagreed.
    """
    with torch.no_grad():
        n_tokens = int(probe.window_size[0]) * int(probe.window_size[1])
        dim = int(probe.dim)
        # A device-local generator keeps the probe off the global RNG stream,
        # so enabling the conversion cannot shift any randomness the run
        # consumes afterwards.
        reference = next(probe.parameters())
        generator = torch.Generator(device=reference.device).manual_seed(0)

        def _randn(*shape: int) -> torch.Tensor:
            return torch.randn(
                *shape, generator=generator, device=reference.device, dtype=reference.dtype
            )

        x = _randn(2, n_tokens, dim)
        stock = probe.forward(x)
        rewrite = sdpa_window_attention_forward(probe, x)
        plain_ok = torch.allclose(stock, rewrite, atol=PARITY_ATOL, rtol=PARITY_RTOL)
        plain_err = float((stock - rewrite).abs().max())

        x = _randn(4, n_tokens, dim)
        shift_mask = _randn(2, n_tokens, n_tokens)
        stock = probe.forward(x, mask=shift_mask)
        rewrite = sdpa_window_attention_forward(probe, x, mask=shift_mask)
        masked_ok = torch.allclose(stock, rewrite, atol=PARITY_ATOL, rtol=PARITY_RTOL)
        masked_err = float((stock - rewrite).abs().max())

        detail = (
            f"fp64 max|delta|: {plain_err:.3e} unmasked, {masked_err:.3e} masked "
            f"(gate {PARITY_ATOL:g})"
        )
        return plain_ok and masked_ok, detail


def _parity_check(module) -> tuple[bool, str]:
    """Semantic parity of the rewrite against this module's own stock forward.

    Probes an **fp64 deep copy** in eval mode under ``no_grad`` — never the live
    module, whose dtype, mode and parameters are left untouched. fp64 removes
    the TF32 and SDPA-backend kernel differences that made an in-dtype probe
    refuse correct modules on CUDA (see the module docstring), which is also
    what lets the gate sit at 1e-10 instead of 1e-4.

    Runs on the module's device; if that device cannot execute fp64 (MPS), the
    probe retries on CPU — the check is about algebra, and CPU is an equally
    valid venue for it. Any exception — an old torch without SDPA's ``scale``
    kwarg, an attribute a future timm renamed — counts as failure, and failure
    means the module keeps its stock forward.
    """
    try:
        probe = copy.deepcopy(module).double().eval()
    except Exception as exc:
        return False, f"could not build the fp64 probe copy: {exc!r}"
    try:
        return _probe_parity(probe)
    except Exception as device_exc:
        try:
            return _probe_parity(probe.cpu())
        except Exception as cpu_exc:
            return False, f"probe raised on device ({device_exc!r}) and on CPU ({cpu_exc!r})"


def convert_swinv2_attention_to_sdpa(model: torch.nn.Module, logger=None) -> int:
    """Rebind every SwinV2 ``WindowAttention`` in ``model`` to the SDPA forward.

    Returns the number of modules converted, which the caller should log the
    way ``enable_fused_attention``'s count is logged: a silent zero here means
    the run is on the eager path and the step-time budget is ~12 GB heavier.

    Modules that fail the parity check, or that use attention dropout, are left
    untouched — see the module docstring for why both refusals are deliberate.
    """
    log = logger or LOGGER
    try:
        from timm.models.swin_transformer_v2 import WindowAttention
    except ImportError:
        log.warning("timm.models.swin_transformer_v2 is unavailable; no SDPA conversion.")
        return 0

    converted = 0
    refused = 0
    for name, module in model.named_modules():
        if not isinstance(module, WindowAttention):
            continue
        # Instance-level forwards (an earlier conversion, someone else's patch)
        # are respected: the parity check below compares against whatever
        # `module.forward` currently is, so a re-run converts nothing twice.
        if getattr(module.forward, "__func__", None) is sdpa_window_attention_forward:
            continue
        if float(getattr(module.attn_drop, "p", 0.0)) > 0.0:
            log.warning(
                "Not converting %s to SDPA: attn_drop=%.3f would match the stock path "
                "in distribution but not in RNG draws.",
                name,
                float(module.attn_drop.p),
            )
            refused += 1
            continue
        parity_ok, parity_detail = _parity_check(module)
        if not parity_ok:
            log.warning(
                "Not converting %s to SDPA: its stock forward does not match the rewrite "
                "on this timm/torch install (%s). It stays on the eager path.",
                name,
                parity_detail,
            )
            refused += 1
            continue
        module.forward = types.MethodType(sdpa_window_attention_forward, module)
        converted += 1

    if refused:
        # A partial conversion is the expensive outcome: every refused module
        # keeps saving its N^2 attention matrices for backward, and one refused
        # 18-block stage is enough to change what batch size fits. Say so once,
        # loudly, next to the per-module reasons above.
        log.warning(
            "SDPA conversion: %s of %s candidate attention modules converted, %s refused. "
            "Refused modules stay on the eager path and keep their O(N^2) attention "
            "activations; run scripts/diagnose_sdpa_parity.py for the per-module report.",
            converted,
            converted + refused,
            refused,
        )
    return converted
