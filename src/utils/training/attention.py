from __future__ import annotations

import logging

import torch


def _capture_timm_last_attention(backbone: torch.nn.Module, x: torch.Tensor) -> torch.Tensor | None:
    blocks = getattr(backbone, "blocks", None)
    if not blocks:
        return None
    last_block = blocks[-1]
    attention_module = getattr(last_block, "attn", None)
    if attention_module is None or not hasattr(attention_module, "attn_drop"):
        return None

    previous_fused = getattr(attention_module, "fused_attn", None)
    if previous_fused is not None:
        attention_module.fused_attn = False

    captured: list[torch.Tensor] = []

    def hook(_module, _inputs, output):
        captured.append(output.detach())

    handle = attention_module.attn_drop.register_forward_hook(hook)
    try:
        backbone(x)
    finally:
        handle.remove()
        if previous_fused is not None:
            attention_module.fused_attn = previous_fused

    return captured[0] if captured else None


def log_attention_maps(
    tracker,
    backbone: torch.nn.Module,
    images: torch.Tensor,
    step: int,
    logger: logging.Logger | None = None,
    max_images: int = 4,
) -> None:
    if not tracker.enabled_for_images:
        return

    with torch.no_grad():
        sample = images[:max_images]
        attention = None
        if hasattr(backbone, "get_last_selfattention"):
            try:
                attention = backbone.get_last_selfattention(sample)
            except Exception:
                attention = None
        if attention is None:
            try:
                attention = _capture_timm_last_attention(backbone, sample)
            except Exception as exc:
                if logger:
                    logger.warning("Unable to extract attention maps: %s", exc)
                return

        if attention is None or attention.ndim != 4:
            if logger:
                logger.warning("Attention maps are not available for this backbone.")
            return

        cls_attention = attention[:, :, 0, 1:].mean(dim=1)
        tokens = cls_attention.shape[-1]
        side = int(tokens**0.5)
        if side * side != tokens:
            if logger:
                logger.warning("Cannot reshape attention tokens into a square map: %s", tokens)
            return

        maps = cls_attention.reshape(cls_attention.shape[0], 1, side, side)
        maps = torch.nn.functional.interpolate(
            maps,
            size=sample.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        maps = maps.detach().cpu()
        maps = (maps - maps.amin(dim=(-2, -1), keepdim=True)) / (
            maps.amax(dim=(-2, -1), keepdim=True) - maps.amin(dim=(-2, -1), keepdim=True) + 1e-8
        )
        tracker.log_images("attention/last_block_cls", maps, step)
