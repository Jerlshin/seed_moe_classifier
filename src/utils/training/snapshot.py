from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


def _redact_environment(env: dict[str, str], patterns: list[str]) -> dict[str, str]:
    redacted: dict[str, str] = {}
    upper_patterns = [pattern.upper() for pattern in patterns]
    for key, value in env.items():
        if any(pattern in key.upper() for pattern in upper_patterns):
            redacted[key] = "<redacted>"
        else:
            redacted[key] = value
    return redacted


def snapshot_run_configuration(cfg: DictConfig, output_dir: str | Path) -> dict[str, Path]:
    snapshot_dir = Path(output_dir) / "snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    snapshot_cfg = OmegaConf.select(cfg, "tracking.snapshot", default={})
    cfg_container: dict[str, Any] = OmegaConf.to_container(cfg, resolve=True)  # type: ignore[assignment]
    written: dict[str, Path] = {}

    if snapshot_cfg.get("save_yaml", True):
        path = snapshot_dir / "config.yaml"
        path.write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")
        written["config_yaml"] = path

    if snapshot_cfg.get("save_json", True):
        path = snapshot_dir / "config.json"
        path.write_text(json.dumps(cfg_container, indent=2, sort_keys=True), encoding="utf-8")
        written["config_json"] = path

    if snapshot_cfg.get("save_cli_args", True):
        path = snapshot_dir / "cli_args.json"
        path.write_text(json.dumps(sys.argv, indent=2), encoding="utf-8")
        written["cli_args"] = path

    if snapshot_cfg.get("save_environment", True):
        redact_keys = list(snapshot_cfg.get("redact_env_keys", []))
        path = snapshot_dir / "environment.json"
        environment = _redact_environment(dict(os.environ), redact_keys)
        path.write_text(json.dumps(environment, indent=2, sort_keys=True), encoding="utf-8")
        written["environment"] = path

    return written
