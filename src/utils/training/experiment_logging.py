import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class JsonlLogHandler(logging.Handler):
    """Writes structured log records for post-run parsing and dashboards."""

    def __init__(self, path: Path):
        super().__init__()
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("a", encoding="utf-8")

    def emit(self, record: logging.LogRecord) -> None:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            formatter = self.formatter or logging.Formatter()
            payload["exception"] = formatter.formatException(record.exc_info)
        for key, value in record.__dict__.items():
            if key.startswith("_") or key in {
                "args",
                "created",
                "exc_info",
                "exc_text",
                "filename",
                "funcName",
                "levelname",
                "levelno",
                "lineno",
                "module",
                "msecs",
                "msg",
                "name",
                "pathname",
                "process",
                "processName",
                "relativeCreated",
                "stack_info",
                "thread",
                "threadName",
            }:
                continue
            try:
                json.dumps(value)
                payload[key] = value
            except TypeError:
                payload[key] = str(value)
        self._file.write(json.dumps(payload, sort_keys=True) + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()
        super().close()


def setup_experiment_logger(
    log_dir: str | Path,
    name: str = "seed_moe",
    level: str = "INFO",
    console: bool = True,
    structured_jsonl: bool = True,
) -> logging.Logger:
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_path / "training.log", mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    if structured_jsonl:
        jsonl_handler = JsonlLogHandler(log_path / "training.log.jsonl")
        logger.addHandler(jsonl_handler)

    return logger
