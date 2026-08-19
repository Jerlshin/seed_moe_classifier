"""Wide-format CSV metric sinks, for analysis that does not want an event parser.

Why this exists next to ``events.jsonl``
----------------------------------------

``events.jsonl`` is the complete record and stays that way: it carries the
non-scalar events (corpus digests, shapes, budgets, checkpoint paths) that a
table cannot hold. What it is *not* is directly loadable. Recovering "the epoch
loss curve" from it means streaming ~4 MB of JSON, filtering on
``type == "metrics"``, and re-keying by prefix -- which is why
``src/utils/evaluation.py`` has a 60-line parser for exactly that job, and why
every downstream consumer has to import it.

These sinks write the same numbers a second time as CSV, one file per metric
family, in **wide** format: one row per ``(step)``, one column per metric. That
is the shape ``pandas.read_csv`` and every plotting tool expect, and it is the
shape an automated analysis can consume without knowing anything about this
repository.

The column set is discovered, not declared
------------------------------------------

Which metrics exist depends on the run: ``grad_scale`` only under fp16,
``gpu_busy_fraction`` only when the run opted in, ``aux_*`` only with an
auxiliary head, and the collapse diagnostics only on logging steps. Declaring a
header up front would therefore either drop metrics or bake in a schema that
each new arm invalidates.

So the sink buffers rows and rewrites the whole file whenever a new column
appears, and appends when it does not. A stage-1 run logs on the order of
10^3-10^4 rows, so the rewrite is milliseconds and happens a handful of times --
the first few hundred steps, then never again. What it buys is that the file on
disk is always complete and always rectangular, including after a crash, which
is the property that makes it worth writing at all.

Missing values are written as empty fields rather than ``nan``, so
``pandas.read_csv`` types the column as float with ``NaN`` and a plain
``csv.DictReader`` sees ``""``. Both are unambiguous; ``nan`` as a literal is
not, because a metric can genuinely *be* NaN and that is a different fact from
"this metric was not logged at this step".
"""

from __future__ import annotations

import csv
import logging
import math
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

#: Written as the first column of every file, so rows from different families
#: can be concatenated and still ordered.
INDEX_COLUMN = "step"

#: Wall-clock seconds since the sink was created. Present in every row because
#: "when did this happen" is the one question the step index cannot answer, and
#: joining against the log file's timestamps is worse than carrying it.
ELAPSED_COLUMN = "elapsed_seconds"


def _format(value: Any) -> str:
    """One CSV cell. Empty for absent, ``nan``/``inf`` preserved literally."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        # `repr` round-trips a float exactly and is shorter than `%.17g` for the
        # values that do not need 17 digits, which is nearly all of them.
        return repr(value)
    return str(value)


class CsvMetricSink:
    """One wide CSV file for one metric family.

    Args:
        path: Destination file. Parent directories are created.
        index_column: Name of the leading column, normally the step or epoch.
        elapsed_start: ``time.perf_counter()`` reading the elapsed column is
            measured against. ``None`` omits the column.
    """

    def __init__(
        self,
        path: str | Path,
        index_column: str = INDEX_COLUMN,
        elapsed_start: float | None = None,
    ):
        self.path = Path(path)
        self.index_column = str(index_column)
        self.elapsed_start = elapsed_start
        self.columns: list[str] = [self.index_column]
        if elapsed_start is not None:
            self.columns.append(ELAPSED_COLUMN)
        self.rows: list[dict[str, Any]] = []
        self._written = 0
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, index: Any, metrics: Mapping[str, Any]) -> None:
        """Add one row. New metric names extend the schema."""
        row: dict[str, Any] = {self.index_column: index}
        if self.elapsed_start is not None:
            row[ELAPSED_COLUMN] = time.perf_counter() - self.elapsed_start

        new_columns = False
        for key, value in metrics.items():
            column = str(key)
            if column in {self.index_column, ELAPSED_COLUMN}:
                continue
            if column not in self.columns:
                self.columns.append(column)
                new_columns = True
            row[column] = value

        self.rows.append(row)
        if new_columns:
            # The schema changed, so every row already on disk is now missing a
            # field. Rewriting is the only way to keep the file rectangular.
            self.flush(rewrite=True)
        else:
            self.flush()

    def flush(self, rewrite: bool = False) -> None:
        """Write pending rows, or the whole file when the schema changed."""
        if rewrite:
            self._written = 0
        pending = self.rows[self._written :]
        if not pending and not rewrite:
            return
        mode = "w" if rewrite else "a"
        with self.path.open(mode, newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            if rewrite:
                writer.writerow(self.columns)
            for row in pending:
                writer.writerow([_format(row.get(column)) for column in self.columns])
        self._written = len(self.rows)

    def close(self) -> None:
        self.flush()


class CsvMetricWriter:
    """Routes prefixed metrics to one :class:`CsvMetricSink` per family.

    ``log_metrics(..., prefix="train")`` lands in ``metrics_train.csv``,
    ``prefix="epoch"`` in ``metrics_epoch.csv``, and unprefixed metrics in
    ``metrics.csv``. The split is by prefix rather than by one wide file because
    the families have genuinely different row cadences -- per logging step, per
    epoch, per probe -- and interleaving them would leave a file that is 90 %
    empty cells and cannot be read as a time series without a filter.

    Args:
        directory: Where the CSV files are written.
        enabled: ``False`` makes every method a no-op and opens no files.
        elapsed_start: Passed through to each sink; see :class:`CsvMetricSink`.
        index_columns: Per-prefix override of the leading column's name, so the
            epoch file says ``epoch`` rather than ``step``.
    """

    #: Prefixes whose leading column is an epoch index rather than a global step.
    DEFAULT_INDEX_COLUMNS = {"epoch": "epoch", "probe": "epoch", "milestone": "epoch"}

    def __init__(
        self,
        directory: str | Path,
        enabled: bool = True,
        elapsed_start: float | None = None,
        index_columns: Mapping[str, str] | None = None,
        logger: logging.Logger | None = None,
    ):
        self.directory = Path(directory)
        self.enabled = bool(enabled)
        self.elapsed_start = elapsed_start
        self.logger = logger or logging.getLogger(__name__)
        self.index_columns = {**self.DEFAULT_INDEX_COLUMNS, **dict(index_columns or {})}
        self.sinks: dict[str, CsvMetricSink] = {}
        if self.enabled:
            self.directory.mkdir(parents=True, exist_ok=True)

    def _sink(self, family: str) -> CsvMetricSink:
        if family not in self.sinks:
            name = f"metrics_{family}.csv" if family else "metrics.csv"
            self.sinks[family] = CsvMetricSink(
                self.directory / name,
                index_column=self.index_columns.get(family, INDEX_COLUMN),
                elapsed_start=self.elapsed_start,
            )
        return self.sinks[family]

    def log(self, metrics: Mapping[str, Any], step: Any, prefix: str | None = None) -> None:
        """Append one row to the family ``prefix`` names."""
        if not self.enabled or not metrics:
            return
        try:
            self._sink(prefix or "").append(step, metrics)
        except Exception as exc:  # pragma: no cover - a CSV must never kill a run
            self.logger.warning("Unable to write CSV metrics for %r: %s", prefix, exc)

    def write_table(
        self,
        name: str,
        rows: Sequence[Mapping[str, Any]],
        columns: Sequence[str] | None = None,
    ) -> str | None:
        """Write a standalone table (per-view geometry, checkpoint selection).

        Unlike the metric sinks this is written once, whole, so the column order
        can be declared. Returns the path, or ``None`` when disabled or empty.
        """
        if not self.enabled or not rows:
            return None
        ordered: list[str] = list(columns) if columns else []
        if not ordered:
            for row in rows:
                for key in row:
                    if key not in ordered:
                        ordered.append(str(key))
        path = self.directory / (name if name.endswith(".csv") else f"{name}.csv")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(ordered)
                for row in rows:
                    writer.writerow([_format(row.get(column)) for column in ordered])
        except Exception as exc:  # pragma: no cover
            self.logger.warning("Unable to write CSV table %s: %s", name, exc)
            return None
        return str(path)

    def close(self) -> None:
        for sink in self.sinks.values():
            try:
                sink.close()
            except Exception as exc:  # pragma: no cover
                self.logger.warning("Unable to flush %s: %s", sink.path, exc)

    @property
    def paths(self) -> dict[str, str]:
        """``{family: path}`` for every file this writer has opened."""
        return {family: str(sink.path) for family, sink in self.sinks.items()}
