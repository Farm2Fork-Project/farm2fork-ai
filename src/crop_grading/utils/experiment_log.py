"""CSV experiment logging helpers."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def append_experiment_log(path: str | Path, row: dict[str, Any]) -> None:
    """Append one experiment row to a CSV, creating headers when needed."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    normalized_row = {key: _format_value(value) for key, value in row.items()}
    write_header = not output_path.exists()

    with output_path.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(normalized_row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(normalized_row)


def utc_timestamp() -> str:
    """Return a compact UTC timestamp for experiment logs."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _format_value(value: Any) -> Any:
    if isinstance(value, float):
        return f"{value:.6f}"
    if value is None:
        return ""
    return value
