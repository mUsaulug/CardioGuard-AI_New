"""Compare CNN and XGBoost test metrics and persist a combined report."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

try:
    import pandas as pd
except ImportError:  # pragma: no cover - optional dependency
    pd = None


METRIC_COLUMNS = ["roc_auc", "pr_auc", "f1", "accuracy", "loss"]


def load_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def extract_metrics(record: Dict[str, object]) -> Dict[str, object]:
    metrics = record.get("test")
    if metrics is None or not isinstance(metrics, dict):
        raise ValueError("Missing 'test' metrics in JSON payload.")
    return metrics


def build_row(name: str, metrics: Dict[str, object]) -> Dict[str, object]:
    row = {"model": name}
    for column in METRIC_COLUMNS:
        row[column] = metrics.get(column)
    return row


def format_table(rows: List[Dict[str, object]]) -> str:
    if pd is not None:
        frame = pd.DataFrame(rows)
        return frame.to_string(index=False)

    headers = ["model", *METRIC_COLUMNS]
    lines = [",".join(headers)]
    for row in rows:
        values = ["" if row.get(key) is None else str(row.get(key)) for key in headers]
        lines.append(",".join(values))
    return "\n".join(lines)


def write_outputs(rows: List[Dict[str, object]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "compare_metrics.csv"
    json_path = output_dir / "compare_metrics.json"

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", *METRIC_COLUMNS])
        writer.writeheader()
        writer.writerows(rows)

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare CNN and XGBoost test metrics")
    parser.add_argument(
        "--cnn-metrics",
        type=Path,
        default=Path("logs/metrics.json"),
        help="Path to CNN metrics.json file",
    )
    parser.add_argument(
        "--xgb-metrics",
        type=Path,
        default=Path("logs/xgb/metrics.json"),
        help="Path to XGBoost metrics.json file",
    )
    args = parser.parse_args()

    cnn_payload = load_json(args.cnn_metrics)
    xgb_payload = load_json(args.xgb_metrics)

    cnn_metrics = extract_metrics(cnn_payload)
    xgb_metrics = extract_metrics(xgb_payload)

    rows = [build_row("cnn", cnn_metrics), build_row("xgb", xgb_metrics)]

    print(format_table(rows))
    write_outputs(rows, Path("logs"))


if __name__ == "__main__":
    main()
