"""Compare CNN and XGBoost test metrics side-by-side."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


METRIC_ORDER = ["roc_auc", "pr_auc", "f1", "accuracy", "loss"]


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Metrics file not found: {path}. "
            "Run the corresponding training script or pass the correct path."
        )
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def extract_test_metrics(payload: Dict[str, Any], label: str) -> Dict[str, Any]:
    if "test" in payload:
        metrics = payload["test"]
        if not isinstance(metrics, dict):
            raise ValueError(f"Invalid test metrics format for {label}.")
        return metrics
    if "metrics" in payload:
        metrics = payload["metrics"]
        if not isinstance(metrics, dict):
            raise ValueError(f"Invalid metrics format for {label}.")
        return metrics
    raise KeyError(
        f"Unable to find test metrics for {label}. Expected a 'test' key in {label} metrics JSON."
    )


def build_row(model_name: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
    row: Dict[str, Any] = {"model": model_name}
    for key in METRIC_ORDER:
        value = metrics.get(key)
        row[key] = value
    return row


def write_outputs(rows: Iterable[Dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "compare_metrics.json"
    csv_path = output_dir / "compare_metrics.csv"

    rows_list: List[Dict[str, Any]] = list(rows)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(rows_list, handle, indent=2)

    fieldnames = ["model", *METRIC_ORDER]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_list)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare CNN and XGBoost test metrics")
    parser.add_argument(
        "--cnn-metrics",
        type=Path,
        default=Path("logs/metrics.json"),
        help="Path to CNN metrics.json (default: logs/metrics.json)",
    )
    parser.add_argument(
        "--xgb-metrics",
        type=Path,
        default=Path("logs/xgb/metrics.json"),
        help="Path to XGBoost metrics.json (default: logs/xgb/metrics.json)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("logs"),
        help="Directory for compare_metrics.json/csv (default: logs)",
    )
    args = parser.parse_args()

    cnn_payload = load_json(args.cnn_metrics)
    xgb_payload = load_json(args.xgb_metrics)

    cnn_metrics = extract_test_metrics(cnn_payload, "CNN")
    xgb_metrics = extract_test_metrics(xgb_payload, "XGBoost")

    rows = [
        build_row("cnn", cnn_metrics),
        build_row("xgb", xgb_metrics),
    ]
    write_outputs(rows, args.output_dir)

    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()