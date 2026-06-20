from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd


DEFAULT_METRICS = [
    "metrics/precision(B)",
    "metrics/recall(B)",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
]


def final_metrics_from_results(results_path: Path) -> dict[str, str | float]:
    results = pd.read_csv(results_path)
    results.columns = [column.strip() for column in results.columns]
    if results.empty:
        raise ValueError(f"No rows found in {results_path}")

    final_row = results.dropna(how="all").iloc[-1]
    selection_metric = "metrics/mAP50-95(B)"
    if selection_metric in results.columns:
        best_index = results[selection_metric].astype(float).idxmax()
        best_row = results.loc[best_index]
    else:
        best_row = final_row

    row: dict[str, str | float] = {
        "run_name": results_path.parent.name,
        "results_path": str(results_path),
        "final_epoch": float(final_row["epoch"]) if "epoch" in results.columns else len(results) - 1,
        "best_epoch": float(best_row["epoch"]) if "epoch" in results.columns else float(best_row.name),
    }

    for metric in DEFAULT_METRICS:
        if metric in results.columns:
            row[f"final_{metric}"] = float(final_row[metric])
            row[f"best_{metric}"] = float(best_row[metric])

    return row


def summarize_metric_rows(rows: list[dict[str, str | float]]) -> list[dict[str, str | float]]:
    metric_names = []
    for prefix in ("best", "final"):
        for metric in DEFAULT_METRICS:
            metric_name = f"{prefix}_{metric}"
            if any(metric_name in row for row in rows):
                metric_names.append(metric_name)
    summary_rows: list[dict[str, str | float]] = []
    for metric in metric_names:
        values = pd.Series([float(row[metric]) for row in rows if metric in row])
        summary_rows.append(
            {
                "metric": metric,
                "folds": int(values.count()),
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if values.count() > 1 else 0.0,
                "min": float(values.min()),
                "max": float(values.max()),
            }
        )
    return summary_rows


def write_csv(path: Path, rows: list[dict[str, str | float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_yolo_cv_results(results_glob: str, output_dir: Path) -> tuple[Path, Path]:
    result_paths = sorted(Path().glob(results_glob))
    if not result_paths:
        raise FileNotFoundError(f"No YOLO results.csv files matched: {results_glob}")

    per_fold_rows = [final_metrics_from_results(path) for path in result_paths]
    summary_rows = summarize_metric_rows(per_fold_rows)

    per_fold_path = output_dir / "yolo_cv_per_fold_metrics.csv"
    summary_path = output_dir / "yolo_cv_summary_metrics.csv"
    write_csv(per_fold_path, per_fold_rows)
    write_csv(summary_path, summary_rows)
    return per_fold_path, summary_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarise YOLO k-fold training metrics from results.csv files."
    )
    parser.add_argument(
        "--results-glob",
        default="runs/detect/cv_stratified_fold*/results.csv",
        help="Glob matching YOLO results.csv files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/yolo_cv_summary"),
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    per_fold_path, summary_path = summarize_yolo_cv_results(
        results_glob=args.results_glob,
        output_dir=args.output_dir,
    )
    print(f"Per-fold metrics: {per_fold_path}")
    print(f"Summary metrics: {summary_path}")


if __name__ == "__main__":
    main()
