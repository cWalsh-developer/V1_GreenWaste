from pathlib import Path

from greenwaste.summarize_yolo_cv_results import summarize_yolo_cv_results


def write_results(path: Path, precision: float, recall: float, map50: float, map5095: float) -> None:
    path.parent.mkdir(parents=True)
    path.write_text(
        "epoch,metrics/precision(B),metrics/recall(B),metrics/mAP50(B),metrics/mAP50-95(B)\n"
        f"0,{precision / 2},{recall / 2},{map50 / 2},{map5095 / 2}\n"
        f"1,{precision},{recall},{map50},{map5095}\n"
    )


def test_summarize_yolo_cv_results_writes_per_fold_and_summary(tmp_path, monkeypatch):
    write_results(tmp_path / "runs/detect/cv_stratified_fold1/results.csv", 0.8, 0.7, 0.9, 0.6)
    write_results(tmp_path / "runs/detect/cv_stratified_fold2/results.csv", 1.0, 0.9, 0.7, 0.4)
    output_dir = tmp_path / "summary"
    monkeypatch.chdir(tmp_path)

    per_fold_path, summary_path = summarize_yolo_cv_results(
        results_glob="runs/detect/cv_stratified_fold*/results.csv",
        output_dir=output_dir,
    )

    per_fold_text = per_fold_path.read_text()
    summary_text = summary_path.read_text()

    assert "cv_stratified_fold1" in per_fold_text
    assert "metrics/mAP50(B)" in summary_text
    assert "0.8" in summary_text
    assert "2" in summary_text
