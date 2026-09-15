from greenwaste.one_vs_rest_ensemble import (
    EnsembleDetection,
    box_iou,
    non_max_suppression,
    parse_key_value_float,
    parse_key_value_path,
)


def test_box_iou() -> None:
    assert box_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert round(box_iou((0, 0, 10, 10), (5, 5, 15, 15)), 3) == 0.143


def test_non_max_suppression_keeps_highest_overlapping_detection() -> None:
    detections = [
        EnsembleDetection("sofa", 0.8, (0, 0, 10, 10), "sofa.pt"),
        EnsembleDetection("chair_seating", 0.6, (1, 1, 11, 11), "chair.pt"),
        EnsembleDetection("storage", 0.7, (50, 50, 80, 80), "storage.pt"),
    ]

    kept = non_max_suppression(detections, iou_threshold=0.5)

    assert [detection.item_class for detection in kept] == ["sofa", "storage"]


def test_parse_key_value_float() -> None:
    assert parse_key_value_float(["sofa=0.4", "chair_seating=0.25"]) == {
        "sofa": 0.4,
        "chair_seating": 0.25,
    }


def test_parse_key_value_path_resolves_nested_ultralytics_runs(tmp_path, monkeypatch) -> None:
    nested = tmp_path / "runs" / "detect" / "runs" / "detect" / "model" / "weights"
    nested.mkdir(parents=True)
    weights = nested / "best.pt"
    weights.write_bytes(b"weights")
    monkeypatch.chdir(tmp_path)

    parsed = parse_key_value_path(["sofa=runs/detect/model/weights/best.pt"])

    assert parsed["sofa"] == weights
