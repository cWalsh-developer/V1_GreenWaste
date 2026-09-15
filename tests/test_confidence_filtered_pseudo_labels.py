from pathlib import Path

from greenwaste.confidence_filtered_pseudo_labels import (
    ModelVote,
    collect_candidates,
    evaluate_votes,
    filter_candidates_by_class,
    limit_candidates_per_class,
    median_box,
    xyxy_to_yolo,
)


def vote(name: str, confidence: float, margin: float, box=(10.0, 10.0, 90.0, 90.0)):
    return ModelVote("model.pt", name, confidence, confidence - margin, margin, box)


def test_consensus_accepts_expected_class_with_overlapping_boxes() -> None:
    votes = [
        vote("storage", 0.91, 0.40),
        vote("storage", 0.88, 0.30, (12.0, 12.0, 92.0, 92.0)),
        vote("storage", 0.86, 0.25, (8.0, 8.0, 88.0, 88.0)),
        vote("sofa", 0.81, 0.40),
        vote("storage", 0.70, 0.30),
    ]

    status, reason, box, ratio, agreement = evaluate_votes(
        votes, "storage", 100, 100, 0.8, 0.2, 3, 0.5, 0.05, 0.95
    )

    assert status == "accepted"
    assert reason == "accepted"
    assert box == (10.0, 10.0, 90.0, 90.0)
    assert ratio == 0.64
    assert agreement == 3


def test_consensus_rejects_low_margin_predictions() -> None:
    votes = [vote("storage", 0.9, 0.1) for _ in range(5)]

    status, reason, *_ = evaluate_votes(
        votes, "storage", 100, 100, 0.8, 0.2, 3, 0.5, 0.05, 0.95
    )

    assert status == "review"
    assert reason == "insufficient_high_confidence_agreement"


def test_consensus_rejects_class_disagreement() -> None:
    votes = [vote("sofa", 0.9, 0.4) for _ in range(5)]

    status, reason, *_ = evaluate_votes(
        votes, "storage", 100, 100, 0.8, 0.2, 3, 0.5, 0.05, 0.95
    )

    assert status == "review"
    assert reason == "teacher_class_disagreement"


def test_collect_candidates_ignores_review_other(tmp_path: Path) -> None:
    for folder in ("storage", "_review_other"):
        (tmp_path / folder).mkdir()
        (tmp_path / folder / "item.jpg").write_bytes(b"image")

    assert collect_candidates(tmp_path) == [(tmp_path / "storage" / "item.jpg", "storage")]


def test_limit_candidates_per_class_produces_balanced_pilot() -> None:
    candidates = [
        (Path(f"{item_class}_{index}.jpg"), item_class)
        for item_class in ("storage", "sofa")
        for index in range(3)
    ]

    selected = limit_candidates_per_class(candidates, 2)

    assert [item_class for _, item_class in selected] == [
        "sofa",
        "sofa",
        "storage",
        "storage",
    ]


def test_limit_candidates_per_class_random_sample_is_reproducible() -> None:
    candidates = [
        (Path(f"storage_{index}.jpg"), "storage") for index in range(10)
    ]

    first = limit_candidates_per_class(candidates, 3, sample_seed=7)
    second = limit_candidates_per_class(candidates, 3, sample_seed=7)

    assert first == second
    assert len(first) == 3


def test_filter_candidates_by_class_keeps_only_requested_classes() -> None:
    candidates = [
        (Path("sofa.jpg"), "sofa"),
        (Path("table.jpg"), "tables_desks"),
        (Path("chair.jpg"), "chairs"),
    ]

    assert filter_candidates_by_class(candidates, ["sofa", "tables_desks"]) == [
        (Path("sofa.jpg"), "sofa"),
        (Path("table.jpg"), "tables_desks"),
    ]


def test_box_helpers() -> None:
    votes = [
        vote("storage", 0.9, 0.4, (0.0, 10.0, 80.0, 90.0)),
        vote("storage", 0.9, 0.4, (10.0, 20.0, 90.0, 100.0)),
        vote("storage", 0.9, 0.4, (20.0, 30.0, 100.0, 110.0)),
    ]
    assert median_box(votes) == (10.0, 20.0, 90.0, 100.0)
    assert xyxy_to_yolo((10.0, 20.0, 90.0, 100.0), 100, 200) == (
        0.5,
        0.3,
        0.8,
        0.4,
    )
