from pathlib import Path

from greenwaste.make_yolo_one_vs_rest_datasets import make_yolo_one_vs_rest_datasets


def write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake image")


def test_make_yolo_one_vs_rest_datasets_preserves_splits_and_writes_negatives(tmp_path: Path) -> None:
    source = tmp_path / "source"
    for split in ("train", "val", "test"):
        (source / "labels" / split).mkdir(parents=True)

    (source / "data.yaml").write_text(
        "\n".join(
            [
                f"path: {source.as_posix()}",
                "train: images/train",
                "val: images/val",
                "test: images/test",
                "names:",
                "  0: beds_mattresses",
                "  1: chair_seating",
            ]
        )
        + "\n"
    )

    write_image(source / "images" / "train" / "bed.jpg")
    (source / "labels" / "train" / "bed.txt").write_text("0 0.5 0.5 0.4 0.4\n")
    write_image(source / "images" / "train" / "chair.jpg")
    (source / "labels" / "train" / "chair.txt").write_text("1 0.5 0.5 0.3 0.3\n")
    write_image(source / "images" / "val" / "mixed.jpg")
    (source / "labels" / "val" / "mixed.txt").write_text(
        "0 0.4 0.4 0.2 0.2\n1 0.6 0.6 0.2 0.2\n"
    )
    write_image(source / "images" / "test" / "empty.jpg")
    (source / "labels" / "test" / "empty.txt").write_text("")

    output = tmp_path / "one_vs_rest"
    rows = make_yolo_one_vs_rest_datasets(source, output)

    assert len(rows) == 8
    assert (output / "beds_mattresses" / "data.yaml").exists()
    assert (output / "chair_seating" / "data.yaml").exists()
    assert (output / "beds_mattresses" / "labels" / "train" / "bed.txt").read_text() == (
        "0 0.5 0.5 0.4 0.4\n"
    )
    assert (output / "beds_mattresses" / "labels" / "train" / "chair.txt").read_text() == ""
    assert (output / "chair_seating" / "labels" / "val" / "mixed.txt").read_text() == (
        "0 0.6 0.6 0.2 0.2\n"
    )
    assert "0: beds_mattresses" in (output / "beds_mattresses" / "data.yaml").read_text()
    assert "test: images/test" in (output / "beds_mattresses" / "data.yaml").read_text()
