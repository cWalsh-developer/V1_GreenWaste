from pathlib import Path

from PIL import Image

from greenwaste.make_yolo_kfolds import make_yolo_kfolds


def write_image(path: Path, colour: str = "white") -> None:
    image = Image.new("RGB", (24, 24), colour)
    image.save(path)


def write_source_dataset(root: Path) -> None:
    for relative_path in (
        Path("images/train"),
        Path("images/val"),
        Path("labels/train"),
        Path("labels/val"),
    ):
        (root / relative_path).mkdir(parents=True)

    for index, class_id in enumerate((0, 0, 1, 1, 2, 2), start=1):
        split = "train" if index <= 4 else "val"
        stem = f"item_{index}"
        write_image(root / "images" / split / f"{stem}.jpg")
        (root / "labels" / split / f"{stem}.txt").write_text(
            f"{class_id} 0.500000 0.500000 0.500000 0.500000\n"
        )

    write_image(root / "images/train/item_1__aug01.jpg")
    (root / "labels/train/item_1__aug01.txt").write_text(
        "0 0.500000 0.500000 0.500000 0.500000\n"
    )

    (root / "data.yaml").write_text(
        f"path: {root.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: beds_mattresses\n"
        "  1: chair_seating\n"
        "  2: sofa\n"
    )


def test_make_yolo_kfolds_assigns_each_group_to_validation_once(tmp_path):
    source_dir = tmp_path / "source"
    output_dir = tmp_path / "folds"
    write_source_dataset(source_dir)

    rows = make_yolo_kfolds(
        source_dir=source_dir,
        output_dir=output_dir,
        folds=3,
        seed=11,
    )

    val_rows = [row for row in rows if row["target_split"] == "val"]
    val_groups = [row["group_key"] for row in val_rows]
    assert sorted(set(val_groups)) == [f"item_{index}" for index in range(1, 7)]

    for fold in range(1, 4):
        fold_dir = output_dir / f"fold_{fold}"
        assert (fold_dir / "data.yaml").exists()
        assert (fold_dir / "images/train").exists()
        assert (fold_dir / "images/val").exists()

    fold_by_group: dict[str, set[int]] = {}
    for row in val_rows:
        fold_by_group.setdefault(str(row["group_key"]), set()).add(int(row["fold"]))

    assert fold_by_group["item_1"] == {
        row["fold"] for row in val_rows if row["group_key"] == "item_1__aug01"
    } or "item_1__aug01" not in fold_by_group
    assert len(fold_by_group["item_1"]) == 1
    assert (output_dir / "cross_validation_manifest.csv").exists()
    assert (output_dir / "cross_validation_counts.csv").exists()


def test_make_yolo_kfolds_keeps_augmented_siblings_out_of_opposite_split(tmp_path):
    source_dir = tmp_path / "source"
    output_dir = tmp_path / "folds"
    write_source_dataset(source_dir)

    rows = make_yolo_kfolds(
        source_dir=source_dir,
        output_dir=output_dir,
        folds=3,
        seed=11,
    )

    item_1_rows = [row for row in rows if row["group_key"] == "item_1"]
    for fold in range(1, 4):
        fold_rows = [row for row in item_1_rows if row["fold"] == fold]
        target_splits = {row["target_split"] for row in fold_rows}
        assert len(target_splits) == 1
