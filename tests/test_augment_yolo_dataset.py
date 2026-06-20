from pathlib import Path

from PIL import Image

from greenwaste.augment_yolo_dataset import augment_yolo_dataset


def write_tiny_yolo_dataset(root: Path) -> None:
    for relative_path in (
        Path("images/train"),
        Path("images/val"),
        Path("labels/train"),
        Path("labels/val"),
    ):
        (root / relative_path).mkdir(parents=True)

    train_image = Image.new("RGB", (32, 32), "white")
    for x in range(10, 22):
        for y in range(8, 24):
            train_image.putpixel((x, y), (40, 80, 140))
    train_image.save(root / "images/train/example.jpg")
    (root / "labels/train/example.txt").write_text("0 0.500000 0.500000 0.500000 0.500000\n")

    val_image = Image.new("RGB", (32, 32), "white")
    val_image.save(root / "images/val/heldout.jpg")
    (root / "labels/val/heldout.txt").write_text("0 0.500000 0.500000 0.500000 0.500000\n")

    (root / "data.yaml").write_text(
        f"path: {root.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: beds_mattresses\n"
    )


def test_augment_yolo_dataset_adds_train_variants_without_touching_val(tmp_path):
    source_dir = tmp_path / "source"
    output_dir = tmp_path / "augmented"
    write_tiny_yolo_dataset(source_dir)

    records = augment_yolo_dataset(
        source_dir=source_dir,
        output_dir=output_dir,
        variants_per_image=2,
        seed=7,
        include_class_ids={0},
        white_threshold=245,
        min_foreground_ratio=0.02,
        background_probability=1.0,
        channel_swap_probability=0.0,
    )

    assert len(records) == 2
    assert (output_dir / "images/train/example.jpg").exists()
    assert (output_dir / "labels/train/example.txt").exists()
    assert (output_dir / "images/train/example__aug01.jpg").exists()
    assert (output_dir / "labels/train/example__aug01.txt").read_text() == (
        output_dir / "labels/train/example.txt"
    ).read_text()
    assert not (output_dir / "images/val/heldout__aug01.jpg").exists()
    assert (output_dir / "images/val/heldout.jpg").exists()
    assert (output_dir / "augmentation_summary.csv").exists()
    assert f"path: {output_dir.as_posix()}" in (output_dir / "data.yaml").read_text()
