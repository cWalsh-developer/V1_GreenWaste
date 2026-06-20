from pathlib import Path

from PIL import Image

from greenwaste.make_yolo_final_dataset import make_yolo_final_dataset


def write_source_dataset(root: Path) -> None:
    for relative_path in (
        Path("images/train"),
        Path("images/val"),
        Path("labels/train"),
        Path("labels/val"),
    ):
        (root / relative_path).mkdir(parents=True)

    for split in ("train", "val"):
        image = Image.new("RGB", (20, 20), "white")
        image.save(root / "images" / split / f"{split}_item.jpg")
        (root / "labels" / split / f"{split}_item.txt").write_text(
            "0 0.500000 0.500000 0.500000 0.500000\n"
        )

    (root / "data.yaml").write_text(
        f"path: {root.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: beds_mattresses\n"
    )


def test_make_yolo_final_dataset_copies_train_and_val_into_train(tmp_path):
    source_dir = tmp_path / "source"
    output_dir = tmp_path / "final"
    write_source_dataset(source_dir)

    copied = make_yolo_final_dataset(source_dir=source_dir, output_dir=output_dir)

    assert copied == 2
    assert (output_dir / "images/train/train__train_item.jpg").exists()
    assert (output_dir / "labels/train/train__train_item.txt").exists()
    assert (output_dir / "images/train/val__val_item.jpg").exists()
    assert (output_dir / "labels/train/val__val_item.txt").exists()
    data_yaml = (output_dir / "data.yaml").read_text()
    assert f"path: {output_dir.as_posix()}" in data_yaml
    assert "train: images/train" in data_yaml
    assert "val: images/train" in data_yaml
