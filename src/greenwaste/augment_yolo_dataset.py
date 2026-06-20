from __future__ import annotations

import argparse
import csv
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass(frozen=True)
class AugmentationRecord:
    source_image: Path
    output_image: Path
    output_label: Path
    variant: int
    class_ids: tuple[int, ...]
    background_composited: bool
    channel_swapped: bool


def find_image_for_label(image_dir: Path, label_path: Path) -> Path | None:
    for extension in IMAGE_EXTENSIONS:
        candidate = image_dir / f"{label_path.stem}{extension}"
        if candidate.exists():
            return candidate
    return None


def class_ids_from_label(label_path: Path) -> tuple[int, ...]:
    class_ids: list[int] = []
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        class_ids.append(int(float(parts[0])))
    return tuple(sorted(set(class_ids)))


def yolo_boxes_to_pixels(label_path: Path, image_size: tuple[int, int]) -> list[tuple[int, int, int, int]]:
    image_width, image_height = image_size
    boxes: list[tuple[int, int, int, int]] = []
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue

        _, x_center, y_center, width, height = (float(value) for value in parts)
        box_width = width * image_width
        box_height = height * image_height
        x1 = int(round((x_center * image_width) - (box_width / 2.0)))
        y1 = int(round((y_center * image_height) - (box_height / 2.0)))
        x2 = int(round((x_center * image_width) + (box_width / 2.0)))
        y2 = int(round((y_center * image_height) + (box_height / 2.0)))
        boxes.append(
            (
                max(0, x1),
                max(0, y1),
                min(image_width, x2),
                min(image_height, y2),
            )
        )
    return boxes


def make_room_background(size: tuple[int, int], rng: random.Random) -> Image.Image:
    width, height = size
    wall_colour = tuple(rng.randint(low, high) for low, high in ((170, 230), (170, 230), (165, 225)))
    floor_colour = tuple(rng.randint(low, high) for low, high in ((105, 185), (95, 170), (80, 155)))
    image = Image.new("RGB", size, wall_colour)
    draw = ImageDraw.Draw(image)

    horizon = rng.randint(int(height * 0.48), int(height * 0.72))
    draw.rectangle((0, horizon, width, height), fill=floor_colour)
    draw.line((0, horizon, width, horizon), fill=tuple(max(0, channel - 45) for channel in wall_colour), width=2)

    for _ in range(rng.randint(5, 12)):
        x1 = rng.randint(0, max(0, width - 1))
        y1 = rng.randint(0, max(0, height - 1))
        x2 = min(width, x1 + rng.randint(max(8, width // 18), max(12, width // 5)))
        y2 = min(height, y1 + rng.randint(max(8, height // 18), max(12, height // 4)))
        colour = tuple(rng.randint(70, 225) for _ in range(3))
        draw.rectangle((x1, y1, x2, y2), fill=colour)

    noise = rng.randint(4, 16)
    array = np.asarray(image).astype(np.int16)
    array += rng.randint(-noise, noise)
    array += np.random.default_rng(rng.randrange(2**32)).integers(
        -noise,
        noise + 1,
        size=array.shape,
    )
    array = np.clip(array, 0, 255).astype(np.uint8)
    return Image.fromarray(array, mode="RGB").filter(ImageFilter.GaussianBlur(radius=0.4))


def white_background_mask(
    image: Image.Image,
    boxes: list[tuple[int, int, int, int]],
    threshold: int,
    min_foreground_ratio: float,
) -> tuple[Image.Image, bool]:
    rgb_image = image.convert("RGB")
    image_width, image_height = rgb_image.size
    full_mask = Image.new("L", rgb_image.size, 0)
    full_mask_draw = ImageDraw.Draw(full_mask)
    image_array = np.asarray(rgb_image)
    used_segmented_mask = False

    for x1, y1, x2, y2 in boxes:
        if x2 <= x1 or y2 <= y1:
            continue

        crop = image_array[y1:y2, x1:x2]
        foreground = np.any(crop < threshold, axis=2)
        foreground_ratio = float(foreground.mean()) if foreground.size else 0.0

        if foreground_ratio >= min_foreground_ratio:
            crop_mask = Image.fromarray((foreground * 255).astype(np.uint8), mode="L")
            full_mask.paste(crop_mask, (x1, y1))
            used_segmented_mask = True
        else:
            full_mask_draw.rectangle((x1, y1, x2, y2), fill=255)

    if not boxes:
        full_mask_draw.rectangle((0, 0, image_width, image_height), fill=255)

    return full_mask.filter(ImageFilter.GaussianBlur(radius=1.2)), used_segmented_mask


def apply_photometric_effects(
    image: Image.Image,
    rng: random.Random,
    channel_swap_probability: float,
) -> tuple[Image.Image, bool]:
    output = image.convert("RGB")
    output = ImageEnhance.Brightness(output).enhance(rng.uniform(0.72, 1.22))
    output = ImageEnhance.Contrast(output).enhance(rng.uniform(0.72, 1.28))
    output = ImageEnhance.Color(output).enhance(rng.uniform(0.55, 1.35))
    channel_swapped = rng.random() < channel_swap_probability
    if channel_swapped:
        red, green, blue = output.split()
        output = Image.merge("RGB", (blue, green, red))

    array = np.asarray(output).astype(np.float32)
    noise = rng.randint(0, 18)
    if noise:
        array += np.random.default_rng(rng.randrange(2**32)).normal(0, noise, size=array.shape)
    array = np.clip(array, 0, 255).astype(np.uint8)
    output = Image.fromarray(array, mode="RGB")

    if rng.random() < 0.25:
        output = output.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.25, 1.1)))

    return output, channel_swapped


def augment_image(
    image_path: Path,
    label_path: Path,
    rng: random.Random,
    white_threshold: int,
    min_foreground_ratio: float,
    background_probability: float,
    channel_swap_probability: float,
) -> tuple[Image.Image, bool, bool]:
    with Image.open(image_path) as image:
        rgb_image = image.convert("RGB")

    boxes = yolo_boxes_to_pixels(label_path, rgb_image.size)
    background_composited = rng.random() < background_probability

    if background_composited:
        background = make_room_background(rgb_image.size, rng)
        mask, _ = white_background_mask(
            image=rgb_image,
            boxes=boxes,
            threshold=white_threshold,
            min_foreground_ratio=min_foreground_ratio,
        )
        rgb_image = Image.composite(rgb_image, background, mask)

    rgb_image, channel_swapped = apply_photometric_effects(
        image=rgb_image,
        rng=rng,
        channel_swap_probability=channel_swap_probability,
    )
    return rgb_image, background_composited, channel_swapped


def copy_dataset(source_dir: Path, output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"{output_dir} already exists. Choose a new output directory or pass --overwrite."
            )
        shutil.rmtree(output_dir)
    for relative_path in (
        Path("images/train"),
        Path("images/val"),
        Path("labels/train"),
        Path("labels/val"),
    ):
        target = output_dir / relative_path
        target.mkdir(parents=True, exist_ok=True)
        source = source_dir / relative_path
        if source.exists():
            for path in source.iterdir():
                if path.is_file():
                    shutil.copy2(path, target / path.name)

    source_yaml = source_dir / "data.yaml"
    if source_yaml.exists():
        lines = source_yaml.read_text().splitlines()
        rewritten = [
            f"path: {output_dir.as_posix()}" if line.startswith("path:") else line
            for line in lines
        ]
        (output_dir / "data.yaml").write_text("\n".join(rewritten) + "\n")


def write_summary(output_dir: Path, records: list[AugmentationRecord]) -> None:
    summary_path = output_dir / "augmentation_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "source_image",
                "output_image",
                "output_label",
                "variant",
                "class_ids",
                "background_composited",
                "channel_swapped",
            ],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "source_image": record.source_image,
                    "output_image": record.output_image,
                    "output_label": record.output_label,
                    "variant": record.variant,
                    "class_ids": " ".join(str(class_id) for class_id in record.class_ids),
                    "background_composited": record.background_composited,
                    "channel_swapped": record.channel_swapped,
                }
            )


def augment_yolo_dataset(
    source_dir: Path,
    output_dir: Path,
    variants_per_image: int,
    seed: int,
    include_class_ids: set[int] | None,
    white_threshold: int,
    min_foreground_ratio: float,
    background_probability: float,
    channel_swap_probability: float,
    overwrite: bool = False,
) -> list[AugmentationRecord]:
    copy_dataset(source_dir=source_dir, output_dir=output_dir, overwrite=overwrite)

    rng = random.Random(seed)
    image_dir = output_dir / "images" / "train"
    label_dir = output_dir / "labels" / "train"
    records: list[AugmentationRecord] = []

    for label_path in sorted(label_dir.glob("*.txt")):
        image_path = find_image_for_label(image_dir, label_path)
        if image_path is None:
            continue

        class_ids = class_ids_from_label(label_path)
        if include_class_ids is not None and not set(class_ids).intersection(include_class_ids):
            continue

        for variant in range(1, variants_per_image + 1):
            augmented, background_composited, channel_swapped = augment_image(
                image_path=image_path,
                label_path=label_path,
                rng=rng,
                white_threshold=white_threshold,
                min_foreground_ratio=min_foreground_ratio,
                background_probability=background_probability,
                channel_swap_probability=channel_swap_probability,
            )

            output_stem = f"{image_path.stem}__aug{variant:02d}"
            output_image = image_dir / f"{output_stem}.jpg"
            output_label = label_dir / f"{output_stem}.txt"
            augmented.save(output_image, quality=92)
            shutil.copy2(label_path, output_label)
            records.append(
                AugmentationRecord(
                    source_image=image_path,
                    output_image=output_image,
                    output_label=output_label,
                    variant=variant,
                    class_ids=class_ids,
                    background_composited=background_composited,
                    channel_swapped=channel_swapped,
                )
            )

    write_summary(output_dir, records)
    return records


def parse_class_ids(value: str) -> set[int]:
    return {int(part.strip()) for part in value.split(",") if part.strip()}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a YOLO dataset copy with domain-oriented training augmentations."
    )
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variants-per-image", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--include-class-ids",
        type=parse_class_ids,
        default=None,
        help="Optional comma-separated class IDs to augment, e.g. 0,3 for beds and storage.",
    )
    parser.add_argument("--white-threshold", type=int, default=245)
    parser.add_argument("--min-foreground-ratio", type=float, default=0.02)
    parser.add_argument("--background-probability", type=float, default=0.9)
    parser.add_argument("--channel-swap-probability", type=float, default=0.15)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    records = augment_yolo_dataset(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        variants_per_image=args.variants_per_image,
        seed=args.seed,
        include_class_ids=args.include_class_ids,
        white_threshold=args.white_threshold,
        min_foreground_ratio=args.min_foreground_ratio,
        background_probability=args.background_probability,
        channel_swap_probability=args.channel_swap_probability,
        overwrite=args.overwrite,
    )
    print(f"Created augmented training images: {len(records)}")
    print(f"Output directory: {args.output_dir}")
    print(f"Summary CSV: {args.output_dir / 'augmentation_summary.csv'}")


if __name__ == "__main__":
    main()
