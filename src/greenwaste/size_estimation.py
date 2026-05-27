from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import cv2
except ImportError as exc:  # pragma: no cover - runtime dependency
    raise ImportError(
        "opencv-python is required for ROI selection. Install with: pip install opencv-python"
    ) from exc


@dataclass(frozen=True)
class CaptureIntrinsics:
    fx: float
    fy: float
    ppx: float
    ppy: float


@dataclass(frozen=True)
class SizeEstimate:
    width_cm: float
    height_cm: float
    distance_cm: float
    depth_cm: float
    roi: tuple[int, int, int, int]
    roi_refined: tuple[int, int, int, int]


def load_capture_images(capture_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    rgb_path = capture_dir / "rgb.png"
    depth_path = capture_dir / "depth.png"
    if not rgb_path.exists():
        raise FileNotFoundError(f"RGB image not found: {rgb_path}")
    if not depth_path.exists():
        raise FileNotFoundError(f"Depth image not found: {depth_path}")

    rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if rgb is None or depth is None:
        raise RuntimeError("Failed to load capture images")
    return rgb, depth


def parse_intrinsics(metadata: dict[str, Any]) -> CaptureIntrinsics:
    intrinsics = metadata.get("intrinsics", {})
    return CaptureIntrinsics(
        fx=float(intrinsics.get("fx")),
        fy=float(intrinsics.get("fy")),
        ppx=float(intrinsics.get("ppx")),
        ppy=float(intrinsics.get("ppy")),
    )


def select_roi(rgb_image: np.ndarray) -> tuple[int, int, int, int]:
    roi = cv2.selectROI("Select ROI", rgb_image, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow("Select ROI")
    return tuple(int(v) for v in roi)


def estimate_size_from_roi(
    depth_image: np.ndarray,
    roi: tuple[int, int, int, int],
    intrinsics: CaptureIntrinsics,
    depth_scale: float,
) -> SizeEstimate:
    x, y, w, h = roi
    if w <= 0 or h <= 0:
        raise ValueError("ROI width and height must be positive")

    depth_roi = depth_image[y : y + h, x : x + w].astype(np.float32)
    valid = depth_roi[depth_roi > 0]
    if valid.size == 0:
        raise ValueError("No valid depth values in ROI")

    depth_median = float(np.median(valid) * depth_scale)
    depth_min = float(np.percentile(valid, 30) * depth_scale)
    depth_max = float(np.percentile(valid, 70) * depth_scale)
    depth_thickness = max(depth_max - depth_min, 0.0)

    inlier_mask = (
        (depth_roi > 0)
        & (depth_roi >= depth_min / depth_scale)
        & (depth_roi <= depth_max / depth_scale)
    )

    if np.any(inlier_mask):
        ys, xs = np.where(inlier_mask)
        x_min, x_max = int(xs.min()), int(xs.max())
        y_min, y_max = int(ys.min()), int(ys.max())
        w_refined = max(x_max - x_min + 1, 1)
        h_refined = max(y_max - y_min + 1, 1)
        roi_refined = (x + x_min, y + y_min, w_refined, h_refined)
    else:
        w_refined, h_refined = w, h
        roi_refined = roi

    width_m = (w_refined / intrinsics.fx) * depth_median
    height_m = (h_refined / intrinsics.fy) * depth_median

    return SizeEstimate(
        width_cm=width_m * 100.0,
        height_cm=height_m * 100.0,
        distance_cm=depth_median * 100.0,
        depth_cm=depth_thickness * 100.0,
        roi=roi,
        roi_refined=roi_refined,
    )
