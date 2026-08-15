from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .lca import (
    DEFAULT_LCA_FACTOR_PATH,
    DEFAULT_MATERIAL_PROFILE_PATH,
    build_composition_factor_rows,
    load_lca_factor_table,
    load_material_profile_table,
    select_lca_factors,
    select_material_profiles,
)
from .lca_estimation import CONDITION_STATUSES, build_lca_payload
from .realsense_capture import (
    CaptureConfig,
    apply_gray_world_white_balance,
    configure_sensors,
    swap_red_blue,
)
from .reference_matching import (
    build_match_payload,
    match_reference_items,
)
from .size_estimation import CaptureIntrinsics, estimate_size_from_roi
from .yolo_depth_size import size_category, xyxy_to_roi

try:
    import cv2
except ImportError as exc:  # pragma: no cover - runtime dependency
    raise ImportError(
        "opencv-python is required for the live demo. Install with: pip install opencv-python"
    ) from exc

try:
    import pyrealsense2 as rs
except ImportError as exc:  # pragma: no cover - runtime dependency
    raise ImportError(
        "pyrealsense2 is required for the live RealSense demo. "
        "Install with: pip install pyrealsense2"
    ) from exc

try:
    import torch
    from ultralytics import YOLO
except ImportError as exc:  # pragma: no cover - runtime dependency
    raise ImportError(
        "ultralytics and torch are required for the live demo. "
        "Install with: pip install ultralytics torch"
    ) from exc


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class LiveDetection:
    item_class: str
    class_id: int
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]


@dataclass(frozen=True)
class LiveRouteResult:
    capture_id: str
    item_class: str
    confidence: float
    width_cm: float
    height_cm: float
    depth_cm: float
    distance_cm: float
    size_category: str
    material_family: str
    weight_range_kg: tuple[float | None, float | None]
    recommended_route: str | None
    recommended_scenario: str | None
    co2e_range_kg: tuple[float | None, float | None]
    condition_status: str
    candidate_count: int
    bbox_xyxy: tuple[float, float, float, float]
    roi: tuple[int, int, int, int]
    roi_refined: tuple[int, int, int, int]
    rationale: str


def _intrinsics_to_capture_intrinsics(intrinsics: rs.intrinsics) -> CaptureIntrinsics:
    return CaptureIntrinsics(
        fx=float(intrinsics.fx),
        fy=float(intrinsics.fy),
        ppx=float(intrinsics.ppx),
        ppy=float(intrinsics.ppy),
    )


def _prepare_bgr_frame(
    color_image: np.ndarray,
    color_format: str,
    software_white_balance: bool = False,
    swap_rb: bool = False,
) -> np.ndarray:
    if color_format.lower() == "rgb":
        rgb = color_image
    else:
        rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

    if software_white_balance:
        rgb = apply_gray_world_white_balance(rgb)
    if swap_rb:
        rgb = swap_red_blue(rgb)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def detect_live_items(
    model: YOLO,
    frame_bgr: np.ndarray,
    confidence: float,
    image_size: int,
    max_detections: int,
    device: int | str,
) -> list[LiveDetection]:
    result = model.predict(
        source=frame_bgr,
        conf=confidence,
        iou=0.7,
        imgsz=image_size,
        device=device,
        verbose=False,
    )[0]

    detections: list[LiveDetection] = []
    for box in result.boxes:
        class_id = int(box.cls.item())
        x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
        detections.append(
            LiveDetection(
                item_class=str(result.names[class_id]),
                class_id=class_id,
                confidence=float(box.conf.item()),
                bbox_xyxy=(x1, y1, x2, y2),
            )
        )

    return sorted(
        detections,
        key=lambda detection: detection.confidence,
        reverse=True,
    )[:max_detections]


def _build_match_summary_row(
    route_capture_id: str,
    detection: LiveDetection,
    width_cm: float,
    height_cm: float,
    depth_cm: float,
    distance_cm: float,
    size_label: str,
    match_payload: dict[str, Any],
) -> pd.Series:
    reference_summary = match_payload["reference_summary"]
    weight_low, weight_high = reference_summary["weight_range_kg"]
    return pd.Series(
        {
            "capture_id": route_capture_id,
            "item_class": detection.item_class,
            "composition_profile": match_payload["composition_profile"],
            "confidence": detection.confidence,
            "width_cm": width_cm,
            "height_cm": height_cm,
            "depth_cm": depth_cm,
            "distance_cm": distance_cm,
            "size_category": size_label,
            "candidate_count": reference_summary["candidate_count"],
            "material_family": reference_summary["material_family"],
            "weight_low_kg": weight_low,
            "weight_high_kg": weight_high,
            "weight_reference_count": reference_summary["weight_reference_count"],
            "weight_imputed_count": reference_summary["weight_imputed_count"],
            "weight_missing_count": reference_summary["weight_missing_count"],
            "weight_source_note": reference_summary["weight_source_note"],
            "reference_size_bin": reference_summary["size_bin"],
        }
    )


def build_live_route_result(
    detection: LiveDetection,
    depth_image: np.ndarray,
    intrinsics: CaptureIntrinsics,
    depth_scale: float,
    reference_df: pd.DataFrame,
    lca_factors: pd.DataFrame,
    material_profiles: pd.DataFrame,
    route_capture_id: str,
    condition_status: str,
    top_n: int,
    use_material_profiles: bool,
) -> LiveRouteResult:
    if depth_image.ndim == 3:
        depth_image = depth_image[:, :, 0]
    image_height, image_width = depth_image.shape[:2]
    roi = xyxy_to_roi(detection.bbox_xyxy, image_width, image_height)
    estimate = estimate_size_from_roi(
        depth_image=depth_image,
        roi=roi,
        intrinsics=intrinsics,
        depth_scale=depth_scale,
    )
    size_label = size_category(
        width_cm=estimate.width_cm,
        height_cm=estimate.height_cm,
        depth_cm=estimate.depth_cm,
    )

    size_row = pd.Series(
        {
            "capture_id": route_capture_id,
            "item_class": detection.item_class,
            "confidence": detection.confidence,
            "width_cm": estimate.width_cm,
            "height_cm": estimate.height_cm,
            "depth_cm": estimate.depth_cm,
            "distance_cm": estimate.distance_cm,
            "size_category": size_label,
        }
    )
    matches = match_reference_items(
        reference_df=reference_df,
        item_class=detection.item_class,
        width_cm=estimate.width_cm,
        height_cm=estimate.height_cm,
        depth_cm=estimate.depth_cm,
        top_n=top_n,
    )
    match_payload = build_match_payload(size_row, matches)
    match_summary_row = _build_match_summary_row(
        route_capture_id=route_capture_id,
        detection=detection,
        width_cm=estimate.width_cm,
        height_cm=estimate.height_cm,
        depth_cm=estimate.depth_cm,
        distance_cm=estimate.distance_cm,
        size_label=size_label,
        match_payload=match_payload,
    )

    if use_material_profiles:
        selected_profiles = select_material_profiles(
            material_profiles,
            str(match_summary_row.get("composition_profile", detection.item_class)),
        )
        composition_factor_rows = build_composition_factor_rows(
            lca_factors,
            selected_profiles,
        )
        lca_payload = build_lca_payload(
            match_summary_row,
            composition_factor_rows=composition_factor_rows,
            condition_status=condition_status,
        )
    else:
        lca_payload = build_lca_payload(
            match_summary_row,
            factor_rows=select_lca_factors(
                lca_factors,
                str(match_summary_row.get("material_family", "unknown")),
            ),
            condition_status=condition_status,
        )

    recommendation = lca_payload["recommendation"]
    co2e_low, co2e_high = recommendation["co2e_range_kg"]
    weight_low, weight_high = match_payload["reference_summary"]["weight_range_kg"]
    return LiveRouteResult(
        capture_id=route_capture_id,
        item_class=detection.item_class,
        confidence=detection.confidence,
        width_cm=estimate.width_cm,
        height_cm=estimate.height_cm,
        depth_cm=estimate.depth_cm,
        distance_cm=estimate.distance_cm,
        size_category=size_label,
        material_family=str(match_payload["reference_summary"]["material_family"]),
        weight_range_kg=(weight_low, weight_high),
        recommended_route=recommendation["recommended_route"],
        recommended_scenario=recommendation["recommended_scenario"],
        co2e_range_kg=(co2e_low, co2e_high),
        condition_status=recommendation["condition_status"],
        candidate_count=int(match_payload["reference_summary"]["candidate_count"]),
        bbox_xyxy=detection.bbox_xyxy,
        roi=roi,
        roi_refined=estimate.roi_refined,
        rationale=str(recommendation["rationale"]),
    )


def _format_range(
    values: tuple[float | None, float | None],
    suffix: str,
    decimals: int = 1,
) -> str:
    low, high = values
    if low is None or high is None or pd.isna(low) or pd.isna(high):
        return "unknown"
    return f"{low:.{decimals}f}-{high:.{decimals}f} {suffix}"


def _draw_text_lines(
    image: np.ndarray,
    lines: list[str],
    origin: tuple[int, int],
    font_scale: float = 0.55,
    color: tuple[int, int, int] = (245, 245, 245),
    line_height: int = 22,
) -> None:
    x, y = origin
    for idx, line in enumerate(lines):
        cv2.putText(
            image,
            line,
            (x, y + idx * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            1,
            cv2.LINE_AA,
        )


def draw_live_overlay(
    frame_bgr: np.ndarray,
    detections: list[LiveDetection],
    route_result: LiveRouteResult | None,
    status: str,
    condition_status: str,
    fps: float,
) -> np.ndarray:
    output = frame_bgr.copy()
    height, width = output.shape[:2]

    for detection in detections:
        x1, y1, x2, y2 = [int(round(value)) for value in detection.bbox_xyxy]
        color = (45, 190, 90) if detection.confidence >= 0.5 else (0, 190, 255)
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        label = f"{detection.item_class} {detection.confidence:.2f}"
        (label_w, label_h), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            1,
        )
        y_text = max(y1 - 8, label_h + 8)
        cv2.rectangle(
            output,
            (x1, y_text - label_h - baseline - 4),
            (x1 + label_w + 8, y_text + baseline),
            color,
            -1,
        )
        cv2.putText(
            output,
            label,
            (x1 + 4, y_text - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (15, 20, 20),
            1,
            cv2.LINE_AA,
        )

    panel_w = min(430, max(320, width // 3))
    overlay = output.copy()
    cv2.rectangle(overlay, (width - panel_w, 0), (width, height), (25, 32, 36), -1)
    cv2.addWeighted(overlay, 0.82, output, 0.18, 0, output)

    x = width - panel_w + 18
    y = 34
    header = [
        "GreenWaste Live Demo",
        f"Condition: {condition_status}",
        f"FPS: {fps:.1f}",
    ]
    _draw_text_lines(output, header, (x, y), font_scale=0.58, line_height=24)

    y += 92
    if route_result is None:
        lines = [
            "No route result yet",
            status,
            "",
            "Keys: q quit | s save",
            "u unknown | r reusable | n not reusable",
        ]
        _draw_text_lines(output, lines, (x, y), font_scale=0.52, line_height=23)
        return output

    size_text = (
        f"{route_result.width_cm:.0f} x {route_result.height_cm:.0f} x "
        f"{route_result.depth_cm:.0f} cm"
    )
    lines = [
        f"Detected: {route_result.item_class}",
        f"Confidence: {route_result.confidence:.2f}",
        f"Size: {size_text}",
        f"Size class: {route_result.size_category}",
        f"Distance: {route_result.distance_cm:.0f} cm",
        "",
        f"Material: {route_result.material_family}",
        f"Weight: {_format_range(route_result.weight_range_kg, 'kg', 1)}",
        f"Matches: {route_result.candidate_count}",
        "",
        f"Route: {route_result.recommended_route or 'unknown'}",
        f"CO2e: {_format_range(route_result.co2e_range_kg, 'kg CO2e', 2)}",
        "",
        status,
        "",
        "Keys: q quit | s save",
        "u unknown | r reusable | n not reusable",
    ]
    _draw_text_lines(output, lines, (x, y), font_scale=0.52, line_height=22)
    return output


def save_live_snapshot(
    output_dir: Path,
    frame_bgr: np.ndarray,
    route_result: LiveRouteResult | None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    image_path = output_dir / f"live_demo_{timestamp}.png"
    cv2.imwrite(str(image_path), frame_bgr)

    if route_result is not None:
        json_path = output_dir / f"live_demo_{timestamp}.json"
        json_path.write_text(json.dumps(asdict(route_result), indent=2))
    return image_path


def run_live_demo(
    model_path: Path,
    reference_path: Path,
    output_dir: Path,
    factor_path: Path = DEFAULT_LCA_FACTOR_PATH,
    material_profile_path: Path = DEFAULT_MATERIAL_PROFILE_PATH,
    confidence: float = 0.25,
    image_size: int = 960,
    max_detections: int = 1,
    top_n: int = 10,
    route_update_seconds: float = 1.0,
    width: int = 640,
    height: int = 480,
    fps: int = 30,
    serial: str | None = None,
    condition_status: str = "unknown",
    use_material_profiles: bool = True,
    auto_exposure: bool = True,
    exposure: int | None = None,
    gain: int | None = None,
    auto_white_balance: bool = True,
    white_balance: int | None = None,
    lock_white_balance: bool = False,
    software_white_balance: bool = False,
    swap_rb: bool = False,
    color_format: str = "bgr",
) -> None:
    if condition_status not in CONDITION_STATUSES:
        raise ValueError(
            "condition_status must be one of: "
            + ", ".join(sorted(CONDITION_STATUSES))
        )

    reference_df = pd.read_csv(reference_path)
    lca_factors = load_lca_factor_table(factor_path)
    material_profiles = (
        load_material_profile_table(material_profile_path)
        if use_material_profiles
        else pd.DataFrame()
    )

    LOGGER.info("Loading YOLO model from %s", model_path)
    model = YOLO(str(model_path))
    device = 0 if torch.cuda.is_available() else "cpu"
    LOGGER.info("Using inference device: %s", device)

    pipeline = rs.pipeline()
    cfg = rs.config()
    if serial:
        cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    color_stream_format = rs.format.bgr8 if color_format.lower() == "bgr" else rs.format.rgb8
    cfg.enable_stream(rs.stream.color, width, height, color_stream_format, fps)

    profile = pipeline.start(cfg)
    configure_sensors(
        profile.get_device(),
        CaptureConfig(
            width=width,
            height=height,
            fps=fps,
            auto_exposure=auto_exposure,
            exposure=exposure,
            gain=gain,
            auto_white_balance=auto_white_balance,
            white_balance=white_balance,
            lock_white_balance=lock_white_balance,
            color_format=color_format,
            software_white_balance=software_white_balance,
            swap_rb=swap_rb,
        ),
    )
    align = rs.align(rs.stream.color)
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

    if lock_white_balance:
        for _ in range(15):
            pipeline.wait_for_frames()
        for sensor in profile.get_device().query_sensors():
            if sensor.supports(rs.option.enable_auto_white_balance):
                sensor.set_option(rs.option.enable_auto_white_balance, 0.0)

    latest_route: LiveRouteResult | None = None
    latest_route_at = 0.0
    latest_status = "Waiting for detections"
    frame_count = 0
    fps_started_at = time.perf_counter()
    smoothed_fps = 0.0

    LOGGER.info("Live demo started. Press q/Esc to quit.")
    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            frame_bgr = _prepare_bgr_frame(
                color_image,
                color_format=color_format,
                software_white_balance=software_white_balance,
                swap_rb=swap_rb,
            )
            intrinsics = _intrinsics_to_capture_intrinsics(
                color_frame.profile.as_video_stream_profile().get_intrinsics()
            )

            detections = detect_live_items(
                model=model,
                frame_bgr=frame_bgr,
                confidence=confidence,
                image_size=image_size,
                max_detections=max_detections,
                device=device,
            )

            now = time.perf_counter()
            if detections and now - latest_route_at >= route_update_seconds:
                try:
                    latest_route = build_live_route_result(
                        detection=detections[0],
                        depth_image=depth_image,
                        intrinsics=intrinsics,
                        depth_scale=depth_scale,
                        reference_df=reference_df,
                        lca_factors=lca_factors,
                        material_profiles=material_profiles,
                        route_capture_id="live_realsense",
                        condition_status=condition_status,
                        top_n=top_n,
                        use_material_profiles=use_material_profiles,
                    )
                    latest_status = "Route updated"
                    latest_route_at = now
                except Exception as exc:  # pragma: no cover - live diagnostic path
                    LOGGER.warning("Failed to build live route result: %s", exc)
                    latest_status = f"Route unavailable: {exc}"
                    latest_route_at = now
            elif not detections:
                latest_status = "No detection above threshold"

            frame_count += 1
            elapsed = now - fps_started_at
            if elapsed >= 1.0:
                smoothed_fps = frame_count / elapsed
                frame_count = 0
                fps_started_at = now

            display = draw_live_overlay(
                frame_bgr=frame_bgr,
                detections=detections,
                route_result=latest_route,
                status=latest_status,
                condition_status=condition_status,
                fps=smoothed_fps,
            )
            cv2.imshow("GreenWaste Live RealSense Demo", display)

            key = cv2.waitKey(1) & 0xFF
            if cv2.getWindowProperty(
                "GreenWaste Live RealSense Demo",
                cv2.WND_PROP_VISIBLE,
            ) < 1:
                break
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                saved_path = save_live_snapshot(output_dir, display, latest_route)
                latest_status = f"Saved {saved_path.name}"
            elif key == ord("u"):
                condition_status = "unknown"
                latest_route_at = 0.0
            elif key == ord("r"):
                condition_status = "reusable"
                latest_route_at = 0.0
            elif key == ord("n"):
                condition_status = "not_reusable"
                latest_route_at = 0.0
    finally:
        cv2.destroyAllWindows()
        pipeline.stop()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a live RealSense + YOLO GreenWaste demo with on-screen route "
            "recommendations."
        )
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("data/interim/ikea_reference_cleaned.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/interim/live_realsense_demo"),
    )
    parser.add_argument("--factor-table", type=Path, default=DEFAULT_LCA_FACTOR_PATH)
    parser.add_argument(
        "--material-profiles",
        type=Path,
        default=DEFAULT_MATERIAL_PROFILE_PATH,
    )
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--image-size", type=int, default=960)
    parser.add_argument("--max-detections", type=int, default=1)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument(
        "--route-update-seconds",
        type=float,
        default=1.0,
        help="Minimum seconds between reference/LCA route refreshes.",
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--serial", type=str, default=None)
    parser.add_argument(
        "--condition",
        choices=sorted(CONDITION_STATUSES),
        default="unknown",
    )
    parser.add_argument(
        "--no-material-profiles",
        action="store_true",
        help="Use direct material-family factors instead of class proxy profiles.",
    )
    parser.add_argument(
        "--manual-exposure",
        action="store_true",
        help="Disable auto exposure and use optional --exposure/--gain values.",
    )
    parser.add_argument("--exposure", type=int, default=None)
    parser.add_argument("--gain", type=int, default=None)
    parser.add_argument(
        "--manual-white-balance",
        action="store_true",
        help="Disable camera auto white balance and use optional --white-balance.",
    )
    parser.add_argument("--white-balance", type=int, default=None)
    parser.add_argument("--lock-white-balance", action="store_true")
    parser.add_argument(
        "--software-white-balance",
        action="store_true",
        help="Apply gray-world white balance before inference/display.",
    )
    parser.add_argument("--swap-rb", action="store_true")
    parser.add_argument(
        "--color-format",
        choices=["bgr", "rgb"],
        default="bgr",
        help="RealSense colour stream format. BGR avoids OpenCV colour swapping.",
    )
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_parser().parse_args()

    run_live_demo(
        model_path=args.model,
        reference_path=args.reference,
        output_dir=args.output_dir,
        factor_path=args.factor_table,
        material_profile_path=args.material_profiles,
        confidence=args.confidence,
        image_size=args.image_size,
        max_detections=args.max_detections,
        top_n=args.top_n,
        route_update_seconds=args.route_update_seconds,
        width=args.width,
        height=args.height,
        fps=args.fps,
        serial=args.serial,
        condition_status=args.condition,
        use_material_profiles=not args.no_material_profiles,
        auto_exposure=not args.manual_exposure,
        exposure=args.exposure,
        gain=args.gain,
        auto_white_balance=not args.manual_white_balance,
        white_balance=args.white_balance,
        lock_white_balance=args.lock_white_balance,
        software_white_balance=args.software_white_balance,
        swap_rb=args.swap_rb,
        color_format=args.color_format,
    )


if __name__ == "__main__":
    main()
