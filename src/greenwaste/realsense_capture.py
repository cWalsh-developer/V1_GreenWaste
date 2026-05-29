from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyrealsense2 as rs

try:
    import cv2
except ImportError as exc:  # pragma: no cover - runtime dependency
    raise ImportError(
        "opencv-python is required for saving frames. Install with: pip install opencv-python"
    ) from exc

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class CaptureConfig:
    width: int = 640
    height: int = 480
    fps: int = 30
    auto_exposure: bool = True
    exposure: int | None = None
    gain: int | None = None
    preview: bool = False
    notes: str | None = None
    warmup_frames: int = 15
    save_bgr_copy: bool = False
    auto_white_balance: bool = True
    white_balance: int | None = None
    lock_white_balance: bool = False
    color_format: str = "rgb"
    software_white_balance: bool = False
    swap_rb: bool = False
    preview_color_space: str = "bgr"


def build_metadata(
    capture_id: str,
    camera_model: str,
    serial: str,
    rgb_shape: tuple[int, int, int],
    depth_shape: tuple[int, int],
    depth_scale: float,
    intrinsics: dict[str, Any],
    notes: str | None = None,
    color_space: str = "rgb",
    software_white_balance: bool = False,
    swap_rb: bool = False,
) -> dict[str, Any]:
    timestamp = datetime.now(timezone.utc).isoformat()
    return {
        "capture_id": capture_id,
        "timestamp_utc": timestamp,
        "camera_model": camera_model,
        "serial": serial,
        "resolution_rgb": {
            "width": int(rgb_shape[1]),
            "height": int(rgb_shape[0]),
        },
        "resolution_depth": {
            "width": int(depth_shape[1]),
            "height": int(depth_shape[0]),
        },
        "depth_scale": float(depth_scale),
        "intrinsics": intrinsics,
        "notes": notes,
        "color_space": color_space,
        "software_white_balance": software_white_balance,
        "swap_rb": swap_rb,
    }


def _intrinsics_to_dict(intrinsics: rs.intrinsics) -> dict[str, Any]:
    return {
        "width": intrinsics.width,
        "height": intrinsics.height,
        "ppx": intrinsics.ppx,
        "ppy": intrinsics.ppy,
        "fx": intrinsics.fx,
        "fy": intrinsics.fy,
        "model": str(intrinsics.model),
        "coeffs": list(intrinsics.coeffs),
    }


def _set_sensor_option(sensor: rs.sensor, option: rs.option, value: float) -> None:
    if sensor.supports(option):
        sensor.set_option(option, value)


def configure_sensors(device: rs.device, config: CaptureConfig) -> None:
    sensors = device.query_sensors()
    for sensor in sensors:
        if sensor.supports(rs.option.enable_auto_exposure):
            _set_sensor_option(
                sensor,
                rs.option.enable_auto_exposure,
                1.0 if config.auto_exposure else 0.0,
            )
        if sensor.supports(rs.option.enable_auto_white_balance):
            _set_sensor_option(
                sensor,
                rs.option.enable_auto_white_balance,
                1.0 if config.auto_white_balance else 0.0,
            )
        if not config.auto_white_balance and config.white_balance is not None:
            if sensor.supports(rs.option.white_balance):
                _set_sensor_option(
                    sensor, rs.option.white_balance, float(config.white_balance)
                )
        if not config.auto_exposure:
            if config.exposure is not None:
                _set_sensor_option(sensor, rs.option.exposure, float(config.exposure))
            if config.gain is not None:
                _set_sensor_option(sensor, rs.option.gain, float(config.gain))


def _depth_colormap(depth_image: np.ndarray) -> np.ndarray:
    depth_normalized = cv2.normalize(
        depth_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
    )
    return cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)


def apply_gray_world_white_balance(image: np.ndarray) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] != 3:
        return image
    img = image.astype(np.float32)
    means = img.reshape(-1, 3).mean(axis=0)
    target = float(means.mean())
    scale = target / np.clip(means, 1e-6, None)
    img *= scale
    return np.clip(img, 0, 255).astype(np.uint8)


def swap_red_blue(image: np.ndarray) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] != 3:
        return image
    return image[:, :, ::-1]


def normalize_to_rgb(image: np.ndarray, color_format: str) -> np.ndarray:
    if color_format.lower() == "bgr":
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def capture_single(
    output_dir: Path,
    config: CaptureConfig,
    device_serial: str | None = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    capture_id = output_dir.name

    pipeline = rs.pipeline()
    cfg = rs.config()
    if device_serial:
        cfg.enable_device(device_serial)

    cfg.enable_stream(
        rs.stream.depth, config.width, config.height, rs.format.z16, config.fps
    )
    color_stream_format = rs.format.rgb8
    if config.color_format.lower() == "bgr":
        color_stream_format = rs.format.bgr8
    cfg.enable_stream(
        rs.stream.color, config.width, config.height, color_stream_format, config.fps
    )

    profile = pipeline.start(cfg)
    configure_sensors(profile.get_device(), config)
    align = rs.align(rs.stream.color)

    try:
        depth_image = None
        color_image = None

        for _ in range(config.warmup_frames):
            pipeline.wait_for_frames()

        if config.lock_white_balance:
            for sensor in profile.get_device().query_sensors():
                if sensor.supports(rs.option.enable_auto_white_balance):
                    _set_sensor_option(sensor, rs.option.enable_auto_white_balance, 0.0)

        if config.preview:
            LOGGER.info("Preview mode: press 's' to save, 'q' or Esc to quit")
            while True:
                frames = pipeline.wait_for_frames()
                aligned_frames = align.process(frames)
                depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()

                if not depth_frame or not color_frame:
                    continue

                depth_image = np.asanyarray(depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())

                depth_vis = _depth_colormap(depth_image)
                rgb_image = normalize_to_rgb(color_image, config.color_format)
                if config.software_white_balance:
                    rgb_image = apply_gray_world_white_balance(rgb_image)
                if config.swap_rb:
                    rgb_image = swap_red_blue(rgb_image)
                preview_image = rgb_image
                if config.preview_color_space.lower() == "bgr":
                    preview_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
                cv2.imshow("RealSense RGB", preview_image)
                cv2.imshow("RealSense Depth", depth_vis)
                key = cv2.waitKey(1) & 0xFF
                if cv2.getWindowProperty("RealSense RGB", cv2.WND_PROP_VISIBLE) < 1:
                    LOGGER.info("Preview window closed by user")
                    return output_dir
                if key == ord("s"):
                    break
                if key in (ord("q"), 27):
                    LOGGER.info("Capture canceled by user")
                    return output_dir
        else:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not depth_frame or not color_frame:
                raise RuntimeError("Failed to capture depth or color frame")

            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

        depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        intrinsics = _intrinsics_to_dict(
            color_frame.profile.as_video_stream_profile().get_intrinsics()
        )
        camera_model = profile.get_device().get_info(rs.camera_info.name)
        serial = profile.get_device().get_info(rs.camera_info.serial_number)

        rgb_path = output_dir / "rgb.png"
        depth_path = output_dir / "depth.png"
        bgr_path = output_dir / "rgb_bgr.png"
        metadata_path = output_dir / "metadata.json"

        saved_rgb = normalize_to_rgb(color_image, config.color_format)
        if config.software_white_balance:
            saved_rgb = apply_gray_world_white_balance(saved_rgb)
        if config.swap_rb:
            saved_rgb = swap_red_blue(saved_rgb)
        cv2.imwrite(str(rgb_path), saved_rgb)
        if config.save_bgr_copy:
            saved_bgr = cv2.cvtColor(saved_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(bgr_path), saved_bgr)
        cv2.imwrite(str(depth_path), depth_image)

        metadata = build_metadata(
            capture_id=capture_id,
            camera_model=camera_model,
            serial=serial,
            rgb_shape=color_image.shape,
            depth_shape=depth_image.shape,
            depth_scale=depth_scale,
            intrinsics=intrinsics,
            notes=config.notes,
            color_space=config.color_format.lower(),
            software_white_balance=config.software_white_balance,
            swap_rb=config.swap_rb,
        )
        metadata_path.write_text(json.dumps(metadata, indent=2))

        LOGGER.info("Saved rgb to %s", rgb_path)
        LOGGER.info("Saved depth to %s", depth_path)
        LOGGER.info("Saved metadata to %s", metadata_path)
    finally:
        if config.preview:
            cv2.destroyAllWindows()
        pipeline.stop()

    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Capture a single RGB/depth frame from RealSense"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save capture outputs",
    )
    parser.add_argument(
        "--serial", type=str, default=None, help="Optional device serial"
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Show live preview; press 's' to save or 'q' to quit",
    )
    parser.add_argument(
        "--preview-color-space",
        choices=["rgb", "bgr"],
        default="bgr",
        help="Color space used for preview window",
    )
    parser.add_argument(
        "--auto-exposure",
        action="store_true",
        default=True,
        help="Enable auto exposure (default: true)",
    )
    parser.add_argument(
        "--manual-exposure",
        action="store_true",
        help="Disable auto exposure and use manual exposure/gain",
    )
    parser.add_argument("--exposure", type=int, default=None, help="Manual exposure")
    parser.add_argument("--gain", type=int, default=None, help="Manual gain")
    parser.add_argument("--notes", type=str, default=None, help="Capture notes")
    parser.add_argument(
        "--warmup-frames",
        type=int,
        default=15,
        help="Frames to skip for auto exposure/white balance",
    )
    parser.add_argument(
        "--save-bgr-copy",
        action="store_true",
        default=True,
        help="Also save rgb_bgr.png for debugging color space",
    )
    parser.add_argument(
        "--no-save-bgr-copy",
        action="store_true",
        help="Disable saving rgb_bgr.png",
    )
    parser.add_argument(
        "--auto-white-balance",
        action="store_true",
        default=True,
        help="Enable auto white balance (default: true)",
    )
    parser.add_argument(
        "--manual-white-balance",
        action="store_true",
        help="Disable auto white balance and use manual setting",
    )
    parser.add_argument(
        "--white-balance",
        type=int,
        default=None,
        help="Manual white balance (Kelvin)",
    )
    parser.add_argument(
        "--lock-white-balance",
        action="store_true",
        help="Lock white balance after warmup frames",
    )
    parser.add_argument(
        "--color-format",
        choices=["rgb", "bgr"],
        default="rgb",
        help="Color stream format from camera",
    )
    parser.add_argument(
        "--software-white-balance",
        action="store_true",
        help="Apply gray-world white balance before saving",
    )
    parser.add_argument(
        "--swap-rb",
        action="store_true",
        help="Swap red/blue channels before saving",
    )
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = build_parser()
    args = parser.parse_args()

    auto_exposure = args.auto_exposure and not args.manual_exposure
    auto_white_balance = args.auto_white_balance and not args.manual_white_balance

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or Path("data/raw/realsense") / f"capture_{timestamp}"

    LOGGER.info("Capturing to %s", output_dir)
    capture_single(
        output_dir=output_dir,
        config=CaptureConfig(
            auto_exposure=auto_exposure,
            exposure=args.exposure,
            gain=args.gain,
            preview=args.preview,
            notes=args.notes,
            warmup_frames=args.warmup_frames,
            save_bgr_copy=bool(args.save_bgr_copy and not args.no_save_bgr_copy),
            auto_white_balance=auto_white_balance,
            white_balance=args.white_balance,
            lock_white_balance=args.lock_white_balance,
            color_format=args.color_format,
            software_white_balance=args.software_white_balance,
            swap_rb=args.swap_rb,
            preview_color_space=args.preview_color_space,
        ),
        device_serial=args.serial,
    )


if __name__ == "__main__":
    main()
