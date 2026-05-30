from greenwaste.realsense_capture import build_metadata


def test_build_metadata_fields():
    metadata = build_metadata(
        capture_id="capture_001",
        camera_model="Intel RealSense D415",
        serial="123456",
        rgb_shape=(480, 640, 3),
        depth_shape=(480, 640),
        depth_scale=0.001,
        intrinsics={"fx": 1.0, "fy": 1.0, "ppx": 0.0, "ppy": 0.0},
    )

    assert metadata["capture_id"] == "capture_001"
    assert metadata["camera_model"] == "Intel RealSense D415"
    assert metadata["serial"] == "123456"
    assert metadata["resolution_rgb"] == {"width": 640, "height": 480}
    assert metadata["resolution_depth"] == {"width": 640, "height": 480}
    assert metadata["depth_scale"] == 0.001
    assert "timestamp_utc" in metadata
